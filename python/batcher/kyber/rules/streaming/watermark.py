"""Pushdown through the watermark-bounded streaming operators.

`WatermarkDedup` and `WatermarkStreamJoin` are streaming-only nodes executed by the
driver rather than lowered to the Rust IR, and until now every optimizer rule simply
walked past them: `dispatch` optimizes their *inputs* separately and the nodes
themselves are opaque. That left the two operators whose cost is dominated by
**retained state** as the two the optimizer could not improve.

Pushdown matters more here than anywhere in batch. Below a `Filter`, a batch rewrite
saves CPU on rows that were going to be discarded anyway. Below one of these, it
shrinks the *seen-key set* or the *join buffer* — memory held for the lifetime of a
query that never ends. A stream whose state does not fit is not slow; it fails.

Every rule here is gated on a proof that the rewrite is semantics-preserving, stated in
its docstring. The dangerous direction is specific and worth naming: `WatermarkDedup`
keeps the **first** row per key, so a filter that can reject some rows of a key but not
others changes *which* row is first, and therefore changes the output. Only a predicate
that is constant across a key group may cross it.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase, RuleCategory
from batcher.plan.expr_ir.walk import referenced_columns, remap_columns
from batcher.plan.logical import Filter, LogicalPlan, WatermarkDedup, WatermarkStreamJoin

__all__ = [
    "push_filter_into_stream_join_sides",
    "push_filter_through_watermark_dedup",
]


@rule(
    name="push_filter_through_watermark_dedup",
    phase=Phase.PUSHDOWN,
    matches=(Filter,),
    category=RuleCategory.REWRITE,
)
def push_filter_through_watermark_dedup(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Push a key-constant filter below a streaming dedup, shrinking its seen-key state.

    `WatermarkDedup` emits the **first** row it sees for each `subset` key and remembers
    that key until the watermark forgets it. Filtering after the dedup discards rows the
    dedup has already paid to remember; filtering before means the rejected keys never
    enter the state at all — the state is bounded by the *surviving* key count rather
    than the total one.

    The rewrite is only sound when the predicate cannot disagree between two rows that
    share a key. If it could, moving it below would let a row that the dedup would have
    suppressed become the first surviving row for its key, changing the output. A
    predicate over `subset` columns alone is constant per key by construction, so it is
    safe; anything referencing a non-key column is refused.

    The `event_time` column is deliberately *not* treated as safe: two rows of one key
    differ in event time (that is the point), so an event-time predicate is exactly the
    reordering hazard above.

    Args:
        node: The `Filter` under consideration.
        _ctx: The optimizer context (unused — this rewrite is purely structural).

    Returns:
        The rewritten plan with the filter below the dedup, or None to leave it alone.
    """
    dedup = node.input
    if not isinstance(dedup, WatermarkDedup):
        return None
    if not referenced_columns(node.predicate) <= set(dedup.subset):
        return None
    pushed = Filter(dedup.input, node.predicate)
    return WatermarkDedup(
        pushed,
        subset=dedup.subset,
        event_time=dedup.event_time,
        lateness_micros=dedup.lateness_micros,
    )


@rule(
    name="push_filter_into_stream_join_sides",
    phase=Phase.PUSHDOWN,
    matches=(Filter,),
    category=RuleCategory.REWRITE,
)
def push_filter_into_stream_join_sides(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Push a single-side predicate into that side of a stream-stream interval join.

    A `WatermarkStreamJoin` buffers both sides until the watermark proves no future row
    can match. A predicate applied above the join filters rows that have already been
    buffered, matched, and emitted; applied below, the rejected rows never occupy the
    buffer. Under a stream that is the difference between bounded and unbounded memory,
    not a constant factor of CPU.

    The join is an inner join, so a row filtered from one side can only remove output
    rows — never add or alter one — which makes pushing a side-pure predicate down
    equivalent to applying it above. The predicate must reference columns from exactly
    one side: a cross-side predicate is a join condition, not a pushable filter, and a
    constant predicate has nothing to gain from moving.

    Column identity comes from the join's own `output` mapping, which records each
    output column's `side` and pre-rename `name`; a predicate is attributed to a side
    only when *every* column it names resolves to that one side.

    Args:
        node: The `Filter` under consideration.
        _ctx: The optimizer context (unused — this rewrite is purely structural).

    Returns:
        The rewritten plan with the filter inside one join side, or None.
    """
    join = node.input
    if not isinstance(join, WatermarkStreamJoin):
        return None

    used = referenced_columns(node.predicate)
    if not used:
        return None

    by_alias = {o.alias: o for o in join.output}
    sides = set()
    renames: dict[str, str] = {}
    for name in used:
        out_col = by_alias.get(name)
        if out_col is None:
            return None  # not a join output — leave it where it is
        sides.add(out_col.side)
        renames[name] = out_col.name

    if len(sides) != 1:
        return None  # a cross-side predicate is a join condition, not a filter

    # The predicate is phrased in the join's output aliases; rewrite it into the side's
    # own column names before attaching it below the rename.
    predicate = remap_columns(node.predicate, renames)
    if sides.pop() == "left":
        return WatermarkStreamJoin(
            left=Filter(join.left, predicate),
            right=join.right,
            left_keys=join.left_keys,
            right_keys=join.right_keys,
            output=join.output,
            left_time=join.left_time,
            right_time=join.right_time,
            within_micros=join.within_micros,
            lateness_micros=join.lateness_micros,
        )
    return WatermarkStreamJoin(
        left=join.left,
        right=Filter(join.right, predicate),
        left_keys=join.left_keys,
        right_keys=join.right_keys,
        output=join.output,
        left_time=join.left_time,
        right_time=join.right_time,
        within_micros=join.within_micros,
        lateness_micros=join.lateness_micros,
    )
