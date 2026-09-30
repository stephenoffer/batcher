"""The cost gates the aggregate-through-join pushdown rules consult.

Each gate is a pure question about a candidate push (does it pay, or does something veto it)
and none rewrites a plan; the rules in `agg_pushdown.rules` ask them. `_reduces_enough` stays
beside the rules, where tests patch it. Layer: kyber (optimizer).
"""

from __future__ import annotations

import pyarrow as pa

from batcher._internal.logging import note_suppressed
from batcher.kyber.pass_base import OptimizerContext
from batcher.plan.expr_ir import Col
from batcher.plan.logical import Aggregate, Join, JoinOutputCol, LogicalPlan

__all__: list[str] = []


#: The reduction a push must clear when it also moves string group keys off the hot path
#: (`_moves_string_keys`). Lower than `_MIN_PREAGG_REDUCTION` because the push saves more than
#: the rows it removes: the join no longer gathers the unique side's strings for every input
#: row, and the outer aggregate no longer row-encodes them for every input row.
_STRING_KEY_MIN_REDUCTION = 2.0

#: How much coarser than the pushed aggregate the outer one may group for `_moves_string_keys`
#: still to pay: the outer aggregate must keep at least 1/`_OUTER_GROUP_SHARE` of its groups.
_OUTER_GROUP_SHARE = 4.0


def _moves_string_keys(
    ctx: OptimizerContext, node: Aggregate, join: Join, pushed: Aggregate
) -> bool:
    """Whether the push is worth it because the group keys the unique side supplies are strings.

    `_reduces_enough` needs a *measured* 8x, and a group count over keys from two relations is
    never measured before the push has run, which it cannot do until the gate lets it. The
    estimate is also biased the wrong way for this shape: it combines the keys' distinct
    counts as though every pair occurred, so it cannot see that each customer buys in only
    some years. TPC-DS q4's `(ss_customer_sk, d_year)` estimates 1.29M groups over 2.75M rows
    (2.1x) against a true 190,587 (14x).

    What licenses the push without a measurement is what it saves besides rows. Grouping
    `customer ⋈ facts` by `c_customer_id, c_first_name, ...` gathers seven strings through the
    join and row-encodes them into the group hash for *every fact row*. Pre-aggregating by the
    integer join key does that work once per group instead. Measured on q4's `year_total`
    branch at sf1: 96 ms -> 69 ms. So this accepts a push when at least one outer group key is
    a string from the unique side, every pushed key has a distinct count, and the product of
    those counts is still at most half the side's rows.
    """
    right_aliases = {o.alias for o in join.output if o.side == "right"}
    schema = node.input.available_schema()
    if schema is None:
        return False
    if not any(
        isinstance(k.expr, Col)
        and k.expr.name in right_aliases
        and k.expr.name in schema.names
        and _is_string(schema.field(k.expr.name).type)
        for k in node.group_keys
    ):
        return False
    side = ctx.estimator.estimate(join.left)
    if any(
        side.columns.get(k.alias) is None or side.columns[k.alias].ndv is None
        for k in pushed.group_keys
    ):
        return False
    # The *product* of the keys' distinct counts, not the estimator's group count: the
    # estimator damps the product on the assumption that keys are correlated, which is how
    # TPC-DS q23's `(ss_item_sk, d_date)` -- 18K items x 1,461 dates, nearly one row per pair
    # -- read as a 2x reduction, and the push ran 205 -> 329 ms for grouping that removed
    # nothing. Independent keys can form at most the product, so a product that still halves
    # the side is a reduction the correlation assumption cannot have invented.
    groups = 1.0
    for k in pushed.group_keys:
        groups *= max(1.0, side.columns[k.alias].ndv)
    if groups * _STRING_KEY_MIN_REDUCTION > side.rows:
        return False
    # The saving is the outer aggregate's string work, which is only large when that aggregate
    # groups about as finely as the pushed one. q4's `year_total` groups by customer and year,
    # so the push replaces a string-keyed grouping of every fact row with an integer-keyed one.
    # `lineitem JOIN orders GROUP BY o_orderpriority` groups into 5: its outer aggregate is
    # already cheap, and the push only adds a 1.5M-group aggregate over 6M rows (26 ms -> 95 ms,
    # against DuckDB's 33). So the outer grouping must keep at least a quarter of the groups.
    outer = ctx.estimator.estimate(node).rows
    return outer * _OUTER_GROUP_SHARE >= groups


def _is_string(dtype: pa.DataType) -> bool:
    return pa.types.is_string(dtype) or pa.types.is_large_string(dtype)


def _join_out_reduces_more(ctx: OptimizerContext, pushed: LogicalPlan, join: Join) -> bool:
    """Whether the join already reduces harder than the pre-aggregate would — a veto.

    :data:`_MIN_PREAGG_REDUCTION` prices the push as a *ratio* against the side being shrunk,
    blind to what the join then does with it. A selective join is the stronger reducer, and
    pre-aggregating in front of one is pure added work: the group-by still reads every source
    row, and the join it feeds emits fewer rows than the group-by produced. TPC-H Q17 motivates
    this — lineitem pre-aggregated 6M rows to 201,152 groups (29.8x, past the ratio gate) to feed
    a join against 195 parts emitting 5,514 rows, taking it from **12 ms to 242 ms**. A **veto,
    not a license**, as :func:`_measured_as_non_reducing` is: `_reduces_enough` must approve
    first, so this only withdraws it — hence reading the join estimate at any provenance.
    """
    join_rows = ctx.estimator.estimate(join).rows
    if join_rows <= 0:
        return False
    return ctx.estimator.estimate(pushed).rows >= join_rows


def _global_aggregate_gains_nothing(
    ctx: OptimizerContext, node: Aggregate, source: LogicalPlan, join: Join
) -> bool:
    """Veto: the outer aggregate is **global**, and the join does not multiply what it reads.

    :func:`_reduces_enough` prices the push as a row-reduction *ratio* and
    :func:`_join_out_reduces_more` vetoes when the join out-reduces it. Neither asks the one
    question that decides an ungrouped aggregate: **what the plan would have cost without the
    push.** For a global aggregate that cost is a streaming reduction with `O(1)` state — the
    cheapest thing the engine can do — so replacing it with a *grouped* hash aggregate over the
    same rows is strictly more work unless the join multiplies them.

    Measured on H2O `x JOIN medium USING (id2)` (10 M probe rows, a unique 10,000-row build
    side). Varying only which column the aggregate reads, so only the push changes:

    ==========================================  =========  ==================
    ``SELECT ... FROM x JOIN medium``            time       pushed?
    ==========================================  =========  ==================
    ``count(v2)``  (right column)                 7.7 ms    no
    ``count(id2)`` (join key)                    19.3 ms    yes, **2.5x**
    ``sum(v1)``    (left column)                 24.2 ms    yes, **3.1x**
    ==========================================  =========  ==================

    The reduction here is ~1000x (10 M rows to 10,000 groups), so it clears
    :data:`_MIN_PREAGG_REDUCTION` by two orders of magnitude and is still a pessimization: the
    push pays a full 10 M-row grouped hash aggregate to save a broadcast probe that was nearly
    free, and the global reduction it replaces never built a hash table at all.

    So the gate is `join_rows > source_rows` — does the join *amplify*? That is the condition
    eager aggregation has always actually needed: it pays when the join duplicates the side being
    pre-aggregated, because then collapsing rows early avoids paying for the copies. When the
    join emits no more rows than it reads, there is nothing downstream to save.

    Two consequences worth stating, because both look like bugs and are not:

    * It effectively withdraws :func:`pre_aggregation_through_join` from *global* aggregates,
      since that rule requires the other side to be **unique on the join key** for correctness —
      which is exactly the condition that makes the join unable to amplify. The rule keeps
      firing for grouped aggregates, where the outer aggregate builds a hash table either way.
    * It does not cost the distributed path the classic "pre-aggregate before the shuffle" win.
      A global aggregate is already `partial → combine` (`.claude/rules/rust-engine.md`), so
      every partition reduces to one row before any exchange; there is no shuffle volume left
      for an eager push to remove.

    A **veto, not a license**: the gates above must approve first, so this only ever withdraws.
    It therefore reads the join estimate at any provenance, as :func:`_join_out_reduces_more`
    does and for the same reason.
    """
    if node.group_keys:
        return False
    source_rows = ctx.estimator.estimate(source).rows
    join_rows = ctx.estimator.estimate(join).rows
    if source_rows <= 0 or join_rows <= 0:
        return False
    return join_rows <= source_rows


def _measured_as_non_reducing(ctx: OptimizerContext, node: Aggregate) -> bool:
    """Whether a *past run* measured this group-by as collapsing almost nothing.

    `record_group_reduction` records every group-by's real `groups / input_rows` against the
    aggregate's own signature, and `learned_partial_agg` reads it back — but nothing called the
    reader, so the measurement was written on every query and consumed by none. This closes
    that loop at the one decision the ratio exists to inform: pre-aggregation pays exactly when
    a group-by collapses many rows into few groups, and is wasted work when nearly every row is
    its own group.

    Deliberately a **veto, not a license**. `_reduces_enough` above still has to approve the
    push from the estimator's `ndv`; a measurement can only *withdraw* that approval, never
    grant it. So a stale or unlucky measurement can at worst skip a beneficial rewrite — it can
    never introduce one the cost model rejected. Absent a measurement (`None`, the cold case)
    nothing changes, so a first run is byte-for-byte the previous behavior.

    Result-invariant either way: engaging or skipping the partial pre-aggregate is an algebraic
    identity, so this moves only how much work the join's left side does.
    """
    if ctx.hub is None:
        return False
    try:
        from batcher.kyber.learned_tuning import learned_partial_agg
        from batcher.kyber.signature import plan_signature

        return learned_partial_agg(ctx.hub, plan_signature(node)) is False
    except Exception as exc:  # pragma: no cover - a learned read must never break planning
        note_suppressed("kyber", "read measured non-reducing evidence", exc)
        return False


def _outer_aggregate_collapses(
    node: Aggregate, join: Join, out_map: dict[str, JoinOutputCol], keep_sources: list[str]
) -> bool:
    """Whether every pushed group lands in its own outer group, so the outer merge is identity.

    The pushed aggregate groups by the left side's group keys plus the join keys, and the
    right side is unique on its join key, so each pushed group meets exactly one right row.
    If the outer group keys name every one of those pushed keys (the join key either through
    the left column or through the right column it equals), distinct pushed groups are
    distinct outer groups, and the remaining outer keys are right columns the join key
    determines. TPC-DS q23's `GROUP BY substr(i_item_desc), i_item_sk, d_date` over
    `store_sales ⋈ item` is the shape: it groups 1.66M rows by a 30-character string to find
    the 52 with `count(*) > 4`, where grouping by `(ss_item_sk, d_date)` first finds the same
    52 on integers and joins `item` for those alone.
    """
    named: set[str] = set()
    for key in node.group_keys:
        col = out_map.get(key.expr.name) if isinstance(key.expr, Col) else None
        if col is None:
            continue
        if col.side == "left":
            named.add(col.name)
        else:
            for lk, rk in zip(join.left_keys, join.right_keys, strict=True):
                if col.name == rk:
                    named.add(lk)
    return set(keep_sources) <= named


def _commuted(join: Join) -> Join:
    """`join` with its inputs swapped; same output columns, same names, same rows."""
    flip = {"left": "right", "right": "left"}
    return Join(
        join.right,
        join.left,
        join.right_keys,
        join.left_keys,
        join.join_type,
        tuple(JoinOutputCol(flip[o.side], o.name, o.alias) for o in join.output),
        join.strategy,
    )
