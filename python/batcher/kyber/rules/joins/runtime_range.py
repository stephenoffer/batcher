"""Runtime join filters — the key range one side of a join implies about the other.

Split from `rewrites` on the seam between changing a join's shape and narrowing its inputs:
this module only ever *adds* a superset filter beneath a join, and owns the table of which
side each join type may narrow (`_FILTERABLE_SIDES`), which the sideways-information-passing
family in `extra.runtime_filters` reads from the package rather than restates.

Registration order is within-phase run order, so `joins/__init__` imports this module
directly after `rewrites`, the position `runtime_join_filter` held at the end of that file.
"""

from __future__ import annotations

import dataclasses

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase, RuleCategory
from batcher.plan.expr_ir import Col, Expr, Lit, referenced_columns, remap_columns
from batcher.plan.logical import Filter, Join, LogicalPlan, Project
from batcher.plan.stats import ColumnStat, RelStats, ambiguous_float_bound

__all__ = ["runtime_join_filter"]


# --- Runtime join filters (sideways information passing) --------------------

# Which side(s) of each join type may be safely reduced by the *other* side's key
# range. "left"/"right" name the side that receives the filter. A side is filterable
# only when its unmatched rows are not required in the output: an outer join's
# preserved side and an anti join's left side must keep their unmatched rows.
_FILTERABLE_SIDES = {
    "inner": ("left", "right"),
    "semi": ("left", "right"),  # semi emits a left row only if it matches → both prunable
    "anti": ("right",),  # left rows without a match MUST survive; only prune the right
    "left": ("right",),  # left rows are preserved; only prune the right
    "right": ("left",),  # right rows are preserved; only prune the left
    # "full" preserves both sides → nothing is safely prunable.
}


@rule(
    name="runtime_join_filter",
    phase=Phase.ENFORCE,
    matches=(Join,),
    category=RuleCategory.ENFORCE,
)
def runtime_join_filter(node: Join, ctx: OptimizerContext) -> LogicalPlan | None:
    """Push a `key BETWEEN other_min AND other_max` filter onto a prunable join side.

    For an equi-join every matching row has equal keys, so a row whose key falls
    outside the *other* side's `[min, max]` range can never match. Pushing that range
    onto the opposite input is a superset filter — it drops only provably-non-matching
    rows, never a real match — the cheap form of the sideways-information-passing /
    bloom-filter join pruning DuckDB and Spark AQE rely on, available here purely from
    the `ColumnStat.min`/`max` Kyber already propagates. When the prunable side is a
    scan, the added `Filter` is captured by `required_predicates_per_source` at
    lowering and pushed to the source, so zonemaps prune whole row-groups / Hive
    partitions — dynamic partition pruning with no new IR node.

    Multi-key joins are handled per key: a matching row must fall inside the other
    side's range on **every** key, so each narrowing key contributes a `BETWEEN`
    conjunct (`k1 BETWEEN .. AND k2 BETWEEN ..`). Runs once in ENFORCE (after physical
    selection) so it never re-adds, and fires only when bounds are known *and*
    genuinely narrower (so the filter prunes rather than adds overhead), on a side the
    join does not preserve.
    """
    sides = _FILTERABLE_SIDES.get(node.join_type)
    if sides is None or not node.left_keys or len(node.left_keys) != len(node.right_keys):
        return None
    left_stats = ctx.estimator.estimate(node.left)
    right_stats = ctx.estimator.estimate(node.right)

    # Per side, collect a BETWEEN conjunct for every key the opposite range narrows.
    right_preds: list[Expr] = []
    left_preds: list[Expr] = []
    for lk, rk in zip(node.left_keys, node.right_keys, strict=True):
        left_col = left_stats.column(lk)
        right_col = right_stats.column(rk)
        if "right" in sides and _narrows(left_col, right_col):
            right_preds.append(_between(rk, left_col))
        if "left" in sides and _narrows(right_col, left_col):
            left_preds.append(_between(lk, right_col))

    new_left, new_right = node.left, node.right
    changed = False
    if right_preds:
        placed = _place_key_range(new_right, _conjoin(right_preds), right_stats, ctx)
        changed = placed is not None
        new_right = placed or new_right
    if left_preds:
        placed = _place_key_range(new_left, _conjoin(left_preds), left_stats, ctx)
        changed = changed or placed is not None
        new_left = placed or new_left
    if not changed:
        return None
    ctx.notes.setdefault("runtime_join_filters", []).append(node.join_type)
    return Join(
        new_left,
        new_right,
        node.left_keys,
        node.right_keys,
        node.join_type,
        node.output,
        node.strategy,
    )


# A key range is attached at all only when it is estimated to discard more than this share of
# the side's rows. `_narrows` asks only that the other side's range be strictly inside, which a
# range one key short of the whole domain satisfies: JOB q13a's `movie_id >= 2` over
# `movie_info` kept 14,834,457 of 14,835,720 rows and cost 100 ms of CPU to keep them, plus a
# compaction of every batch. A filter has to remove rows to pay for the pass and the copy.
_MAX_KEPT = 0.8

# A key range sinks beneath the side's own filters only when it is estimated to discard at least
# this share of the rows there. Below it, the range runs over every row instead of the
# survivors and its stacked `Filter` compacts each batch once more, which a range that keeps
# most rows does not repay.
_SINK_MAX_KEPT = 0.5


def _place_key_range(
    side: LogicalPlan, pred: Expr, stats: RelStats, ctx: OptimizerContext
) -> LogicalPlan | None:
    """Attach a join's key-range filter to `side`, beneath the side's own filters if it pays.

    `runtime_join_filter` runs in ENFORCE, after the filters a leaf carries were fused and
    split, so wrapping `side` puts the range *above* them: every predicate the leaf already
    had is evaluated over every row, and the integer compare that discards most of them runs
    last, over the survivors. The data plane's `AND` has no short circuit (see
    `extra.filter_split`), but a stacked `Filter` compacts its batch, so the order of stacked
    filters is the order work is saved in. On JOB the leaf is `movie_info` (14.8M rows) under
    `info IN (...nine strings...)`, and the range a 30,000-row `movie_link` implies on
    `movie_id` keeps 3.6% of it; above the `IN`, the string membership test ran on all 14.8M
    rows and was two thirds of the query's operator time.

    So the range descends through the side's `Filter`s and through any `Project` that passes
    its key columns through unchanged, and is attached directly above the first node it
    cannot see through -- usually the `Scan`, where lowering also hands it to the source as a
    pushed predicate. It descends only when it has a filter to get ahead of and is estimated
    selective (`_SINK_MAX_KEPT`); otherwise it wraps `side` as before. A range estimated to
    keep nearly every row (`_MAX_KEPT`) is not attached at all.

    Exact: a filter commutes with another filter (both keep a row only when their predicate
    is TRUE) and with a projection whose items it reads as bare columns. The range is a pair
    of comparisons against literals, which cannot raise, so moving it earlier cannot make a
    query fail that succeeded; it can only spare a later predicate rows it would have seen.

    Args:
        side: The join input the range filters.
        pred: The range predicate, phrased in `side`'s output columns.
        stats: `side`'s estimated statistics, which the range's selectivity is read from.
        ctx: The optimizer context, for its estimator.

    Returns:
        `side` with the range applied, or None when the range would not pay for itself.
    """
    kept = ctx.estimator.expr_selectivity(pred, stats)
    if kept > _MAX_KEPT:
        return None
    passed: list[LogicalPlan] = []
    node, local = side, pred
    while True:
        if isinstance(node, Filter):
            passed.append(node)
            node = node.input
            continue
        if isinstance(node, Project):
            items = {it.alias: it.expr for it in node.items}
            sources = {c: items.get(c) for c in referenced_columns(local)}
            if all(isinstance(e, Col) for e in sources.values()):
                passed.append(node)
                local = remap_columns(local, {c: e.name for c, e in sources.items()})
                node = node.input
                continue
        break
    if kept > _SINK_MAX_KEPT or not any(isinstance(n, Filter) for n in passed):
        return Filter(side, pred)
    out: LogicalPlan = Filter(node, local)
    for above in reversed(passed):
        out = dataclasses.replace(above, input=out)
    return out


def _narrows(source: ColumnStat, target: ColumnStat) -> bool:
    """Whether `source`'s key range is known and strictly inside `target`'s — so a
    `target BETWEEN source.min AND source.max` filter would actually drop rows.

    Both ranges must be known: without the target's spread we cannot tell the filter
    is selective, and adding a non-selective filter is pure overhead.

    An **ambiguous float bound refuses outright**, and this is a soundness gate, not a
    heuristic. The rule's whole licence is that the pushed `BETWEEN` "drops only
    provably-non-matching rows, never a real match" — and on a float key that is false. An
    equi-join *canonicalizes* its key (`bc_runtime::keys` folds `-0.0` into `0.0` and every NaN
    into one value), so the join matches `-0.0` on one side to `0.0` on the other; a `BETWEEN`
    does not canonicalize, and on the engine's total order `-0.0 < 0.0`, so the filter deletes
    precisely that matching row. Joining `k = [-0.0, 1.5, 2.0]` to `k = [0.0, 1.5]` returned one
    row where the join returns two.

    The bug was latent for as long as float join-key bounds were never *fetched* (they weren't:
    `column_bounds_needed` only collected filter columns). It became reachable the moment they
    were, which is the honest reason it is being fixed here and not earlier.
    """
    if source.min is None or source.max is None or target.min is None or target.max is None:
        return False
    if any(ambiguous_float_bound(v) for v in (source.min, source.max, target.min, target.max)):
        return False
    try:
        return source.min > target.min or source.max < target.max
    except TypeError:
        return False  # incomparable bound types → leave the join untouched


def _between(column: str, bounds: ColumnStat) -> Expr:
    """`column >= bounds.min AND column <= bounds.max`."""
    col = Col(column)
    return (col >= Lit(bounds.min)) & (col <= Lit(bounds.max))


def _conjoin(preds: list[Expr]) -> Expr:
    """AND a non-empty list of predicates (a single predicate is returned as-is)."""
    out = preds[0]
    for pred in preds[1:]:
        out = out & pred
    return out
