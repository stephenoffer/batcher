"""Aggregate-through-join pushdown — pre-aggregate a join side to shrink its input.

`eager_aggregation` pushes a partial **min/max** below an inner join: those are
idempotent under the join's row duplication, so it is correct for *any* fan-out.
`pre_aggregation_through_join` extends this to the additive **sum/count** aggregates,
which are only fan-out-safe when the *other* side is provably unique on the join key
(so each partial row matches exactly one row and is counted once); the final aggregate
then *merges* the partials (sum of partial sums, sum of partial counts).

Both reuse `_right_unique_on_keys` from `joins` (the uniqueness oracle) and are
cost-gated on a measured row reduction — which also makes them idempotent. They are
registered via the `@rule` decorator (auto-discovered on import from
`kyber.rules.__init__`).
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.rules.joins import _right_unique_on_keys
from batcher.plan.expr_ir import AggExpr, Col
from batcher.plan.logical import (
    Aggregate,
    AggregateSpec,
    Join,
    JoinOutputCol,
    LogicalPlan,
    Projection,
)
from batcher.plan.stats import Provenance

__all__ = ["eager_aggregation", "pre_aggregation_through_join"]

# Aggregates idempotent under row duplication — safe to push below any-fan-out join.
_FANOUT_SAFE_AGGS = frozenset({"min", "max"})
# Decomposable aggregates and the function that *merges* their partials. sum/count are
# additive (count of counts is a sum); min/max are idempotent. avg/median/distinct are
# excluded — they can't be re-aggregated from a single partial column.
_PREAGG_MERGE = {"sum": "sum", "count": "sum", "count_star": "sum", "min": "min", "max": "max"}
_ADDITIVE_AGGS = frozenset({"sum", "count", "count_star"})


@rule(name="eager_aggregation", phase=Phase.REWRITE, matches=(Aggregate,))
def eager_aggregation(node: Aggregate, ctx: OptimizerContext) -> LogicalPlan | None:
    """Push a fan-out-safe partial aggregate below an inner join.

    `Aggregate(group=G, [min/max(L.x)])(Join(L, R))` → the same aggregate over
    `Join(Aggregate(L, group=G_L + keys, [min/max]), R)`. Pre-aggregating the join
    side that the aggregate collapses shrinks the join's input. Restricted to
    `min`/`max`, which are idempotent under the join's row duplication, so the
    rewrite is correct for *any* fan-out (no uniqueness assumption needed).

    Cost-gated: fires only when the pushed aggregate actually reduces the side's row
    count (per the estimator's `ndv`) — which both avoids pessimizing a non-reducing
    group-by and makes the rule idempotent (a second push finds no further
    reduction). All aggregate inputs and group keys must be plain columns drawn from
    the left side. Returns None otherwise.
    """
    join = node.input
    if not isinstance(join, Join) or join.join_type != "inner" or not node.aggregates:
        return None
    out_map = {o.alias: o for o in join.output}
    left_aliases = {a for a, o in out_map.items() if o.side == "left"}

    left_group_sources: list[str] = []
    for key in node.group_keys:
        if not isinstance(key.expr, Col) or key.expr.name not in out_map:
            return None  # only plain column group keys, drawn from the join output
        if key.expr.name in left_aliases:
            left_group_sources.append(out_map[key.expr.name].name)

    agg_sources: list[str] = []
    for spec in node.aggregates:
        agg = spec.agg
        if agg.func not in _FANOUT_SAFE_AGGS or not isinstance(agg.input, Col):
            return None
        if agg.input.name not in left_aliases or agg.input2 is not None:
            return None
        src = out_map[agg.input.name].name
        if src in left_group_sources:
            return None  # a column both grouped and aggregated — leave it alone
        agg_sources.append(src)

    # Build the pushed (partial) aggregate on the left input: group by the left group
    # columns plus the join keys (needed for the join), aggregate the same way.
    keep_sources: list[str] = list(dict.fromkeys([*left_group_sources, *join.left_keys]))
    partial_keys = tuple(Projection(s, Col(s)) for s in keep_sources)
    partials = tuple(
        AggregateSpec(f"__eag_{i}", AggExpr(spec.agg.func, Col(src)))
        for i, (spec, src) in enumerate(zip(node.aggregates, agg_sources, strict=True))
    )
    pushed = Aggregate(join.left, partial_keys, partials)

    # Cost gate: fire only on a *measured* reduction (a learned `ndv`, not the
    # estimator's default guess), so a stats-less plan never pushes a pointless
    # group-by and a second push (already reduced) finds no further gain → no-op.
    pushed_stats = ctx.estimator.estimate(pushed)
    if pushed_stats.provenance is Provenance.DEFAULT:
        return None
    if not pushed_stats.rows < ctx.estimator.estimate(join.left).rows:
        return None

    # Rewrite the join output: keep right columns and the left columns the pushed
    # aggregate still provides (group keys / join keys); drop aggregated-away columns;
    # add the partial columns. Then combine the partials in the final aggregate.
    provided = set(keep_sources) | {f"__eag_{i}" for i in range(len(partials))}
    new_output = [o for o in join.output if o.side == "right" or o.name in provided]
    new_output += [JoinOutputCol("left", f"__eag_{i}", f"__eag_{i}") for i in range(len(partials))]
    new_join = Join(
        pushed,
        join.right,
        join.left_keys,
        join.right_keys,
        "inner",
        tuple(new_output),
        join.strategy,
    )
    final_aggs = tuple(
        AggregateSpec(spec.alias, AggExpr(spec.agg.func, Col(f"__eag_{i}")))
        for i, spec in enumerate(node.aggregates)
    )
    return Aggregate(new_join, node.group_keys, final_aggs)


@rule(name="pre_aggregation_through_join", phase=Phase.REWRITE, matches=(Aggregate,))
def pre_aggregation_through_join(node: Aggregate, ctx: OptimizerContext) -> LogicalPlan | None:
    """Push a partial **sum/count** aggregate below an inner join whose other side is
    unique on the join key.

    `Aggregate(group=G, [SUM(L.x), …])(Join(L, R))` → the same aggregate over
    `Join(Aggregate(L, group=G_L+keys, [SUM partial]), R)`, finalised by *merging* the
    partials (SUM of partial sums, SUM of partial counts). Generalises
    `eager_aggregation` (min/max, any fan-out) to the additive aggregates — which are
    only fan-out-safe when **R is provably unique on the join key**, so each partial
    row matches exactly one R row and is counted once. Cost-gated on a measured row
    reduction (also what makes it idempotent). All aggregate inputs and group keys must
    be plain columns from the left side; aggregates must be decomposable
    (sum/count/count_star/min/max) with at least one additive among them (pure min/max
    is `eager_aggregation`'s job). Returns None otherwise.
    """
    join = node.input
    if not isinstance(join, Join) or join.join_type != "inner" or not node.aggregates:
        return None
    out_map = {o.alias: o for o in join.output}
    left_aliases = {a for a, o in out_map.items() if o.side == "left"}

    left_group_sources: list[str] = []
    for key in node.group_keys:
        if not isinstance(key.expr, Col) or key.expr.name not in out_map:
            return None
        if key.expr.name in left_aliases:
            left_group_sources.append(out_map[key.expr.name].name)

    agg_inputs: list[str | None] = []
    for spec in node.aggregates:
        agg = spec.agg
        if agg.func not in _PREAGG_MERGE or agg.input2 is not None:
            return None
        if agg.func == "count_star":
            agg_inputs.append(None)  # counts rows, no input column
            continue
        if not isinstance(agg.input, Col) or agg.input.name not in left_aliases:
            return None
        src = out_map[agg.input.name].name
        if src in left_group_sources:
            return None  # column both grouped and aggregated — leave it alone
        agg_inputs.append(src)
    if not any(spec.agg.func in _ADDITIVE_AGGS for spec in node.aggregates):
        return None  # pure min/max → eager_aggregation handles it (no uniqueness needed)
    if not _right_unique_on_keys(join, ctx):
        return None  # additive partials are only safe without fan-out

    keep_sources = list(dict.fromkeys([*left_group_sources, *join.left_keys]))
    partial_keys = tuple(Projection(s, Col(s)) for s in keep_sources)
    partials = tuple(
        AggregateSpec(
            f"__pre_{i}",
            AggExpr(spec.agg.func, Col(src) if src is not None else None),
        )
        for i, (spec, src) in enumerate(zip(node.aggregates, agg_inputs, strict=True))
    )
    pushed = Aggregate(join.left, partial_keys, partials)

    # Cost gate: fire only on a measured reduction (not the estimator's default guess).
    pushed_stats = ctx.estimator.estimate(pushed)
    if pushed_stats.provenance is Provenance.DEFAULT:
        return None
    if not pushed_stats.rows < ctx.estimator.estimate(join.left).rows:
        return None

    provided = set(keep_sources) | {f"__pre_{i}" for i in range(len(partials))}
    new_output = [o for o in join.output if o.side == "right" or o.name in provided]
    new_output += [JoinOutputCol("left", f"__pre_{i}", f"__pre_{i}") for i in range(len(partials))]
    new_join = Join(
        pushed,
        join.right,
        join.left_keys,
        join.right_keys,
        "inner",
        tuple(new_output),
        join.strategy,
    )
    final_aggs = tuple(
        AggregateSpec(spec.alias, AggExpr(_PREAGG_MERGE[spec.agg.func], Col(f"__pre_{i}")))
        for i, spec in enumerate(node.aggregates)
    )
    return Aggregate(new_join, node.group_keys, final_aggs)
