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
from batcher.plan.expr_ir import (
    AggExpr,
    Coalesce,
    Col,
    Lit,
    referenced_columns,
    remap_columns,
)
from batcher.plan.logical import (
    Aggregate,
    AggregateSpec,
    Distinct,
    Join,
    JoinOutputCol,
    LogicalPlan,
    Project,
    Projection,
)
from batcher.plan.stats import Provenance

__all__ = [
    "count_distinct_to_distinct_count",
    "eager_aggregation",
    "pre_aggregate_join_measures",
    "pre_aggregation_through_join",
]

# Synthetic column the count-distinct rewrite materializes the distinct input under.
_COUNT_DISTINCT_VALUE = "__count_distinct_value"

# Pushing the *measure* (right) side of a join down: each decomposable aggregate and the
# function that merges its per-key partials. `avg`/`median`/`distinct` are excluded —
# they cannot be recombined from a single partial column.
_MEASURE_MERGE = {"sum": "sum", "count": "sum", "min": "min", "max": "max"}
# On a LEFT join, an unmatched left row has a NULL partial. For `count` the empty result
# must be 0 (count over no rows is 0); for `sum`/`min`/`max` it is NULL (their identity),
# which a SUM/MIN/MAX over the group already produces — so only `count` needs a coalesce.
_MEASURE_ZERO_ON_EMPTY = frozenset({"count"})

# Aggregates idempotent under row duplication — safe to push below any-fan-out join.
_FANOUT_SAFE_AGGS = frozenset({"min", "max"})
# Decomposable aggregates and the function that *merges* their partials. sum/count are
# additive (count of counts is a sum); min/max are idempotent. avg/median/distinct are
# excluded — they can't be re-aggregated from a single partial column.
_PREAGG_MERGE = {"sum": "sum", "count": "sum", "count_star": "sum", "min": "min", "max": "max"}
_ADDITIVE_AGGS = frozenset({"sum", "count", "count_star"})


@rule(name="count_distinct_to_distinct_count", phase=Phase.REWRITE, matches=(Aggregate,))
def count_distinct_to_distinct_count(
    node: Aggregate, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """Rewrite a lone ``COUNT(DISTINCT x)`` group-by into a distinct then a plain count.

    ``Aggregate(group=G, [count_distinct(x) AS a])`` →
    ``Aggregate(group=G, [count(x) AS a])`` over ``Distinct(Project(G…, x AS v))``.

    Deduping the ``(G, x)`` pairs and counting the non-null ``x`` per group equals the
    distinct count, but reuses the radix-parallel distinct + count kernels — which
    parallelize across the distinct *values*. The direct ``count_distinct`` combine
    partitions by ``G``, so a query with few groups but many distinct values per group
    (the common shape) runs on only a handful of cores. ``COUNT(x)`` (not ``COUNT(*)``)
    drops the single ``(G, NULL)`` row a group with null inputs contributes, matching
    SQL's NULL-excluding ``COUNT(DISTINCT)``.

    Restricted to a *lone* ``count_distinct`` (no other aggregate, exact only — not
    ``approx_count_distinct``): mixing it with row-level aggregates would need them to
    see the un-deduped rows, which this single distinct can't serve.
    """
    if len(node.aggregates) != 1:
        return None
    spec = node.aggregates[0]
    if spec.agg.func != "count_distinct" or spec.agg.input is None:
        return None
    # Alias clash with the synthetic value column (vanishingly rare) → leave it alone.
    if any(key.alias == _COUNT_DISTINCT_VALUE for key in node.group_keys):
        return None

    # Inner: project the group keys + the distinct input, then dedup the whole row.
    proj_items = (*node.group_keys, Projection(_COUNT_DISTINCT_VALUE, spec.agg.input))
    deduped = Distinct(Project(node.input, proj_items))
    # Outer: group by the projected key columns, count the now-distinct non-null values.
    group_keys = tuple(Projection(key.alias, Col(key.alias)) for key in node.group_keys)
    aggregates = (AggregateSpec(spec.alias, AggExpr("count", Col(_COUNT_DISTINCT_VALUE))),)
    return Aggregate(deduped, group_keys, aggregates)


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


@rule(name="pre_aggregate_join_measures", phase=Phase.REWRITE, matches=(Aggregate,))
def pre_aggregate_join_measures(node: Aggregate, ctx: OptimizerContext) -> LogicalPlan | None:
    """Pre-aggregate the *measure* (right) side of a join, grouped by the join key.

    The aggregates all reference one join side (the *measure* side); the group keys
    reference the *other* side. Pre-aggregating the measure side by its join key shrinks
    the join's input to one row per key, so the join no longer materialises the full
    fan-out before the group-by. Two canonical shapes:

    * **measure on the right** — `customer LEFT JOIN orders GROUP BY c_custkey,
      COUNT(o_orderkey)` (TPC-H Q13): pre-aggregate orders by `o_custkey`.
    * **measure on the left** — `lineitem JOIN orders GROUP BY o_orderpriority,
      SUM(l_extendedprice*(1-l_discount))` (the operator-mix join→agg): pre-aggregate
      lineitem by `l_orderkey` (the aggregate input may be a *computed expression* over
      the measure side, not just a bare column).

    Correctness for **any** fan-out: pre-aggregating the measure side makes it unique on
    the join key, and the outer aggregate recombines the partials (COUNT→SUM, SUM→SUM,
    MIN→MIN, MAX→MAX) — so duplicating a partial across N rows of the other side and then
    re-summing reproduces the direct result. For a LEFT join an unmatched left row gets a
    NULL partial; only `count` coalesces it to 0 (empty count is 0; empty sum/min/max is
    NULL, which the merge already yields). To keep the LEFT semantics simple the measure
    side must be the **right** (non-preserved) side on a left join. Group keys are plain
    columns of the group side; single equi-key; cost-gated on a measured row reduction
    (also what makes it idempotent — a second push finds the measure side already unique).
    """
    join = node.input
    if not isinstance(join, Join) or join.join_type not in {"inner", "left"} or not node.aggregates:
        return None
    if len(join.left_keys) != 1 or len(join.right_keys) != 1:
        return None  # single equi-key only (conservative)
    out_map = {o.alias: o for o in join.output}
    side_aliases = {
        "left": {a for a, o in out_map.items() if o.side == "left"},
        "right": {a for a, o in out_map.items() if o.side == "right"},
    }

    # Every aggregate must be a decomposable function whose input references exactly one
    # side — that single side is the measure side to pre-aggregate.
    measure_side: str | None = None
    for spec in node.aggregates:
        agg = spec.agg
        if agg.func not in _MEASURE_MERGE or agg.input2 is not None or agg.input is None:
            return None
        cols = referenced_columns(agg.input)
        side = next((s for s in ("left", "right") if cols and cols <= side_aliases[s]), None)
        if side is None:
            return None  # input is constant, or spans both sides
        if measure_side is None:
            measure_side = side
        elif measure_side != side:
            return None  # aggregates reference different sides
    group_side = "left" if measure_side == "right" else "right"

    # Group keys: plain columns of the group side.
    for key in node.group_keys:
        if not isinstance(key.expr, Col) or key.expr.name not in side_aliases[group_side]:
            return None

    # On a LEFT join only the right (non-preserved) side may be pre-aggregated.
    if join.join_type == "left" and measure_side != "right":
        return None

    m_input = join.left if measure_side == "left" else join.right
    m_key = (join.left_keys if measure_side == "left" else join.right_keys)[0]
    alias_to_src = {a: out_map[a].name for a in side_aliases[measure_side]}

    # Partial aggregate on the measure side: group by its join key, the (remapped to
    # source columns) aggregate expressions.
    partials = tuple(
        AggregateSpec(
            f"__pm_{i}", AggExpr(spec.agg.func, remap_columns(spec.agg.input, alias_to_src))
        )
        for i, spec in enumerate(node.aggregates)
    )
    pushed = Aggregate(m_input, (Projection(m_key, Col(m_key)),), partials)

    # Cost gate: fire only on a *measured* reduction of the measure side (a learned `ndv`,
    # not the estimator's default), so a stats-less plan never pushes a pointless
    # group-by and a second push (measure side already unique) is a no-op.
    pushed_stats = ctx.estimator.estimate(pushed)
    if pushed_stats.provenance is Provenance.DEFAULT:
        return None
    if not pushed_stats.rows < ctx.estimator.estimate(m_input).rows:
        return None

    # New join: keep the group side's outputs (the group keys read them); replace the
    # measure side's outputs with the partial-aggregate columns.
    new_output = [o for o in join.output if o.side == group_side]
    new_output += [
        JoinOutputCol(measure_side, f"__pm_{i}", f"__pm_{i}") for i in range(len(partials))
    ]
    if measure_side == "left":
        new_join = Join(
            pushed, join.right, (m_key,), join.right_keys, join.join_type, tuple(new_output)
        )
    else:
        new_join = Join(
            join.left, pushed, join.left_keys, (m_key,), join.join_type, tuple(new_output)
        )

    # A LEFT join's unmatched (left) rows carry a NULL measure-side partial here (measure
    # is always the right side on a left join), so `count` coalesces to 0.
    coalesce_empty = join.join_type == "left"
    final_aggs = tuple(
        AggregateSpec(
            spec.alias,
            AggExpr(
                _MEASURE_MERGE[spec.agg.func],
                Coalesce([Col(f"__pm_{i}"), Lit(0)])
                if coalesce_empty and spec.agg.func in _MEASURE_ZERO_ON_EMPTY
                else Col(f"__pm_{i}"),
            ),
        )
        for i, spec in enumerate(node.aggregates)
    )
    return Aggregate(new_join, node.group_keys, final_aggs)
