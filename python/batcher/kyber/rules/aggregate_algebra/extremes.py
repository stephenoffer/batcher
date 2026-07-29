"""An aggregate whose arguments make it the group's minimum or maximum.

Two shapes, both of which a query writes without noticing:

* **`quantile(x, 0)` is `min(x)` and `quantile(x, 1)` is `max(x)`.** The 0th and 100th
  percentiles are the extremes under every interpolation convention, so no assumption
  about the engine's quantile method is involved. The same holds for `approx_quantile`:
  a KLL sketch stores the true minimum and maximum exactly, because they are the first and
  last items of every compacted level, so its rank-0 and rank-1 queries are not
  approximate at all. This is the shape a dashboard emits when a percentile control is
  dragged to either end.
* **`arg_min(x, x)` is `min(x)`.** Asking for the value of `x` at the row where `x` is
  smallest is asking for the smallest `x`; ties cannot disagree, because the candidate rows
  all carry the same value. Generated SQL produces this whenever the measure and the
  ordering key are bound to the same column.

The rewrite is worth far more than the saved comparison. `quantile` materializes the whole
group in a sorted structure and `approx_quantile` allocates a sketch per group;
`min`/`max` carry one scalar and merge by comparison. On a wide group-by that is the
difference between spilling and not.

The type guard on the quantile rules is not optional. `quantile` and `approx_quantile`
always produce `Float64`, while `min` and `max` preserve their input type — so on an
integer column the rewrite would silently change the output column's type. It fires only
when the schema proves the input is already floating point. `arg_min`/`arg_max` need no
guard: they preserve the input type exactly as `min`/`max` do.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.rules.exprs.guards import is_float
from batcher.plan.expr_ir import AggExpr
from batcher.plan.expr_rewrite import expr_key
from batcher.plan.logical import Aggregate, AggregateSpec, LogicalPlan
from batcher.plan.schema import SchemaRef

__all__ = ["extreme_quantile_to_min_max", "self_ordered_arg_extreme_to_min_max"]

#: `(quantile function, parameter) -> the extreme it names`.
_EXTREME_QUANTILES = {0.0: "min", 1.0: "max"}
_QUANTILE_FNS = frozenset({"quantile", "approx_quantile"})
#: `arg_min`/`arg_max` collapse to these when their ordering key is their own value.
_ARG_EXTREMES = {"arg_min": "min", "arg_max": "max"}


def _input_schema(node: Aggregate) -> SchemaRef | None:
    try:
        return node.input.available_schema()
    except Exception:
        return None


def _rebuilt(node: Aggregate, aggregates: list[AggregateSpec]) -> LogicalPlan | None:
    """A copy of `node` carrying `aggregates`, or ``None`` when nothing changed."""
    if all(new is old for new, old in zip(aggregates, node.aggregates, strict=True)):
        return None
    return Aggregate(node.input, node.group_keys, tuple(aggregates), watermark=node.watermark)


@rule(name="extreme_quantile_to_min_max", phase=Phase.REWRITE, matches=(Aggregate,))
def extreme_quantile_to_min_max(node: Aggregate, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`quantile(x, 0) -> min(x)` and `quantile(x, 1) -> max(x)`, likewise for
    `approx_quantile`, on a floating-point column.

    The extreme percentiles are the extremes under every interpolation convention, and a
    KLL sketch stores both exactly rather than approximately, so neither rewrite depends on
    how the engine computes an interior quantile. The `is_float` guard is what keeps the
    output type unchanged: the quantile family always answers `Float64` while `min`/`max`
    preserve their input, so on an integer column this would move the column's type.
    """
    schema = _input_schema(node)
    if schema is None:
        return None
    rewritten: list[AggregateSpec] = []
    for spec in node.aggregates:
        agg = spec.agg
        extreme = _EXTREME_QUANTILES.get(agg.param) if agg.param is not None else None
        if (
            agg.func in _QUANTILE_FNS
            and extreme is not None
            and agg.input is not None
            and is_float(agg.input, schema)
        ):
            rewritten.append(AggregateSpec(spec.alias, AggExpr(extreme, agg.input)))
        else:
            rewritten.append(spec)
    return _rebuilt(node, rewritten)


@rule(name="self_ordered_arg_extreme_to_min_max", phase=Phase.REWRITE, matches=(Aggregate,))
def self_ordered_arg_extreme_to_min_max(
    node: Aggregate, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`arg_min(x, x) -> min(x)` and `arg_max(x, x) -> max(x)`.

    The value at the row where `x` is smallest *is* the smallest `x`. Ties are not a
    hazard here even though `arg_min` breaks them by arrival order: every candidate row
    carries the same `x`, so whichever one wins reports the same value.

    No type guard is needed — `arg_min`/`arg_max` and `min`/`max` all preserve the input
    type — and no null guard either, since all four ignore nulls and answer null for a
    group that has none.
    """
    rewritten: list[AggregateSpec] = []
    for spec in node.aggregates:
        agg = spec.agg
        extreme = _ARG_EXTREMES.get(agg.func)
        if (
            extreme is not None
            and agg.input is not None
            and agg.input2 is not None
            and expr_key(agg.input) == expr_key(agg.input2)
        ):
            rewritten.append(AggregateSpec(spec.alias, AggExpr(extreme, agg.input)))
        else:
            rewritten.append(spec)
    return _rebuilt(node, rewritten)
