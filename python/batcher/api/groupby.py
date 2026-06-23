"""`GroupBy` — an in-progress grouped aggregation produced by `Dataset.group_by`.

A `GroupBy` is coupled to `Dataset`: `Dataset.group_by()` returns one, and
`GroupBy.agg()` returns a new `Dataset`. To avoid an import cycle, `Dataset` is
only referenced for typing here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import AggExpr, Col, Expr
from batcher.plan.logical import Aggregate, AggregateSpec, Projection

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["GroupBy"]


class GroupBy:
    """An in-progress grouped aggregation. Finish with `.agg(...)`."""

    __slots__ = ("_keys", "_named", "_source")

    def __init__(
        self, source: Dataset, keys: tuple[str, ...], named: dict[str, Expr] | None = None
    ) -> None:
        self._source = source
        self._keys = keys
        self._named = named or {}

    def agg(self, **named: AggExpr) -> Dataset:
        """Compute named aggregates per group, returning a new `Dataset`.

        Each keyword binds an output column name to an aggregate expression
        (`col("x").sum()`, `count()`, ...). The result columns are the group keys
        followed by the named aggregates.
        """
        if not named:
            raise PlanError("agg() requires at least one named aggregate")
        for alias, agg in named.items():
            if not isinstance(agg, AggExpr):
                raise PlanError(
                    f"agg() value for {alias!r} must be an aggregate expression, "
                    f"e.g. col('x').sum() or count()"
                )
        group_keys = tuple(Projection(k, Col(k)) for k in self._keys) + tuple(
            Projection(alias, expr) for alias, expr in self._named.items()
        )
        specs = tuple(AggregateSpec(alias, agg) for alias, agg in named.items())
        plan = Aggregate(self._source._plan, group_keys, specs)
        return self._source._derive(plan)
