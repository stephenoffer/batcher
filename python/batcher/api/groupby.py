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
    """An in-progress grouped aggregation, produced by `Dataset.group_by`.

    Not constructed directly: `Dataset.group_by(*keys)` returns one, holding the
    chosen group keys (and any derived keys). It is a builder with a single
    finisher — call `agg` with the named aggregates to get back a new `Dataset`
    whose columns are the group keys followed by those aggregates. Like everything
    in the API it is lazy; no work runs until the resulting `Dataset` hits a
    terminal operation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "v": [1, 2, 3]})
            >>> ds.group_by("g").agg(total=bt.col("v").sum()).sort("g").to_pydict()
            {'g': ['a', 'b'], 'total': [3, 3]}
    """

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

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict(
                ...     {"dept": ["eng", "eng", "sales"], "salary": [100, 120, 90]}
                ... )
                >>> ds.group_by("dept").agg(
                ...     total=bt.col("salary").sum(), n=bt.count()
                ... ).sort("dept").to_pydict()
                {'dept': ['eng', 'sales'], 'total': [220, 90], 'n': [2, 1]}
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
        plan = Aggregate(self._source._plan, group_keys, specs, watermark=self._source._watermark)
        return self._source._derive(plan)
