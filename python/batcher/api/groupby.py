"""`GroupBy` — an in-progress grouped aggregation produced by `Dataset.group_by`.

A `GroupBy` is coupled to `Dataset`: `Dataset.group_by()` returns one, and
`GroupBy.agg()` returns a new `Dataset`. To avoid an import cycle, `Dataset` is
only referenced for typing here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import AggExpr, Col, Expr
from batcher.plan.expr_ir.selectors import Selector
from batcher.plan.logical import Aggregate, AggregateSpec, Project, Projection

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["GroupBy"]


def _is_numeric(schema: object, column: str) -> bool:
    """Whether `column` has an integer/float/decimal Arrow type in `schema`."""
    import pyarrow as pa

    idx = schema.get_field_index(column)  # type: ignore[attr-defined]
    if idx < 0:
        return True  # unknown to the schema — don't pre-filter it out
    t = schema.field(idx).type  # type: ignore[attr-defined]
    return pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_decimal(t)


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
        """Hold the source dataset and grouping keys until `agg` finishes the aggregation."""
        self._source = source
        self._keys = keys
        self._named = named or {}

    def __repr__(self) -> str:
        """Show the grouping keys, e.g. ``GroupBy(keys=['region', 'day'])``."""
        keys = [*self._keys, *self._named]
        return f"GroupBy(keys={keys!r})"

    def agg(self, *aggs: AggExpr, **named: AggExpr | Expr) -> Dataset:
        """Compute aggregates per group, returning a new `Dataset`.

        Keyword args bind an output name to an aggregate (`col("x").sum()`,
        `count()`, ...). A positional arg is a bare single-column aggregate
        (``col("x").sum()``) that keeps its source column's name — use a keyword when
        you want a different output name. The result columns are the group keys
        followed by the aggregates, in the order given.

        A keyword value may also be a whole **expression over aggregates** —
        ``col("x").sum() / col("y").sum()``, ``col("v").max() - col("v").min()`` — not
        just a single one. The engine still runs one mergeable aggregate pass; the
        surrounding arithmetic is computed in a projection over the aggregated columns, so
        the result is identical single-node and distributed. Aggregates cannot be nested
        (``sum(x).mean()``).

        For the common case of reducing every value column the same way, prefer the
        shortcut methods (`sum`, `mean`, `count`, ...) over spelling out `agg`.

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

                >>> ds.group_by("dept").agg(
                ...     avg=bt.col("salary").sum() / bt.count()
                ... ).sort("dept").to_pydict()
                {'dept': ['eng', 'sales'], 'avg': [110.0, 90.0]}

        Args:
            *aggs: Bare single-column aggregates (``col(name).<agg>()``) that keep
                ``name`` as the output column.
            **named: Output column name to an aggregate, or an expression over aggregates.

        Returns:
            A new lazy `Dataset` of the group keys followed by the aggregates.

        Raises:
            PlanError: If no aggregate is given, or a value neither is nor contains an
                aggregate expression.
        """
        resolved = {**self._named_aggs(aggs), **named}
        if not resolved:
            raise PlanError("agg() requires at least one aggregate")
        return self._source._derive(self._lower_aggregates(resolved))

    def _group_key_projections(self) -> tuple[Projection, ...]:
        """The group-by output columns: positional keys by name, plus named key exprs."""
        return tuple(Projection(k, Col(k)) for k in self._keys) + tuple(
            Projection(alias, expr) for alias, expr in self._named.items()
        )

    def _lower_aggregates(self, resolved: dict[str, AggExpr | Expr]):
        """Lower ``{alias: aggregate-or-expression-over-aggregates}`` to a logical plan.

        A pure aggregate becomes one `AggregateSpec`. An expression *over* aggregates has
        its aggregate leaves hoisted into hidden columns (deduplicated) and the surrounding
        scalar expression re-evaluated in a following `Project`. When every output is a
        bare aggregate the projection is skipped — the plan shape is exactly as before.
        """
        from batcher.plan.expr_ir.walk import (
            AggregateLeafRegistry,
            contains_aggregate,
            split_aggregate_leaves,
        )

        registry = AggregateLeafRegistry()
        pure_specs: list[AggregateSpec] = []
        project_items: list[Projection] = []
        has_composite = False
        for alias, value in resolved.items():
            if isinstance(value, AggExpr):
                pure_specs.append(AggregateSpec(alias, value))
                project_items.append(Projection(alias, Col(alias)))
            elif isinstance(value, Expr) and contains_aggregate(value):
                has_composite = True
                project_items.append(Projection(alias, split_aggregate_leaves(value, registry)))
            else:
                raise PlanError(
                    f"agg() value for {alias!r} must be an aggregate expression or an "
                    "expression over aggregates, e.g. col('x').sum() or "
                    "col('x').sum() / col('y').sum()"
                )

        group_keys = self._group_key_projections()
        watermark = self._source._watermark
        if not has_composite:
            return Aggregate(self._source._plan, group_keys, tuple(pure_specs), watermark=watermark)

        hidden = tuple(AggregateSpec(name, agg) for name, agg in registry.leaves())
        agg_plan = Aggregate(
            self._source._plan, group_keys, tuple(pure_specs) + hidden, watermark=watermark
        )
        passthrough = tuple(Projection(k.alias, Col(k.alias)) for k in group_keys)
        return Project(agg_plan, passthrough + tuple(project_items))

    def len(self, name: str = "len") -> Dataset:
        """Count the rows in each group.

        The grouped analogue of ``len(ds)``: one integer column of per-group row
        counts (not per-column non-null counts — for those use
        ``agg(n=col("x").count())``).

        Args:
            name: The output column name for the count.

        Returns:
            A new `Dataset` of the group keys followed by the row-count column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "v": [1, 2, 3]})
                >>> ds.group_by("g").len().sort("g").to_pydict()
                {'g': ['a', 'b'], 'len': [2, 1]}
        """
        return self._finish((AggregateSpec(name, AggExpr("count_star", None)),))

    def count(self, *columns: str | Selector) -> Dataset:
        """Count non-null values of each column per group (all non-key columns by default).

        The per-column complement of `len`: `len` counts rows, `count` counts the
        non-null entries of each value column (SQL ``COUNT(col)``, like pandas
        ``groupby().count()``). Name columns or pass a selector to count a subset.

        Args:
            *columns: Columns (names or selectors) to count; defaults to every
                non-key column.

        Returns:
            A new `Dataset` of the group keys followed by the per-group non-null counts.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, None, 3]})
                >>> ds.group_by("g").count().sort("g").to_pydict()
                {'g': ['a', 'b'], 'x': [1, 1]}
        """
        return self._reduce("count", columns)

    def quantile(self, q: float, *columns: str | Selector) -> Dataset:
        """The `q`-quantile of each column per group (every non-key numeric column by default).

        Args:
            q: The quantile to compute, in ``[0, 1]`` (``0.5`` is the median).
            *columns: Columns (names or selectors) to reduce; defaults to every
                non-key numeric column.

        Returns:
            A new `Dataset` of the group keys followed by the per-group quantiles.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "a", "b"], "x": [1.0, 2.0, 3.0, 9.0]})
                >>> ds.group_by("g").quantile(0.5).sort("g").to_pydict()
                {'g': ['a', 'b'], 'x': [2.0, 9.0]}
        """
        if not 0.0 <= q <= 1.0:
            raise PlanError(f"quantile q must be in [0, 1], got {q}")
        targets = self._resolve_columns(columns, numeric_only=True)
        if not targets:
            raise PlanError(
                "group_by().quantile() has no numeric value columns to reduce — "
                "name the columns to reduce explicitly"
            )
        specs = tuple(AggregateSpec(c, Col(c).quantile(q)) for c in targets)
        return self._finish(specs)

    def sum(self, *columns: str | Selector) -> Dataset:
        """Sum each value column per group (every non-key numeric column by default).

        Like pandas' ``numeric_only``, the no-argument form reduces only numeric
        columns; name a non-numeric column explicitly to attempt to sum it.

        Args:
            *columns: Columns (names or selectors) to reduce; defaults to every
                non-key numeric column.

        Returns:
            A new `Dataset` of the group keys followed by the per-group sums, each
            keeping its source column name.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 3], "y": [10, 20, 30]})
                >>> ds.group_by("g").sum().sort("g").to_pydict()
                {'g': ['a', 'b'], 'x': [3, 3], 'y': [30, 30]}
        """
        return self._reduce("sum", columns)

    def mean(self, *columns: str | Selector) -> Dataset:
        """Average each value column per group (every non-key numeric column by default).

        Args:
            *columns: Columns (names or selectors) to reduce; defaults to every
                non-key numeric column.

        Returns:
            A new `Dataset` of the group keys followed by the per-group means.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1.0, 3.0, 8.0]})
                >>> ds.group_by("g").mean().sort("g").to_pydict()
                {'g': ['a', 'b'], 'x': [2.0, 8.0]}
        """
        return self._reduce("mean", columns)

    def min(self, *columns: str | Selector) -> Dataset:
        """Minimum of each value column per group (all non-key columns by default).

        Args:
            *columns: Columns (names or selectors) to reduce; defaults to every
                non-key column.

        Returns:
            A new `Dataset` of the group keys followed by the per-group minima.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [3, 1, 2]})
                >>> ds.group_by("g").min().sort("g").to_pydict()
                {'g': ['a', 'b'], 'x': [1, 2]}
        """
        return self._reduce("min", columns)

    def max(self, *columns: str | Selector) -> Dataset:
        """Maximum of each value column per group (all non-key columns by default).

        Args:
            *columns: Columns (names or selectors) to reduce; defaults to every
                non-key column.

        Returns:
            A new `Dataset` of the group keys followed by the per-group maxima.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [3, 1, 2]})
                >>> ds.group_by("g").max().sort("g").to_pydict()
                {'g': ['a', 'b'], 'x': [3, 2]}
        """
        return self._reduce("max", columns)

    def median(self, *columns: str | Selector) -> Dataset:
        """Median of each value column per group (every non-key numeric column by default).

        Args:
            *columns: Columns (names or selectors) to reduce; defaults to every
                non-key numeric column.

        Returns:
            A new `Dataset` of the group keys followed by the per-group medians.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "a", "b"], "x": [1, 3, 5, 9]})
                >>> ds.group_by("g").median().sort("g").to_pydict()
                {'g': ['a', 'b'], 'x': [3.0, 9.0]}
        """
        return self._reduce("median", columns)

    def n_unique(self, *columns: str | Selector) -> Dataset:
        """Count distinct values of each column per group (all non-key columns by default).

        Args:
            *columns: Columns (names or selectors) to reduce; defaults to every
                non-key column.

        Returns:
            A new `Dataset` of the group keys followed by the per-group distinct counts.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 1, 5]})
                >>> ds.group_by("g").n_unique().sort("g").to_pydict()
                {'g': ['a', 'b'], 'x': [1, 1]}
        """
        return self._reduce("n_unique", columns)

    def std(self, *columns: str | Selector) -> Dataset:
        """Sample standard deviation per group (every non-key numeric column by default).

        Args:
            *columns: Columns (names or selectors) to reduce; defaults to every
                non-key numeric column.

        Returns:
            A new `Dataset` of the group keys followed by the per-group standard deviations.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b", "b"], "x": [1.0, 3.0, 10.0, 10.0]})
                >>> ds.group_by("g").std().sort("g").to_pydict()
                {'g': ['a', 'b'], 'x': [1.4142135623730951, 0.0]}
        """
        return self._reduce("std", columns)

    def var(self, *columns: str | Selector) -> Dataset:
        """Sample variance of each column per group (every non-key numeric column by default).

        Args:
            *columns: Columns (names or selectors) to reduce; defaults to every
                non-key numeric column.

        Returns:
            A new `Dataset` of the group keys followed by the per-group variances.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b", "b"], "x": [1.0, 3.0, 10.0, 10.0]})
                >>> ds.group_by("g").var().sort("g").to_pydict()
                {'g': ['a', 'b'], 'x': [2.0, 0.0]}
        """
        return self._reduce("var", columns)

    # Reductions whose default (all non-key columns) is restricted to numeric columns,
    # mirroring pandas' `numeric_only`: averaging or summing a string column is an error,
    # so an explicit-columns call is required to attempt it.
    _NUMERIC_ONLY = frozenset({"sum", "mean", "median", "std", "var"})

    def _value_columns(self, numeric_only: bool) -> list[str]:
        """The default reduction targets: non-key columns, optionally numeric ones only."""
        keys = set(self._keys) | set(self._named)
        cols = [c for c in self._source._plan.available_columns() if c not in keys]
        if not numeric_only:
            return cols
        schema = self._source._plan.available_schema()
        if schema is None:
            return cols  # can't tell types; fall back to all and let the engine judge
        arrow = schema.arrow
        return [c for c in cols if _is_numeric(arrow, c)]

    def _resolve_columns(
        self, columns: tuple[str | Selector, ...], numeric_only: bool
    ) -> list[str]:
        if not columns:
            return self._value_columns(numeric_only)
        keys = set(self._keys) | set(self._named)
        out: list[str] = []
        for c in columns:
            if isinstance(c, Selector):
                out.extend(
                    n
                    for n in c.matched_columns(
                        self._source._plan.available_columns(),
                        self._source._plan.available_schema(),
                    )
                    if n not in keys
                )
            else:
                out.append(c)
        return out

    def _reduce(self, fn: str, columns: tuple[str | Selector, ...]) -> Dataset:
        targets = self._resolve_columns(columns, numeric_only=fn in self._NUMERIC_ONLY)
        if not targets:
            hint = (
                "no numeric value columns to reduce"
                if not columns and fn in self._NUMERIC_ONLY
                else "no value columns to reduce; every column is a group key"
            )
            raise PlanError(f"group_by().{fn}() has {hint} — name the columns to reduce explicitly")
        specs = tuple(AggregateSpec(c, getattr(Col(c), fn)()) for c in targets)
        return self._finish(specs)

    def _named_aggs(self, aggs: tuple[AggExpr, ...]) -> dict[str, AggExpr]:
        """Resolve bare positional aggregates to an ordered {source_column: agg} map."""
        out: dict[str, AggExpr] = {}
        for a in aggs:
            if isinstance(a, AggExpr) and isinstance(a.input, Col):
                if a.input.name in out:
                    raise PlanError(
                        f"agg() got two positional aggregates over column {a.input.name!r}, "
                        "which would both be named after it; give one a keyword name, "
                        "e.g. agg(total=col('x').sum(), avg=col('x').mean())"
                    )
                out[a.input.name] = a
            else:
                raise PlanError(
                    "a positional agg() argument must be a single-column aggregate that "
                    "names its output, e.g. col('x').sum(); for a custom name or a "
                    "count()/multi-column aggregate use a keyword (agg(total=...))"
                )
        return out

    def _finish(self, specs: tuple[AggregateSpec, ...]) -> Dataset:
        plan = Aggregate(
            self._source._plan,
            self._group_key_projections(),
            specs,
            watermark=self._source._watermark,
        )
        return self._source._derive(plan)
