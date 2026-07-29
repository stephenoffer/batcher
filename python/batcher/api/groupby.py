"""`GroupBy` — an in-progress grouped aggregation produced by `Dataset.group_by`.

A `GroupBy` is coupled to `Dataset`: `Dataset.group_by()` returns one, and
`GroupBy.agg()` returns a new `Dataset`. To avoid an import cycle, `Dataset` is
only referenced for typing here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NoReturn

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.api.dataset.compat.guidance import groupby_attribute_error
from batcher.plan.expr_ir import AggExpr, Col, Expr
from batcher.plan.expr_ir.selectors import Selector
from batcher.plan.logical import Aggregate, AggregateSpec, Project, Projection

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["GroupBy"]


def _numeric_columns(schema: pa.Schema) -> set[str]:
    """The names in `schema` with an integer/float/decimal Arrow type.

    One pass over the fields, rather than a name lookup per candidate column: the caller
    is filtering *every* non-key column of the relation, so a per-column `get_field_index`
    made the default reduction target list quadratic in the width of a wide table.
    """
    return {
        f.name
        for f in schema
        if pa.types.is_integer(f.type)
        or pa.types.is_floating(f.type)
        or pa.types.is_decimal(f.type)
    }


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

    def __iter__(self) -> NoReturn:
        """Reject iteration, pointing at the relational spelling instead.

        pandas lets you loop over ``(key, sub_frame)`` pairs. Batcher cannot: that
        materializes one frame per distinct key in the control plane, which is
        ``O(groups)`` Python over data the engine is meant to reduce in Rust — and it
        caps the operation at a single machine.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "b"], "v": [1, 2]})
                >>> try:
                ...     iter(ds.group_by("g"))
                ... except Exception as exc:
                ...     print(type(exc).__name__)
                PlanError

        Raises:
            PlanError: Always.
        """
        raise PlanError(
            "a GroupBy is not iterable — looping over groups would pull every group to the "
            "driver. Aggregate instead: .agg(total=col('x').sum()); use "
            ".window(partition_by=[...]) to compute per-group values while keeping every "
            "row; or .map_groups(fn) to run a Python function on each group, which does the "
            "same work inside the workers."
        )

    def __getattr__(self, name: str) -> Any:
        """Raise an `AttributeError` that names the window or aggregate to use instead.

        Only reached when normal lookup fails. A pandas/Spark migrant reaches for a
        per-group operation Batcher spells differently (``gb.transform``, ``gb.apply``,
        ``gb.cumcount``, ``gb.get_group``), so the traceback carries the mapping — see
        `batcher.api.dataset.compat.guidance`.

        Args:
            name: The attribute name that was not found.

        Raises:
            AttributeError: Always, with guidance for `name`.
        """
        # Dunder and private probes (copy/pickle/inspect) must fail plainly.
        if name.startswith("_"):
            raise AttributeError(name)
        raise groupby_attribute_error(self, name)

    @property
    def keys(self) -> list[str]:
        """The grouping key column names, in order.

        Returns:
            The positional key names followed by any named key expressions.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"g": ["a"], "v": [1]}).group_by("g").keys
                ['g']
        """
        return [*self._keys, *self._named]

    def agg(self, *aggs: AggExpr | dict[str, Any], **named: AggExpr | Expr) -> Dataset:
        """Compute aggregates per group, returning a new `Dataset`.

        A single positional ``dict`` is the pandas spelling:
        ``agg({"salary": "sum"})`` names the output after the column, and
        ``agg({"salary": ["min", "max"]})`` suffixes each reducer
        (``salary_min``, ``salary_max``), as pandas does when it flattens.

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
                ``name`` as the output column, or a single pandas-style
                ``{column: reducer}`` / ``{column: [reducers]}`` dict.
            **named: Output column name to an aggregate, or an expression over aggregates.

        Returns:
            A new lazy `Dataset` of the group keys followed by the aggregates.

        Raises:
            PlanError: If no aggregate is given, or a value neither is nor contains an
                aggregate expression.
        """
        if len(aggs) == 1 and isinstance(aggs[0], dict):
            return self.agg(**{**self._spec_to_aggs(aggs[0]), **named})
        resolved = {**self._named_aggs(aggs), **named}
        if not resolved:
            raise PlanError("agg() requires at least one aggregate")
        return self._source._derive(self._lower_aggregates(resolved))

    def map_groups(self, fn: Callable, **options: Any) -> Dataset:
        """Apply a Python function to each group as one whole batch.

        `fn` receives a `pyarrow.RecordBatch` holding every row of one group, in the
        source's column order, and returns the batch that replaces it — the per-entity
        shape that `agg` cannot express: building a user's session sequence, fitting a
        curve per device, ranking a document's chunks. Groups may return different row
        counts, and the results are concatenated.

        This is the operation `map_batches` after a `group_by` looks like but is not.
        `map_batches` sees whatever batches the engine produces, and a group spans several
        of them, so a callback written that way runs on fragments and returns a wrong
        answer rather than an error. `map_groups` first reduces each group to one row with
        a **mergeable** aggregate, which is what makes it produce exactly one call per
        group however many partitions the input has.

        The cost is that one group is materialized at a time, so a single key holding
        hundreds of millions of rows needs a reduction (`agg`) or a window rather than a
        callback. Row order *within* a group is not guaranteed; sort inside `fn` when the
        function depends on it. And because the result is a `map_batches` above a
        relational breaker, whether `collect(distributed=True)` accepts the plan is the same
        question as for ``group_by(...).agg(...).map_batches(fn)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import pyarrow as pa
                >>> ds = bt.from_pydict({"k": ["a", "b", "a"], "v": [1, 10, 3]})
                >>> def spread(group):
                ...     v = group.column("v").to_pylist()
                ...     return {"k": [group.column("k")[0].as_py()], "spread": [max(v) - min(v)]}
                >>> out = ds.group_by("k").map_groups(spread, output_columns=["k", "spread"])
                >>> out.sort("k").to_pydict()
                {'k': ['a', 'b'], 'spread': [2, 0]}

        Args:
            fn: Called once per group with that group's rows as a `pyarrow.RecordBatch`.
            **options: `map_batches` options for the per-group stage, such as
                ``output_columns``, ``batch_format``, ``num_gpus``, or ``concurrency``.

        Returns:
            A new lazy `Dataset` holding what `fn` returned for each group, concatenated.

        Raises:
            PlanError: if every column is a group key, leaving nothing to hand `fn`.
        """
        from batcher.api.group_apply import build_map_groups

        if self._named:
            raise PlanError(
                "map_groups needs plain column keys; group_by was given a derived key "
                f"({sorted(self._named)}). Add the derived column with with_columns first, "
                "then group by its name."
            )
        return build_map_groups(self._source, self._keys, fn, options)

    def _spec_to_aggs(self, spec: dict[str, Any]) -> dict[str, AggExpr]:
        """Expand a pandas ``{column: "sum"}`` / ``{column: ["min", "max"]}`` agg spec.

        A single reducer keeps the source column's name (``{"v": "sum"}`` → ``v``);
        a list disambiguates with a suffix (``{"v": ["min", "max"]}`` → ``v_min``,
        ``v_max``), which is how pandas names the flattened result.
        """
        out: dict[str, AggExpr] = {}
        available = self._source._plan.available_columns()
        for column, reducers in spec.items():
            if column not in available:
                raise PlanError(f"agg(): unknown column {column!r}; available: {sorted(available)}")
            names = [reducers] if isinstance(reducers, str) else list(reducers)
            for fn in names:
                reducer = getattr(Col(column), fn, None)
                if reducer is None or not callable(reducer):
                    raise PlanError(
                        f"agg(): {fn!r} is not an aggregate; try 'sum', 'mean', 'min', "
                        "'max', 'count', 'median', 'std', 'var', or 'n_unique'"
                    )
                out[column if len(names) == 1 else f"{column}_{fn}"] = reducer()
        return out

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

    def nunique(self, *columns: str | Selector) -> Dataset:
        """Count distinct values per group — the pandas ``nunique`` spelling of :meth:`n_unique`.

        Args:
            *columns: Columns (names or selectors) to reduce; defaults to every
                non-key column.

        Returns:
            A new `Dataset` of the group keys followed by the per-group distinct counts.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 1, 5]})
                >>> ds.group_by("g").nunique().sort("g").to_pydict()
                {'g': ['a', 'b'], 'x': [1, 1]}
        """
        return self._reduce("n_unique", columns)

    def size(self, name: str = "size") -> Dataset:
        """Count the rows in each group — the pandas ``size`` spelling of :meth:`len`.

        Args:
            name: Name of the output count column.

        Returns:
            A new `Dataset` of the group keys followed by the per-group row count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"]})
                >>> ds.group_by("g").size().sort("g").to_pydict()
                {'g': ['a', 'b'], 'size': [2, 1]}
        """
        return self.len(name)

    def first(self, *columns: str | Selector, order_by: str | Expr) -> Dataset:
        """The first value of each column per group, along an explicit `order_by`.

        `order_by` is required, and deliberately so: a relation has no inherent row
        order, so a "first" without one would return whichever row a morsel happened
        to reach first — an answer that changes between runs and between one machine
        and a cluster. pandas and Polars leave this implicit; Batcher does not.

        Args:
            *columns: Columns (names or selectors) to reduce; defaults to every
                non-key column.
            order_by: The column or expression defining "first" within each group.

        Returns:
            A new `Dataset` of the group keys followed by each column's first value.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a"], "t": [2, 1], "v": [20, 10]})
                >>> ds.group_by("g").first("v", order_by="t").to_pydict()
                {'g': ['a'], 'v': [10]}
        """
        return self._ordered_reduce("first", columns, order_by)

    def last(self, *columns: str | Selector, order_by: str | Expr) -> Dataset:
        """The last value of each column per group, along an explicit `order_by`.

        `order_by` is required for the same reason as in :meth:`first`: without a
        defined order, "last" is whichever row arrived last, which is not a result.

        Args:
            *columns: Columns (names or selectors) to reduce; defaults to every
                non-key column.
            order_by: The column or expression defining "last" within each group.

        Returns:
            A new `Dataset` of the group keys followed by each column's last value.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a"], "t": [2, 1], "v": [20, 10]})
                >>> ds.group_by("g").last("v", order_by="t").to_pydict()
                {'g': ['a'], 'v': [20]}
        """
        return self._ordered_reduce("last", columns, order_by)

    def head(self, n: int = 5, *, order_by: str | Expr) -> Dataset:
        """The first `n` rows of each group along `order_by`, keeping every column.

        The pandas ``groupby().head(n)`` idiom, and unlike the reducers it returns
        *rows*, not aggregates. It lowers to a `row_number` window partitioned by the
        group keys plus a filter, so it stays one streaming pass and is identical
        single-node and distributed.

        `order_by` is required for the same reason as in :meth:`first`: without a
        defined order, "the first n" is whichever rows a morsel reached first.

        Args:
            n: How many rows to keep per group.
            order_by: The column or expression defining the order within each group.

        Returns:
            A new `Dataset` of the surviving rows, with the original columns.

        Raises:
            PlanError: If `n` is less than 1.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "v": [2, 1, 3]})
                >>> ds.group_by("g").head(1, order_by="v").sort("g").to_pydict()
                {'g': ['a', 'b'], 'v': [1, 3]}
        """
        return self._ranked_rows(n, order_by, descending=False)

    def tail(self, n: int = 5, *, order_by: str | Expr) -> Dataset:
        """The last `n` rows of each group along `order_by`, keeping every column.

        The pandas ``groupby().tail(n)`` idiom. Implemented as :meth:`head` over the
        reversed order, so it carries the same streaming and distribution properties.

        Args:
            n: How many rows to keep per group.
            order_by: The column or expression defining the order within each group.

        Returns:
            A new `Dataset` of the surviving rows, with the original columns.

        Raises:
            PlanError: If `n` is less than 1.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "v": [2, 1, 3]})
                >>> ds.group_by("g").tail(1, order_by="v").sort("g").to_pydict()
                {'g': ['a', 'b'], 'v': [2, 3]}
        """
        return self._ranked_rows(n, order_by, descending=True)

    def _ranked_rows(self, n: int, order_by: str | Expr, *, descending: bool) -> Dataset:
        """Keep the `n` lowest- (or highest-) ranked rows per group, via `row_number`."""
        if n < 1:
            raise PlanError(f"group_by().head()/tail(): n must be >= 1, got {n}")
        if self._named:
            raise PlanError(
                "group_by().head()/tail() needs plain column keys — a derived key "
                "(group_by(k=expr)) has no column on the input rows to partition by. "
                "Add the derived key with with_columns() first, then group by its name."
            )
        rank = "__bc_group_rank"
        ranked = self._source.window(
            partition_by=list(self._keys),
            order_by=[(order_by, descending)] if isinstance(order_by, str) else [order_by],
            functions={rank: "row_number"},
        )
        return ranked.filter(Col(rank) <= n).drop(rank)

    def _ordered_reduce(
        self, fn: str, columns: tuple[str | Selector, ...], order_by: str | Expr
    ) -> Dataset:
        """Reduce with an order-dependent aggregate (`first`/`last`) along `order_by`."""
        targets = self._resolve_columns(columns, numeric_only=False)
        if not targets:
            raise PlanError(
                f"group_by().{fn}() has no value columns to reduce; name them explicitly"
            )
        key = Col(order_by) if isinstance(order_by, str) else order_by
        specs = tuple(AggregateSpec(c, getattr(Col(c), fn)(key)) for c in targets)
        return self._finish(specs)

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

    def product(self, *columns: str | Selector) -> Dataset:
        """Product of each value column per group (every non-key numeric column by default).

        Args:
            *columns: Columns (names or selectors) to reduce; defaults to every
                non-key numeric column.

        Returns:
            A new `Dataset` of the group keys followed by the per-group products.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [2, 3, 5]})
                >>> ds.group_by("g").product().sort("g").to_pydict()
                {'g': ['a', 'b'], 'x': [6.0, 5.0]}
        """
        return self._reduce("product", columns)

    def array_agg(self, *columns: str | Selector) -> Dataset:
        """Collect each value column's values into a list per group (all non-key by default).

        The group-wise ``array_agg`` / ``list`` aggregate — gather each group's values into a
        `List` column, e.g. to build a per-entity sequence of features for a model. Values
        appear in input order.

        Args:
            *columns: Columns (names or selectors) to collect; defaults to every non-key
                column.

        Returns:
            A new `Dataset` of the group keys followed by a `List` column per collected column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 3]})
                >>> ds.group_by("g").array_agg().sort("g").to_pydict()
                {'g': ['a', 'b'], 'x': [[1, 2], [3]]}
        """
        return self._reduce("array_agg", columns)

    def mode(self, *columns: str | Selector) -> Dataset:
        """The most frequent value of each column per group (all non-key columns by default).

        Args:
            *columns: Columns (names or selectors) to reduce; defaults to every non-key column.

        Returns:
            A new `Dataset` of the group keys followed by the per-group modes.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "a", "b"], "x": [5, 5, 7, 9]})
                >>> ds.group_by("g").mode().sort("g").to_pydict()
                {'g': ['a', 'b'], 'x': [5, 9]}
        """
        return self._reduce("mode", columns)

    def skewness(self, *columns: str | Selector) -> Dataset:
        """Sample skewness of each column per group (every non-key numeric column by default).

        Args:
            *columns: Columns (names or selectors) to reduce; defaults to every non-key
                numeric column.

        Returns:
            A new `Dataset` of the group keys followed by the per-group skewness.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "a"], "x": [1.0, 2.0, 3.0]})
                >>> ds.group_by("g").skewness().to_pydict()
                {'g': ['a'], 'x': [0.0]}
        """
        return self._reduce("skewness", columns)

    def kurtosis(self, *columns: str | Selector) -> Dataset:
        """Sample excess kurtosis of each column per group (numeric columns by default).

        Args:
            *columns: Columns (names or selectors) to reduce; defaults to every non-key
                numeric column.

        Returns:
            A new `Dataset` of the group keys followed by the per-group kurtosis.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "a", "a"], "x": [1.0, 2.0, 3.0, 4.0]})
                >>> round(ds.group_by("g").kurtosis().to_pydict()["x"][0], 2)
                -1.2
        """
        return self._reduce("kurtosis", columns)

    # Reductions whose default (all non-key columns) is restricted to numeric columns,
    # mirroring pandas' `numeric_only`: averaging or summing a string column is an error,
    # so an explicit-columns call is required to attempt it.
    _NUMERIC_ONLY = frozenset(
        {"sum", "mean", "median", "std", "var", "product", "skewness", "kurtosis"}
    )

    def _value_columns(self, numeric_only: bool) -> list[str]:
        """The default reduction targets: non-key columns, optionally numeric ones only."""
        keys = set(self._keys) | set(self._named)
        cols = [c for c in self._source._plan.available_columns() if c not in keys]
        if not numeric_only:
            return cols
        schema = self._source._plan.available_schema()
        if schema is None:
            return cols  # can't tell types; fall back to all and let the engine judge
        # A column the schema does not know about is kept rather than pre-filtered out,
        # matching what the per-column check did on an unknown name.
        known = {f.name for f in schema.arrow}
        numeric = _numeric_columns(schema.arrow)
        return [c for c in cols if c in numeric or c not in known]

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
