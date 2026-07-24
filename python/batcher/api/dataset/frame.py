"""`Dataset` — the lazy, immutable, fluent entry point.

A `Dataset` is a handle to a `LogicalPlan` plus its bound input relations. Every
operation returns a new `Dataset` (nothing mutates); no work happens until a
terminal operation (`collect`, `to_pydict`, ...). At that point `api` orchestrates
the layers: Kyber optimizes, Carbonite checks feasibility, Core executes.

One obvious way to do each thing: expressions everywhere (no lambdas), `select`
for choosing/deriving the full output, `with_columns` for adding/replacing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, TypeVar, overload

import pyarrow as pa

from batcher._internal.errors import PlanError, require_float, require_int
from batcher.api._join_helpers import (
    _as_expr,
    _as_key_expr,
    _as_str_list,
    _asof_output,
    _broadcast,
    _join_output,
    _resolve_join_keys,
)
from batcher.api.dataset._build import (
    RepartitionSpec,
    build_cast,
    build_distinct,
    build_explode,
    build_pivot,
    build_sample,
    build_unnest,
    build_unpivot,
    build_window,
    build_with_random,
    expand_selector_expr,
    selector_columns,
)
from batcher.api.dataset._nulls import (
    build_drop_nulls,
    build_fill_null,
    build_fill_null_strategy,
)
from batcher.api.dataset._window import (
    build_window_columns,
    windowed_filter,
    windowed_project,
)
from batcher.api.dataset.compat import (
    attribute_error_for,
    build_collect_schema,
    build_first,
    build_glimpse,
    build_info,
    build_item,
    build_iter_rows,
    build_iter_slices,
    build_last,
    build_memory_usage,
)
from batcher.api.dataset.dq import DatasetDQ
from batcher.api.dataset.meta import DatasetMeta
from batcher.api.dataset.ml import DatasetML
from batcher.api.dataset.scd import DatasetSCD
from batcher.api.groupby import GroupBy
from batcher.api.terminal import (
    _collect,
    _count,
    _explain,
    _is_empty,
    _iter_batches,
    _schema,
    _show,
    _stats,
    _to_pandas,
    _to_polars,
    _to_pydict,
    _to_pylist,
)
from batcher.io.source import Source
from batcher.plan.expr_ir import Aliased, Col, Expr
from batcher.plan.expr_ir.selectors import Selector, has_selector
from batcher.plan.expr_rewrite import is_bare_window
from batcher.plan.logical import (
    AsofJoin,
    Distinct,
    Join,
    Limit,
    LogicalPlan,
    Project,
    Projection,
    RowId,
    Sort,
    SortKeySpec,
    Union,
    remap_sources,
)
from batcher.plan.schema import suggest_columns
from batcher.plan.streaming import Watermark

if TYPE_CHECKING:
    from batcher.api.io_namespace import Writer
    from batcher.api.stats import RunStats

__all__ = ["Dataset", "GroupBy"]

# The return of a user function passed to `Dataset.pipe` — `pipe` is transparent.
_T = TypeVar("_T")
# The value of an argument that also has an ecosystem-spelling alias (see `_one_of`).
_V = TypeVar("_V")


def _one_of(primary: _V, alias: _V, primary_name: str, alias_name: str) -> _V:
    """Collapse a Batcher argument and its ecosystem-spelling alias into one value.

    Several methods accept the pandas/Polars name for an argument alongside the
    Batcher one (`sample(frac=)` for `fraction`, `sort(by=)` for the positional
    keys). Passing both is a mistake worth naming rather than silently resolving in
    some undocumented precedence order.
    """
    if primary is not None and alias is not None:
        raise PlanError(
            f"pass {primary_name} or {alias_name}, not both "
            f"({primary_name}={primary!r}, {alias_name}={alias!r})"
        )
    return primary if primary is not None else alias


# The dtype families `select_dtypes` understands, plus every ecosystem spelling that
# unambiguously means one of them: a Python type, a NumPy/Arrow/Polars dtype name.
# Concrete widths map to their family because a relation's column is whatever width
# the engine resolved it to, and a user asking for "int32" means "the integer one".
_DTYPE_FAMILY_ALIASES: dict[Any, str] = {
    int: "integer",
    float: "floating",
    str: "string",
    bool: "boolean",
    "int": "integer",
    "int8": "integer",
    "int16": "integer",
    "int32": "integer",
    "int64": "integer",
    "uint8": "integer",
    "uint16": "integer",
    "uint32": "integer",
    "uint64": "integer",
    "float16": "floating",
    "float32": "floating",
    "float64": "floating",
    "double": "floating",
    "number": "numeric",
    "object": "string",
    "str": "string",
    "utf8": "string",
    "string": "string",
    "large_string": "string",
    "bool": "boolean",
    "boolean": "boolean",
    "date": "temporal",
    "date32": "temporal",
    "date64": "temporal",
    "datetime": "temporal",
    "timestamp": "temporal",
    "datetime64[ns]": "temporal",
}


def _as_family_list(wanted: Any) -> list[Any]:
    """Normalize `select_dtypes`'s argument to a list of family specifications."""
    return list(wanted) if isinstance(wanted, (list, tuple, set)) else [wanted]


def _resolve_dtype_family(family: Any) -> Callable[[], Any]:
    """Resolve one `select_dtypes` family specification to a column-selector factory."""
    from batcher.plan.expr_ir import selectors

    known = {
        "numeric": selectors.numeric,
        "integer": selectors.integer,
        "floating": selectors.floating,
        "string": selectors.string,
        "boolean": selectors.boolean,
        "temporal": selectors.temporal,
    }
    # A family name wins as itself; anything else resolves through the alias table.
    name = family if family in known else _DTYPE_FAMILY_ALIASES.get(family)
    factory = known.get(name)
    if factory is None:
        raise PlanError(
            f"select_dtypes(): cannot resolve {family!r} to a dtype family; expected "
            f"one of {sorted(known)}, a Python type (int/float/str/bool), or a dtype "
            "name such as 'int64'"
        )
    return factory


def _as_opt_str_list(value: str | list[str] | None) -> list[str] | None:
    """Accept a single column name where a list is expected, as pandas does."""
    return [value] if isinstance(value, str) else value


def _unknown_cols(missing: set[str], available: list[str]) -> str:
    """Render an unknown-column list with a 'did you mean' hint for the first miss."""
    ordered = sorted(missing)
    return f"{ordered}{suggest_columns(ordered[0], available)}" if ordered else "[]"


def _empty_projection_message(method: str, positional: tuple[object, ...]) -> str:
    """Explain an empty projection — a selector that matched nothing reads as 'no columns'."""
    selectors = [repr(p) for p in positional if has_selector(p)]
    if selectors:
        return f"{method}(): the column selector(s) {', '.join(selectors)} matched no columns"
    return f"{method}() requires at least one column"


def _reject_sliding_window_key(alias: str, expr: Expr) -> None:
    """Refuse a sliding `window(...)` used directly as a group key.

    A row belongs to *several* overlapping sliding windows, so `window(ts, w, slide)`
    evaluates to the **list** of the starts that contain it. Grouping by that list groups
    by the list — every row whose overlap set happens to be identical lands in one group,
    keyed by an array. It returns rows, and they are wrong: the windows never overlap, so
    a row is counted once instead of once per window it belongs to.

    Exploding first is the whole operation ("one row per window this row is in"), and it
    cannot be inferred: silently fanning the rows out here would change the cardinality of
    a `group_by` under the caller. So reject, and say what to write instead. A tumbling
    window (no `slide`) is a scalar start and groups directly, as it should.
    """
    from batcher.plan.expr_ir.func_nodes import WindowBuckets

    if isinstance(expr, WindowBuckets):
        raise PlanError(
            f"group_by({alias}=window(..., slide=...)): a sliding window puts each row in "
            "several overlapping windows, so the expression is the *list* of their starts "
            "— grouping by it groups by the list, not by the windows. Fan the rows out "
            "first:\n"
            f"    ds.select({alias}=window(ts, '1h', '30m'), ...)"
            f".explode({alias!r}).group_by({alias!r}).agg(...)\n"
            "A tumbling window (no slide) is a single start and can be grouped directly."
        )


class Dataset:
    """A lazy, immutable relation — the fluent entry point to the engine.

    A `Dataset` is a handle to a query plan plus its bound inputs. Construct one
    with a session constructor (`batcher.from_pydict`, `from_arrow`, `read`, …),
    then build it up with transformations (`filter`, `select`, `with_columns`,
    `group_by`, `join`, …). Every transformation is **lazy** and returns a *new*
    `Dataset`; nothing mutates and no work runs until a **terminal** operation
    (`collect`, `to_pydict`, `to_pylist`, `iter_batches`, `write`, `count`, …)
    executes the optimized plan. Expressions (`batcher.col("x") * 2`) describe
    column work that runs in the Rust data plane; per-row Python never enters the
    hot path.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, 2, 3], "g": ["a", "a", "b"]})
            >>> ds.filter(bt.col("x") > 1).select("x").to_pydict()
            {'x': [2, 3]}
    """

    __slots__ = ("_cache", "_plan", "_repartition", "_sources", "_watermark")

    def __init__(
        self,
        plan: LogicalPlan,
        sources: list[Source],
        repartition: RepartitionSpec | None = None,
        watermark: Watermark | None = None,
        cache: bool = False,
    ) -> None:
        """Bind a logical plan to its sources; prefer a session constructor over this."""
        self._plan = plan
        self._sources = sources
        # An optional output-layout hint consumed by `write` (set by `repartition`);
        # transformations drop it (it is a pre-write concern), so it never propagates.
        self._repartition = repartition
        # An event-time watermark set by `with_watermark`; carried through
        # breaker-free transforms so the next `group_by().agg()` can attach it.
        self._watermark = watermark
        # Set by `cache()`: this dataset's collected result is stored in the process
        # result cache. Deliberately *not* propagated by `_derive` — caching marks
        # this exact result; a further transform is a new (uncached) result.
        self._cache = cache

    # --- introspection -----------------------------------------------------
    @property
    def columns(self) -> list[str]:
        """The output column names of the current plan.

        Returns:
            The output column names, in order.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"a": [1], "b": [2]}).columns
                ['a', 'b']
        """
        return self._plan.available_columns()

    @property
    def is_streaming(self) -> bool:
        """Whether any bound source is unbounded (e.g. Kafka, incremental files).

        A streaming dataset cannot be `collect()`-ed (it would never finish); consume
        it incrementally with `iter_batches()` or write it to a sink instead.

        Returns:
            ``True`` if any bound source is unbounded.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3]}).is_streaming
                False
        """
        from batcher.io.source import is_bounded

        return any(not is_bounded(s) for s in self._sources)

    def __repr__(self) -> str:
        """Show the lazy plan's output columns (no execution)."""
        return f"Dataset(columns={self.columns})"

    def _repr_html_(self) -> str:
        """Notebook display: the lazy plan's output columns (no execution).

        A `Dataset` is lazy and possibly unbounded, so the rich repr shows the schema
        rather than silently running the query; call `show()`/`collect()` for data.
        """
        cols = "".join(f"<th>{c}</th>" for c in self.columns)
        return (
            "<div><strong>Dataset</strong> "
            f"<em>(lazy, {len(self.columns)} columns — call .show() to preview)</em>"
            f"<table><thead><tr>{cols}</tr></thead></table></div>"
        )

    @overload
    def __getitem__(self, key: str) -> Expr: ...

    @overload
    def __getitem__(self, key: list[str] | slice | Expr) -> Dataset: ...

    def __getitem__(self, key: str | list[str] | slice | Expr) -> Expr | Dataset:
        """Index sugar: a column `Expr`, a projected `Dataset`, a row slice, or a filter.

        ``ds["x"]`` returns an `Expr`; ``ds[["a", "b"]]`` returns a projected
        `Dataset`; ``ds[:n]`` / ``ds[i:j]`` returns a row slice (like `limit`/`offset`);
        ``ds[ds["a"] > 1]`` returns the rows matching a boolean expression, the
        pandas/Polars ``df[df.a > 1]`` idiom (equivalent to `filter`).

        Args:
            key: A column name, a list of names, a slice, or a boolean `Expr` mask.

        Returns:
            An `Expr` for a single column name, else a projected, sliced, or filtered
            `Dataset`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [1, 2, 3], "b": [4, 5, 6]})
                >>> ds[ds["a"] > 1].to_pydict()
                {'a': [2, 3], 'b': [5, 6]}
                >>> ds[["a"]].to_pydict()
                {'a': [1, 2, 3]}
                >>> ds[:2].to_pydict()
                {'a': [1, 2], 'b': [4, 5]}
        """
        if isinstance(key, str):
            return Col(key)
        # A boolean expression is a row mask: ds[ds["a"] > 1] == ds.filter(...). Checked
        # before `list` so a list *of* expressions is still a projection error, not this.
        if isinstance(key, Expr):
            return self.filter(key)
        if isinstance(key, list):
            return self.select(*key)
        if isinstance(key, slice):
            if key.step not in (None, 1):
                raise PlanError("Dataset slice step is not supported")
            start = key.start or 0
            if start < 0 or (key.stop is not None and key.stop < 0):
                raise PlanError("Dataset slice bounds must be non-negative")
            n = (key.stop - start) if key.stop is not None else None
            sliced = self if start == 0 else self.limit(2**63 - 1, offset=start)
            return sliced if n is None else sliced.limit(n)
        raise PlanError(
            "Dataset index must be a column name, a list of names, a slice, or a "
            "boolean expression (ds[ds['x'] > 0]); got " + type(key).__name__
        )

    def __len__(self) -> int:
        """Row count — ``len(ds)`` is sugar for `count()` (a terminal operation).

        Returns:
            The number of result rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> len(bt.from_pydict({"x": [1, 2, 3]}))
                3
        """
        return self.count()

    def __bool__(self) -> bool:
        """Always raises — the truth value of a lazy `Dataset` is ambiguous.

        Without this, ``if ds:`` would silently fall back to `__len__` and execute a
        full ``count`` just to decide a branch. Ask for what you mean instead:
        `has_rows` / `is_empty` for emptiness, ``ds is not None`` for existence.

        Raises:
            PlanError: Always.
        """
        raise PlanError(
            "the truth value of a Dataset is ambiguous (a lazy plan, not a result); "
            "use ds.has_rows / ds.is_empty() to test for rows, or `ds is not None` "
            "to test that a Dataset was produced"
        )

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        """Iterate the result as Arrow ``RecordBatch``es — ``for batch in ds``.

        Sugar for `iter_batches()`. The unit is a **batch**, never a row: per-row
        Python iteration would touch tuples in the control plane (forbidden), so to
        process individual rows, work on each batch's columns (Arrow/NumPy) instead.
        A terminal, streaming operation.

        Returns:
            An iterator over the result's Arrow record batches.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [1, 2, 3], "b": [4, 5, 6]})
                >>> for batch in ds:
                ...     print(batch.column_names)
                ['a', 'b']
        """
        return iter(self.iter_batches())

    def __arrow_c_stream__(self, requested_schema: object = None) -> object:
        """Export the result over the Arrow **PyCapsule stream interface** (zero-copy).

        This is the bridge out of Batcher: any library that speaks the C Data Interface
        — Polars, DuckDB, pyarrow, pandas, nanoarrow — consumes a `Dataset` directly,
        with no ``to_arrow()`` call and no copy::

            pl.DataFrame(ds)            # Polars
            duckdb.sql("SELECT * FROM ds")
            pa.table(ds)

        The stream is **lazy**: batches are pulled from `iter_batches()` as the consumer
        reads them, so a larger-than-memory result streams into DuckDB rather than
        materializing first. Executing the plan is therefore a side effect of the
        consumer iterating, which makes this a terminal operation.

        Args:
            requested_schema: A schema capsule the consumer would prefer, per the
                protocol. Honoured only when it matches; otherwise the stream's own
                schema is exported (the consumer must then cast).

        Returns:
            An ``ArrowArrayStream`` PyCapsule.
        """
        reader = pa.RecordBatchReader.from_batches(self.schema, self.iter_batches())
        return reader.__arrow_c_stream__(requested_schema)

    def __contains__(self, name: object) -> bool:
        """Column-membership test — ``"x" in ds`` is true if ``x`` is an output column.

        Resolved from the schema with no execution (never a value scan).

        Args:
            name: The candidate column name.

        Returns:
            ``True`` if `name` is an output column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1], "y": [2]})
                >>> "x" in ds
                True
                >>> "z" in ds
                False
        """
        return isinstance(name, str) and name in self.columns

    def __getattr__(self, name: str) -> Any:
        """Raise an `AttributeError` that says what to type instead.

        Only reached when normal lookup fails. A migrant types what they already
        know (``ds.set_index``, ``ds.iterrows``, ``ds.amount``), so the traceback is
        where the mapping onto Batcher's spelling has to live — see
        `batcher.api.dataset.compat.guidance`.

        Args:
            name: The attribute name that was not found.

        Raises:
            AttributeError: Always, with guidance for `name`.
        """
        # Dunder and private lookups must fail plainly: copy/pickle/inspect probe for
        # `__deepcopy__`, `__getstate__`, and friends, and a decorated message here
        # would turn "this object has no custom deepcopy" into a hard error.
        if name.startswith("_"):
            raise AttributeError(name)
        raise attribute_error_for(self, name)

    def __add__(self, other: Dataset) -> Dataset:
        """``ds1 + ds2`` — concatenate rows (UNION ALL). Operator sugar for
        ``union(other)``; use `union` directly for ``distinct=True`` or many inputs.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> a = bt.from_pydict({"x": [1, 2]})
                >>> b = bt.from_pydict({"x": [2, 3]})
                >>> sorted((a + b).to_pydict()["x"])
                [1, 2, 2, 3]
        """
        if not isinstance(other, Dataset):
            return NotImplemented
        return self.union(other)

    def __or__(self, other: Dataset) -> Dataset:
        """``ds1 | ds2`` — concatenate and deduplicate rows (UNION). Operator sugar
        for ``union(other, distinct=True)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> a = bt.from_pydict({"x": [1, 2]})
                >>> b = bt.from_pydict({"x": [2, 3]})
                >>> sorted((a | b).to_pydict()["x"])
                [1, 2, 3]
        """
        if not isinstance(other, Dataset):
            return NotImplemented
        return self.union(other, distinct=True)

    def __and__(self, other: Dataset) -> Dataset:
        """``ds1 & ds2`` — distinct rows in BOTH (SQL INTERSECT). Sugar for `intersect`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> a = bt.from_pydict({"x": [1, 2]})
                >>> b = bt.from_pydict({"x": [2, 3]})
                >>> sorted((a & b).to_pydict()["x"])
                [2]
        """
        if not isinstance(other, Dataset):
            return NotImplemented
        return self.intersect(other)

    def __sub__(self, other: Dataset) -> Dataset:
        """``ds1 - ds2`` — distinct rows in this but not `other` (SQL EXCEPT). Sugar
        for `except_`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> a = bt.from_pydict({"x": [1, 2]})
                >>> b = bt.from_pydict({"x": [2, 3]})
                >>> sorted((a - b).to_pydict()["x"])
                [1]
        """
        if not isinstance(other, Dataset):
            return NotImplemented
        return self.except_(other)

    def _derive(self, plan: LogicalPlan) -> Dataset:
        # Carry the watermark through breaker-free transforms so a `with_watermark`
        # before a `filter`/`select` still reaches the downstream `group_by().agg()`.
        return Dataset(plan, self._sources, watermark=self._watermark)

    def _named_positionals(self, exprs: tuple[Expr, ...]) -> dict[str, Expr]:
        """Resolve self-naming positional expressions to an ordered name -> expr map."""
        out: dict[str, Expr] = {}
        for e in exprs:
            if has_selector(e):
                out.update(expand_selector_expr(self, e))
            elif isinstance(e, Aliased):
                out[e.name] = e.inner
            elif isinstance(e, Col):
                out[e.name] = e
            else:
                raise PlanError(
                    "a positional with_columns() argument must name its output: pass a "
                    "column selector, an aliased expression (expr.alias('total')), or a "
                    "bare col(...); otherwise use a keyword (with_columns(total=expr))"
                )
        return out

    def cache(self) -> Dataset:
        """Mark this dataset's result to be cached in memory after it is computed.

        The first terminal op (``collect`` and friends) on the returned dataset
        executes normally and stores its Arrow result in a process-wide,
        memory-bounded LRU cache keyed by the plan and its inputs; later terminals on
        an equivalent dataset return the cached result without re-executing. The cache
        is bounded by ``memory.result_cache_max_bytes`` and yields its memory back to
        running queries under pressure, so caching never grows the process without
        bound. Like Spark/Polars ``cache``, it marks *this* result; a further
        transform is a new, uncached result. Single-node relational results only.

        Returns:
            A new `Dataset` whose first computed result is cached.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> hot = bt.from_pydict({"k": [1, 1, 2], "v": [10, 20, 30]}).cache()
                >>> hot.collect().num_rows  # computed once, then served from cache
                3
                >>> hot.collect().num_rows  # cache hit
                3
        """
        return Dataset(
            self._plan,
            self._sources,
            repartition=self._repartition,
            watermark=self._watermark,
            cache=True,
        )

    def with_watermark(self, time_col: str, lateness: str) -> Dataset:
        """Declare an event-time watermark on `time_col` (Spark ``withWatermark``).

        `lateness` is how late a row may arrive and still be counted (a fixed
        duration like ``"10m"`` / ``"1h"``). On a windowed streaming aggregation the
        watermark bounds state: once it passes a window's end, that window is emitted
        and evicted, and rows older than the watermark are dropped as late. Carried
        through to the next ``group_by(window(...)).agg(...)``.

        Args:
            time_col: The event-time column the watermark advances on.
            lateness: How late a row may arrive and still count (e.g. ``"10m"``).

        Returns:
            A new `Dataset` carrying the watermark.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"ts": [1, 2, 3], "v": [1, 2, 3]})
                >>> ds.with_watermark("ts", "10m").columns
                ['ts', 'v']
        """
        from batcher.plan.functions.temporal import _duration_micros

        if time_col not in self._plan.available_columns():
            raise PlanError(f"with_watermark(): unknown column {time_col!r}")
        wm = Watermark(time_col, _duration_micros(lateness, arg="watermark lateness"))
        return Dataset(self._plan, self._sources, repartition=self._repartition, watermark=wm)

    # --- transformations ---------------------------------------------------
    def pipe(self, fn: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
        """Apply `fn(self, *args, **kwargs)` and return its result, to keep a chain fluent.

        The escape hatch for composing your own transformations without breaking the
        method chain: ``ds.pipe(add_features).filter(...)`` reads in the order it runs,
        where ``add_features(ds.filter(...))`` would not. `pipe` is transparent — it
        adds no plan node and returns whatever `fn` returns, so it stays lazy when `fn`
        does.

        Args:
            fn: A callable taking this `Dataset` as its first argument.
            *args: Extra positional arguments forwarded to `fn`.
            **kwargs: Extra keyword arguments forwarded to `fn`.

        Returns:
            Whatever `fn` returns — typically a new `Dataset`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> def scale(ds, factor):
                ...     return ds.with_columns(x=bt.col("x") * factor)
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.pipe(scale, 10).filter(bt.col("x") > 10).to_pydict()
                {'x': [20, 30]}
        """
        return fn(self, *args, **kwargs)

    def filter(self, *predicates: Expr, **equals: Any) -> Dataset:
        """Keep only the rows where every predicate is true.

        A predicate is an expression built from columns, e.g. ``col("amount") >
        100``. Combine conditions with ``&`` (and), ``|`` (or), and ``~`` (not),
        parenthesizing each side because those operators bind tighter than
        comparisons. Rows where a predicate is null are dropped. Like every
        transformation this is lazy and returns a new `Dataset`.

        Several predicates are ANDed together, and a keyword argument is an
        equality shorthand: ``filter(status="paid", region="eu")`` means
        ``filter((col("status") == "paid") & (col("region") == "eu"))``. Both
        spellings save the parenthesizing that ``&`` would otherwise need.

        A predicate may compose window expressions — ``filter(col("x") >
        col("x").mean().over(partition_by=["g"]))`` keeps rows above their group
        mean. The window sees every input row, as in the SQL subquery it desugars to.

        Args:
            *predicates: Boolean expressions evaluated per row, ANDed together.
            **equals: Column-equals-value shorthands, ANDed with `predicates`.

        Returns:
            A new `Dataset` with the matching rows.

        Raises:
            PlanError: If no condition is given, an argument is not an expression,
                or a keyword names a column the dataset does not have.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 5, 9], "ok": [True, False, True]})
                >>> ds.filter((bt.col("x") > 2) & bt.col("ok")).to_pydict()
                {'x': [9], 'ok': [True]}

                >>> ds = bt.from_pydict({"g": ["a", "b", "a"], "x": [1, 2, 3]})
                >>> ds.filter(g="a").to_pydict()
                {'g': ['a', 'a'], 'x': [1, 3]}
        """
        conditions = list(predicates)
        for name, value in equals.items():
            self._require_column(name, "filter")
            conditions.append(Col(name) == value)
        if not conditions:
            raise PlanError(
                "filter() requires a condition, e.g. filter(col('x') > 0) or filter(x=1)"
            )
        for cond in conditions:
            if not isinstance(cond, Expr):
                raise PlanError(
                    "filter() requires an expression, e.g. col('x') > 0; got "
                    f"{type(cond).__name__}. A SQL string goes to ds.sql(...) instead."
                )
        combined = conditions[0]
        for cond in conditions[1:]:
            combined = combined & cond
        return windowed_filter(self, combined)

    def select(self, *columns: str | Expr, **named: Expr | int | float | bool | str) -> Dataset:
        """Project to exactly the given columns.

        Positional args are column names (strings), bare ``col(...)`` references,
        aliased expressions (``expr.alias("name")``), or column selectors
        (``bt.exclude("id")``, ``bt.numeric() * 2``) which expand to one output per
        matched column; keyword args bind a new name to an expression:
        ``ds.select("id", total=col("price") * col("qty"))``.

        Args:
            *columns: Column names, ``col(...)`` references, aliased expressions, or
                column selectors.
            **named: New column names bound to expressions.

        Returns:
            A new `Dataset` with exactly the selected columns.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3], "g": ["a", "b", "a"]})
                >>> ds.select("x", y=bt.col("x") * 2).to_pydict()
                {'x': [1, 2, 3], 'y': [2, 4, 6]}

                >>> ds.select(bt.exclude("g")).to_pydict()
                {'x': [1, 2, 3]}
        """
        items: list[Projection] = []
        for c in columns:
            if isinstance(c, str):
                items.append(Projection(c, Col(c)))
            elif has_selector(c):
                items.extend(Projection(n, e) for n, e in expand_selector_expr(self, c))
            elif isinstance(c, Aliased):
                items.append(Projection(c.name, c.inner))
            elif isinstance(c, Col):
                items.append(Projection(c.name, c))
            else:
                raise PlanError(
                    "positional select() arguments must be column names, col(...) "
                    "references, aliased expressions, or column selectors; name other "
                    "derived columns via a keyword (select(total=expr)) or .alias('total')"
                )
        for alias, expr in named.items():
            items.append(Projection(alias, _as_expr(expr)))
        if not items:
            raise PlanError(_empty_projection_message("select", columns))
        return windowed_project(self, items)

    def with_columns(self, *exprs: Expr, **named: Expr | int | float | bool | str) -> Dataset:
        """Add or replace columns, keeping all existing ones.

        Values may be expressions, scalars, or window expressions from
        ``agg.over(...)`` (e.g. ``with_columns(total=col("x").sum().over(partition_by=["g"]))``).
        A window may be composed with ordinary arithmetic —
        ``with_columns(share=col("x") / col("x").sum().over())`` — and window and
        non-window columns may be mixed freely in one call.

        Positional args must already carry their output name: a column selector
        (``bt.numeric().round(2)`` replaces each numeric column in place), an aliased
        expression, or a bare ``col(...)``. Anything else needs a keyword.

        Args:
            *exprs: Self-naming expressions — column selectors, aliased expressions,
                or bare ``col(...)`` references.
            **named: Column names bound to expressions (or scalars) to add or replace.

        Returns:
            A new `Dataset` with the columns added or replaced.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.with_columns(y=bt.col("x") + 1).to_pydict()
                {'x': [1, 2, 3], 'y': [2, 3, 4]}

                >>> ds = bt.from_pydict({"a": [1.234], "b": [5.678], "s": ["x"]})
                >>> ds.with_columns(bt.numeric().round(1)).to_pydict()
                {'a': [1.2], 'b': [5.7], 's': ['x']}
        """
        positional = self._named_positionals(exprs)
        clashing = sorted(positional.keys() & named.keys())
        if clashing:
            raise PlanError(
                f"with_columns() got column(s) {clashing} both positionally and as a "
                "keyword; give each output column exactly one definition"
            )
        named = {**positional, **named}
        if not named:
            raise PlanError(_empty_projection_message("with_columns", exprs))
        # Bare `agg.over(...)` columns need no surrounding projection: name each
        # window by its own alias and append it. Anything composed goes through the
        # hoisting path below.
        if all(is_bare_window(e) for e in named.values()):
            return build_window_columns(self, named)
        existing = self._plan.available_columns()
        items: list[Projection] = []
        for name in existing:
            if name in named:
                items.append(Projection(name, _as_expr(named[name])))
            else:
                items.append(Projection(name, Col(name)))
        for alias, expr in named.items():
            if alias not in existing:
                items.append(Projection(alias, _as_expr(expr)))
        return windowed_project(self, items)

    def sort(
        self,
        *keys: str | Expr,
        descending: bool | list[bool] = False,
        nulls_first: bool | list[bool] = False,
        by: str | Expr | list[str | Expr] | None = None,
        ascending: bool | list[bool] | None = None,
        na_position: str | None = None,
    ) -> Dataset:
        """Order rows by one or more keys (column names or expressions).

        `descending`/`nulls_first` are either a single bool applied to all keys or
        a list matching the number of keys.

        The pandas ``sort_values`` spellings are accepted too: `by` for the keys,
        `ascending` as the inverse of `descending`, and `na_position`
        (``"first"``/``"last"``) as the spelling of `nulls_first`.

        Args:
            *keys: The sort keys, as column names or expressions.
            descending: Sort descending — one bool for all keys or a per-key list.
            nulls_first: Order nulls first — one bool for all keys or a per-key list.
            by: The pandas spelling of `keys`; a single key or a list of them.
            ascending: The pandas inverse of `descending`.
            na_position: The pandas spelling of `nulls_first`: ``"first"`` or ``"last"``.

        Returns:
            A new `Dataset` with rows ordered by the keys.

        Raises:
            PlanError: If no key is given, if an alias conflicts with the name it
                aliases, or if `na_position` is not ``"first"``/``"last"``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [3, 1, 2]})
                >>> ds.sort("x", descending=True).to_pydict()
                {'x': [3, 2, 1]}

                >>> ds.sort(by="x", ascending=False).to_pydict()
                {'x': [3, 2, 1]}
        """
        if by is not None:
            if keys:
                raise PlanError("sort() takes keys positionally or as `by`, not both")
            keys = tuple(by) if isinstance(by, list) else (by,)
        if ascending is not None:
            if descending is not False:
                raise PlanError("pass descending or ascending, not both")
            descending = (
                [not a for a in ascending] if isinstance(ascending, list) else not ascending
            )
        if na_position is not None:
            if nulls_first is not False:
                raise PlanError("pass nulls_first or na_position, not both")
            if na_position not in ("first", "last"):
                raise PlanError(
                    f"sort(): na_position must be 'first' or 'last', got {na_position!r}"
                )
            nulls_first = na_position == "first"
        by = keys
        if not by:
            raise PlanError("sort() requires at least one key")
        desc = _broadcast(descending, len(by), "descending")
        nf = _broadcast(nulls_first, len(by), "nulls_first")
        keys = tuple(
            SortKeySpec(_as_key_expr(k), descending=d, nulls_first=n)
            for k, d, n in zip(by, desc, nf, strict=True)
        )
        return self._derive(Sort(self._plan, keys))

    def window(
        self,
        *,
        partition_by: list[str | Expr] = (),
        order_by: list[str | tuple[str, bool] | Expr] = (),
        functions: dict[str, str | tuple[str, str]],
        frame: tuple[int | None, int | None] | None = None,
    ) -> Dataset:
        """Append window-function columns, preserving all input columns.

        Rows are partitioned by `partition_by` (empty → one partition) and ordered
        by `order_by` (column names, ``(name, descending)`` tuples, or expressions).
        Each `functions` entry maps an output name to a ranking function
        (``"row_number"``/``"rank"``/``"dense_rank"``, no input, needs `order_by`)
        or an aggregate (``("sum"|"mean"|"min"|"max"|"count", "col")``; ``"avg"`` is
        accepted as a synonym for ``"mean"``) — whole-partition without `order_by`,
        else running/cumulative.

        `frame` sets an explicit ``ROWS`` frame on the aggregates: a ``(start,
        end)`` pair of signed row offsets (negative = preceding, ``0`` = current,
        positive = following, ``None`` = unbounded), so ``frame=(-2, 0)`` is a
        trailing 3-row window.

        Args:
            partition_by: Columns or expressions to partition rows by.
            order_by: Ordering keys — names, ``(name, descending)`` tuples, or expressions.
            functions: Output name to a ranking function or an ``(agg, column)`` pair.
            frame: An explicit ``ROWS`` frame as a ``(start, end)`` offset pair.

        Returns:
            A new `Dataset` with the window columns appended.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a", "b"], "v": [1, 2, 3]})
                >>> ds.window(partition_by=["g"], functions={"s": ("sum", "v")}).to_pydict()
                {'g': ['a', 'a', 'b'], 'v': [1, 2, 3], 's': [3, 3, 3]}
        """
        return build_window(
            self,
            partition_by=partition_by,
            order_by=order_by,
            functions=functions,
            frame=frame,
        )

    @property
    def ml(self) -> DatasetML:
        """ML/multimodal accessor: batch inference, embedding, and mapping.

        Runs `infer`/`embed`/`map_batches` with GPU and actor-pool scheduling
        (``ds.ml.infer(model, num_gpus=1, concurrency=4)``).

        Returns:
            The ML accessor bound to this dataset.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.ml.map(lambda r: {"x": r["x"] * 10}).to_pydict()
                {'x': [10, 20, 30]}
        """
        return DatasetML(self)

    @property
    def dq(self) -> DatasetDQ:
        """Data-quality accessor: accumulate expectations then act on the failures.

        Chain expectations
        (`not_null`/`unique`/`in_range`/`matches`/`accepted_values`/`check`) then
        `fail()` (raise), `drop()` (keep valid), `quarantine()` (split valid/rejected),
        or `validate()` (counts). E.g.
        ``ds.dq.not_null("id").unique(["id"]).in_range("age", 0, 120).quarantine()``.

        Returns:
            The data-quality accessor bound to this dataset.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1, 2, 3], "age": [30, 200, -5]})
                >>> ds.dq.in_range("age", 0, 120).drop().to_pydict()
                {'id': [1], 'age': [30]}
        """
        return DatasetDQ(self)

    @property
    def meta(self) -> DatasetMeta:
        """Metadata accessor: answer a question from statistics instead of from the data.

        A footer, a manifest, a catalog, and an immutable in-memory relation already know a
        great deal a query would otherwise be run to rediscover — the row count, a column's
        extremes, how many values are missing, whether a key is unique, whether a join can
        match at all. Every shortcut under `meta` asks for that first and only executes when
        the answer is not provable, so what it returns is always what executing would return.

        Reach the breadth through the sub-accessors: ``.col("x")`` (and ``.col("x").check``),
        ``.schema``, ``.nulls``, ``.approx``, ``.storage``, and ``.against(other)``.

        Returns:
            The metadata accessor bound to this dataset.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.meta.col("x").bounds()
                (1, 3)
                >>> ds.meta.none_match(bt.col("x") > 100)
                True
        """
        return DatasetMeta(self)

    @property
    def scd(self) -> DatasetSCD:
        """Slowly-changing-dimension accessor: upsert this snapshot into a target.

        Apply as `scd.type1` (overwrite), `scd.type2` (effective-dated history), or
        `scd.type3` (previous-value column).

        Returns:
            The SCD accessor bound to this dataset.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1], "v": ["a"]})
                >>> hasattr(ds.scd, "type2")
                True
        """
        return DatasetSCD(self)

    def map_batches(
        self,
        fn: Callable | type,
        *,
        batch_size: int | None = None,
        input_columns: list[str] | None = None,
        output_columns: list[str] | None = None,
        num_workers: int | str = "auto",
        num_gpus: float = 0.0,
        concurrency: int | None = None,
        batch_format: str = "pyarrow",
        multiprocessing: bool = False,
        max_errored_rows: int = 0,
        timeout: float = 0.0,
        max_retries: int = 0,
        retry_backoff: float = 0.5,
        retry_on: type[BaseException] | tuple[type[BaseException], ...] | None = None,
        max_concurrency: int = 0,
    ) -> Dataset:
        """Apply a Python function to each Arrow batch (sugar for `ds.ml.map_batches`).

        Kept top-level for the familiar spelling; see `ds.ml` for the full ML surface.

        `num_workers` defaults to ``"auto"`` — the per-batch calls fan across all
        local cores, so a batch transform is parallel by default rather than
        single-threaded (the Ray Data foot-gun). Threads only speed up a
        GIL-releasing `fn` (Arrow/NumPy/torch); pass ``multiprocessing=True`` for a
        CPU-bound pure-Python `fn`. An explicit int wins.

        Args:
            fn: A callable (or stateful class) applied to each batch.
            batch_size: Rows per batch handed to `fn`; ``None`` uses the engine default.
            input_columns: The columns `fn` reads, letting projection pushdown prune the
                scan to just those; ``None`` keeps every column alive. Omitting one `fn`
                does read is a correctness bug — it gets pruned out from under it.
            output_columns: The output column names, when `fn` reshapes the schema.
            num_workers: Worker fan-out; ``"auto"`` spreads across local cores.
            num_gpus: GPUs reserved per worker.
            concurrency: Maximum concurrent workers; ``None`` lets the engine choose.
            batch_format: The batch type passed to `fn` (``"pyarrow"`` by default).
            multiprocessing: Use processes instead of threads for a CPU-bound `fn`.
            max_errored_rows: How many per-row errors to tolerate before failing.
            timeout: Wall-clock ceiling (seconds) for one `fn` call; 0 = no timeout.
            max_retries: Times to retry a batch whose `fn` raises before failing.
            retry_backoff: Base retry backoff (seconds); attempt `k` waits `retry_backoff * 2**k`.
            retry_on: Exception type(s) worth retrying; ``None`` retries any `Exception`.
            max_concurrency: Max in-flight batches for an ``async def`` `fn`; 0 = a default.

        Returns:
            A new `Dataset` of the transformed batches.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import pyarrow.compute as pc
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> def add_one(batch):
                ...     return batch.set_column(0, "x", pc.add(batch.column("x"), 1))
                >>> ds.map_batches(add_one).to_pydict()
                {'x': [2, 3, 4]}
        """
        return self.ml.map_batches(
            fn,
            batch_size=batch_size,
            input_columns=input_columns,
            output_columns=output_columns,
            num_workers=num_workers,
            num_gpus=num_gpus,
            concurrency=concurrency,
            batch_format=batch_format,
            multiprocessing=multiprocessing,
            max_errored_rows=max_errored_rows,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_on=retry_on,
            max_concurrency=max_concurrency,
        )

    def offload_blobs(
        self,
        column: str = "bytes",
        *,
        uri_column: str = "uri",
        root: str | None = None,
        batch_size: int = 8,
    ) -> Dataset:
        """Offload a large-payload column to a content-addressed store, leaving handles.

        The write-side dual of reference-mode reads: each row's ``column`` payload is
        written to ``{root}/{sha256}`` (deduped by content) and replaced with a tiny
        ``uri_column`` handle, with the payload column nulled. The blobs then stay out
        of every shuffle and spill buffer until `materialize_blobs` reads them back
        right before they are needed. `root` defaults to the configured spill store.

        Args:
            column: The payload column to offload.
            uri_column: The handle column that replaces the payload.
            root: The content-addressed store root; defaults to the spill store.
            batch_size: Rows per batch while writing blobs.

        Returns:
            A new `Dataset` with the payload column replaced by handles.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, pyarrow as pa
                >>> ds = bt.from_arrow(pa.table({"id": [1], "bytes": [b"payload"]}))
                >>> handles = ds.offload_blobs(root=tempfile.mkdtemp()).collect()
                >>> handles.column("bytes").to_pylist()  # payload moved out of line
                [None]
                >>> handles.column("id").to_pylist()
                [1]
        """
        from functools import partial

        from batcher.io.formats.multimodal.blob import default_blob_root, offload_blob_bytes

        resolved = root or default_blob_root()
        out_cols = list(self.columns)
        if uri_column not in out_cols:
            out_cols.append(uri_column)
        return self.map_batches(
            partial(offload_blob_bytes, root=resolved, src=column, uri_col=uri_column),
            batch_size=batch_size,
            output_columns=out_cols,
        )

    def materialize_blobs(
        self,
        *,
        uri_column: str = "uri",
        into: str = "bytes",
        batch_size: int = 8,
    ) -> Dataset:
        """Read offloaded payloads back from their handles into the ``into`` column.

        The inverse of `offload_blobs` (and the same primitive reference-mode reads
        use): each ``uri_column`` handle is fetched into ``into`` as ``large_binary``.
        Run it right before the operator that needs raw bytes — with a small
        ``batch_size`` the GB payloads never all co-reside.

        Args:
            uri_column: The handle column to read payloads from.
            into: The output column that receives the payload bytes.
            batch_size: Rows per batch while reading blobs.

        Returns:
            A new `Dataset` with the payloads materialized into `into`.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, pyarrow as pa
                >>> root = tempfile.mkdtemp()
                >>> ds = bt.from_arrow(pa.table({"id": [1], "bytes": [b"payload"]}))
                >>> handles = ds.offload_blobs(root=root)
                >>> handles.materialize_blobs().collect().column("bytes").to_pylist()
                [b'payload']
        """
        from functools import partial

        from batcher.io.formats.multimodal.blob import read_blob_bytes

        out_cols = list(self.columns)
        if into not in out_cols:
            out_cols.append(into)
        return self.map_batches(
            partial(read_blob_bytes, uri_col=uri_column, into=into),
            batch_size=batch_size,
            output_columns=out_cols,
        )

    def map(
        self,
        fn: Callable,
        *,
        output_columns: list[str] | None = None,
        batch_size: int | None = None,
        num_workers: int | str = "auto",
        max_concurrency: int = 0,
    ) -> Dataset:
        """Apply a per-row function ``fn(row) -> row`` (Ray Data ``map``).

        Sugar for `ds.ml.map`. Prefer `map_batches` (vectorized) when you can; see `ds.ml`.
        Pass an ``async def`` `fn` for a per-row I/O-bound call (a per-row LLM/API request):
        each batch's rows are awaited concurrently, up to `max_concurrency`.

        Args:
            fn: A callable (or ``async def``) mapping one row dict to a new row dict.
            output_columns: The output column names when `fn` changes the schema.
            batch_size: Rebatch to this many rows before processing.
            num_workers: Concurrent calls within a worker (``"auto"`` sizes it).
            max_concurrency: In-flight per-row awaits within a batch for an ``async`` `fn`.

        Returns:
            A new `Dataset` of the mapped rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.map(lambda row: {"x": row["x"] * 2}).to_pydict()
                {'x': [2, 4, 6]}
        """
        return self.ml.map(
            fn,
            output_columns=output_columns,
            batch_size=batch_size,
            num_workers=num_workers,
            max_concurrency=max_concurrency,
        )

    def flat_map(
        self,
        fn: Callable,
        *,
        output_columns: list[str] | None = None,
        batch_size: int | None = None,
        num_workers: int | str = "auto",
        max_concurrency: int = 0,
    ) -> Dataset:
        """Apply a per-row function ``fn(row) -> iterable[row]`` and flatten.

        The Ray Data ``flat_map`` — sugar for `ds.ml.flat_map`; see `ds.ml`. An ``async def``
        `fn` has its rows awaited concurrently within a batch.

        Args:
            fn: A callable (or ``async def``) mapping one row dict to an iterable of row dicts.
            output_columns: The output column names when `fn` changes the schema.
            batch_size: Rebatch to this many rows before processing.
            num_workers: Concurrent calls within a worker (``"auto"`` sizes it).
            max_concurrency: In-flight per-row awaits within a batch for an ``async`` `fn`.

        Returns:
            A new `Dataset` of the flattened rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2]})
                >>> ds.flat_map(lambda row: [{"x": row["x"]}, {"x": row["x"]}]).to_pydict()
                {'x': [1, 1, 2, 2]}
        """
        return self.ml.flat_map(
            fn,
            output_columns=output_columns,
            batch_size=batch_size,
            num_workers=num_workers,
            max_concurrency=max_concurrency,
        )

    def sql(self, query: str, *, table_name: str = "self", dialect: str | None = None) -> Dataset:
        """Run a SQL query with this dataset bound to `table_name` (default ``self``).

        The Polars-style ``ds.sql("SELECT ... FROM self")``: a lazy `Dataset` that
        composes with the rest of the API. Tables and functions registered on the
        default catalog (via `bt.register_function` or ``CREATE TABLE``) resolve too,
        so the query can join ``self`` against them. For multi-table SQL with several
        ad-hoc inputs, use `bt.sql(query, a=ds1, b=ds2)`.

        Args:
            query: A SQL statement referring to this dataset as `table_name`.
            table_name: The name this dataset is bound to in the query.
            dialect: Override the sqlglot read dialect (default ``duckdb``).

        Returns:
            A lazy `Dataset` of the query result.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [1, 2, 3]})
                >>> ds.sql("SELECT a, a * 2 AS d FROM self WHERE a > 1").to_pydict()
                {'a': [2, 3], 'd': [4, 6]}
        """
        from batcher.api.session import _catalog

        session = _catalog if dialect is None else _catalog._with_dialect(dialect)
        return session._run(query, {table_name: self})

    def with_column(self, name: str, expr: Expr) -> Dataset:
        """Add a column `name` from `expr`, or replace it if it already exists.

        The single-column sugar for `with_columns`; all existing columns are kept.
        Lazy — returns a new `Dataset`.

        Args:
            name: The output column name (replaced if present).
            expr: The expression computing the column.

        Returns:
            A new `Dataset` with the column added or replaced.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.with_column("y", bt.col("x") * 10).to_pydict()
                {'x': [1, 2, 3], 'y': [10, 20, 30]}
        """
        return self.with_columns(**{name: expr})

    def drop(
        self,
        *names: str | Selector,
        columns: str | list[str] | None = None,
        labels: str | list[str] | None = None,
    ) -> Dataset:
        """Return a dataset without the named columns, preserving the rest in order.

        The complement of `select`: name the columns to remove rather than the ones
        to keep, either by name or with a column selector (``ds.drop(bt.temporal())``).
        Lazy. Raises `PlanError` on an unknown column name (with a suggestion) or if
        every column would be dropped.

        The pandas keyword spellings ``drop(columns=[...])`` and
        ``drop(labels=[...])`` are accepted too.

        Args:
            *names: Names of the columns to remove, or column selectors matching them.
            columns: The pandas keyword spelling of `names`.
            labels: The older pandas spelling of `columns`.

        Returns:
            A new `Dataset` with the remaining columns.

        Raises:
            PlanError: On an unknown column, a non-column argument, or if every
                column would be dropped.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
                >>> ds.drop("b").to_pydict()
                {'a': [1, 2], 'c': [5, 6]}

                >>> ds.drop(columns=["b", "c"]).to_pydict()
                {'a': [1, 2]}

                >>> ds.drop(bt.matches("^[bc]$")).to_pydict()
                {'a': [1, 2]}
        """
        keyword = _as_opt_str_list(_one_of(columns, labels, "columns", "labels"))
        targets: tuple[str | Selector, ...] = (*names, *(keyword or ()))
        if not targets:
            raise PlanError("drop() requires at least one column name or selector")
        available = self._plan.available_columns()
        to_drop: set[str] = set()
        for c in targets:
            if isinstance(c, Selector):
                to_drop.update(selector_columns(self, c))
            elif isinstance(c, str):
                to_drop.add(c)
            else:
                raise PlanError(
                    f"drop() takes column names or column selectors, got {type(c).__name__}"
                )
        missing = to_drop - set(available)
        if missing:
            raise PlanError(f"drop(): unknown column(s) {_unknown_cols(missing, available)}")
        keep = [c for c in available if c not in to_drop]
        if not keep:
            raise PlanError("drop() would remove all columns")
        return self.select(*keep)

    def rename(
        self,
        mapping: dict[str, str] | Callable[[str], str] | None = None,
        *,
        columns: dict[str, str] | Callable[[str], str] | None = None,
        **renames: str,
    ) -> Dataset:
        """Rename columns, preserving order.

        Pass a ``{old: new}`` dict or kwargs (``rename(old="new")``); a dict and
        kwargs may be combined. A callable is applied to every column name, which is
        how pandas and Polars spell a bulk rename: ``ds.rename(str.lower)``. The
        pandas keyword form ``rename(columns={...})`` is accepted too.

        Args:
            mapping: An ``{old: new}`` rename mapping, or a function applied to
                every column name.
            columns: The pandas keyword spelling of `mapping`.
            **renames: Renames given as ``old="new"`` keyword arguments.

        Returns:
            A new `Dataset` with the columns renamed.

        Raises:
            PlanError: If a name to rename is not a column, or if a callable
                collapses two columns onto the same output name.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [1], "b": [2]})
                >>> ds.rename(a="x").to_pydict()
                {'x': [1], 'b': [2]}

                >>> ds.rename(str.upper).columns
                ['A', 'B']
        """
        mapping = _one_of(mapping, columns, "mapping", "columns")
        available = self._plan.available_columns()
        if callable(mapping):
            renamed = {c: mapping(c) for c in available}
            produced = list(renamed.values())
            collisions = sorted({n for n in produced if produced.count(n) > 1})
            if collisions:
                raise PlanError(
                    f"rename(): the function maps several columns onto {collisions}; "
                    "column names must stay unique"
                )
            mapping = renamed
        merged = {**(mapping or {}), **renames}
        missing = set(merged) - set(available)
        if missing:
            raise PlanError(f"rename(): unknown column(s) {_unknown_cols(missing, available)}")
        items = tuple(Projection(merged.get(c, c), Col(c)) for c in available)
        return self._derive(Project(self._plan, items))

    def distinct(
        self,
        subset: list[str] | None = None,
        *,
        keep: str = "any",
        order_by: str | list[str] | list[tuple[str, bool]] | None = None,
    ) -> Dataset:
        """Remove duplicate rows.

        With no `subset`, DISTINCT over all columns. With `subset`, keep one row per
        distinct key combination: `keep="first"`/`"last"` picks the first/last row in
        `order_by` order (required for first/last); `keep="any"` keeps an arbitrary
        deterministic row. Lowers to ``row_number() OVER (PARTITION BY subset
        ORDER BY ...)`` + filter — no new IR.

        Args:
            subset: Columns defining the key; ``None`` deduplicates over all columns.
            keep: Which row to keep per key — ``"any"``, ``"first"``, or ``"last"``.
            order_by: The order defining first/last (required for those `keep` modes).

        Returns:
            A new `Dataset` with duplicate rows removed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 1, 2, 2, 3]})
                >>> ds.distinct().sort("x").to_pydict()
                {'x': [1, 2, 3]}
        """
        if subset is None:
            return self._derive(Distinct(self._plan))
        return build_distinct(self, subset, keep, order_by)

    def unique(
        self,
        subset: list[str] | None = None,
        *,
        keep: str = "any",
        order_by: str | list[str] | list[tuple[str, bool]] | None = None,
    ) -> Dataset:
        """Remove duplicate rows — the Polars ``unique`` spelling of :meth:`distinct`.

        Args:
            subset: Columns defining the key; ``None`` deduplicates over all columns.
            keep: Which row to keep per key — ``"any"``, ``"first"``, or ``"last"``.
            order_by: The order defining first/last (required for those `keep` modes).

        Returns:
            A new `Dataset` with duplicate rows removed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 1, 2, 2, 3]})
                >>> ds.unique().sort("x").to_pydict()
                {'x': [1, 2, 3]}
        """
        return self.distinct(subset, keep=keep, order_by=order_by)

    def repartition(
        self,
        num_files: int | None = None,
        *,
        by: str | list[str] | None = None,
        target_size_mb: float | None = None,
    ) -> Dataset:
        """Set how the next `write` lays out its files (the data is unchanged).

        Pass exactly one sizing option: `num_files` (split into that many files),
        `target_size_mb` (coalesce into ~that-size files — the small-files fix), or
        neither with only `by` to Hive-partition by column(s). `by` may combine with
        a sizing option. ``ds.repartition(target_size_mb=128).write("out/")``;
        ``ds.repartition(by="dt").write("out/")``. See `bt.compact` for in-place use.

        Args:
            num_files: Split the output into this many files.
            by: Column(s) to Hive-partition the output by.
            target_size_mb: Coalesce into files of about this size.

        Returns:
            A new `Dataset` carrying the write layout hint.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.repartition(num_files=2).to_pydict()
                {'x': [1, 2, 3]}
        """
        if num_files is not None and target_size_mb is not None:
            raise PlanError("repartition(): pass num_files or target_size_mb, not both")
        if num_files is not None:
            num_files = require_int(num_files, func="repartition", arg="num_files", minimum=1)
        if target_size_mb is not None and target_size_mb <= 0:
            raise PlanError(f"repartition(): target_size_mb must be > 0, got {target_size_mb}")
        by_cols = () if by is None else ((by,) if isinstance(by, str) else tuple(by))
        if num_files is None and target_size_mb is None and not by_cols:
            raise PlanError("repartition(): provide num_files, target_size_mb, or by")
        spec = RepartitionSpec(num_files=num_files, by=by_cols, target_size_mb=target_size_mb)
        return Dataset(self._plan, self._sources, spec)

    def value_counts(
        self,
        column: str,
        *,
        name: str | None = None,
        sort: bool = True,
        normalize: bool = False,
    ) -> Dataset:
        """Count occurrences of each distinct value of `column` (pandas/Polars ``value_counts``).

        Returns ``[column, name]``, sorted by count descending unless `sort=False`.
        Sugar over ``group_by(column).agg(count())``. With `normalize` the counts
        become each value's share of the total, and the output column is named
        ``proportion`` rather than ``count``, as pandas names it.

        Args:
            column: The column whose values to count.
            name: The name of the output column; defaults to ``"count"``, or
                ``"proportion"`` when `normalize` is set.
            sort: Sort by count descending (the default).
            normalize: Report each value's share of the total instead of its count.

        Returns:
            A new `Dataset` of value and count (or proportion) columns.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"c": ["a", "a", "b"]})
                >>> ds.value_counts("c").to_pydict()
                {'c': ['a', 'b'], 'count': [2, 1]}

                >>> ds.value_counts("c", normalize=True).to_pydict()
                {'c': ['a', 'b'], 'proportion': [0.6666666666666666, 0.3333333333333333]}
        """
        from batcher.api.functions import count

        name = name or ("proportion" if normalize else "count")
        out = self.group_by(column).agg(**{name: count()})
        if normalize:
            # The share is computed against a whole-relation window total, so it is
            # one pass and identical single-node or distributed.
            out = out.window(functions={"__vc_total": ("sum", name)})
            out = out.with_columns(**{name: Col(name) / Col("__vc_total")}).drop("__vc_total")
        return out.sort(name, descending=True) if sort else out

    def describe(self, *, percentiles: tuple[float, ...] = (0.25, 0.5, 0.75)) -> Dataset:
        """Summary statistics per column (pandas/Polars ``describe``).

        **Executes** the query and returns a small `Dataset` with a ``statistic``
        label column and one Float64 column per input column. Numeric columns report
        count / null_count / mean / std / min / the requested `percentiles` (default
        quartiles) / max; non-numeric columns report count and null_count only.
        Composes the already-tested aggregates — no per-row work in Python.

        Args:
            percentiles: The quantiles to report for numeric columns.

        Returns:
            A `Dataset` of summary statistics with a ``statistic`` label column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3, 4]})
                >>> ds.describe().columns
                ['statistic', 'x']
        """
        from batcher.api.dataset._describe import describe

        return describe(self, percentiles)

    def null_count(self) -> Dataset:
        """A one-row dataset of each column's null count (pandas ``isnull().sum()``).

        Lazy: lowers to a single global aggregate and a `select`, so it stays
        mergeable and identical single-node and distributed.

        Returns:
            A one-row `Dataset` of each column's null count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, None, 3], "y": [1, 2, 3]})
                >>> ds.null_count().to_pydict()
                {'x': [1], 'y': [0]}
        """
        from batcher.api.dataset._describe import null_count

        return null_count(self)

    def profile(self) -> Dataset:
        """A per-column data-quality profile that **executes** the query.

        Returns one row per column with
        ``count``/``null_count``/``null_fraction``/``approx_distinct`` (HyperLogLog
        cardinality). The quick "what does this column look like" check before a load.

        Returns:
            A `Dataset` with one profile row per input column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 2]})
                >>> ds.profile().columns
                ['column', 'count', 'null_count', 'null_fraction', 'approx_distinct']
        """
        from batcher.api.dataset._describe import profile

        return profile(self)

    def top_k(self, k: int, by: str | list[str], *, descending: bool = True) -> Dataset:
        """The `k` rows ranked highest (or lowest) by `by`.

        Sugar for ``sort(by, descending).limit(k)`` — the engine fuses sort+limit to
        a top-N.

        Args:
            k: The number of rows to keep.
            by: The ranking key column(s).
            descending: Rank highest-first (the default); ``False`` ranks lowest-first.

        Returns:
            A new `Dataset` with the top `k` rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [5, 1, 3, 2, 4]})
                >>> ds.top_k(2, "x").to_pydict()
                {'x': [5, 4]}
        """
        k = require_int(k, func="top_k", arg="k", minimum=0)
        keys = by if isinstance(by, list) else [by]
        return self.sort(*keys, descending=descending).limit(k)

    def cross_join(self, other: Dataset, *, suffix: str = "_right") -> Dataset:
        """Cartesian product — every left row paired with every right row.

        Lowered to an equi-join on a constant key, so it reuses the join engine; the
        temporary key is dropped from the output (colliding names get `suffix`).

        Args:
            other: The right-hand dataset to pair every left row with.
            suffix: Suffix appended to right columns whose names collide.

        Returns:
            A new `Dataset` of the Cartesian product.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> left = bt.from_pydict({"a": [1, 2]})
                >>> right = bt.from_pydict({"b": ["x"]})
                >>> left.cross_join(right).sort("a").to_pydict()
                {'a': [1, 2], 'b': ['x', 'x']}

            The join emits rows in no particular order, so sort when you need one.
        """
        from batcher.plan.expr_ir import lit

        # The temporary equi-join key must not shadow a real column on either side:
        # `with_columns` replaces a same-named column, so a user column literally named
        # `__cross_key__` would be silently overwritten and then dropped — losing its
        # data. Pick a name absent from both schemas.
        taken = set(self.columns) | set(other.columns)
        key = "__cross_key__"
        while key in taken:
            key += "_"
        left = self.with_columns(**{key: lit(1)})
        right = other.with_columns(**{key: lit(1)})
        return left.join(right, on=key, suffix=suffix).drop(key)

    def explode(
        self,
        column: str,
        *,
        alias: str | None = None,
        outer: bool = False,
        index: str | None = None,
    ) -> Dataset:
        """Explode a list/array column into one row per element (SQL ``UNNEST``).

        Other columns repeat per element. The exploded column replaces `column` in place
        (renamed to `alias` if given) and streams (no breaker). Raises `PlanError` if
        `column` is not a column.

        By default a null or empty list produces **no** rows, which is DuckDB's ``UNNEST``
        semantics — and a trap for document pipelines, where a row that chunked to nothing
        then disappears along with its id and metadata. Pass `outer=True` to keep it with a
        NULL element instead (Spark ``explode_outer``).

        `index` names an extra column holding each element's 0-based position within its
        own list (Spark ``posexplode``), which is what lets chunks be reassembled in order
        after a shuffle. It is NULL for a row kept only by `outer`.

        Args:
            column: The list/array column to explode.
            alias: Rename the exploded column to this name.
            outer: Keep rows whose list is null or empty, with a NULL element.
            index: Name for an appended 0-based element-position column.

        Returns:
            A new `Dataset` with one row per list element.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1, 2], "xs": [[1, 2], [3]]})
                >>> ds.explode("xs").to_pydict()
                {'id': [1, 1, 2], 'xs': [1, 2, 3]}

                >>> # A document that chunked to nothing survives, and chunks are ordered.
                >>> docs = bt.from_pydict({"doc": ["a", "b"], "chunks": [["p", "q"], []]})
                >>> docs.explode("chunks", outer=True, index="i").to_pydict()
                {'doc': ['a', 'a', 'b'], 'chunks': ['p', 'q', None], 'i': [0, 1, None]}
        """
        return build_explode(self, column, alias, outer=outer, index=index)

    def with_row_index(self, name: str = "index", *, offset: int = 0) -> Dataset:
        """Add a sequential row-index column (Polars ``with_row_index``).

        The new `name` column numbers rows ``offset, offset+1, …`` in their current
        order (a single counter, so the single-node and parallel paths agree on an
        order-preserving pipeline). Add it after any reorder you want it to reflect.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": ["a", "b", "c"]}).with_row_index().to_pydict()
                {'index': [0, 1, 2], 'x': ['a', 'b', 'c']}

        Args:
            name: The index column's name.
            offset: The value assigned to the first row.

        Returns:
            A new `Dataset` with the index column appended.
        """
        return self._derive(RowId(self._plan, name, offset))

    def with_random(self, name: str = "random", *, seed: int = 0, normal: bool = False) -> Dataset:
        """Add a reproducible pseudo-random column (`seed`-keyed, one value per row).

        Values are uniform in ``[0, 1)`` by default, or standard normal when `normal`
        is set. The sequence is keyed by ``seed`` and each row's position, so it is
        reproducible across runs and identical on the single-node and parallel paths
        (unlike a wall-clock-seeded RNG). Use it for deterministic sampling/shuffling.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> a = ds.with_random(seed=7).to_pydict()["random"]
                >>> b = ds.with_random(seed=7).to_pydict()["random"]
                >>> a == b and all(0.0 <= v < 1.0 for v in a)
                True

        Args:
            name: The output column's name.
            seed: Seeds the sequence; the same seed reproduces the same values.
            normal: Draw from the standard normal instead of the uniform.

        Returns:
            A new `Dataset` with the random column appended.
        """
        return build_with_random(self, name, seed=seed, normal=normal)

    def drop_duplicates_within_watermark(
        self, subset: list[str], *, event_time: str, lateness: str
    ) -> Dataset:
        """Deduplicate a stream by `subset`, bounding state with a watermark.

        Keeps the first row per `subset` key seen within the event-time watermark
        (``max(event_time) - lateness``); once the watermark passes a key it is
        forgotten, so seen-key memory stays bounded (Spark
        ``dropDuplicatesWithinWatermark``). Over a *bounded* source this is exact
        deduplication (plain `distinct`); over a stream it runs the watermark-bounded
        driver. Consume with `iter_batches()` (or `for_each_batch`).

        Args:
            subset: The columns whose combination defines a duplicate.
            event_time: The event-time column the watermark advances on.
            lateness: How late a row may arrive and still be deduplicated (e.g. ``"10m"``).

        Returns:
            A new `Dataset` with duplicates removed within the watermark.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime
                >>> t0 = datetime.datetime(2024, 1, 1)
                >>> ds = bt.from_pydict({"id": [1, 1, 2], "ts": [t0, t0, t0]})
                >>> out = ds.drop_duplicates_within_watermark(
                ...     ["id"], event_time="ts", lateness="10m"
                ... )
                >>> sorted(out.to_pydict()["id"])
                [1, 2]
        """
        from batcher.io.source import is_bounded
        from batcher.plan.functions.temporal import _duration_micros
        from batcher.plan.logical import WatermarkDedup

        missing = [c for c in [*subset, event_time] if c not in self.columns]
        if missing:
            raise PlanError(f"drop_duplicates_within_watermark(): unknown column(s) {missing}")
        if all(is_bounded(s) for s in self._sources):
            return self.distinct(subset, keep="first", order_by=[(event_time, False)])
        lateness_us = _duration_micros(lateness, arg="watermark lateness")
        return self._derive(WatermarkDedup(self._plan, tuple(subset), event_time, lateness_us))

    def session_window(
        self,
        time_col: str,
        gap: str,
        *,
        partition_by: list[str] | None = None,
        **aggs: Expr,
    ) -> Dataset:
        """Aggregate by event-time **session** windows (Spark ``session_window``).

        A session groups consecutive events (within each `partition_by` group) whose
        inter-arrival gap is below `gap` (a fixed duration like ``"10m"``); a larger
        gap starts a new session. Returns one row per session with `partition_by`,
        ``session_start``/``session_end``, and the named aggregates::

            ds.session_window("ts", "5m", partition_by=["user"], hits=col("v").sum())

        Composed from the window + group-by engine (no new operator), so it is
        differential-tested against DuckDB and runs single-node or distributed.

        Args:
            time_col: The event-time column that orders events into sessions.
            gap: The maximum inter-arrival gap within a session (e.g. ``"10m"``).
            partition_by: Columns whose groups sessionize independently.
            **aggs: Named aggregate expressions computed per session.

        Returns:
            A new `Dataset` with one row per session.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime
                >>> t0 = datetime.datetime(2024, 1, 1)
                >>> dt = datetime.timedelta(seconds=60)
                >>> ds = bt.from_pydict(
                ...     {"ts": [t0, t0 + dt, t0 + 10 * dt, t0 + 11 * dt], "v": [1, 2, 3, 4]}
                ... )
                >>> out = ds.session_window("ts", "5m", total=bt.col("v").sum())
                >>> sorted(out.to_pydict()["total"])
                [3, 7]
        """
        from batcher.api.dataset._build import build_session_window

        return build_session_window(self, time_col, gap, partition_by or [], aggs)

    def unnest(self, *columns: str) -> Dataset:
        """Expand each struct `column` into its fields as top-level columns.

        Matches Polars ``unnest`` / Spark ``select("s.*")``. Each struct field becomes
        a column where the struct was; non-struct columns are unchanged. Raises
        `PlanError` if a column is not a struct or if an expanded field name would
        collide with an existing column.

        Args:
            *columns: The struct columns to expand.

        Returns:
            A new `Dataset` with each struct's fields promoted to columns.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": [{"a": 1, "b": 2}]})
                >>> ds.unnest("s").to_pydict()
                {'a': [1], 'b': [2]}
        """
        return build_unnest(self, list(columns))

    def sample(
        self,
        fraction: float | int | None = None,
        *,
        n: int | None = None,
        frac: float | None = None,
        seed: int | None = None,
        random_state: int | None = None,
    ) -> Dataset:
        """Sample rows by a `fraction` (``0.0`` to ``1.0``) or a fixed count `n`.

        Deterministic and partition-independent: rows are kept by a stable seeded
        hash of their values, so the sampled set is identical single-node or
        distributed and reproducible for a given `seed`. `fraction` streams (no
        breaker, each row kept iff its hash is under `fraction`); `n` keeps exactly
        the `n` smallest-hash rows (a breaker). Pass exactly one of `fraction`/`n`.
        With `seed=None` a fresh seed is baked at plan-build.

        The positional argument reads the way both neighbouring libraries spell it:
        an `int` is a row count (``sample(100)``, as in Polars) and a `float` is a
        fraction (``sample(0.1)``). `frac` and `random_state` are accepted as the
        pandas spellings of `fraction` and `seed`.

        Args:
            fraction: A row count when an `int`, or a fraction in ``[0.0, 1.0]``
                when a `float`.
            n: An exact number of rows to keep (mutually exclusive with `fraction`).
            frac: The pandas spelling of `fraction`; must be a fraction.
            seed: Seeds the sampling; ``None`` bakes a fresh seed at plan-build.
            random_state: The pandas spelling of `seed`.

        Returns:
            A new `Dataset` of the sampled rows.

        Raises:
            PlanError: If both a row count and a fraction are given, if neither is,
                or if an alias conflicts with the name it aliases.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": list(range(100))})
                >>> ds.sample(n=3, seed=1).count()
                3

                >>> ds.sample(3, seed=1).count()
                3
        """
        seed = _one_of(seed, random_state, "seed", "random_state")
        # A bare int positional is a row count, not a >100% fraction. bool is an int
        # subclass, so exclude it rather than reading `True` as "sample one row".
        if isinstance(fraction, int) and not isinstance(fraction, bool):
            n = _one_of(n, fraction, "n", "the positional row count")
            fraction = None
        fraction = _one_of(fraction, frac, "fraction", "frac")
        if fraction is not None and n is not None:
            raise PlanError(
                f"sample() takes a row count or a fraction, not both; got n={n} "
                f"and fraction={fraction}"
            )
        return build_sample(self, fraction, seed, n)

    def pivot(
        self,
        *,
        index: list[str],
        on: str,
        values: str,
        aggregate: str = "sum",
        columns: list | None = None,
        aggfunc: str | None = None,
    ) -> Dataset:
        """Reshape long → wide (SQL ``PIVOT`` / pandas ``pivot_table``).

        Groups by `index` and spreads the distinct values of column `on` into their
        own columns, each holding ``aggregate(values)`` for the matching rows
        (`aggregate` ∈ sum/mean/min/max/count). With `columns` omitted the pivot
        values are discovered by an eager pre-pass over `on`; pass `columns=[...]` to
        fix them (and avoid the pre-pass). Lowers to a grouped conditional aggregate.

        Args:
            index: The columns to group by (the output row key).
            on: The column whose distinct values become output columns.
            values: The column aggregated into each pivoted cell.
            aggregate: The aggregate to apply — sum/mean/min/max/count.
            columns: Fix the pivot values explicitly, skipping the discovery pre-pass.
                Note this is *not* the pandas ``pivot_table(columns=...)``, which
                names the spread column — that is `on` here.
            aggfunc: The pandas spelling of `aggregate`.

        Returns:
            A new `Dataset` reshaped from long to wide.

        Raises:
            PlanError: If `aggregate` and `aggfunc` are both given.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict(
                ...     {"idx": ["r", "r", "s"], "k": ["a", "b", "a"], "v": [1, 2, 3]}
                ... )
                >>> ds.pivot(index=["idx"], on="k", values="v").sort("idx").to_pydict()
                {'idx': ['r', 's'], 'a': [1, 3], 'b': [2, None]}
        """
        if aggfunc is not None:
            if aggregate != "sum":
                raise PlanError("pass aggregate or aggfunc, not both")
            aggregate = aggfunc
        return build_pivot(self, index, on, values, aggregate, columns)

    def unpivot(
        self,
        *,
        index: list[str] | None = None,
        on: list[str] | None = None,
        variable_name: str = "variable",
        value_name: str = "value",
    ) -> Dataset:
        """Reshape wide → long (SQL ``UNPIVOT`` / pandas ``melt`` / Polars ``unpivot``).

        Each row becomes one row per `on` column: the `index` columns repeat, plus a
        `variable_name` column (the melted column's name) and a `value_name` column
        (its value). Omit `on` to melt every non-`index` column, or omit `index` to
        keep every non-`on` column as an identifier. The `on` columns must share a type.

        Args:
            index: The identifier columns that repeat per melted column.
            on: The columns to melt; ``None`` melts every non-`index` column.
            variable_name: The name of the column holding each melted column's name.
            value_name: The name of the column holding each melted value.

        Returns:
            A new `Dataset` reshaped from wide to long.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1], "a": [10], "b": [20]})
                >>> ds.unpivot(index=["id"]).to_pydict()
                {'id': [1, 1], 'variable': ['a', 'b'], 'value': [10, 20]}
        """
        return build_unpivot(self, index, on, variable_name, value_name)

    def fill_null(
        self,
        value: Any | dict[str, Any] | None = None,
        *,
        strategy: str | None = None,
        subset: list[str] | None = None,
        order_by: list[str] | None = None,
        partition_by: list[str] | None = None,
    ) -> Dataset:
        """Replace nulls with `value` (one for all columns, or a ``{col: value}`` dict).

        Pass `strategy` instead of `value` to fill from a statistic — ``"mean"``,
        ``"min"``, ``"max"`` (the column's whole-relation aggregate) or ``"zero"`` — or
        to carry a neighbouring value: ``"forward"`` / ``"backward"``. The carrying
        strategies **require `order_by`**, because a fill moves values along a row order
        and a relation has none by itself; `partition_by` keeps each series independent.
        `subset` limits a strategy fill to specific columns; the `order_by` /
        `partition_by` keys are never filled, being the frame of reference.

        Args:
            value: A fill value for every column, or a ``{column: value}`` mapping.
            strategy: ``"zero"``, ``"mean"``, ``"min"``, ``"max"``, ``"forward"``, or
                ``"backward"``. Mutually exclusive with `value`.
            subset: Columns the fill applies to; ``None`` means every column whose
                type can hold `value` (and every column, for a strategy fill).
            order_by: Columns defining the row order the fill carries along. Required
                for ``"forward"`` / ``"backward"``, ignored by the other strategies.
            partition_by: Columns whose groups the fill must not cross.

        Returns:
            A new lazy `Dataset` with nulls replaced.

        Raises:
            PlanError: If both `value` and `strategy` are given, if neither is, if the
                strategy is unknown, or if a carrying strategy has no `order_by`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, None, 3]})
                >>> ds.fill_null(0).to_pydict()
                {'x': [1, 0, 3]}

                >>> readings = bt.from_pydict(
                ...     {"t": [1, 2, 3, 4], "temp": [20.0, None, None, 23.0]}
                ... )
                >>> readings.fill_null(strategy="forward", order_by=["t"]).to_pydict()
                {'t': [1, 2, 3, 4], 'temp': [20.0, 20.0, 20.0, 23.0]}
        """
        if strategy is not None:
            if value is not None:
                raise PlanError("fill_null(): pass either `value` or `strategy`, not both")
            return build_fill_null_strategy(self, strategy, subset, order_by, partition_by)
        if value is None:
            raise PlanError("fill_null(): provide a `value` or a `strategy`")
        return build_fill_null(self, value, subset)

    def drop_nulls(self, subset: list[str] | None = None, *, how: str = "any") -> Dataset:
        """Drop rows that are null in any of `subset` (default: any column).

        The row-filtering counterpart to `fill_null`: with ``how="any"`` a row
        survives only if all of the considered columns are non-null. ``how="all"``
        drops a row only when *every* considered column is null, which is the pandas
        ``dropna(how="all")`` behaviour. Lazy.

        Args:
            subset: Columns to check for nulls; ``None`` checks every column.
            how: ``"any"`` drops a row with any null; ``"all"`` only when all of the
                considered columns are null.

        Returns:
            A new `Dataset` with the null-containing rows removed.

        Raises:
            PlanError: If `how` is not ``"any"`` or ``"all"``, or a column is unknown.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, None, 3]})
                >>> ds.drop_nulls().to_pydict()
                {'x': [1, 3]}

                >>> ds = bt.from_pydict({"x": [1, None], "y": [None, None]})
                >>> ds.drop_nulls(how="all").to_pydict()
                {'x': [1], 'y': [None]}
        """
        if how == "any":
            return build_drop_nulls(self, subset)
        if how != "all":
            raise PlanError(f"drop_nulls(): how must be 'any' or 'all', got {how!r}")
        cols = list(self.columns) if subset is None else list(subset)
        unknown = set(cols) - set(self.columns)
        if unknown:
            raise PlanError(
                f"drop_nulls(): unknown column(s) {_unknown_cols(unknown, self.columns)}"
            )
        keep = Col(cols[0]).is_not_null()
        for c in cols[1:]:
            keep = keep | Col(c).is_not_null()
        return self.filter(keep)

    def cast(self, dtypes: str | dict[str, str], *, strict: bool = True) -> Dataset:
        """Cast columns to `dtypes` — one dtype for all, or per-column via a dict.

        With `strict=False`, values that cannot be converted become NULL (DuckDB
        ``TRY_CAST``) instead of erroring the query — the safe-ingest spelling.

        Args:
            dtypes: One dtype for all columns, or a ``{column: dtype}`` mapping.
            strict: Error on an invalid value (the default); ``False`` casts it to NULL.

        Returns:
            A new `Dataset` with the columns cast.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.cast({"x": "float64"}).to_pydict()
                {'x': [1.0, 2.0, 3.0]}
        """
        return build_cast(self, dtypes, strict=strict)

    def union(self, *others: Dataset, distinct: bool = False) -> Dataset:
        """Concatenate with other datasets (UNION ALL, or UNION if `distinct`).

        All datasets must have identical columns. Sources are merged so each
        side's scans resolve correctly.

        Args:
            *others: The datasets to concatenate; each must share this one's columns.
            distinct: Deduplicate the result (UNION) instead of keeping all rows.

        Returns:
            A new `Dataset` concatenating the inputs.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> a = bt.from_pydict({"x": [1, 2]})
                >>> b = bt.from_pydict({"x": [3, 4]})
                >>> a.union(b).to_pydict()
                {'x': [1, 2, 3, 4]}
        """
        plans: list[LogicalPlan] = [self._plan]
        sources = list(self._sources)
        for other in others:
            plans.append(remap_sources(other._plan, len(sources)))
            sources.extend(other._sources)
        return Dataset(Union(tuple(plans), distinct), sources)

    def intersect(self, other: Dataset, *, distinct: bool = True) -> Dataset:
        """Rows present in BOTH datasets (SQL INTERSECT, or INTERSECT ALL if not `distinct`).

        NULLs compare equal, matching SQL set semantics: a row that is identical —
        nulls included — in both inputs is in the result. `distinct` (the default)
        returns each such row once; ``distinct=False`` is INTERSECT ALL, keeping a row
        ``min(left_count, right_count)`` times.

        Args:
            other: The dataset to intersect with; must share this one's columns.
            distinct: Deduplicate the result (INTERSECT) instead of keeping multiplicity.

        Returns:
            A new `Dataset` of the rows present in both inputs.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> a = bt.from_pydict({"x": [1, 2, 3]})
                >>> b = bt.from_pydict({"x": [2, 3, 4]})
                >>> a.intersect(b).sort("x").to_pydict()
                {'x': [2, 3]}

                >>> a = bt.from_pydict({"x": [1, 1, 2]})
                >>> b = bt.from_pydict({"x": [1, 1, 3]})
                >>> a.intersect(b, distinct=False).sort("x").to_pydict()
                {'x': [1, 1]}
        """
        cols = self._same_columns(other, "intersect")
        return self._set_membership(other, cols, both=True, distinct=distinct)

    def except_(self, other: Dataset, *, distinct: bool = True) -> Dataset:
        """Rows in this dataset but NOT in `other` (SQL EXCEPT, or EXCEPT ALL if not `distinct`).

        NULLs compare equal (a wholly-null row in both inputs is excluded), matching
        SQL set semantics. `distinct` (the default) returns each surviving row once;
        ``distinct=False`` is EXCEPT ALL, keeping a row
        ``max(left_count - right_count, 0)`` times.

        Args:
            other: The dataset whose rows to subtract; must share this one's columns.
            distinct: Deduplicate the result (EXCEPT) instead of keeping multiplicity.

        Returns:
            A new `Dataset` of the rows in this but not `other`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> a = bt.from_pydict({"x": [1, 2, 3]})
                >>> b = bt.from_pydict({"x": [2]})
                >>> a.except_(b).sort("x").to_pydict()
                {'x': [1, 3]}

                >>> a = bt.from_pydict({"x": [1, 1, 2]})
                >>> b = bt.from_pydict({"x": [1]})
                >>> a.except_(b, distinct=False).sort("x").to_pydict()
                {'x': [1, 2]}
        """
        cols = self._same_columns(other, "except")
        return self._set_membership(other, cols, both=False, distinct=distinct)

    def _set_membership(
        self, other: Dataset, cols: list[str], *, both: bool, distinct: bool
    ) -> Dataset:
        """INTERSECT/EXCEPT via group-by membership flags.

        Tag each side, union, then group by *all* columns. Grouping treats NULL as a
        single group, so NULLs compare equal — the SQL set-operation semantics a hash
        join cannot give (it drops NULL keys). `bool_or` records presence on each side
        per group; keep groups in both (INTERSECT) or only the left (EXCEPT). One row
        per distinct combination, so the result is DISTINCT by construction, and the
        whole thing is mergeable aggregation, so it distributes.

        The ALL forms (`distinct=False`) need multiplicity, which a membership flag
        cannot carry. Number each row within its run of identical rows first, and the
        k-th copy on the left then meets the k-th copy on the right under the very same
        membership group-by, now keyed on (columns, ordinal). Keeping the groups in both
        sides leaves ordinals 1..min(cl, cr) — INTERSECT ALL; keeping the left-only ones
        leaves cr+1..cl — EXCEPT ALL. The ordinal's ORDER BY is the partition columns
        themselves: every row in a partition is identical, so the order is a pure
        tie-break and any assignment yields the same multiset.
        """
        from batcher.plan.expr_ir import col, lit
        from batcher.plan.expr_ir.nodes import row_number

        keys = list(cols)
        left, right = self.select(*cols), other.select(*cols)
        if not distinct:
            ordinal = row_number().over(partition_by=cols, order_by=cols)
            left = left.with_columns(__bc_n__=ordinal)
            right = right.with_columns(__bc_n__=ordinal)
            keys = [*cols, "__bc_n__"]
        left = left.with_columns(__bc_l__=lit(True), __bc_r__=lit(False))
        right = right.with_columns(__bc_l__=lit(False), __bc_r__=lit(True))
        grouped = (
            left.union(right)
            .group_by(*keys)
            .agg(__bc_in_l__=col("__bc_l__").bool_or(), __bc_in_r__=col("__bc_r__").bool_or())
        )
        in_l, in_r = col("__bc_in_l__"), col("__bc_in_r__")
        keep = (in_l & in_r) if both else (in_l & ~in_r)
        return grouped.filter(keep).select(*cols)

    def _same_columns(self, other: Dataset, op: str) -> list[str]:
        if self.columns != other.columns:
            raise PlanError(f"{op} requires identical columns: {self.columns} vs {other.columns}")
        return list(self.columns)

    def limit(self, n: int, offset: int = 0) -> Dataset:
        """Take at most `n` rows, after skipping the first `offset`.

        The SQL ``LIMIT`` / ``OFFSET``. Pair it with `sort` for a deterministic
        result — without an order, which rows you get is unspecified. To find the
        largest or smallest rows, prefer `top_k`, which the optimizer can push down
        instead of sorting the whole dataset.

        Args:
            n: Maximum number of rows to return.
            offset: Number of leading rows to skip first.

        Returns:
            A new `Dataset` with at most `n` rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3, 4, 5]})
                >>> ds.sort("x").limit(2, offset=1).to_pydict()
                {'x': [2, 3]}
        """
        n = require_int(n, func="limit", arg="n", minimum=0)
        offset = require_int(offset, func="limit", arg="offset", minimum=0)
        return self._derive(Limit(self._plan, n, offset))

    def head(self, n: int = 5) -> Dataset:
        """Keep the first `n` rows (alias for ``limit(n)``).

        Lazy — returns a new `Dataset`. Without a preceding `sort` the rows are in
        an unspecified order, so pair it with `sort` (or use `top_k`) when you need
        a deterministic preview.

        Args:
            n: Maximum number of rows to keep.

        Returns:
            A new `Dataset` with at most `n` rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3, 4, 5]}).sort("x").head(2).to_pydict()
                {'x': [1, 2]}
        """
        return self.limit(n)

    def tail(self, n: int = 5) -> Dataset:
        """Keep the last `n` rows.

        Unlike `head`, this needs to know how many rows there are, so it **executes a
        `count` eagerly** (often answered from metadata with no scan) before building
        the lazy plan that selects the trailing rows. Without a preceding `sort` the
        rows are in an unspecified order.

        Args:
            n: Maximum number of rows to keep.

        Returns:
            A new `Dataset` with at most `n` rows.

        Raises:
            PlanError: If `n` is negative.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3, 4, 5]}).sort("x").tail(2).to_pydict()
                {'x': [4, 5]}
        """
        if n < 0:
            raise PlanError(f"tail(): n must be non-negative, got {n}")
        total = self.count()
        if n >= total:
            return self
        idx = "__bc_tail_idx"
        return self.with_row_index(idx).filter(Col(idx) >= total - n).drop(idx)

    def gather_every(self, n: int, offset: int = 0) -> Dataset:
        """Keep every `n`-th row, starting at `offset` — Polars ``gather_every``.

        A lazy downsample: rows ``offset, offset + n, offset + 2n, …`` in current order
        (put a `sort` first for a defined order). Composes a row index with a filter, so
        it stays streaming and adds no operator.

        Args:
            n: Keep one row out of every `n` (must be >= 1).
            offset: The 0-based index of the first row kept.

        Returns:
            A new `Dataset` with every `n`-th row.

        Raises:
            PlanError: If `n` < 1 or `offset` < 0.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [10, 20, 30, 40, 50]}).gather_every(2).to_pydict()
                {'x': [10, 30, 50]}
        """
        n = require_int(n, func="gather_every", arg="n", minimum=1)
        offset = require_int(offset, func="gather_every", arg="offset", minimum=0)
        idx = "__bc_gather_idx"
        keep = (Col(idx) >= offset) & ((Col(idx) - offset) % n == 0)
        return self.with_row_index(idx).filter(keep).drop(idx)

    def reverse(self) -> Dataset:
        """Reverse the row order — Polars ``reverse``.

        Materializes a row index and sorts on it descending, so the last row becomes the
        first. A pipeline breaker (like any sort).

        Returns:
            A new `Dataset` with the rows in reverse order.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3]}).reverse().to_pydict()
                {'x': [3, 2, 1]}
        """
        idx = "__bc_reverse_idx"
        return self.with_row_index(idx).sort(idx, descending=True).drop(idx)

    def bottom_k(self, k: int, by: str | list[str]) -> Dataset:
        """The `k` rows with the smallest `by` — the Polars ``bottom_k`` spelling of ``top_k``.

        The ascending-order companion to :meth:`top_k`; equivalent to
        ``top_k(k, by, descending=False)``.

        Args:
            k: How many rows to keep.
            by: The column(s) to rank by, ascending.

        Returns:
            A new `Dataset` of the `k` rows with the smallest `by` values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [5, 3, 8, 1]}).bottom_k(2, "x").sort("x").to_pydict()
                {'x': [1, 3]}
        """
        return self.top_k(require_int(k, func="bottom_k", arg="k"), by, descending=False)

    def slice(self, offset: int, length: int | None = None) -> Dataset:
        """Rows ``[offset, offset + length)`` — the Polars ``slice`` spelling of ``limit``.

        Args:
            offset: 0-based index of the first row kept.
            length: How many rows to keep; to the end when ``None``.

        Returns:
            A new `Dataset` of the selected row range.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3, 4, 5]}).slice(1, 2).to_pydict()
                {'x': [2, 3]}
        """
        offset = require_int(offset, func="slice", arg="offset")
        length = self.count() if length is None else require_int(length, func="slice", arg="length")
        return self.limit(length, offset)

    def melt(
        self,
        *,
        index: list[str] | None = None,
        on: list[str] | None = None,
        variable_name: str = "variable",
        value_name: str = "value",
        id_vars: str | list[str] | None = None,
        value_vars: str | list[str] | None = None,
        var_name: str | None = None,
    ) -> Dataset:
        """Reshape wide → long — the pandas ``melt`` spelling of :meth:`unpivot`.

        Accepts the pandas argument names as well: `id_vars` for `index`,
        `value_vars` for `on`, and `var_name` for `variable_name`. A single column
        name may be given where pandas allows one instead of a list.

        Args:
            index: Columns to keep as identifiers (repeated per melted column).
            on: Columns to melt; all non-`index` columns when ``None``.
            variable_name: Name of the output column holding the melted column names.
            value_name: Name of the output column holding the melted values.
            id_vars: The pandas spelling of `index`.
            value_vars: The pandas spelling of `on`.
            var_name: The pandas spelling of `variable_name`.

        Returns:
            A new long-format `Dataset`.

        Raises:
            PlanError: If an alias conflicts with the name it aliases.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1], "a": [10], "b": [20]})
                >>> ds.melt(index=["id"]).sort("variable").to_pydict()
                {'id': [1, 1], 'variable': ['a', 'b'], 'value': [10, 20]}

                >>> ds.melt(id_vars="id").sort("variable").to_pydict()
                {'id': [1, 1], 'variable': ['a', 'b'], 'value': [10, 20]}
        """
        index = _as_opt_str_list(_one_of(index, id_vars, "index", "id_vars"))
        on = _as_opt_str_list(_one_of(on, value_vars, "on", "value_vars"))
        if var_name is not None:
            if variable_name != "variable":
                raise PlanError("pass variable_name or var_name, not both")
            variable_name = var_name
        return self.unpivot(index=index, on=on, variable_name=variable_name, value_name=value_name)

    # --- row-oriented terminal consumers ---------------------------------------------
    # The boundary where a finished result becomes Python values. These stream batch
    # by batch rather than collecting, so walking a larger-than-memory result stays
    # bounded — and none of them puts Python inside the query.

    def iter_rows(self, *, named: bool = False) -> Iterator[tuple[Any, ...] | dict[str, Any]]:
        """Stream the result one row at a time, as tuples or dicts.

        A terminal operation. Rows arrive batch by batch, so this stays bounded on a
        result far larger than memory — unlike `to_pylist`, which materializes it.
        Per-row Python is fine *here*, at the end of a pipeline; inside a query, use
        expressions or `map_batches` instead.

        Args:
            named: Yield ``{column: value}`` dicts instead of positional tuples.

        Yields:
            One row per result row, as a tuple or a dict.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2], "y": ["a", "b"]})
                >>> list(ds.iter_rows())
                [(1, 'a'), (2, 'b')]

                >>> next(ds.iter_rows(named=True))
                {'x': 1, 'y': 'a'}
        """
        return build_iter_rows(self, named)

    def iter_slices(self, n_rows: int | None = None) -> Iterator[pa.RecordBatch]:
        """Stream the result as `RecordBatch` slices of at most `n_rows` rows.

        A terminal operation and the Polars spelling of `iter_batches`.

        Args:
            n_rows: Maximum rows per slice; ``None`` uses the engine's batch size.

        Yields:
            The result's `pyarrow.RecordBatch` slices, in order.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> sum(s.num_rows for s in ds.iter_slices())
                3
        """
        return build_iter_slices(self, n_rows)

    def first(self, *, named: bool = False) -> tuple[Any, ...] | dict[str, Any] | None:
        """The first result row, or ``None`` if the result is empty.

        A terminal operation. A relation has no inherent row order, so sort first
        when "first" has to mean something specific.

        Args:
            named: Return a ``{column: value}`` dict instead of a positional tuple.

        Returns:
            The first row, or ``None`` when there are no rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [3, 1, 2]}).sort("x").first()
                (1,)
        """
        return build_first(self, named)

    def last(self, *, named: bool = False) -> tuple[Any, ...] | dict[str, Any] | None:
        """The last result row, or ``None`` if the result is empty.

        A terminal operation. Unlike `first` this must drain the whole result, since
        a relation cannot be read backwards; sort first when "last" has to mean
        something specific.

        Args:
            named: Return a ``{column: value}`` dict instead of a positional tuple.

        Returns:
            The last row, or ``None`` when there are no rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [3, 1, 2]}).sort("x").last()
                (3,)
        """
        return build_last(self, named)

    def item(self, *, column: str | None = None) -> Any:
        """The single value of a one-row result — the Polars ``item``.

        A terminal operation for the "I just want the number" case. Raises rather
        than guessing if the result has no rows or more than one, so a query that
        silently started returning several rows fails loudly instead of returning
        the first one.

        Args:
            column: Which column to take; required when the result has several.

        Returns:
            The single scalar value.

        Raises:
            PlanError: If the result is not exactly one row, or `column` is needed
                and missing or unknown.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3]}).agg(total=bt.col("x").sum()).item()
                6
        """
        return build_item(self, column)

    # --- introspection a REPL user reaches for ---------------------------------------

    @property
    def width(self) -> int:
        """The number of output columns — the Polars ``width`` (free, no execution).

        Returns:
            The column count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1], "y": [2]}).width
                2
        """
        return len(self.columns)

    @property
    def height(self) -> int:
        """The number of result rows — the Polars ``height`` (executes a `count`).

        Returns:
            The row count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2]}).height
                2
        """
        return self.count()

    @property
    def empty(self) -> bool:
        """Whether the result has no rows — the pandas ``empty`` (executes a `count`).

        Returns:
            ``True`` if the result is empty.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1]}).empty
                False
        """
        return self.is_empty()

    def collect_schema(self) -> dict[str, pa.DataType]:
        """The output schema as an ordered ``{column: arrow_type}`` mapping.

        The dict-shaped counterpart of `schema` (which returns a `pyarrow.Schema`),
        matching how Polars spells the same question.

        Returns:
            Each output column mapped to its Arrow type, in column order.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> {k: str(v) for k, v in bt.from_pydict({"x": [1]}).collect_schema().items()}
                {'x': 'int64'}
        """
        return build_collect_schema(self)

    def info(self) -> None:
        """Print a pandas-style summary: row count, and each column's type and nulls.

        A terminal operation for interactive use: it executes a `count` and a
        `null_count`, never a full scan of the values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2]}).info()  # doctest: +SKIP
        """
        build_info(self)

    def glimpse(self, *, max_items_per_column: int = 10) -> None:
        """Print a transposed preview — one line per column — the Polars ``glimpse``.

        A terminal operation for interactive use: it reads a single bounded head
        slice, so it is cheap on a wide or long dataset.

        Args:
            max_items_per_column: How many sample values to show per column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2]}).glimpse()  # doctest: +SKIP
        """
        build_glimpse(self, max_items_per_column)

    def memory_usage(self) -> dict[str, int]:
        """An *estimated* in-memory size in bytes per column — the pandas ``memory_usage``.

        Estimated, not measured: it multiplies the row count by each Arrow type's
        width, using a nominal width for variable-width types (string, binary, list),
        whose real footprint cannot be known without reading the data.

        Returns:
            Each output column mapped to its estimated size in bytes.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3]}).memory_usage()
                {'x': 24}
        """
        return build_memory_usage(self)

    def equals(self, other: Dataset, *, ordered: bool = False) -> bool:
        """Whether `other` computes the same result as this dataset.

        Compares *results*, not plans: both sides execute and their rows are
        compared, so two differently-built queries that agree are equal. Column
        names and types must match. By default row order is ignored, because a
        relation is an unordered multiset; pass ``ordered=True`` after a `sort` to
        compare the emitted order too.

        Args:
            other: The dataset to compare against.
            ordered: Compare row order as well as row content.

        Returns:
            ``True`` if both sides produce the same result.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2]})
                >>> ds.equals(ds.filter(bt.col("x") > 0))
                True
                >>> ds.equals(ds.filter(bt.col("x") > 1))
                False
        """
        if self.columns != other.columns:
            return False
        left, right = self.collect(), other.collect()
        if left.schema != right.schema:
            return False
        if ordered:
            return left.equals(right)
        return sorted(map(repr, left.to_pylist())) == sorted(map(repr, right.to_pylist()))

    # --- interoperability protocols ---------------------------------------------------
    # Standard Python/Arrow protocols, so a Dataset drops into code that was never
    # written for Batcher: `np.asarray(ds)`, `pd.api.interchange.from_dataframe(ds)`.

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> Any:
        """Materialize as a 2-D NumPy array so ``np.asarray(ds)`` works.

        A terminal operation. Every column must share a common dtype for the result
        to be meaningful, which is NumPy's constraint, not Batcher's; use
        `to_numpy` for a per-column ``{name: array}`` mapping instead.

        Args:
            dtype: The NumPy dtype to coerce to; inferred when ``None``.
            copy: Accepted for NumPy 2 compatibility. The result is always a fresh
                array (it is computed), so ``copy=False`` cannot be honoured.

        Returns:
            A ``(rows, columns)`` NumPy array of the result.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import numpy as np
                >>> np.asarray(bt.from_pydict({"x": [1, 2]})).shape
                (2, 1)
        """
        import numpy as np

        columns = self.to_numpy()
        stacked = np.column_stack([columns[name] for name in self.columns])
        return stacked.astype(dtype) if dtype is not None else stacked

    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True) -> Any:
        """Expose the result through the DataFrame Interchange Protocol.

        A terminal operation. Lets any consumer of the protocol (pandas, Polars,
        Vaex, plotting libraries) read a `Dataset` without knowing about Batcher:
        ``pandas.api.interchange.from_dataframe(ds)``.

        Args:
            nan_as_null: Passed through to the underlying Arrow implementation.
            allow_copy: Passed through to the underlying Arrow implementation.

        Returns:
            The interchange object for the collected Arrow table.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2]}).__dataframe__() is not None
                True
        """
        return self.collect().__dataframe__(nan_as_null=nan_as_null, allow_copy=allow_copy)

    # --- ecosystem spellings ----------------------------------------------------------
    # A migrant finds the operation under the name they already type. Each of these
    # delegates to the Batcher primary — same plan, same semantics, no second
    # implementation to keep in step.

    def to_dicts(self) -> list[dict[str, Any]]:
        """Row-oriented list of dicts — the Polars ``to_dicts`` spelling of :meth:`to_pylist`.

        Returns:
            One ``{column: value}`` dict per row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"a": [1, 2]}).to_dicts()
                [{'a': 1}, {'a': 2}]
        """
        return self.to_pylist()

    def to_dict(self) -> dict[str, list[Any]]:
        """Column-oriented dict — the pandas ``to_dict("list")`` spelling of :meth:`to_pydict`.

        Returns:
            Each column name mapped to its list of values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"a": [1, 2]}).to_dict()
                {'a': [1, 2]}
        """
        return self.to_pydict()

    def drop_duplicates(self, subset: list[str] | None = None) -> Dataset:
        """Remove duplicate rows — the pandas ``drop_duplicates`` spelling of :meth:`distinct`.

        Args:
            subset: Consider only these columns when comparing; all when ``None``.

        Returns:
            A new `Dataset` without duplicate rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 1, 2]}).drop_duplicates().count()
                2
        """
        return self.distinct(subset)

    def with_row_count(self, name: str = "index", *, offset: int = 0) -> Dataset:
        """Add a row-number column — the older Polars spelling of :meth:`with_row_index`.

        Args:
            name: Name of the new row-number column.
            offset: The value of the first row's index.

        Returns:
            A new `Dataset` with the row-number column prepended.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [7, 8]}).with_row_count().columns
                ['index', 'x']
        """
        return self.with_row_index(name, offset=offset)

    def vstack(self, other: Dataset) -> Dataset:
        """Stack `other`'s rows below this one — the Polars ``vstack`` spelling of :meth:`union`.

        Args:
            other: The dataset whose rows to append.

        Returns:
            A new `Dataset` with both sets of rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1]})
                >>> ds.vstack(ds).count()
                2
        """
        return self.union(other)

    def append(self, other: Dataset) -> Dataset:
        """Append `other`'s rows — the pandas ``append`` spelling of :meth:`union`.

        Returns a new `Dataset`; nothing is appended in place, because a `Dataset`
        is immutable.

        Args:
            other: The dataset whose rows to append.

        Returns:
            A new `Dataset` with both sets of rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1]})
                >>> ds.append(ds).count()
                2
        """
        return self.union(other)

    def difference(self, other: Dataset) -> Dataset:
        """Rows in this dataset but not `other` — the SQL ``EXCEPT`` spelling of :meth:`except_`.

        Args:
            other: The dataset whose rows to subtract.

        Returns:
            A new `Dataset` of the rows only this side has.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> a = bt.from_pydict({"x": [1, 2]})
                >>> b = bt.from_pydict({"x": [2]})
                >>> a.difference(b).to_pydict()
                {'x': [1]}
        """
        return self.except_(other)

    def persist(self) -> Dataset:
        """Keep this result in the process cache — the Spark ``persist`` spelling of :meth:`cache`.

        Returns:
            A new `Dataset` whose collected result is cached.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1]}).persist().count()
                1
        """
        return self.cache()

    def coalesce(self, n: int) -> Dataset:
        """Reduce the output to `n` partitions — the Spark ``coalesce`` spelling.

        A pre-write layout hint, like :meth:`repartition`, that controls how many
        files a subsequent `write` produces.

        Args:
            n: The target number of output partitions.

        Returns:
            A new `Dataset` carrying the output-partition hint.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2]}).coalesce(1).count()
                2
        """
        return self.repartition(require_int(n, func="coalesce", arg="n", minimum=1))

    def lazy(self) -> Dataset:
        """Return this dataset unchanged — a `Dataset` is always lazy.

        Present so a Polars script that calls ``.lazy()`` runs unmodified. There is
        no eager mode to switch out of.

        Returns:
            This same `Dataset`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1]})
                >>> ds.lazy() is ds
                True
        """
        return self

    def copy(self) -> Dataset:
        """Return this dataset unchanged — a `Dataset` is immutable, so copying is a no-op.

        Present so a pandas script that defensively copies runs unmodified. Every
        Batcher operation already returns a new `Dataset`, so there is no shared
        mutable state to defend against.

        Returns:
            This same `Dataset`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1]})
                >>> ds.copy() is ds
                True
        """
        return self

    def query(self, expr: str) -> Dataset:
        """Keep rows matching a SQL boolean `expr` — the pandas ``query`` spelling.

        The string is a SQL ``WHERE`` clause over this dataset's columns, evaluated
        by the same SQL front end as :meth:`sql`. Prefer expressions
        (``ds.filter(col("x") > 1)``) in code you own: they are checked when the plan
        is built rather than when the string is parsed.

        Args:
            expr: A SQL boolean expression over this dataset's columns.

        Returns:
            A new `Dataset` with the matching rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 5, 9]})
                >>> ds.query("x > 2").to_pydict()
                {'x': [5, 9]}
        """
        return self.sql(f"SELECT * FROM self WHERE {expr}")

    def to_csv(self, path: str, **options: Any) -> Any:
        """Write the result as CSV — the pandas ``to_csv`` spelling of ``ds.write.csv``.

        Args:
            path: Destination path or URI.
            **options: Forwarded to ``ds.write.csv`` (``partition_by=``, ``mode=``, …).

        Returns:
            The `WriteManifest` describing what was written.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import os, tempfile
                >>> ds = bt.from_pydict({"x": [1, 2]})
                >>> with tempfile.TemporaryDirectory() as d:
                ...     _ = ds.to_csv(os.path.join(d, "out.csv"))
                ...     bt.read.csv(os.path.join(d, "out.csv")).count()
                2
        """
        return self.write.csv(path, **options)

    def to_parquet(self, path: str, **options: Any) -> Any:
        """Write the result as Parquet — the pandas ``to_parquet`` spelling of ``ds.write.parquet``.

        Args:
            path: Destination path or URI.
            **options: Forwarded to ``ds.write.parquet`` (``partition_by=``, ``mode=``, …).

        Returns:
            The `WriteManifest` describing what was written.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import os, tempfile
                >>> ds = bt.from_pydict({"x": [1, 2]})
                >>> with tempfile.TemporaryDirectory() as d:
                ...     _ = ds.to_parquet(os.path.join(d, "out.parquet"))
                ...     bt.read(os.path.join(d, "out.parquet")).count()
                2
        """
        return self.write.parquet(path, **options)

    def to_json(self, path: str, **options: Any) -> Any:
        """Write the result as JSON — the pandas ``to_json`` spelling of ``ds.write.json``.

        Args:
            path: Destination path or URI.
            **options: Forwarded to ``ds.write.json`` (``partition_by=``, ``mode=``, …).

        Returns:
            The `WriteManifest` describing what was written.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import os, tempfile
                >>> ds = bt.from_pydict({"x": [1, 2]})
                >>> with tempfile.TemporaryDirectory() as d:
                ...     _ = ds.to_json(os.path.join(d, "out.json"))
                ...     bt.read.json(os.path.join(d, "out.json")).count()
                2
        """
        return self.write.json(path, **options)

    def transform(self, fn: Callable[[Dataset], _T], *args: Any, **kwargs: Any) -> _T:
        """Apply `fn` to this whole dataset — the ``transform`` spelling of :meth:`pipe`.

        Args:
            fn: A function taking this `Dataset` and returning anything.
            *args: Extra positional arguments forwarded to `fn`.
            **kwargs: Extra keyword arguments forwarded to `fn`.

        Returns:
            Whatever `fn` returns.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.transform(lambda d: d.filter(bt.col("x") > 1)).count()
                2
        """
        return self.pipe(fn, *args, **kwargs)

    # --- pandas-compatible spellings ------------------------------------------------
    # A data scientist arriving from pandas finds the operation under the name they
    # already type. Each delegates to the Batcher primary — same plan, same semantics.

    def fillna(self, value: Any | dict[str, Any], *, subset: list[str] | None = None) -> Dataset:
        """Replace nulls with `value` — the pandas ``fillna`` spelling of :meth:`fill_null`.

        As in pandas, a scalar fills every column whose type can hold it, so
        ``fillna(0)`` on a frame of numbers and strings fills the numbers.

        Args:
            value: A scalar for every column, or a ``{column: value}`` mapping.
            subset: Columns to fill; ``None`` means every compatible column.

        Returns:
            A new `Dataset` with nulls replaced.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, None, 3]}).fillna(0).to_pydict()
                {'x': [1, 0, 3]}
        """
        return self.fill_null(value, subset=subset)

    def dropna(self, subset: list[str] | None = None) -> Dataset:
        """Drop rows containing nulls — the pandas ``dropna`` spelling of :meth:`drop_nulls`.

        Args:
            subset: Only consider these columns; all columns when ``None``.

        Returns:
            A new `Dataset` without the null-bearing rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, None, 3]}).dropna().to_pydict()
                {'x': [1, 3]}
        """
        return self.drop_nulls(subset)

    def isna(self) -> Dataset:
        """A same-shaped dataset of null indicators — the pandas ``isna`` null mask.

        Every column becomes a boolean column, true where the original was null. The
        quickest way to profile or visualize missingness.

        Returns:
            A new `Dataset` of booleans, one column per input column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, None]}).isna().to_pydict()
                {'x': [False, True]}
        """
        return self.select(**{name: Col(name).is_null() for name in self.columns})

    def notna(self) -> Dataset:
        """A same-shaped dataset of presence indicators — the pandas ``notna`` mask.

        Returns:
            A new `Dataset` of booleans, true where the original value is present.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, None]}).notna().to_pydict()
                {'x': [True, False]}
        """
        return self.select(**{name: Col(name).is_not_null() for name in self.columns})

    def astype(self, dtypes: str | dict[str, str]) -> Dataset:
        """Cast columns — the pandas ``astype`` spelling of :meth:`cast`.

        Args:
            dtypes: One Arrow type name for every column, or a ``{column: type}`` map.

        Returns:
            A new `Dataset` with the columns cast.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2]}).astype({"x": "float64"}).to_pydict()
                {'x': [1.0, 2.0]}
        """
        return self.cast(dtypes)

    def assign(self, **named: Expr | int | float | bool | str) -> Dataset:
        """Add or replace columns — the pandas ``assign`` spelling of :meth:`with_columns`.

        Args:
            named: Output column name to the expression (or constant) computing it.

        Returns:
            A new `Dataset` with the columns added or replaced.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2]}).assign(y=bt.col("x") * 10).to_pydict()
                {'x': [1, 2], 'y': [10, 20]}
        """
        return self.with_columns(**named)

    def groupby(self, *keys: str, **named: Expr) -> GroupBy:
        """Start an aggregation — the pandas ``groupby`` spelling of :meth:`group_by`.

        Args:
            keys: Column names to group by.
            named: Derived grouping keys, as ``name=expression``.

        Returns:
            A `GroupBy` to finalize with ``.agg(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "a"], "x": [1, 2]})
                >>> ds.groupby("g").agg(t=bt.col("x").sum()).to_pydict()
                {'g': ['a'], 't': [3]}
        """
        return self.group_by(*keys, **named)

    def merge(
        self,
        other: Dataset,
        on: str | list[str] | None = None,
        *,
        left_on: str | list[str] | None = None,
        right_on: str | list[str] | None = None,
        how: str = "inner",
        suffix: str = "_right",
    ) -> Dataset:
        """Join two datasets — the pandas ``merge`` spelling of :meth:`join`.

        Args:
            other: The right-hand dataset.
            on: Shared key column name(s).
            left_on: Left key column(s) when the names differ.
            right_on: Right key column(s) when the names differ.
            how: Join type — inner, left, right, full, outer, semi, or anti.
            suffix: Appended to right-hand columns whose names collide.

        Returns:
            A new joined `Dataset`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> a = bt.from_pydict({"k": [1], "x": ["a"]})
                >>> b = bt.from_pydict({"k": [1], "y": ["b"]})
                >>> a.merge(b, on="k").to_pydict()
                {'k': [1], 'x': ['a'], 'y': ['b']}
        """
        return self.join(other, on, left_on=left_on, right_on=right_on, how=how, suffix=suffix)

    def sort_values(self, by: str | list[str], *, ascending: bool | list[bool] = True) -> Dataset:
        """Order rows — the pandas ``sort_values`` spelling of :meth:`sort`.

        Args:
            by: Column name(s) to order by.
            ascending: Sort ascending (pandas' sense); the inverse of `sort`'s
                ``descending``.

        Returns:
            A new ordered `Dataset`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [3, 1, 2]}).sort_values("x").to_pydict()
                {'x': [1, 2, 3]}
        """
        keys = [by] if isinstance(by, str) else list(by)
        desc = [not a for a in ascending] if isinstance(ascending, list) else not ascending
        return self.sort(*keys, descending=desc)

    def nlargest(self, n: int, columns: str | list[str]) -> Dataset:
        """The `n` rows with the largest `columns` — the pandas ``nlargest`` of :meth:`top_k`.

        Args:
            n: How many rows to keep.
            columns: The column(s) to rank by, descending.

        Returns:
            A new `Dataset` of the `n` largest rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [5, 3, 8]}).nlargest(2, "x").sort("x").to_pydict()
                {'x': [5, 8]}
        """
        return self.top_k(require_int(n, func="nlargest", arg="n"), columns)

    def nsmallest(self, n: int, columns: str | list[str]) -> Dataset:
        """The `n` rows with the smallest `columns` — the pandas ``nsmallest``.

        Args:
            n: How many rows to keep.
            columns: The column(s) to rank by, ascending.

        Returns:
            A new `Dataset` of the `n` smallest rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [5, 3, 8]}).nsmallest(2, "x").sort("x").to_pydict()
                {'x': [3, 5]}
        """
        return self.bottom_k(require_int(n, func="nsmallest", arg="n"), columns)

    def round(self, decimals: int = 0) -> Dataset:
        """Round every numeric column to `decimals` places — the pandas ``round``.

        Non-numeric columns pass through untouched (the numeric selector picks the
        columns).

        Args:
            decimals: How many decimal places to keep.

        Returns:
            A new `Dataset` with the numeric columns rounded.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1.234], "s": ["a"]}).round(1).to_pydict()
                {'x': [1.2], 's': ['a']}
        """
        from batcher.plan.expr_ir.selectors import numeric

        return self.with_columns(numeric().round(decimals))

    def abs(self) -> Dataset:
        """Absolute value of every numeric column — the pandas ``abs``.

        Returns:
            A new `Dataset` with the numeric columns made non-negative.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [-1.5], "s": ["a"]}).abs().to_pydict()
                {'x': [1.5], 's': ['a']}
        """
        from batcher.plan.expr_ir.selectors import numeric

        return self.with_columns(numeric().abs())

    def clip(self, lower: float | None = None, upper: float | None = None) -> Dataset:
        """Clamp every numeric column into ``[lower, upper]`` — the pandas ``clip``.

        Args:
            lower: Lower bound; omit for no lower clamp.
            upper: Upper bound; omit for no upper clamp.

        Returns:
            A new `Dataset` with the numeric columns clamped.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [-5, 5, 50]}).clip(0, 10).to_pydict()
                {'x': [0, 5, 10]}
        """
        from batcher.plan.expr_ir.selectors import numeric

        return self.with_columns(numeric().clip(lower, upper))

    def nunique(self) -> Dataset:
        """Distinct value count per column, as a single row (pandas ``nunique``).

        The companion to :meth:`null_count` for a first look at a table: which columns
        are keys, which are low-cardinality categoricals.

        Returns:
            A one-row `Dataset` with the same column names, holding distinct counts.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"a": [1, 1, 2], "b": [1, 2, 3]}).nunique().to_pydict()
                {'a': [2], 'b': [3]}
        """
        return self.agg(**{name: Col(name).n_unique() for name in self.columns})

    def select_dtypes(self, include: Any = None, exclude: Any = None) -> Dataset:
        """Keep only the columns of a dtype family (pandas ``select_dtypes``).

        A family is named the Batcher way (``"numeric"``, ``"integer"``,
        ``"floating"``, ``"string"``, ``"boolean"``, ``"temporal"``), or with any
        spelling pandas accepts for the same idea: a Python type (``int``,
        ``float``, ``str``, ``bool``), a concrete dtype name (``"int64"``,
        ``"float32"``, ``"utf8"``), or a list mixing them. Passing `exclude`
        instead keeps everything the families do *not* match.

        Args:
            include: A family, Python type, dtype name, or list of them to keep.
            exclude: The same, but for columns to drop. Mutually exclusive with
                `include`.

        Returns:
            A new `Dataset` with only the matching columns.

        Raises:
            PlanError: If neither or both of `include`/`exclude` is given, or if a
                family cannot be resolved.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [1], "s": ["x"]})
                >>> ds.select_dtypes("numeric").columns
                ['a']

                >>> ds.select_dtypes(int).columns
                ['a']

                >>> ds.select_dtypes(exclude="string").columns
                ['a']
        """
        if (include is None) == (exclude is None):
            raise PlanError("select_dtypes() takes exactly one of `include` or `exclude`")
        wanted = include if include is not None else exclude
        families = {_resolve_dtype_family(f) for f in _as_family_list(wanted)}
        matched = {c for family in families for c in selector_columns(self, family())}
        keep = [c for c in self.columns if (c in matched) is (include is not None)]
        if not keep:
            raise PlanError(
                f"select_dtypes(): no column matches {wanted!r}; the dataset's types "
                f"are {[str(t) for t in self.dtypes]}"
            )
        return self.select(*keep)

    def sample_frac(self, frac: float, *, seed: int | None = None) -> Dataset:
        """Sample a fraction of the rows — the pandas ``sample(frac=…)`` spelling.

        Args:
            frac: The fraction of rows to keep, in ``[0, 1]``.
            seed: Seed making the sample reproducible.

        Returns:
            A new `Dataset` holding roughly `frac` of the rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": list(range(100))})
                >>> 0 < ds.sample_frac(0.5, seed=1).count() < 100
                True
        """
        return self.sample(fraction=require_float(frac, func="sample_frac", arg="frac"), seed=seed)

    def drop_constant_columns(self) -> Dataset:
        """Drop every column holding a single distinct value — the zero-variance filter.

        Constant columns carry no signal for a model and no information for a report.
        This inspects the data (it executes a distinct-count pass) and then builds the
        lazy projection that keeps the rest.

        Returns:
            A new `Dataset` without the constant columns.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"same": [1, 1, 1], "varies": [1, 2, 3]})
                >>> ds.drop_constant_columns().columns
                ['varies']
        """
        counts = self.nunique().to_pydict()
        constant = [name for name, values in counts.items() if values[0] <= 1]
        return self.drop(*constant) if constant else self

    def crosstab(self, index: str, columns: str) -> Dataset:
        """Contingency table of two categorical columns — the pandas ``crosstab``.

        Counts co-occurrences of `index` and `columns` and pivots them wide: one row per
        `index` value, one column per `columns` value. Combinations that never occur are
        null.

        Args:
            index: The column whose values become the rows.
            columns: The column whose values become the output columns.

        Returns:
            A new wide `Dataset` of co-occurrence counts.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": ["x", "x", "y"], "b": ["p", "q", "p"]})
                >>> ds.crosstab("a", "b").sort("a").to_pydict()
                {'a': ['x', 'y'], 'p': [1, 1], 'q': [1, None]}
        """
        from batcher.plan.expr_ir.constructors import count

        counted = self.group_by(index, columns).agg(__bc_n=count())
        return counted.pivot(index=[index], on=columns, values="__bc_n", aggregate="sum")

    def get_dummies(self, column: str, *, prefix: str | None = None) -> Dataset:
        """One-hot encode a categorical column — the pandas ``get_dummies``.

        Adds one 0/1 indicator column per distinct value, named ``{prefix}_{value}``.
        The distinct values are read from the data (an eager pass), then the indicators
        are built as an ordinary lazy projection.

        Args:
            column: The categorical column to encode.
            prefix: Prefix for the generated column names; the column name by default.

        Returns:
            A new `Dataset` with the indicator columns appended.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": ["x", "y"]})
                >>> ds.get_dummies("a").to_pydict()
                {'a': ['x', 'y'], 'a_x': [1, 0], 'a_y': [0, 1]}
        """
        from batcher.plan.expr_ir.core import Lit

        values = self.select(column).distinct().to_pydict()[column]
        present = sorted(v for v in values if v is not None)
        tag = column if prefix is None else prefix
        return self.with_columns(
            **{f"{tag}_{value}": (Col(column) == Lit(value)).cast("int64") for value in present}
        )

    # --- AI / ML pipeline helpers ---------------------------------------------------

    def shuffle(self, *, seed: int = 0) -> Dataset:
        """Randomly reorder the rows, reproducibly for a given `seed`.

        Training-set order matters: a corpus grouped by source teaches the model the
        grouping. This sorts on a seeded random key, so the permutation is identical
        across runs and across single-node, parallel, and distributed execution.

        Args:
            seed: Seed selecting the permutation.

        Returns:
            A new `Dataset` with the rows reordered.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3, 4, 5]}).shuffle(seed=7).to_pydict()
                {'x': [1, 2, 5, 3, 4]}
        """
        key = "__bc_shuffle_key"
        return self.with_random(key, seed=seed).sort(key).drop(key)

    def sample_per_group(
        self, by: str | list[str], n: int, *, order_by: str | None = None
    ) -> Dataset:
        """Keep at most `n` rows from each group — a balanced/capped sample.

        Caps over-represented classes or sources without dropping rare ones, which is how
        a skewed corpus is balanced before training.

        Args:
            by: The column(s) defining a group.
            n: Maximum rows to keep per group.
            order_by: Which rows to prefer; the first column of `by` order when omitted.

        Returns:
            A new `Dataset` with at most `n` rows per group.

        Raises:
            PlanError: If `n` < 1.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"y": ["a", "a", "a", "b"], "x": [1, 2, 3, 4]})
                >>> ds.sample_per_group("y", 2, order_by="x").to_pydict()
                {'y': ['a', 'a', 'b'], 'x': [1, 2, 4]}
        """
        from batcher.plan.expr_ir.nodes import row_number

        n = require_int(n, func="sample_per_group", arg="n", minimum=1)
        keys = [by] if isinstance(by, str) else list(by)
        order = order_by if order_by is not None else keys[0]
        rank = "__bc_group_rank"
        ranked = self.with_columns(**{rank: row_number().over(partition_by=keys, order_by=[order])})
        return ranked.filter(Col(rank) <= n).drop(rank)

    def stratified_split(
        self, by: str | list[str], test_size: float = 0.25, *, seed: int = 0
    ) -> tuple[Dataset, Dataset]:
        """Split into train/test keeping each group's proportion — a stratified split.

        A plain random split can starve a rare class. This ranks rows *within* each group
        by a stable hash of their own values, so each group contributes the same
        `test_size` fraction. Being value-hashed rather than position-based, the split is
        identical single-node, parallel, and distributed.

        Args:
            by: The column(s) whose proportions the split preserves (the label).
            test_size: Fraction of each group routed to the test side.
            seed: Seed for the row hash, selecting a different split.

        Returns:
            A ``(train, test)`` pair of disjoint `Dataset` objects covering every row.

        Raises:
            PlanError: If `test_size` is not in ``[0, 1]``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"y": ["a"] * 8 + ["b"] * 4, "x": list(range(12))})
                >>> train, test = ds.stratified_split("y", 0.25, seed=5)
                >>> test.group_by("y").agg(n=bt.count()).sort("y").to_pydict()
                {'y': ['a', 'b'], 'n': [2, 1]}
        """
        from batcher.plan.expr_ir import hash_rows

        if not 0.0 <= test_size <= 1.0:
            raise PlanError(f"stratified_split(): test_size must be in [0, 1], got {test_size}")
        keys = [by] if isinstance(by, str) else list(by)
        digest_col, pct = "__bc_stratify_hash", "__bc_stratify_pct"
        digest = hash_rows(*[Col(name) for name in self.columns], seed=seed)
        scored = self.with_columns(**{digest_col: digest}).with_columns(
            **{pct: Col(digest_col).rank_pct(keys)}
        )
        test = scored.filter(Col(pct) < test_size).drop(digest_col, pct)
        train = scored.filter(Col(pct) >= test_size).drop(digest_col, pct)
        return train, test

    def train_val_test_split(
        self, by: str | list[str], val_size: float = 0.15, test_size: float = 0.15, *, seed: int = 0
    ) -> tuple[Dataset, Dataset, Dataset]:
        """Three-way stratified split into train / validation / test.

        Applies :meth:`stratified_split` twice, so every class keeps its proportion in all
        three parts and the parts stay disjoint and complete. Value-hashed, so the split
        is identical single-node and distributed.

        Args:
            by: The column(s) whose proportions each part preserves (the label).
            val_size: Fraction of the whole routed to validation.
            test_size: Fraction of the whole routed to test.
            seed: Seed for the row hash.

        Returns:
            A ``(train, val, test)`` triple of disjoint `Dataset` objects.

        Raises:
            PlanError: If `val_size` + `test_size` is not below 1.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"y": ["a"] * 8 + ["b"] * 4, "x": list(range(12))})
                >>> train, val, test = ds.train_val_test_split("y", 0.25, 0.25, seed=1)
                >>> train.count() + val.count() + test.count()
                12
        """
        if val_size + test_size >= 1.0:
            raise PlanError(
                "train_val_test_split(): val_size + test_size must be < 1, got "
                f"{val_size} + {test_size}"
            )
        rest, test = self.stratified_split(by, test_size, seed=seed)
        # Rescale: `val_size` is a fraction of the whole, but `rest` is what remains.
        val_of_rest = val_size / (1.0 - test_size)
        train, val = rest.stratified_split(by, val_of_rest, seed=seed + 1)
        return train, val, test

    def balance_classes(self, label: str, *, order_by: str | None = None) -> Dataset:
        """Downsample every class to the size of the rarest — a balanced training set.

        The simplest fix for a skewed target when weighting is not an option. Inspects the
        class counts (an eager pass), then keeps that many rows from each class.

        Args:
            label: The categorical column to balance.
            order_by: Which rows to prefer within a class; the label order when omitted.

        Returns:
            A new `Dataset` holding an equal number of rows per class.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"y": ["a"] * 8 + ["b"] * 4, "x": list(range(12))})
                >>> ds.balance_classes("y", order_by="x").group_by("y").agg(
                ...     n=bt.count()
                ... ).sort("y").to_pydict()
                {'y': ['a', 'b'], 'n': [4, 4]}
        """
        from batcher.plan.expr_ir import count as count_star

        counts = self.group_by(label).agg(__bc_n=count_star()).to_pydict()["__bc_n"]
        smallest = min(counts) if counts else 0
        return self.sample_per_group(label, smallest, order_by=order_by)

    def filter_by_length(
        self, column: str, min_chars: int = 1, max_chars: int | None = None
    ) -> Dataset:
        """Keep rows whose text length falls in ``[min_chars, max_chars]``.

        The first filter of a corpus pipeline: drop stubs and runaway documents before
        anything expensive touches them.

        Args:
            column: The text column to measure.
            min_chars: Inclusive minimum length.
            max_chars: Inclusive maximum length; unbounded when ``None``.

        Returns:
            A new `Dataset` with the out-of-range rows removed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": ["hi", "a longer document"]})
                >>> ds.filter_by_length("t", 5).to_pydict()
                {'t': ['a longer document']}
        """
        length = Col(column).str.len()
        kept = self.filter(length >= min_chars)
        return kept if max_chars is None else kept.filter(length <= max_chars)

    def filter_by_token_budget(
        self, column: str, budget: int, *, chars_per_token: float = 4.0
    ) -> Dataset:
        """Keep rows whose estimated token count fits `budget` — the context-window filter.

        Uses the tokenizer-free estimate, so a corpus is sized without paying to tokenize
        it. Pair with `truncate_words` when you would rather trim than drop.

        Args:
            column: The text column to measure.
            budget: Maximum estimated tokens per row.
            chars_per_token: Characters per token to assume.

        Returns:
            A new `Dataset` holding only the rows that fit.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": ["abcd", "abcdefghijklmnop"]})
                >>> ds.filter_by_token_budget("t", 2).to_pydict()
                {'t': ['abcd']}
        """
        return self.filter(
            Col(column).str.fits_token_budget(budget, chars_per_token=chars_per_token)
        )

    def drop_empty(self, column: str) -> Dataset:
        """Drop rows where the text column is null, empty, or only whitespace.

        Args:
            column: The text column to check.

        Returns:
            A new `Dataset` without the blank rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"t": ["hi", "   ", None]})
                >>> ds.drop_empty("t").to_pydict()
                {'t': ['hi']}
        """
        text = Col(column)
        return self.filter(text.is_not_null() & ~text.str.is_blank())

    def class_balance(self, label: str) -> Dataset:
        """The fraction of rows in each class — the label distribution.

        The first thing to check before training: whether the target is skewed enough to
        need weighting or resampling.

        Args:
            label: The categorical column to summarize.

        Returns:
            A `Dataset` of one row per class, with a ``fraction`` column summing to 1.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"y": ["a", "a", "a", "b"]})
                >>> ds.class_balance("y").sort("y").to_pydict()
                {'y': ['a', 'b'], 'fraction': [0.75, 0.25]}
        """
        from batcher.plan.expr_ir import count, lit

        total = float(self.count())
        counts = self.group_by(label).agg(__bc_n=count())
        return counts.select(label, fraction=Col("__bc_n") / lit(total))

    def class_weights(self, label: str) -> Dataset:
        """Inverse-frequency weight per class — ``n_rows / (n_classes * n_in_class)``.

        The scikit-learn ``class_weight="balanced"`` formula: rare classes get a weight
        above 1, common ones below, so a weighted loss treats them equally. Join the
        result back on `label` to attach a per-row sample weight.

        Args:
            label: The categorical column to weight.

        Returns:
            A `Dataset` of one row per class, with a ``weight`` column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"y": ["a", "a", "a", "b"]})
                >>> ds.class_weights("y").sort("y").to_pydict()
                {'y': ['a', 'b'], 'weight': [0.6666666666666666, 2.0]}
        """
        from batcher.plan.expr_ir import count, lit

        total = float(self.count())
        counts = self.group_by(label).agg(__bc_n=count())
        n_classes = float(counts.count())
        return counts.select(label, weight=lit(total) / (lit(n_classes) * Col("__bc_n")))

    @property
    def shape(self) -> tuple[int, int]:
        """The ``(rows, columns)`` of the dataset — the pandas ``shape``.

        Eager in the row count (it executes a `count`, often answered from metadata).

        Returns:
            A ``(row_count, column_count)`` tuple.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2], "y": [3, 4]}).shape
                (2, 2)
        """
        return (self.count(), len(self.columns))

    @property
    def size(self) -> int:
        """The total number of cells (``rows * columns``) — the pandas ``size``.

        Returns:
            The cell count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2], "y": [3, 4]}).size
                4
        """
        rows, cols = self.shape
        return rows * cols

    def join(
        self,
        other: Dataset,
        on: str | list[str] | None = None,
        *,
        left_on: str | list[str] | None = None,
        right_on: str | list[str] | None = None,
        how: str = "inner",
        suffix: str = "_right",
    ) -> Dataset:
        """Equi-join with another dataset.

        Specify keys with `on` (shared column names) or `left_on`/`right_on`.
        `how` is one of inner/left/right/semi/anti. Output keeps the key columns
        (named after the left keys), then the remaining left columns, then the
        remaining right columns (colliding names get `suffix`).

        Args:
            other: The right-hand dataset.
            on: Shared key column name(s) present on both sides.
            left_on: The left key column(s), when the key names differ.
            right_on: The right key column(s), when the key names differ.
            how: The join type — inner/left/right/full/outer/cross/semi/anti.
                ``"cross"`` takes no keys and delegates to :meth:`cross_join`.
            suffix: Suffix appended to right columns whose names collide.

        Returns:
            A new `Dataset` of the joined rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> left = bt.from_pydict({"id": [1, 2], "v": ["a", "b"]})
                >>> right = bt.from_pydict({"id": [1, 2], "w": ["x", "y"]})
                >>> left.join(right, on="id").to_pydict()
                {'id': [1, 2], 'v': ['a', 'b'], 'w': ['x', 'y']}
        """
        how = "full" if how == "outer" else how
        if how == "cross":
            # SQL and every neighbouring library spell the unconditional join this
            # way; it is keyless, so it routes to the dedicated node rather than
            # through key resolution.
            if on is not None or left_on is not None or right_on is not None:
                raise PlanError("join(how='cross') takes no keys — a cross join is unconditional")
            return self.cross_join(other, suffix=suffix)
        if how not in {"inner", "left", "right", "full", "semi", "anti"}:
            raise PlanError(
                f"unsupported join type {how!r} (inner|left|right|full|outer|cross|semi|anti)"
            )
        left_keys, right_keys = _resolve_join_keys(on, left_on, right_on)

        left_cols = self.columns
        right_cols = other.columns
        output = _join_output(left_cols, right_cols, left_keys, right_keys, how, suffix)

        # Append the right side's sources after the left's and shift its scans.
        offset = len(self._sources)
        right_plan = remap_sources(other._plan, offset)
        combined_sources = self._sources + other._sources

        node = Join(self._plan, right_plan, tuple(left_keys), tuple(right_keys), how, tuple(output))
        if how != "full":
            return Dataset(node, combined_sources)

        # Full outer join: coalesce each side's key columns into the final key and
        # drop the temporaries, keeping the standard [keys, left, right] layout.
        from batcher.plan.expr_ir import Coalesce

        items = [
            Projection(lk, Coalesce([Col(f"__fk_l_{i}"), Col(f"__fk_r_{i}")]))
            for i, lk in enumerate(left_keys)
        ]
        items += [
            Projection(c, Col(c)) for c in node.available_columns() if not c.startswith("__fk_")
        ]
        return Dataset(Project(node, tuple(items)), combined_sources)

    def join_stream(
        self,
        other: Dataset,
        on: str | list[str] | None = None,
        *,
        left_on: str | list[str] | None = None,
        right_on: str | list[str] | None = None,
        left_time: str,
        right_time: str,
        within: str,
        lateness: str | None = None,
    ) -> Dataset:
        """Watermark-bounded stream-stream interval inner join (Spark stream-stream join).

        Joins two streams on equality keys (`on` / `left_on`+`right_on`) **and** an
        event-time interval — a row pair matches only if
        ``|left_time - right_time| <= within``. That time bound is what lets buffered
        state be evicted once the watermark passes, keeping memory bounded over two
        unbounded streams. Over bounded sources it is a plain inner join plus the
        interval filter. Consume the streaming result with `iter_batches()`.

        Args:
            other: The right-hand stream.
            on: Shared equality key column name(s).
            left_on: The left equality key(s), when the names differ.
            right_on: The right equality key(s), when the names differ.
            left_time: The left event-time column.
            right_time: The right event-time column.
            within: The maximum time difference for a pair to match (e.g. ``"1h"``).
            lateness: Extra grace before evicting buffered state; ``None`` for none.

        Returns:
            A new `Dataset` of the interval-joined rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime
                >>> t0 = datetime.datetime(2024, 1, 1)
                >>> left = bt.from_pydict({"k": [1, 2], "lt": [t0, t0]})
                >>> right = bt.from_pydict({"k": [1, 2], "rt": [t0, t0]})
                >>> joined = left.join_stream(
                ...     right, on="k", left_time="lt", right_time="rt", within="1h"
                ... )
                >>> joined.count()
                2
        """
        from batcher.io.source import is_bounded
        from batcher.plan.functions.temporal import _duration_micros
        from batcher.plan.logical import WatermarkStreamJoin

        left_keys, right_keys = _resolve_join_keys(on, left_on, right_on)
        within_us = _duration_micros(within, arg="join within")
        lateness_us = _duration_micros(lateness, arg="join lateness") if lateness else 0
        offset = len(self._sources)
        combined = self._sources + other._sources

        if all(is_bounded(s) for s in combined):
            joined = self.join(other, left_on=left_keys, right_on=right_keys, how="inner")
            diff = Col(left_time).cast("int64") - Col(right_time).cast("int64")
            return joined.filter((diff <= within_us) & (diff >= -within_us))

        output = _join_output(self.columns, other.columns, left_keys, right_keys, "inner", "_right")
        node = WatermarkStreamJoin(
            self._plan,
            remap_sources(other._plan, offset),
            tuple(left_keys),
            tuple(right_keys),
            tuple(output),
            left_time,
            right_time,
            within_us,
            lateness_us,
        )
        return Dataset(node, combined)

    def join_asof(
        self,
        other: Dataset,
        *,
        on: str | None = None,
        left_on: str | None = None,
        right_on: str | None = None,
        by: str | list[str] | None = None,
        left_by: str | list[str] | None = None,
        right_by: str | list[str] | None = None,
        direction: str = "backward",
        suffix: str = "_right",
    ) -> Dataset:
        """ASOF (nearest-match) join — match each left row to the nearest right row.

        The match is on the right row whose `on` key is nearest (`direction`:
        ``"backward"`` ≤, ``"forward"`` ≥), within the same `by` group (exact).
        Left-style: every left row is kept (null right columns when unmatched). Both
        sides should be sorted on `on` within `by` for the intended semantics. Specify
        keys via `on`/`by` (shared) or `*_on`/`*_by`.

        Args:
            other: The right-hand dataset to match against.
            on: The shared nearest-match key column.
            left_on: The left match key, when the names differ.
            right_on: The right match key, when the names differ.
            by: Shared exact-match grouping column(s).
            left_by: The left grouping column(s), when the names differ.
            right_by: The right grouping column(s), when the names differ.
            direction: ``"backward"`` (≤) or ``"forward"`` (≥) nearest match.
            suffix: Suffix appended to right columns whose names collide.

        Returns:
            A new `Dataset` with each left row matched to its nearest right row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> left = bt.from_pydict({"t": [1, 5, 10], "v": ["a", "b", "c"]})
                >>> right = bt.from_pydict({"t": [2, 6], "w": ["x", "y"]})
                >>> left.join_asof(right, on="t").to_pydict()
                {'t': [1, 5, 10], 'v': ['a', 'b', 'c'], 'w': [None, 'x', 'y']}
        """
        l_on, r_on = left_on or on, right_on or on
        if l_on is None or r_on is None:
            raise PlanError("join_asof() requires `on` (or both left_on and right_on)")
        l_by = _as_str_list(left_by if left_by is not None else by)
        r_by = _as_str_list(right_by if right_by is not None else by)
        output = _asof_output(self.columns, other.columns, r_on, r_by, suffix)
        right_plan = remap_sources(other._plan, len(self._sources))
        node = AsofJoin(
            self._plan, right_plan, l_on, r_on, tuple(l_by), tuple(r_by), direction, tuple(output)
        )
        return Dataset(node, self._sources + other._sources)

    def group_by(self, *keys: str, **named: Expr) -> GroupBy:
        """Begin a grouped aggregation over the given keys.

        Positional args are key columns by name; keyword args bind a derived key
        column to an expression (e.g. ``group_by("dept", decade=col("year") // 10)``).
        Follow with ``.agg(name=expr)``:
        ``ds.group_by("dept").agg(total=col("salary").sum(), n=count())``.
        Global aggregation (no keys) is ``ds.group_by().agg(...)``.

        Args:
            *keys: Key columns by name.
            **named: Derived key columns bound to expressions.

        Returns:
            A `GroupBy` to finish with ``.agg(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "b", "a"], "v": [1, 2, 3]})
                >>> ds.group_by("g").agg(s=bt.col("v").sum()).sort("g").to_pydict()
                {'g': ['a', 'b'], 's': [4, 2]}
        """
        available = set(self._plan.available_columns())
        for k in keys:
            if not isinstance(k, str):
                raise PlanError(
                    "positional group_by() keys must be column names; give a derived "
                    "key a name, e.g. group_by(bucket=col('x') % 10)"
                )
            if k not in available:
                cols = sorted(available)
                raise PlanError(
                    f"group_by key {k!r} is not a column; available: {cols}"
                    f"{suggest_columns(k, cols)}"
                )
        for alias, expr in named.items():
            if not isinstance(expr, Expr):
                raise PlanError(f"group_by() value for {alias!r} must be an expression")
            _reject_sliding_window_key(alias, expr)
        return GroupBy(self, keys, named)

    def agg(self, *aggs: Expr, **aggregates: Expr) -> Dataset:
        """Aggregate over the whole dataset (no grouping).

        Shorthand for ``group_by().agg(...)``: ``ds.agg(total=col("x").sum())`` returns
        a single-row dataset. Positional args are self-naming aggregates —
        ``ds.agg(bt.sum("x"), bt.mean("y"))`` keeps each source column's name.

        Args:
            *aggs: Self-naming aggregate expressions (e.g. ``bt.sum("x")``).
            **aggregates: Named aggregate expressions over the whole dataset.

        Returns:
            A single-row `Dataset` of the aggregates.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3, 4]})
                >>> ds.agg(total=bt.col("x").sum()).to_pydict()
                {'total': [10]}

                >>> ds.agg(bt.sum("x")).to_pydict()
                {'x': [10]}
        """
        return self.group_by().agg(*aggs, **aggregates)

    # --- terminal operations ----------------------------------------------
    def collect(
        self,
        distributed: bool | str = "auto",
        num_workers: int | None = None,
        spill: bool = False,
        num_partitions: int | None = None,
        adaptive: bool | str = "auto",
        transport: str = "auto",
        backend: str = "cpu",
    ) -> pa.Table:
        """Execute the plan and materialize the result as a `pyarrow.Table`.

        Zero-config by default; every argument is an optional override.
        `distributed="auto"` uses Ray on a multi-node cluster, else single-node.
        Out-of-core spilling is automatic under memory pressure, with worker fan-out
        and partition count sized from the estimated data volume; `spill=True` forces
        it and `num_partitions` overrides the bucket count. `adaptive="auto"` turns on
        intra-query re-optimization only when a join's input size is a pure estimate
        (so measured cardinality could change a build-side/join-order choice), and
        stays one-shot otherwise; `True`/`False` force it. `backend` selects where a
        supported shape runs: `"cpu"` (default) the native engine, `"gpu"` forces the GPU
        (cuDF) for any supported shape, and `"auto"` lets Kyber's cost policy decide — GPU
        only when the estimated input is large enough to amortize the device overhead and
        fits the cluster's GPU memory (sharding across GPUs when it exceeds one), else the
        CPU engine. Any unsupported shape or a GPU-less cluster falls back to the CPU engine,
        so every value is safe to request and the result is identical whichever way it runs.
        Raises `PlanError` if the dataset is unbounded (a streaming source) — use
        `iter_batches()` / `write()`.

        Args:
            distributed: ``"auto"`` uses Ray on a cluster; ``True``/``False`` force it.
            num_workers: Worker fan-out; ``None`` sizes it from the data volume.
            spill: Force out-of-core spilling on (it is automatic under pressure).
            num_partitions: Override the shuffle bucket count.
            adaptive: Enable intra-query re-optimization (``"auto"``/``True``/``False``).
            transport: The shuffle transport; ``"auto"`` selects one.
            backend: ``"cpu"``, ``"gpu"``, or ``"auto"`` to let Kyber's cost policy decide.

        Returns:
            The materialized result table.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.collect().num_rows
                3
        """
        return _collect(
            self._plan,
            self._sources,
            self.columns,
            distributed=distributed,
            num_workers=num_workers,
            spill=spill,
            num_partitions=num_partitions,
            adaptive=adaptive,
            transport=transport,
            cache=self._cache,
            backend=backend,
        )

    def lineage(self) -> dict[str, list[str]]:
        """Return, per output column, the source columns its values are derived from.

        Column-level lineage, read straight off the plan — nothing executes. This is what
        turns a governance tag into an answer: tag ``customers.ssn`` as PII, and lineage
        names every downstream column that carries it.

        Origins are rendered ``"<table>.<column>"``, where the table is the path a source
        is read from. A column built only from literals, or generated (`with_row_index`),
        has no origin and maps to an empty list.

        Lineage tracks *data* flow, not control flow: filtering on a column does not put
        it in the lineage of the surviving columns. An opaque `map_batches` stage is
        over-approximated — every output column is assumed to derive from every input
        column — because for a governance answer a false positive costs a review and a
        false negative costs a breach.

        Examples:
            .. doctest::

                >>> import os
                >>> import tempfile

                >>> import batcher as bt
                >>> path = os.path.join(tempfile.mkdtemp(), "people.parquet")
                >>> _ = bt.from_pydict({"first": ["a"], "last": ["b"], "age": [3]}).write(
                ...     path, format="parquet"
                ... )
                >>> ds = bt.read.parquet(path).select(
                ...     name=bt.concat(bt.col("first"), bt.col("last")),
                ...     decade=bt.col("age") / 10,
                ... )
                >>> sorted(ds.lineage()["name"]) == sorted(
                ...     [f"{path}.first", f"{path}.last"]
                ... )
                True

        Returns:
            A mapping from output column name to its sorted ``"table.column"`` origins.
        """
        from batcher.api.security import table_name
        from batcher.governance import column_lineage

        tables = [table_name(s) or f"<source {i}>" for i, s in enumerate(self._sources)]
        lineage = column_lineage(self._plan, tables)
        return {
            alias: sorted(f"{table}.{column}" for table, column in origins)
            for alias, origins in lineage.items()
        }

    def explain(self, analyze: bool = False, *, format: str = "text") -> str:
        """Return the query plan as a tree, optionally with measured execution profile.

        With ``analyze=False`` (the default) it renders the *planned* operator tree —
        per-operator cardinality estimate, provenance, and chosen strategy — without
        executing, the way DuckDB's ``EXPLAIN`` and Spark's plan display do. With
        ``analyze=True`` it runs the query and renders each operator's *estimate vs
        actual* rows, wall time and share, peak memory, spill, and backend (DuckDB's
        ``EXPLAIN ANALYZE``), so you can see where time and memory actually went.
        ``format="json"`` returns the same profile as a machine-readable document.

        Args:
            analyze: Execute the query and include measured per-operator metrics.
            format: ``"text"`` (or its alias ``"tree"``, as Polars and Spark spell it)
                for the rendered tree, ``"json"`` for the profile dict.

        Returns:
            The plan (and, when ``analyze``, the measured profile) as text or JSON.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> len(ds.filter(bt.col("x") > 1).explain()) > 0
                True
                >>> len(ds.filter(bt.col("x") > 1).explain(analyze=True)) > 0
                True
        """
        fmt = "text" if format == "tree" else format
        return _explain(self._plan, self._sources, self.columns, analyze=analyze, fmt=fmt)

    def stats(self) -> RunStats:
        """Execute (single-node) and return measured per-operator `RunStats`.

        Where `explain()` shows the *planned* shape with estimates, `stats()` runs
        the query and reports what the engine *measured* — rows in/out, wall time,
        peak bytes, spill, and backend per operator, plus a bottleneck call (the
        answer to "where is my time going"). Not available for `map_batches`/ML
        pipelines (raises `BackendError`).

        Returns:
            The measured per-operator run statistics.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"k": ["a", "a", "b"], "v": [1, 2, 3]})
                >>> print(ds.group_by("k").agg(s=bt.col("v").sum()).stats())  # doctest: +SKIP
        """
        return _stats(self._plan, self._sources, self.columns)

    def count(self) -> int:
        """Return the number of result rows.

        Answered from metadata without execution whenever the count is provably
        exact — ``ds.limit(n).count()`` is ``min(n, ds.count())``, a global
        aggregate is ``1``, an empty source is ``0`` — and falls back to a full
        run otherwise. The result is always identical to executing.

        Returns:
            The number of result rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3]}).count()
                3
        """
        return _count(self._plan, self._sources, self.columns)

    def is_empty(self) -> bool:
        """Whether the result has no rows.

        Answered from metadata when the row count is provably known; otherwise a
        single-row probe (which the streaming path reads without scanning the
        whole source).

        Returns:
            ``True`` if the result has no rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1]}).filter(bt.col("x") > 10).is_empty()
                True
        """
        return _is_empty(self._plan, self._sources, self.columns)

    @property
    def schema(self) -> pa.Schema:
        """The output Arrow schema (column names and types), without scanning rows.

        A scan returns its source schema directly; other plans resolve derived
        column types via a zero-row execution. Use `columns` for just the names
        (always free).

        Returns:
            The output Arrow schema.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3]}).schema.names
                ['x']
        """
        return _schema(self._plan, self._sources, self.columns)

    @property
    def dtypes(self) -> list[pa.DataType]:
        """The output column Arrow types, in order (see `schema`).

        Returns:
            The output column Arrow types, in order.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> [str(t) for t in bt.from_pydict({"x": [1, 2, 3]}).dtypes]
                ['int64']
        """
        return list(self.schema.types)

    def _require_column(self, column: str, op: str) -> None:
        """Validate that `column` is an output column, else raise `PlanError`."""
        available = self._plan.available_columns()
        if column not in available:
            raise PlanError(f"{op}(): unknown column {_unknown_cols({column}, available)}")

    def _exec_scalar(self, agg_expr: Expr) -> Any:
        """Execute a single global aggregate and return its one scalar value."""
        res = self.agg(**{"__bc_scalar__": agg_expr}).to_pydict()["__bc_scalar__"]
        return res[0] if res else None

    def _exec_null_total(self, column: str) -> tuple[int, int]:
        """Execute `(null_count, row_count)` for `column` in one aggregate pass."""
        from batcher.api.functions import count

        res = self.agg(__bc_n__=count(), __bc_c__=Col(column).count()).to_pydict()
        total = res["__bc_n__"][0] if res["__bc_n__"] else 0
        nonnull = res["__bc_c__"][0] if res["__bc_c__"] else 0
        return int(total) - int(nonnull), int(total)

    def min(self, column: str) -> Any:
        """The minimum value of `column` (SQL ``MIN``), answered from metadata when exact.

        A scalar terminal: when an EXACT footer/manifest bound is available (a Parquet
        scan, a rename/sort/distinct over one) the answer comes straight from the
        metadata with no scan; otherwise a single ``MIN`` aggregate runs. Nulls are
        ignored; an all-null or empty `column` yields ``None`` — always identical to
        executing.

        Args:
            column: The column to reduce.

        Returns:
            The minimum value, or ``None`` for an empty/all-null column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [3, 1, 2]}).min("x")
                1
        """
        self._require_column(column, "min")
        from batcher.api.terminal.metadata_answer import metadata_min

        answer = metadata_min(self._plan, self._sources, column)
        return answer if answer is not None else self._exec_scalar(Col(column).min())

    def max(self, column: str) -> Any:
        """The maximum value of `column` (SQL ``MAX``), answered from metadata when exact.

        The upper-bound mirror of `min`: an EXACT footer bound answers with no scan,
        else one ``MAX`` aggregate runs. Nulls are ignored; empty/all-null yields ``None``.

        Args:
            column: The column to reduce.

        Returns:
            The maximum value, or ``None`` for an empty/all-null column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [3, 1, 2]}).max("x")
                3
        """
        self._require_column(column, "max")
        from batcher.api.terminal.metadata_answer import metadata_max

        answer = metadata_max(self._plan, self._sources, column)
        return answer if answer is not None else self._exec_scalar(Col(column).max())

    def n_unique(self, column: str) -> int:
        """The exact number of distinct values in `column` (SQL ``COUNT(DISTINCT)``).

        Answered from metadata only when an **exact** distinct count is known (never a
        sketch — use `approx_n_unique` for the fast approximate answer); otherwise an
        exact ``COUNT(DISTINCT)`` runs. Nulls are not counted as a distinct value.

        Args:
            column: The column whose distinct values to count.

        Returns:
            The number of distinct non-null values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 1, 2, 3, 3]}).n_unique("x")
                3
        """
        self._require_column(column, "n_unique")
        from batcher.api.terminal.metadata_answer import metadata_n_unique

        answer = metadata_n_unique(self._plan, self._sources, column)
        return answer if answer is not None else int(self._exec_scalar(Col(column).n_unique()))

    def median(self, column: str) -> Any:
        """The exact median of `column` (SQL ``MEDIAN``), ignoring nulls.

        A scalar terminal, exact rather than sketched — use `approx_median` on a large
        column when a bounded-error answer is enough and a sort is not affordable.

        Args:
            column: The column to reduce.

        Returns:
            The median value, or ``None`` for an empty/all-null column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 5, 2, 4, 3]}).median("x")
                3.0
        """
        self._require_column(column, "median")
        return self._exec_scalar(Col(column).median())

    def mean(self, column: str) -> Any:
        """The arithmetic mean of `column` (SQL ``AVG``), ignoring nulls.

        A scalar terminal, the whole-dataset counterpart of
        ``group_by(...).mean()``; runs one aggregate pass.

        Args:
            column: The column to reduce.

        Returns:
            The mean value, or ``None`` for an empty/all-null column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3, 4]}).mean("x")
                2.5
        """
        self._require_column(column, "mean")
        return self._exec_scalar(Col(column).mean())

    def sum(self, column: str) -> Any:
        """The sum of `column` (SQL ``SUM``), ignoring nulls.

        A scalar terminal; runs one aggregate pass. An empty or all-null column sums
        to ``None`` (matching SQL), not ``0``.

        Args:
            column: The column to reduce.

        Returns:
            The sum, or ``None`` for an empty/all-null column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3, 4]}).sum("x")
                10
        """
        self._require_column(column, "sum")
        return self._exec_scalar(Col(column).sum())

    def std(self, column: str) -> Any:
        """The sample standard deviation of `column` (SQL ``STDDEV_SAMP``), ignoring nulls.

        A scalar terminal; runs one aggregate pass. Fewer than two non-null values
        yields ``None``.

        Args:
            column: The column to reduce.

        Returns:
            The sample standard deviation, or ``None`` when undefined.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [2, 4, 4, 4, 5, 5, 7, 9]}).std("x")
                2.138089935299395
        """
        self._require_column(column, "std")
        return self._exec_scalar(Col(column).std())

    def var(self, column: str) -> Any:
        """The sample variance of `column` (SQL ``VAR_SAMP``), ignoring nulls.

        A scalar terminal; runs one aggregate pass. Fewer than two non-null values
        yields ``None``.

        Args:
            column: The column to reduce.

        Returns:
            The sample variance, or ``None`` when undefined.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3, 4, 5]}).var("x")
                2.5
        """
        self._require_column(column, "var")
        return self._exec_scalar(Col(column).var())

    def quantile(self, column: str, q: float) -> Any:
        """The exact `q`-quantile of `column` (SQL ``QUANTILE_CONT``), ignoring nulls.

        The exact counterpart of `approx_quantile`, which answers from a mergeable
        TDigest instead.

        Args:
            column: The column to reduce.
            q: The quantile to compute, in ``[0, 1]`` (``0.5`` is the median).

        Returns:
            The quantile value, or ``None`` for an empty/all-null column.

        Raises:
            PlanError: If `q` is outside ``[0, 1]``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3, 4]}).quantile("x", 0.25)
                1.75
        """
        self._require_column(column, "quantile")
        if not 0.0 <= q <= 1.0:
            raise PlanError(f"quantile(): q must be in [0, 1], got {q}")
        return self._exec_scalar(Col(column).quantile(q))

    def corr(self, x: str, y: str) -> float | None:
        """The Pearson correlation of columns `x` and `y` (SQL ``CORR``).

        A scalar terminal. Rows where either column is null are ignored; the result is
        ``None`` when fewer than two such rows remain, or when either column is constant.

        Args:
            x: The first column.
            y: The second column.

        Returns:
            The correlation coefficient in ``[-1, 1]``, or ``None``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> round(bt.from_pydict({"a": [1, 2, 3], "b": [2, 4, 6]}).corr("a", "b"), 6)
                1.0
        """
        from batcher.plan.functions.aggregate import corr

        self._require_column(x, "corr")
        self._require_column(y, "corr")
        return self._exec_scalar(corr(Col(x), Col(y)))

    def corr_matrix(self, columns: list[str] | None = None) -> Dataset:
        """The pairwise Pearson correlation matrix over numeric columns.

        **Executes** and returns a small `Dataset`: a ``column`` label column plus one
        Float64 column per correlated column, forming a symmetric matrix (diagonal ``1.0``,
        or ``None`` for a constant column). Every pair is computed in a **single** pass —
        not ``N**2`` separate scans — the standard first step of exploratory data analysis
        and feature selection. Non-numeric columns are skipped unless named explicitly
        (which errors).

        Args:
            columns: optional subset of numeric columns to correlate (default: all numeric).

        Returns:
            A `Dataset` holding the correlation matrix with a ``column`` label column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [1, 2, 3], "b": [2, 4, 6], "c": [3, 2, 1]})
                >>> m = ds.corr_matrix().to_pydict()
                >>> m["column"], round(m["a"][1], 4), round(m["c"][0], 4)
                (['a', 'b', 'c'], 1.0, -1.0)
        """
        from batcher.api.dataset._describe import corr_matrix

        return corr_matrix(self, columns)

    def cov_matrix(self, columns: list[str] | None = None) -> Dataset:
        """The pairwise sample covariance matrix over numeric columns.

        The covariance companion to `corr_matrix`: **executes** and returns a small
        symmetric `Dataset` (a ``column`` label plus one Float64 column per column), every
        pair computed in a **single** pass. The diagonal holds each column's variance. The
        input to PCA / whitening and multivariate-Gaussian modeling.

        Args:
            columns: optional subset of numeric columns (default: all numeric).

        Returns:
            A `Dataset` holding the covariance matrix with a ``column`` label column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
                >>> m = ds.cov_matrix().to_pydict()
                >>> m["column"], m["a"][0], m["b"][0]
                (['a', 'b'], 1.0, 2.0)
        """
        from batcher.api.dataset._describe import cov_matrix

        return cov_matrix(self, columns)

    def cov(self, x: str, y: str, *, ddof: int = 1) -> float | None:
        """The covariance of columns `x` and `y` (SQL ``COVAR_SAMP``/``COVAR_POP``).

        A scalar terminal. Rows where either column is null are ignored.

        Args:
            x: The first column.
            y: The second column.
            ddof: Delta degrees of freedom — ``1`` for the sample covariance (the
                default, ``COVAR_SAMP``) or ``0`` for the population one (``COVAR_POP``).

        Returns:
            The covariance, or ``None`` when too few non-null row pairs remain.

        Raises:
            PlanError: If `ddof` is neither 0 nor 1.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"a": [1, 2, 3], "b": [2, 4, 6]}).cov("a", "b")
                2.0
        """
        from batcher.plan.functions.aggregate import covar_pop, covar_samp

        self._require_column(x, "cov")
        self._require_column(y, "cov")
        if ddof not in (0, 1):
            raise PlanError(f"cov(): ddof must be 0 (population) or 1 (sample), got {ddof}")
        fn = covar_samp if ddof == 1 else covar_pop
        return self._exec_scalar(fn(Col(x), Col(y)))

    def n_null(self, column: str) -> int:
        """The exact number of null values in `column` (``count(*) - count(column)``).

        Answered from metadata when an EXACT per-column null count is known (a Parquet/
        ORC footer records it), else computed in one aggregate pass. The scalar,
        single-column counterpart of `null_count` (which returns one row for every
        column).

        Args:
            column: The column whose nulls to count.

        Returns:
            The number of null values in `column`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, None, 3, None]}).n_null("x")
                2
        """
        self._require_column(column, "n_null")
        from batcher.api.terminal.metadata_answer import metadata_null_count

        answer = metadata_null_count(self._plan, self._sources, column)
        return answer if answer is not None else self._exec_null_total(column)[0]

    def has_nulls(self, column: str) -> bool:
        """Whether `column` contains at least one null, answered from metadata when exact.

        A no-scan answer when an EXACT null count is known, else a single aggregate.

        Args:
            column: The column to test.

        Returns:
            ``True`` if `column` has any null value.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, None, 3]}).has_nulls("x")
                True
                >>> bt.from_pydict({"x": [1, 2, 3]}).has_nulls("x")
                False
        """
        self._require_column(column, "has_nulls")
        from batcher.api.terminal.metadata_answer import metadata_has_nulls

        answer = metadata_has_nulls(self._plan, self._sources, column)
        return answer if answer is not None else self._exec_null_total(column)[0] > 0

    def all_null(self, column: str) -> bool:
        """Whether every value of `column` is null, answered from metadata when exact.

        ``True`` only for a non-empty column whose null count equals its row count (an
        empty dataset is not reported all-null). No-scan when EXACT counts are known.

        Args:
            column: The column to test.

        Returns:
            ``True`` if `column` is non-empty and entirely null.

        Examples:
            .. doctest::

                >>> import batcher as bt, pyarrow as pa
                >>> t = pa.table({"x": pa.array([None, None], type=pa.int64())})
                >>> bt.from_arrow(t).all_null("x")
                True
                >>> bt.from_pydict({"x": [1, None]}).all_null("x")
                False
        """
        self._require_column(column, "all_null")
        from batcher.api.terminal.metadata_answer import metadata_all_null

        answer = metadata_all_null(self._plan, self._sources, column)
        if answer is not None:
            return answer
        nulls, total = self._exec_null_total(column)
        return total > 0 and nulls == total

    @property
    def has_rows(self) -> bool:
        """Whether the result has at least one row (the complement of `is_empty`).

        Answered from metadata when the row count is provably known, else a single-row
        probe (which the streaming path reads without scanning the whole source).

        Returns:
            ``True`` if the result has at least one row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3]}).has_rows
                True
                >>> bt.from_pydict({"x": [1]}).filter(bt.col("x") > 10).has_rows
                False
        """
        return not self.is_empty()

    def approx_n_unique(self, column: str) -> int | None:
        """Approximate number of distinct values in `column` (HyperLogLog).

        Opt-in and explicitly approximate — the fast analog of `n_unique`. Answered
        from a learned sketch ndv with no scan when available, else an HLL pass over the
        data. Returns ``None`` only when neither is possible.

        Args:
            column: The column whose distinct values to estimate.

        Returns:
            The approximate distinct count, or ``None`` if unavailable.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": list(range(1000)) * 2})
                >>> ds.approx_n_unique("x") is not None
                True
        """
        self._require_column(column, "approx_n_unique")
        from batcher.api.terminal.metadata_answer import metadata_approx_n_unique

        answer = metadata_approx_n_unique(self._plan, self._sources, column)
        if answer is not None:
            return answer
        res = self._exec_scalar(Col(column).approx_count_distinct())
        return int(res) if res is not None else None

    def approx_quantile(self, column: str, q: float) -> float | None:
        """Approximate quantile `q` (in ``[0, 1]``) of a numeric `column`.

        Opt-in and explicitly approximate. Answered from the hub's learned quantile
        grid (a KLL sketch from a past run) with no scan when available; otherwise a
        TDigest is streamed over the data — tail-accurate (p99/p999) and far cheaper
        than the exact sort `quantile` would need. Returns ``None`` for a non-numeric
        or empty column. Use the exact aggregate when precision matters.

        Args:
            column: The numeric column to summarize.
            q: The quantile to estimate, in ``[0, 1]``.

        Returns:
            The approximate quantile value, or ``None`` for a non-numeric/empty column.

        Raises:
            PlanError: If `q` is outside ``[0, 1]``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": list(range(1, 101))})
                >>> ds.approx_quantile("x", 0.5) is not None
                True
        """
        q = require_float(q, func="approx_quantile", arg="q")
        if not 0.0 <= q <= 1.0:
            raise PlanError(f"approx_quantile(q) requires q in [0, 1], got {q}")
        self._require_column(column, "approx_quantile")
        from batcher.api.terminal.metadata_answer import metadata_learned_quantile

        learned = metadata_learned_quantile(column, q, self._sources)
        if learned is not None:
            return learned
        from batcher.api.orchestration import approx_quantile

        # Stream just the target column (projected, so only it crosses the boundary)
        # through the mergeable TDigest — the driver never holds the whole column, and a
        # distributed plan streams it back one bounded bucket at a time.
        return approx_quantile(self.select(column).iter_batches(), column, q)

    def approx_median(self, column: str) -> float | None:
        """Approximate median of a numeric `column` — `approx_quantile(column, 0.5)`.

        Args:
            column: The numeric column to summarize.

        Returns:
            The approximate median, or ``None`` if unavailable.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": list(range(1, 101))})
                >>> ds.approx_median("x") is not None
                True
        """
        return self.approx_quantile(column, 0.5)

    def approx_percentile(self, column: str, p: float) -> float | None:
        """Approximate percentile `p` (in ``[0, 100]``) of a numeric `column`.

        The percentile spelling of `approx_quantile` (``p=99`` is ``q=0.99``).

        Args:
            column: The numeric column.
            p: The percentile in ``[0, 100]``.

        Returns:
            The approximate percentile value, or ``None`` if unavailable.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": list(range(1, 101))})
                >>> ds.approx_percentile("x", 90) is not None
                True
        """
        p = require_float(p, func="approx_percentile", arg="p")
        if not 0.0 <= p <= 100.0:
            raise PlanError(f"approx_percentile(p) requires p in [0, 100], got {p}")
        return self.approx_quantile(column, p / 100.0)

    def iter_batches(
        self,
        batch_size: int | None = None,
        *,
        distributed: bool | str = False,
        num_workers: int | None = None,
        transport: str = "auto",
    ):
        """Execute and yield the result as Arrow record batches.

        The execution mode is automatic: a breaker-free pipeline (filter / project /
        map_batches over a single source) — and top-level aggregate / distinct /
        top-N over such an input — is consumed one source batch at a time in bounded
        memory, so a larger-than-memory or unbounded source streams incrementally.
        Other plans (sort / join / window / multi-source) materialize first; if the
        source is unbounded and the plan cannot stream, a `PlanError` is raised
        rather than hanging. `batch_size` rebatches the output.

        With `distributed` (``True`` or ``"auto"`` on a multi-node cluster), a
        top-level breaker fans out across Ray workers and its result streams back one
        reducer bucket at a time, so the driver never holds the whole distributed
        result — the bounded-memory way to pull a large distributed output.

        Args:
            batch_size: Rebatch the output to this many rows; ``None`` keeps engine batches.
            distributed: Fan a top-level breaker across Ray workers (``True``/``"auto"``).
            num_workers: Worker fan-out for the distributed path.
            transport: The shuffle transport; ``"auto"`` selects one.

        Yields:
            The result as Arrow record batches.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> sum(batch.num_rows for batch in ds.iter_batches())
                3
        """
        from batcher.api.terminal.core import _resolve_distributed
        from batcher.api.terminal.event_log import pipeline_signature, report_stream

        batches = _iter_batches(
            self._plan,
            self._sources,
            self.columns,
            batch_size=batch_size,
            distributed=_resolve_distributed(distributed, self._plan, self._sources),
            num_workers=num_workers,
            transport=transport,
        )
        # Wrapped here, at the single public entry, rather than inside `_iter_batches` —
        # which recurses on the `batch_size` path and would double-count every row.
        yield from report_stream(
            batches,
            label=type(self._plan).__name__.lower(),
            signature=pipeline_signature(self._plan),
        )

    @property
    def write(self) -> Writer:
        """The write namespace — ``ds.write(path)`` writes, ``ds.write.<format>(...)`` is explicit.

        ``ds.write(path)`` autodetects the sink format from the path;
        ``ds.write.parquet(...)`` / ``ds.write.delta(...)`` name it. All accept
        `partition_by=`/`distributed=`/`num_workers=` and return a `WriteManifest`.

        Returns:
            The `Writer` namespace bound to this dataset.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import tempfile, os
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> with tempfile.TemporaryDirectory() as d:
                ...     path = os.path.join(d, "out.parquet")
                ...     _ = ds.write(path)
                ...     bt.read(path).count()
                3
        """
        from batcher.api.io_namespace import Writer

        return Writer(self)

    def to_arrow(self) -> pa.Table:
        """Execute the plan and return the result as a `pyarrow.Table`.

        The named form of `collect` with its default settings — a terminal
        operation that runs the optimized query and materializes the output (raises
        `PlanError` on an unbounded streaming source; stream with `iter_batches`).

        Returns:
            The materialized result table.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3]}).to_arrow().num_rows
                3
        """
        return _collect(self._plan, self._sources, self.columns)

    def to_pandas(self):
        """Execute the plan and return the result as a pandas `DataFrame`.

        A terminal operation. Materializes the Arrow result and converts it via
        pyarrow's pandas bridge, so it needs pandas installed
        (``pip install 'batcher-engine[pandas]'``); otherwise raises `BackendError`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3]}).to_pandas().shape  # doctest: +SKIP
                (3, 1)
        """
        return _to_pandas(self._plan, self._sources, self.columns)

    def to_polars(self):
        """Execute the plan and return the result as a Polars `DataFrame`.

        A terminal operation. Polars is Arrow-backed, so the materialized result is
        handed over without a row-wise copy. Needs polars installed
        (``pip install 'batcher-engine[polars]'``); otherwise raises `BackendError`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3]}).to_polars().height  # doctest: +SKIP
                3
        """
        return _to_polars(self._plan, self._sources, self.columns)

    def to_numpy(self, columns: list[str] | None = None) -> dict[str, Any]:
        """Execute the plan and return the result as a ``{column: numpy.ndarray}`` dict.

        A terminal operation for numeric / scientific work: each column becomes a NumPy
        array, and a **fixed-shape-tensor or fixed-size-list column** (an image, embedding,
        or feature-vector column) comes back as a real ``(n, *shape)`` array rather than an
        opaque per-row object array — so the result feeds NumPy / scikit-learn directly.
        Streams the output batches, so it holds one materialized copy, not two.

        Args:
            columns: optional subset of output columns to return (default: all).

        Returns:
            A dict mapping each column name to its NumPy array.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> out = bt.from_pydict({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]}).to_numpy()
                >>> out["x"].tolist(), out["y"].tolist()
                ([1, 2, 3], [4.0, 5.0, 6.0])
        """
        from batcher.api.dataset._export import to_numpy

        return to_numpy(self, columns)

    def to_jax(self, columns: list[str] | None = None) -> dict[str, Any]:
        """Execute the plan and return the result as a ``{column: jax.Array}`` dict.

        The JAX counterpart of `to_numpy`: each column becomes a ``jax.numpy`` array, with a
        tensor/fixed-size-list column reshaped to ``(n, *shape)`` — so an embedding or image
        column feeds a JAX/Flax model directly. Needs JAX installed (``pip install jax``);
        otherwise raises `BackendError`.

        Args:
            columns: optional subset of output columns to return (default: all).

        Returns:
            A dict mapping each column name to its ``jax.Array``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> out = bt.from_pydict({"x": [1, 2, 3]}).to_jax()  # doctest: +SKIP
                >>> out["x"].shape  # doctest: +SKIP
                (3,)
        """
        from batcher.api.dataset._export import to_jax

        return to_jax(self, columns)

    def to_pydict(self) -> dict[str, list[Any]]:
        """Execute the plan and return the result as a column-oriented dict.

        A terminal operation: the inverse of `from_pydict`, mapping each output
        column name to its list of values (pyarrow-style). Materializes the whole
        result in memory — use `iter_batches` for larger-than-memory output.

        Returns:
            Column name to its list of values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"a": [1, 2], "b": ["x", "y"]}).to_pydict()
                {'a': [1, 2], 'b': ['x', 'y']}
        """
        return _to_pydict(self._plan, self._sources, self.columns)

    def to_pylist(self) -> list[dict[str, Any]]:
        """Execute the plan and return the result as a row-oriented list of dicts.

        A terminal operation: one ``{column: value}`` dict per row (pyarrow-style),
        the row-major counterpart of `to_pydict`. Materializes the whole result in
        memory.

        Returns:
            One ``{column: value}`` dict per row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"a": [1, 2], "b": ["x", "y"]}).to_pylist()
                [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}]
        """
        return _to_pylist(self._plan, self._sources, self.columns)

    def to_torch(self, *, columns: list[str] | None = None, batch_size: int | None = None) -> Any:
        """A re-iterable ``torch.utils.data.IterableDataset`` of per-batch tensor dicts.

        Each item is a ``{column: torch.Tensor}`` for one engine batch (non-numeric
        columns are skipped). Re-iterating runs the query again, so it is safe for
        multi-epoch training and streams in bounded memory. Needs `torch`.

        Args:
            columns: The columns to include; ``None`` uses all numeric columns.
            batch_size: Rows per emitted tensor batch; ``None`` uses engine batches.

        Returns:
            A re-iterable ``torch.utils.data.IterableDataset`` of tensor dicts.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> next(iter(ds.to_torch()))["x"].shape  # doctest: +SKIP
                torch.Size([3])
        """
        from batcher.api.dataset._export import to_torch

        return to_torch(self, columns, batch_size)

    def to_torch_dataloader(
        self, *, columns: list[str] | None = None, batch_size: int | None = None, **dl_kwargs: Any
    ) -> Any:
        """A ``torch.utils.data.DataLoader`` over the engine-batched tensor dicts.

        The engine already batches, so the loader wraps :meth:`to_torch` with
        ``batch_size=None``; pass `batch_size` to size engine batches and forward
        any other `DataLoader` kwargs (`num_workers`, `pin_memory`, …). Needs `torch`.

        Args:
            columns: The columns to include; ``None`` uses all numeric columns.
            batch_size: Rows per engine batch; ``None`` keeps the engine's batching.
            **dl_kwargs: Extra ``DataLoader`` kwargs (``num_workers``, ``pin_memory``, …).

        Returns:
            A ``torch.utils.data.DataLoader`` over the tensor dicts.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> type(ds.to_torch_dataloader()).__name__  # doctest: +SKIP
                'DataLoader'
        """
        from batcher.api.dataset._export import to_torch_dataloader

        return to_torch_dataloader(self, columns, batch_size, **dl_kwargs)

    def to_tf(self, *, columns: list[str] | None = None, batch_size: int | None = None) -> Any:
        """A re-iterable ``tf.data.Dataset`` of per-batch tensor dicts (needs `tensorflow`).

        Each element is one engine batch's numeric columns as TensorFlow tensors;
        non-numeric columns are skipped.

        Args:
            columns: The columns to include; ``None`` uses all numeric columns.
            batch_size: Rows per emitted tensor batch; ``None`` uses engine batches.

        Returns:
            A re-iterable ``tf.data.Dataset`` of tensor dicts.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> next(iter(ds.to_tf()))["x"].shape  # doctest: +SKIP
                TensorShape([3])
        """
        from batcher.api.dataset._export import to_tf

        return to_tf(self, columns, batch_size)

    def show(self, limit: int = 10) -> None:
        """Print a preview of the first `limit` result rows to stdout.

        A terminal operation for interactive use: it executes the plan (capped at
        `limit` rows) and prints the result, returning nothing. For programmatic
        access to the data use `to_pydict` / `to_pylist` / `collect`.

        Args:
            limit: Maximum number of rows to print.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 3]}).show()  # doctest: +SKIP
        """
        _show(self._plan, self._sources, self.columns, limit)
