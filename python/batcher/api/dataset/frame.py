"""`Dataset` — the lazy, immutable, fluent entry point.

A `Dataset` is a handle to a `LogicalPlan` plus its bound input relations. Every
operation returns a new `Dataset` (nothing mutates); no work happens until a
terminal operation (`collect`, `to_pydict`, ...). At that point `api` orchestrates
the layers: Kyber optimizes, Carbonite checks feasibility, Core executes.

One obvious way to do each thing: expressions everywhere (no lambdas), `select`
for choosing/deriving the full output, `with_columns` for adding/replacing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from datetime import timedelta
from itertools import accumulate, pairwise
from typing import TYPE_CHECKING, Any, TypeVar, overload

import pyarrow as pa

from batcher._internal.errors import (
    ColumnNotFoundError,
    PlanError,
    require_float,
    require_int,
)
from batcher.api._join_helpers import (
    _as_expr,
    _as_key_expr,
    _as_str_list,
    _asof_output,
    _broadcast,
    _join_output,
    _resolve_join_keys,
)
from batcher.api._varargs import flatten_varargs
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
from batcher.api.dataset._build.combine import (
    OrderSpec,
    build_join_where,
    build_update,
    build_zip,
)
from batcher.api.dataset._build.conform import build_drop_nans, build_match_to_schema
from batcher.api.dataset._build.reshape import build_partition_by, build_split, build_transpose
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
from batcher.api.groupby import GroupBy
from batcher.api.multi_group import MultiLevelGroupBy, cube_levels, rollup_levels
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
from batcher.plan.expr_ir import AggExpr, Aliased, CaseBuilder, Col, Expr
from batcher.plan.expr_ir.selectors import Selector, has_selector, resolve_names
from batcher.plan.expr_rewrite import is_bare_window
from batcher.plan.expr_rewrite.naming import output_name
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
    align_join_key_types,
    asof_tolerance,
    remap_sources,
)
from batcher.plan.resource import StorageLevel
from batcher.plan.schema import column_name, suggest_columns
from batcher.plan.streaming import Watermark

if TYPE_CHECKING:
    from batcher.api.dataset.dq import DatasetDQ
    from batcher.api.dataset.meta import DatasetMeta
    from batcher.api.dataset.ml import DatasetML
    from batcher.api.dataset.scd import DatasetSCD
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


def _as_opt_str_list(
    value: str | list[str] | Selector | None, ds: Dataset | None = None, where: str = ""
) -> list[str] | None:
    """Accept a single column name where a list is expected, as pandas does.

    With `ds`, a column selector (bare or in the list) resolves to the names it matches;
    without it a selector is refused by name rather than failing as "not iterable".
    """
    if ds is not None:
        plan = ds._plan
        value = resolve_names(value, plan.available_columns(), plan.available_schema(), where=where)
    elif has_selector(value) or (isinstance(value, list) and any(map(has_selector, value))):
        raise PlanError(f"{where or 'this argument'} takes column names, not a column selector")
    return [value] if isinstance(value, str) else value


def _require_columns(
    available: list[str], names: list[str] | None, *, where: str
) -> list[str] | None:
    """Check every name exists, raising the same typed error the rest of the API raises.

    The framework converters and the blob helpers reached pyarrow with an unchecked name
    and surfaced ``KeyError: 'Field "x" does not exist in schema'`` -- a message naming
    Arrow's schema object rather than the argument, and the only place on the frame where
    a column typo was not a `ColumnNotFoundError` with a did-you-mean.
    """
    if names is None:
        return None
    for name in names:
        if name not in available:
            raise ColumnNotFoundError.of(name, sorted(available), where=where)
    return names


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


def _multiset_sortable(schema: pa.Schema) -> bool:
    """Whether sorting by every column is a sound way to compare two relations as multisets.

    `Dataset.equals(ordered=False)` asks whether two results hold the same rows in any
    order. Sorting both by all columns answers that in compiled Arrow code — but only for
    types where the sort's ordering and Arrow's equality agree on which values are the
    same value. Three families where they do not:

    - **Floating point.** Arrow's equality treats ``-0.0 == 0.0`` and ``NaN != NaN``,
      neither of which the row-wise comparison does, so the two spellings disagree in
      *both* directions on exactly the ``-0.0``/``NaN`` edge cases the engine is
      elsewhere careful about.
    - **Nested types** (list, map). `sort_indices` raises on them outright.
    - **Dictionary-encoded** columns. `sort_indices` is not implemented for them.

    Anything not provably in the clear falls back to the row-wise comparison, which is
    slower but is the behavior that shipped.
    """
    return all(
        not (
            pa.types.is_floating(field.type)
            or pa.types.is_nested(field.type)
            or pa.types.is_dictionary(field.type)
        )
        for field in schema
    )


def _warn_watermark_dropped(operation: str) -> None:
    """Announce that a multi-input operation is not carrying its inputs' watermark through.

    Every single-input transform carries the watermark (`filter`, `select`, `sort`, even
    `distinct`), so a user who called `with_watermark` reasonably expects it to still be
    there. `join` and `union` do not carry it, and neither consumer of it fails loudly when
    it is gone. `groupby().agg()` builds an aggregate with ``watermark=None``, which is a
    *valid* aggregate -- an unbounded one, whose state accumulates for the life of the stream
    instead of being emitted and evicted as the watermark advances. A streaming session window
    takes ``lateness=0`` (`_build/sessions.py`), so a row that arrives late is excluded from
    its session rather than folded into it. `drop_duplicates_within_watermark` is the one
    stateful operator *not* affected: it takes its own `event_time` and `lateness` arguments
    and never reads this watermark at all.

    Carrying it automatically is the tempting fix and is not obviously right: for a
    stream-to-stream join the correct watermark is the *minimum* of the two sides, not the
    left's, and picking one would quietly change emission on a path this project's own gate
    never executes. So this announces the loss and names the one-line repair instead --
    re-applying `with_watermark` after the join restores the bound exactly.
    """
    import warnings

    warnings.warn(
        f"{operation}() does not carry the event-time watermark through: a `groupby().agg()` "
        "below it is unbounded and never evicts its state, and a streaming session "
        "window below it takes zero allowed lateness. Re-apply "
        "`.with_watermark(time_col, lateness)` to the result to restore the bound.",
        UserWarning,
        stacklevel=3,
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
        cache: StorageLevel | None = None,
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
        # Set by `cache()`: the `StorageLevel` this dataset's collected result is stored
        # at in the process result cache, or `None` for an uncached result. Deliberately
        # *not* propagated by `_derive` — caching marks this exact result; a further
        # transform is a new (uncached) result.
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
        """Notebook display: the lazy plan's output columns and types (no execution).

        A `Dataset` is lazy and possibly unbounded, so the rich repr shows the schema
        rather than silently running the query; call `show()`/`collect()` for data.

        Every name is HTML-escaped. Column names are *data* — they come out of a CSV header,
        a JSON key, or a database catalog — and a notebook renders this string as markup, so
        an unescaped name is a document written by whoever produced the file.
        """
        import html

        names = self.columns
        types = self._html_types(names)
        head = "".join(f"<th>{html.escape(str(c))}</th>" for c in names)
        row = "".join(f"<td><code>{html.escape(t)}</code></td>" for t in types)
        return (
            "<div><strong>Dataset</strong> "
            f"<em>(lazy, {len(names)} columns — call .show() to preview)</em>"
            f"<table><thead><tr>{head}</tr></thead><tbody><tr>{row}</tr></tbody></table></div>"
        )

    def _html_types(self, names: list[str]) -> list[str]:
        """The column types for the notebook repr, or blanks when they cannot be inferred.

        Pure plan analysis with no execution, and best-effort: a source whose schema is only
        knowable by reading it must not be read by a *repr*. An unknown type shows as an
        em-dash rather than making the whole repr fail.
        """
        try:
            schema = self._plan.available_schema()
            if schema is None:
                return ["—"] * len(names)
            arrow = schema.arrow
            return [str(arrow.field(n).type) if n in arrow.names else "—" for n in names]
        except Exception:  # a repr must never raise; an unknown schema is not an error
            return ["—"] * len(names)

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

        **Routed like `collect()`, not like `iter_batches()`.** The protocol takes no
        execution arguments, so the export has to pick a routing policy, and the two
        available defaults disagree: `collect()` resolves ``distributed="auto"`` while
        `iter_batches()` defaults to ``False``. Taking the latter meant that on a
        multi-node cluster ``pl.DataFrame(ds)`` silently ran single-node while
        ``pl.DataFrame(ds.collect())`` distributed — the same query, the same cluster, one
        of them not using it. A caller who wants a specific mode has `iter_batches`; an
        export that cannot be told has to match the terminal op it stands in for. On one
        node ``"auto"`` resolves to single-node, so nothing changes there, and it never
        starts a cluster that was not already connected.

        Args:
            requested_schema: A schema capsule the consumer would prefer, per the
                protocol. Honoured only when it matches; otherwise the stream's own
                schema is exported (the consumer must then cast).

        Returns:
            An ``ArrowArrayStream`` PyCapsule.
        """
        batches = self.iter_batches(distributed="auto")
        reader = pa.RecordBatchReader.from_batches(self.schema, batches)
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

    def _named_positionals(self, exprs: tuple[Expr, ...], api: str) -> dict[str, Expr]:
        """Resolve positional expressions to an ordered name -> expr map.

        A selector expands to one entry per matched column; anything else is named by
        `output_name` (its alias, else its leftmost column, else ``"literal"``). Two
        entries landing on one name are refused rather than silently overwritten.
        """
        pairs: list[tuple[str, Expr]] = []
        for e in exprs:
            if isinstance(e, str):
                pairs.append((e, Col(e)))
            elif has_selector(e):
                pairs.extend(expand_selector_expr(self, e))
            elif isinstance(e, (Expr, AggExpr, CaseBuilder)):
                expr = _as_expr(e)
                pairs.append((output_name(expr), expr))
            else:
                # A bare scalar is almost always a mistake (a column *position*), so it is not
                # lifted to a literal here; `bt.lit(...)` says a constant on purpose.
                raise PlanError(
                    f"positional {api}() arguments must be column names, expressions, or "
                    f"column selectors, got {type(e).__name__}; spell a constant bt.lit(...)"
                )
        out: dict[str, Expr] = {}
        for name, expr in pairs:
            if name in out:
                raise PlanError(
                    f"{api}() would produce the duplicate output column {name!r}: an unnamed "
                    "expression is named after its leftmost column (or 'literal'), so two "
                    "can collide -- rename one with .alias('...') or pass it as a keyword"
                )
            out[name] = expr.inner if isinstance(expr, Aliased) else expr
        return out

    def cache(self, storage_level: StorageLevel | str | None = None) -> Dataset:
        """Mark this dataset's result to be cached after it is computed.

        The first terminal op (``collect`` and friends) on the returned dataset
        executes normally and stores its Arrow result in a process-wide, byte-bounded
        cache keyed by the plan and its inputs; later terminals on an equivalent dataset
        return the cached result without re-executing. Eviction is cost-aware rather than
        purely recent: an expensive, small, often-served result outlives a cheap, large,
        cold one.

        The memory half is bounded by ``memory.result_cache_max_bytes`` and yields its RAM
        back to running queries under pressure, so caching never grows the process without
        bound. What that pressure sheds is written to a local disk tier
        (``memory.result_cache_disk_max_bytes``) rather than dropped, so a working set
        larger than the memory budget costs a read-back rather than a full recompute.
        `storage_level` chooses how far that goes.

        Like Spark and Polars ``cache``, this marks *this* result; a further transform is
        a new, uncached result. Single-node relational results only.

        A terminal that materializes the result (``collect``, ``to_pydict``, ``to_arrow``,
        and the framework conversions) is what *fills* the cache. ``count()``,
        ``is_empty()`` and ``iter_batches()`` are served from a warm one but never fill it,
        because filling it would mean materializing the very result those three exist to
        avoid materializing.

        Args:
            storage_level: Which media the result may occupy — a
                :class:`~batcher.StorageLevel` or its name. Defaults to
                ``MEMORY_AND_DISK``.

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
                >>> big = bt.from_pydict({"x": [1, 2]}).cache("disk_only")
                >>> big.collect().num_rows  # never charged against the memory budget
                2
        """
        return Dataset(
            self._plan,
            self._sources,
            repartition=self._repartition,
            watermark=self._watermark,
            cache=StorageLevel.parse(storage_level),
        )

    def uncache(self) -> Dataset:
        """Drop this dataset's cached result from both cache tiers, if it is held.

        The counterpart to :meth:`cache`, spelled ``unpersist`` in Spark (both names work
        here). Caching is otherwise self-managing — the budget evicts, and pressure
        reclaims — so this is for the case the budget cannot see: a result you know is
        stale or will not be read again, whose RAM and disk you want back *now* rather
        than at the next eviction.

        A no-op when nothing is cached under this plan, so it is always safe to call.
        Returns the dataset so it can sit in a chain; it changes no plan and no result.

        This drops **this process's** copy. An entry in a shared store
        (``memory.shared_cache_uri``) is left alone, because it belongs to every process
        reading it and dropping it on their behalf is not a decision one caller should
        make. Shared entries are keyed by their inputs' content versions, so rewriting the
        data a result came from already retires it.

        Returns:
            This same `Dataset`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2]}).cache()
                >>> ds.collect().num_rows
                2
                >>> ds.uncache().collect().num_rows  # recomputed, not served
                2
        """
        from batcher import carbonite
        from batcher.api.executors import result_cache_key

        carbonite.result_cache().invalidate(result_cache_key(self._plan, self._sources))
        return self

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

        time_col = column_name(time_col, arg="time_col", api="with_watermark")
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

    def filter(
        self,
        *predicates: Expr | str | Callable | type,
        batch_size: int | None = None,
        batch_format: str = "pyarrow",
        input_columns: list[str] | None = None,
        num_workers: int | str = "auto",
        num_cpus: float | None = None,
        num_gpus: float = 0.0,
        memory: float | None = None,
        compute: Any = None,
        concurrency: int | tuple[int, ...] | None = None,
        ray_remote_args: dict[str, Any] | None = None,
        ray_remote_args_fn: Callable[[], dict[str, Any]] | None = None,
        zero_copy_batch: bool = True,
        fn_args: tuple | None = None,
        fn_kwargs: dict | None = None,
        fn_constructor_args: tuple | None = None,
        fn_constructor_kwargs: dict | None = None,
        max_concurrency: int = 0,
        max_errored_rows: int = 0,
        **equals: Any,
    ) -> Dataset:
        """Keep only the rows where every predicate is true.

        A predicate takes one of three forms, and the argument decides which:

        - An **expression** built from columns, such as ``col("amount") > 100``. Combine
          conditions with ``&`` (and), ``|`` (or), and ``~`` (not), parenthesizing each side
          because those operators bind tighter than comparisons. This is the form to prefer:
          it runs in Rust and the optimizer can push it into the scan.
        - A **SQL string** such as ``"amount > 100 AND region = 'eu'"``, parsed as the
          ``WHERE`` clause of a query over this dataset by the same front end as :meth:`sql`.
        - A **callable** (a function or a class) for a condition the expression language
          cannot say, such as a model's verdict. It is batch-level, never per row: `fn`
          receives a whole batch in `batch_format` and returns one boolean per row (an Arrow
          or NumPy boolean array, a pandas or Polars Series, or a list). A class is built
          once per worker, like a `map_batches` model, and the Ray Data resource parameters
          schedule it the way they schedule `map_batches`. The rows are masked as Arrow, so
          every column keeps its exact type.

        Rows where a predicate is null are dropped. Several expression or SQL predicates are
        ANDed together, and a keyword argument is an equality shorthand: ``filter(status="paid",
        region="eu")`` means ``filter((col("status") == "paid") & (col("region") == "eu"))``.
        A column whose name is one of this method's parameters cannot use the shorthand;
        compare it with ``col(...)`` instead.

        A predicate may compose window expressions — ``filter(col("x") >
        col("x").mean().over(partition_by=["g"]))`` keeps rows above their group
        mean. The window sees every input row, as in the SQL subquery it desugars to.

        The keyword-only options below apply only to a callable predicate, and passing one
        with an expression or SQL string raises. Ray Data's ``num_cpus``, ``memory`` and
        ``ray_remote_args_fn`` cannot be honoured by the map scheduler and raise when set.

        Args:
            *predicates: Boolean expressions or SQL predicate strings, ANDed together, or
                exactly one callable batch predicate. A list is accepted in place of
                separate arguments.
            batch_size: Rows per batch handed to a callable predicate.
            batch_format: What a callable predicate sees — ``"pyarrow"``, ``"numpy"``,
                ``"pandas"``, ``"torch"``, ``"polars"`` or ``"jax"``.
            input_columns: The columns a callable predicate reads. Prunes the scan, and
                narrows the batch the predicate is handed to exactly these columns.
            num_workers: Concurrent predicate calls within a worker (``"auto"`` sizes it).
            num_cpus: Ray Data's per-worker CPU request; raises when set.
            num_gpus: GPUs to reserve per worker.
            memory: Ray Data's per-worker memory request; raises when set.
            compute: A Ray Data ``ActorPoolStrategy``/``TaskPoolStrategy``, or
                ``"actors"``/``"tasks"``; folded into `concurrency`.
            concurrency: Size of the distributed actor pool: an int, ``(min, max)``, or
                ``(min, max, initial)`` with ``initial == min``.
            ray_remote_args: Ray options; ``num_gpus``, ``resources`` and
                ``accelerator_type`` are honoured and any other key raises.
            ray_remote_args_fn: Ray Data's per-task options callback; raises when set.
            zero_copy_batch: Hand the predicate read-only zero-copy views. ``False`` copies
                a NumPy, pandas or torch batch first so the predicate may mutate it.
            fn_args: Positional arguments appended to every call: ``fn(batch, *fn_args)``.
            fn_kwargs: Keyword arguments forwarded to every call.
            fn_constructor_args: Positional arguments for a class predicate's construction.
            fn_constructor_kwargs: Keyword arguments for a class predicate's construction.
            max_concurrency: In-flight batches for an ``async def`` predicate; 0 = a default.
            max_errored_rows: Rows a raising predicate may drop per worker before failing.
            **equals: Column-equals-value shorthands, ANDed with `predicates`.

        Returns:
            A new `Dataset` with the matching rows.

        Raises:
            PlanError: If no condition is given, an argument is none of the three forms, a
                callable is mixed with another predicate, a keyword names a column the dataset
                does not have, or a callable-only option is given without a callable.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 5, 9], "ok": [True, False, True]})
                >>> ds.filter((bt.col("x") > 2) & bt.col("ok")).to_pydict()
                {'x': [9], 'ok': [True]}

                >>> ds.filter("x > 2 AND ok").to_pydict()
                {'x': [9], 'ok': [True]}

                >>> import pyarrow.compute as pc
                >>> ds.filter(lambda batch: pc.greater(batch["x"], 2)).to_pydict()
                {'x': [5, 9], 'ok': [False, True]}

                >>> ds = bt.from_pydict({"g": ["a", "b", "a"], "x": [1, 2, 3]})
                >>> ds.filter(g="a").to_pydict()
                {'g': ['a', 'a'], 'x': [1, 3]}
        """
        from batcher.api.dataset._udf import build_filter, resolve_placement
        from batcher.api.dataset._udf.build import refuse_callable_options

        conditions = list(flatten_varargs(predicates))
        fns = [c for c in conditions if callable(c) and not isinstance(c, Expr)]
        options = {
            "batch_size": batch_size,
            "batch_format": batch_format,
            "input_columns": input_columns,
            "num_workers": num_workers,
            "zero_copy_batch": zero_copy_batch,
            "max_concurrency": max_concurrency,
            "max_errored_rows": max_errored_rows,
        }
        ray = {
            "num_cpus": num_cpus,
            "num_gpus": num_gpus,
            "memory": memory,
            "compute": compute,
            "concurrency": concurrency,
            "ray_remote_args": ray_remote_args,
            "ray_remote_args_fn": ray_remote_args_fn,
        }
        bindings = (fn_args, fn_kwargs, fn_constructor_args, fn_constructor_kwargs)
        if fns:
            if len(conditions) != 1 or equals:
                raise PlanError(
                    "filter() takes one callable predicate on its own; AND an expression with "
                    "it in a second filter(...) call, which Kyber can push below the callable"
                )
            fn = fns[0]
            return build_filter(
                self,
                fn,
                placement=resolve_placement("filter", fn, **ray),
                bindings=bindings,
                **options,
            )
        refuse_callable_options(
            Dataset.filter,
            {
                **options,
                **ray,
                "fn_args": fn_args,
                "fn_kwargs": fn_kwargs,
                "fn_constructor_args": fn_constructor_args,
                "fn_constructor_kwargs": fn_constructor_kwargs,
            },
        )
        for name, value in equals.items():
            self._require_column(name, "filter")
            conditions.append(Col(name) == value)
        if not conditions:
            raise PlanError(
                "filter() requires a condition, e.g. filter(col('x') > 0) or filter(x=1)"
            )
        dataset = self
        exprs: list[Expr] = []
        for cond in conditions:
            if isinstance(cond, str):
                dataset = dataset._filter_sql(cond)
            elif isinstance(cond, Expr):
                exprs.append(cond)
            else:
                raise PlanError(
                    "filter() takes an expression (col('x') > 0), a SQL predicate string "
                    "('x > 0'), or a callable batch predicate; got "
                    f"{type(cond).__name__}"
                )
        if not exprs:
            return dataset
        combined = exprs[0]
        for cond in exprs[1:]:
            combined = combined & cond
        return windowed_filter(dataset, combined)

    def _filter_sql(self, predicate: str) -> Dataset:
        """Keep the rows a SQL predicate string accepts, parsed as a ``WHERE`` clause."""
        from batcher._internal.sql_errors import parse_sql

        parsed = parse_sql(predicate, dialect="duckdb")
        from sqlglot import expressions as exp

        if isinstance(parsed, exp.Query | exp.Command) or not predicate.strip():
            raise PlanError(
                f"filter() got {predicate!r}, which is not a SQL predicate; pass the "
                "condition alone, such as 'x > 1', or run a whole query with ds.sql(...)"
            )
        return self.sql(f"SELECT * FROM self WHERE ({predicate})")

    def select(self, *columns: str | Expr, **named: Expr | int | float | bool | str) -> Dataset:
        """Project to exactly the given columns.

        Positional args are column names (strings), expressions, or column selectors
        (``bt.exclude("id")``, ``bt.numeric() * 2``, ``bt.col("a", "b")``) which expand
        to one output per matched column; keyword args bind a new name to an
        expression: ``ds.select("id", total=col("price") * col("qty"))``.

        A positional expression is named the way Polars names it: by its
        ``.alias(...)`` if it has one, else by its leftmost column, else
        ``"literal"`` -- so ``select(col("a") + 1)`` yields a column ``a``. Two
        outputs landing on one name raise rather than overwrite each other.

        An aggregate (``col("x").sum()``) is the whole-frame aggregate. When every
        output is an aggregate or a constant the result is one row; mixed with
        row-level outputs it is broadcast to every row, as ``sum(x) OVER ()`` is.

        Args:
            *columns: Column names, expressions, or column selectors. A list of them is
                accepted in place of separate arguments, as Polars and PySpark accept one.
            **named: New column names bound to expressions.

        Returns:
            A new `Dataset` with exactly the selected columns.

        Raises:
            PlanError: If two outputs would share a name.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3], "g": ["a", "b", "a"]})
                >>> ds.select("x", y=bt.col("x") * 2).to_pydict()
                {'x': [1, 2, 3], 'y': [2, 4, 6]}

                >>> ds.select(bt.exclude("g")).to_pydict()
                {'x': [1, 2, 3]}

                >>> ds.select(bt.col("x") + 1, bt.col("x").sum().alias("total")).to_pydict()
                {'x': [2, 3, 4], 'total': [6, 6, 6]}
        """
        columns = flatten_varargs(columns)
        items = [Projection(n, e) for n, e in self._named_positionals(columns, "select").items()]
        taken = {p.alias for p in items}
        for alias, expr in named.items():
            if alias in taken:
                raise PlanError(
                    f"select() got the duplicate output column {alias!r} both positionally "
                    "and as a keyword; give each output column exactly one definition"
                )
            items.append(Projection(alias, _as_expr(expr)))
        if not items:
            raise PlanError(_empty_projection_message("select", columns))
        return windowed_project(self, items, collapse=True)

    def with_columns(self, *exprs: Expr, **named: Expr | int | float | bool | str) -> Dataset:
        """Add or replace columns, keeping all existing ones.

        Values may be expressions, scalars, or window expressions from
        ``agg.over(...)`` (e.g. ``with_columns(total=col("x").sum().over(partition_by=["g"]))``).
        A window may be composed with ordinary arithmetic —
        ``with_columns(share=col("x") / col("x").sum().over())`` — and window and
        non-window columns may be mixed freely in one call.

        A positional column selector (``bt.numeric().round(2)``) replaces each matched
        column in place. Any other positional expression is named as `select` names
        one: by its alias, else its leftmost column, else ``"literal"`` -- so
        ``with_columns(col("x") * 2)`` replaces ``x``.

        Args:
            *exprs: Column selectors or expressions, named by their alias or leftmost
                column. A list of them is accepted in place of separate arguments.
            **named: Column names bound to expressions (or scalars) to add or replace.

        Returns:
            A new `Dataset` with the columns added or replaced.

        Raises:
            PlanError: If two outputs would share a name.

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
        exprs = flatten_varargs(exprs)
        positional = self._named_positionals(exprs, "with_columns")
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
    ) -> Dataset:
        """Order rows by one or more keys (column names or expressions).

        `descending`/`nulls_first` are either a single bool applied to all keys or
        a list matching the number of keys.

        Args:
            *keys: The sort keys, as column names or expressions. A list of them is
                accepted in place of separate arguments.
            descending: Sort descending — one bool for all keys or a per-key list.
            nulls_first: Order nulls first — one bool for all keys or a per-key list.

        Returns:
            A new `Dataset` with rows ordered by the keys.

        Raises:
            PlanError: If no key is given, or a `descending`/`nulls_first` list does not
                match the number of keys.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [3, 1, 2]})
                >>> ds.sort("x", descending=True).to_pydict()
                {'x': [3, 2, 1]}

                >>> ds.sort("x", descending=True).to_pydict()
                {'x': [3, 2, 1]}
        """
        by = flatten_varargs(keys)
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
        order_by: list[str | tuple[str, bool] | tuple[str, bool, bool] | Expr] = (),
        functions: dict[str, str | tuple[str, str]],
        frame: tuple[int | None, int | None] | None = None,
    ) -> Dataset:
        """Append window-function columns, preserving all input columns.

        Rows are partitioned by `partition_by` (empty → one partition) and ordered
        by `order_by` (column names, ``(name, descending)`` tuples, or expressions).
        A third element places the nulls, ``(name, descending, nulls_first)``; without
        it nulls sort last, as SQL's ``ORDER BY`` does. Where the nulls sit changes
        rankings and, under a running frame, which rows the frame contains.
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
            frame: ``(start, end)`` signed row offsets, optionally with a third
                units element as ``(start, end, units)`` -- ``"rows"`` (default),
                ``"range"`` or ``"groups"``.

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
                >>> ds.map(lambda r: {"x": r["x"] * 10}).to_pydict()
                {'x': [10, 20, 30]}
        """
        # Imported here, not at module scope: the four accessor namespaces pull in the ML,
        # data-quality, metadata-shortcut, and SCD stacks — and through the metadata one,
        # the whole optimizer — none of which a pipeline that never touches an accessor
        # needs. Deferring them is most of what `import batcher` used to spend.
        from batcher.api.dataset.ml import DatasetML

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
        from batcher.api.dataset.dq import DatasetDQ

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
        from batcher.api.dataset.meta import DatasetMeta

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
        from batcher.api.dataset.scd import DatasetSCD

        return DatasetSCD(self)

    def map_batches(
        self,
        fn: Callable | type,
        *,
        batch_size: int | None = None,
        batch_format: str = "pyarrow",
        input_columns: list[str] | None = None,
        preserves_columns: list[str] | None = None,
        output_columns: list[str] | None = None,
        num_workers: int | str = "auto",
        num_cpus: float | None = None,
        num_gpus: float = 0.0,
        memory: float | None = None,
        compute: Any = None,
        concurrency: int | tuple[int, ...] | None = None,
        ray_remote_args: dict[str, Any] | None = None,
        ray_remote_args_fn: Callable[[], dict[str, Any]] | None = None,
        zero_copy_batch: bool = True,
        fn_args: tuple | None = None,
        fn_kwargs: dict | None = None,
        fn_constructor_args: tuple | None = None,
        fn_constructor_kwargs: dict | None = None,
        accelerator_type: str | None = None,
        resources: dict[str, float] | None = None,
        model_memory_gb: float = 0.0,
        multiprocessing: bool = False,
        max_errored_rows: int = 0,
        timeout: float = 0.0,
        max_retries: int = 0,
        retry_backoff: float = 0.5,
        retry_on: type[BaseException] | tuple[type[BaseException], ...] | None = None,
        max_concurrency: int = 0,
    ) -> Dataset:
        """Apply a Python function to each batch.

        `fn` receives one batch and returns the transformed batch — the building block for
        batch inference, embeddings, and custom preprocessing. Pass a **class** instead of a
        function to load a model *once per worker*: it is instantiated once, and the callable
        instance handles each batch. That is the stateful GPU-inference pattern, and a class,
        `num_gpus`, or `concurrency` is what puts the stage on a long-lived actor pool under
        ``collect(distributed=True)``.

        `batch_format` chooses what `fn` sees and returns: ``"pyarrow"`` (a
        `pyarrow.RecordBatch`, zero-copy, the default), ``"numpy"`` (a ``{column: ndarray}``
        dict, Ray Data's default), ``"pandas"`` (a `DataFrame`), ``"torch"`` (a
        ``{column: tensor}`` dict over numeric columns), ``"polars"`` (a `polars.DataFrame`),
        or ``"jax"`` (a ``{column: jax.Array}`` dict over numeric columns). Conversion happens
        only around the call; the engine boundary stays Arrow. A `pyarrow`/`numpy` `fn` may
        also return a Table or column dict. With the default ``zero_copy_batch=True`` a
        NumPy batch is a read-only view where the conversion allows one; pass ``False`` to
        receive writable copies, as Ray Data does by default.

        `num_workers` defaults to ``"auto"``: the per-batch calls fan across all local cores
        for a CPU stage, or one model and CUDA context for a GPU stage, so a batch transform
        is parallel by default rather than single-threaded. Threads only speed up a
        GIL-releasing `fn` (Arrow/NumPy/torch); pass ``multiprocessing=True`` for a CPU-bound
        pure-Python `fn`, which needs import-safe calling code (an
        ``if __name__ == "__main__":`` guard) because the process pool spawns.

        `max_errored_rows` gives dirty-data tolerance: a batch whose `fn` raises is bisected to
        isolate the offending rows, and a failing row is dropped, up to this many per worker.
        `max_retries`/`timeout`/`retry_on` add resilience for a flaky external `fn`: a batch
        is retried with exponential backoff (`retry_backoff * 2**attempt` seconds), and a call
        past `timeout` raises `TimeoutError`, retried like any transient. Pass an
        ``async def`` `fn` for an I/O-bound stage; its batches run concurrently on one event
        loop, up to `max_concurrency` in flight.

        Under `distributed=True`, a partition whose worker is preempted is recomputed from its
        durable input, so `fn` must be idempotent: make an external sink an upsert on a stable
        key rather than a blind insert.

        The Ray Data resource parameters land on what the map scheduler honours: `num_gpus`,
        `concurrency` (an int, ``(min, max)``, or ``(min, max, initial)`` with
        ``initial == min``), `compute` (folded into `concurrency`), and the ``num_gpus``,
        ``resources`` and ``accelerator_type`` keys of `ray_remote_args`. ``num_cpus``,
        ``memory``, ``ray_remote_args_fn`` and every other `ray_remote_args` key cannot be
        honoured and raise `PlanError` when set, rather than being accepted and ignored.

        Warns (`PerformanceWarning`) when a GPU stage is given a plain function rather than a
        class, because a function rebuilds the model on every batch.

        Args:
            fn: A function (or class/factory) applied to each batch.
            batch_size: Rebatch to this many rows before each call; ``None`` uses the
                engine's batches.
            batch_format: What `fn` sees — ``"pyarrow"``, ``"numpy"``, ``"pandas"``,
                ``"torch"``, ``"polars"`` or ``"jax"``.
            input_columns: The columns `fn` reads. Declaring them lets the optimizer prune
                everything else out of the scan. Omitting a column the `fn` actually reads is
                a correctness bug, not a slow path: it is pruned out from under the function.
            preserves_columns: The columns `fn` returns unchanged, same name and same value in
                every output row. A later `filter` reading only these runs *below* the UDF, so
                the model scores fewer rows. Naming a column `fn` rewrites changes the result.
            output_columns: The result schema when `fn` changes the columns.
            num_workers: Concurrent per-batch calls within a worker (``"auto"`` sizes to the
                stage), or an explicit int.
            num_cpus: Ray Data's per-worker CPU request; raises when set.
            num_gpus: GPUs to reserve per distributed worker.
            memory: Ray Data's per-worker memory request; raises when set. Use
                `model_memory_gb` to budget a model's footprint.
            compute: A Ray Data ``ActorPoolStrategy``/``TaskPoolStrategy``, or
                ``"actors"``/``"tasks"``; folded into `concurrency`.
            concurrency: Size of the distributed actor pool: an int, ``(min, max)``, or
                ``(min, max, initial)`` with ``initial == min``.
            ray_remote_args: Ray options; ``num_gpus``, ``resources`` and
                ``accelerator_type`` are honoured and any other key raises.
            ray_remote_args_fn: Ray Data's per-task options callback; raises when set.
            zero_copy_batch: Hand `fn` read-only zero-copy views. ``False`` copies a NumPy,
                pandas or torch batch first so `fn` may mutate it.
            fn_args: Positional arguments appended to every call: ``fn(batch, *fn_args)``.
            fn_kwargs: Keyword arguments forwarded to every ``fn(batch, ...)`` call.
            fn_constructor_args: Positional arguments for a class `fn`'s one-per-worker
                construction, such as a checkpoint path; invalid for a function.
            fn_constructor_kwargs: Keyword arguments for a class `fn`'s construction.
            accelerator_type: Pin actors to a device model (e.g. ``"NVIDIA_A100"``).
            resources: Custom Ray resources per worker, e.g. ``{"TPU": 4}``, for an
                accelerator Ray does not report as ``GPU``.
            model_memory_gb: The model's footprint, for host-RAM budgeting and VRAM packing.
            multiprocessing: Run CPU-bound pure-Python calls across processes.
            max_errored_rows: Rows a raising `fn` may drop per worker before failing.
            timeout: Wall-clock ceiling (seconds) for one `fn` call; 0 = no timeout.
            max_retries: Times to retry a batch whose `fn` raises a retryable error.
            retry_backoff: Base backoff (seconds); attempt `k` waits `retry_backoff * 2**k`.
            retry_on: Exception type(s) worth retrying; ``None`` retries any `Exception`.
            max_concurrency: Max in-flight batches for an ``async def`` `fn`; 0 = a default.

        Returns:
            A new lazy `Dataset` with `fn` applied to every batch.

        Raises:
            PlanError: If an option is invalid, or a resource parameter cannot be honoured.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import pyarrow.compute as pc
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> def add_one(batch):
                ...     return batch.set_column(0, "x", pc.add(batch.column("x"), 1))
                >>> ds.map_batches(add_one).to_pydict()
                {'x': [2, 3, 4]}

                >>> class AddN:
                ...     def __init__(self, n):
                ...         self.n = n
                ...     def __call__(self, batch):
                ...         return {"x": batch["x"] + self.n}
                >>> added = ds.map_batches(AddN, fn_constructor_args=(10,), batch_format="numpy")
                >>> added.to_pydict()
                {'x': [11, 12, 13]}
        """
        from batcher.api.dataset._udf import bind_fn, build_map_batches, resolve_placement
        from batcher.api.dataset._udf.build import writable_format
        from batcher.api.dataset._udf.checks import validate_fn

        validate_fn(fn)  # before binding, which wraps a class in one that is always callable
        placement = resolve_placement(
            "map_batches",
            fn,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            compute=compute,
            concurrency=concurrency,
            ray_remote_args=ray_remote_args,
            ray_remote_args_fn=ray_remote_args_fn,
            accelerator_type=accelerator_type,
            resources=resources,
        )
        bound = bind_fn(
            fn,
            fn_args,
            fn_kwargs,
            fn_constructor_args,
            fn_constructor_kwargs,
            writable_format("map_batches", batch_format, zero_copy_batch),
        )
        return build_map_batches(
            self,
            bound,
            placement=placement,
            batch_size=batch_size,
            batch_format=batch_format,
            input_columns=input_columns,
            preserves_columns=preserves_columns,
            output_columns=output_columns,
            num_workers=num_workers,
            model_memory_gb=model_memory_gb,
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

        _require_columns(self.columns, [column], where="in offload_blobs()")
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

        _require_columns(self.columns, [uri_column], where="in materialize_blobs()")
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
        fn: Callable | type,
        *,
        batch_size: int | None = None,
        batch_format: str = "pyarrow",
        input_columns: list[str] | None = None,
        output_columns: list[str] | None = None,
        num_workers: int | str = "auto",
        num_cpus: float | None = None,
        num_gpus: float = 0.0,
        memory: float | None = None,
        compute: Any = None,
        concurrency: int | tuple[int, ...] | None = None,
        ray_remote_args: dict[str, Any] | None = None,
        ray_remote_args_fn: Callable[[], dict[str, Any]] | None = None,
        zero_copy_batch: bool = True,
        fn_args: tuple | None = None,
        fn_kwargs: dict | None = None,
        fn_constructor_args: tuple | None = None,
        fn_constructor_kwargs: dict | None = None,
        max_concurrency: int = 0,
        max_errored_rows: int = 0,
    ) -> Dataset:
        """Apply a per-row Python function ``fn(row) -> row`` (Ray Data ``map``).

        Each row is passed to `fn` as a ``{column: value}`` dict **inside the worker**, never
        the driver, and the per-row cost is yours. Prefer the vectorized `map_batches` when
        the work can be expressed over whole columns; it is far faster.

        `input_columns` matters more here than anywhere else. A row callback pays for every
        column twice, once to read it and once to box it into a Python object per row, so
        declaring the columns it reads lets projection pushdown prune the scan.

        `batch_format` picks the row values: ``"pyarrow"`` rows hold Python values, and
        ``"numpy"`` rows hold NumPy values the way Ray Data's rows do, so a tensor column
        arrives as an ``ndarray`` per row. Pass a class to build a model once per worker, and
        an ``async def`` `fn` for an I/O-bound per-row call, whose rows are awaited
        concurrently within each batch, up to `max_concurrency` at a time. The Ray Data
        resource parameters behave as they do on `map_batches`.

        Args:
            fn: A ``row -> row`` function, ``async def``, or class, applied per row.
            batch_size: Rows handed to each worker call.
            batch_format: The row values — ``"pyarrow"`` (Python values) or ``"numpy"``.
            input_columns: The columns `fn` reads, so projection pushdown can prune the scan.
                Omitting a column the callback reads is a correctness bug: the pruned column
                is missing from the row dict.
            output_columns: The result schema when `fn` changes the columns.
            num_workers: Concurrent calls within a worker (``"auto"`` sizes it).
            num_cpus: Ray Data's per-worker CPU request; raises when set.
            num_gpus: GPUs to reserve per distributed worker.
            memory: Ray Data's per-worker memory request; raises when set.
            compute: A Ray Data ``ActorPoolStrategy``/``TaskPoolStrategy``, or
                ``"actors"``/``"tasks"``; folded into `concurrency`.
            concurrency: Size of the distributed actor pool: an int, ``(min, max)``, or
                ``(min, max, initial)`` with ``initial == min``.
            ray_remote_args: Ray options; ``num_gpus``, ``resources`` and
                ``accelerator_type`` are honoured and any other key raises.
            ray_remote_args_fn: Ray Data's per-task options callback; raises when set.
            zero_copy_batch: With ``batch_format="numpy"``, ``False`` copies read-only
                arrays so `fn` may mutate a row's values.
            fn_args: Positional arguments appended to every call: ``fn(row, *fn_args)``.
            fn_kwargs: Keyword arguments forwarded to every call.
            fn_constructor_args: Positional arguments for a class `fn`'s construction.
            fn_constructor_kwargs: Keyword arguments for a class `fn`'s construction.
            max_concurrency: In-flight per-row awaits within a batch for an ``async`` `fn`.
            max_errored_rows: Rows a raising `fn` may drop per worker before failing.

        Returns:
            A new lazy `Dataset` with `fn` applied to every row.

        Raises:
            PlanError: If an option is invalid, or a resource parameter cannot be honoured.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.map(lambda row: {"x": row["x"] * 2}).to_pydict()
                {'x': [2, 4, 6]}
        """
        return self._map_rows(fn, False, locals())

    def flat_map(
        self,
        fn: Callable | type,
        *,
        batch_size: int | None = None,
        batch_format: str = "pyarrow",
        input_columns: list[str] | None = None,
        output_columns: list[str] | None = None,
        num_workers: int | str = "auto",
        num_cpus: float | None = None,
        num_gpus: float = 0.0,
        memory: float | None = None,
        compute: Any = None,
        concurrency: int | tuple[int, ...] | None = None,
        ray_remote_args: dict[str, Any] | None = None,
        ray_remote_args_fn: Callable[[], dict[str, Any]] | None = None,
        zero_copy_batch: bool = True,
        fn_args: tuple | None = None,
        fn_kwargs: dict | None = None,
        fn_constructor_args: tuple | None = None,
        fn_constructor_kwargs: dict | None = None,
        max_concurrency: int = 0,
        max_errored_rows: int = 0,
    ) -> Dataset:
        """Apply ``fn(row) -> iterable[row]`` per row and flatten (Ray Data ``flat_map``).

        A one-to-many row transform. Like `map`, `fn` runs per row inside the worker, and each
        call returns zero or more output rows, all concatenated. The options are `map`'s.

        Args:
            fn: A ``row -> iterable[row]`` function, ``async def``, or class, applied per row.
            batch_size: Rows handed to each worker call.
            batch_format: The row values — ``"pyarrow"`` (Python values) or ``"numpy"``.
            input_columns: The columns `fn` reads, so projection pushdown can prune the scan.
            output_columns: The result schema when `fn` changes the columns.
            num_workers: Concurrent calls within a worker (``"auto"`` sizes it).
            num_cpus: Ray Data's per-worker CPU request; raises when set.
            num_gpus: GPUs to reserve per distributed worker.
            memory: Ray Data's per-worker memory request; raises when set.
            compute: A Ray Data ``ActorPoolStrategy``/``TaskPoolStrategy``, or
                ``"actors"``/``"tasks"``; folded into `concurrency`.
            concurrency: Size of the distributed actor pool: an int, ``(min, max)``, or
                ``(min, max, initial)`` with ``initial == min``.
            ray_remote_args: Ray options; ``num_gpus``, ``resources`` and
                ``accelerator_type`` are honoured and any other key raises.
            ray_remote_args_fn: Ray Data's per-task options callback; raises when set.
            zero_copy_batch: With ``batch_format="numpy"``, ``False`` copies read-only
                arrays so `fn` may mutate a row's values.
            fn_args: Positional arguments appended to every call: ``fn(row, *fn_args)``.
            fn_kwargs: Keyword arguments forwarded to every call.
            fn_constructor_args: Positional arguments for a class `fn`'s construction.
            fn_constructor_kwargs: Keyword arguments for a class `fn`'s construction.
            max_concurrency: In-flight per-row awaits within a batch for an ``async`` `fn`.
            max_errored_rows: Rows a raising `fn` may drop per worker before failing.

        Returns:
            A new lazy `Dataset` of the flattened rows.

        Raises:
            PlanError: If an option is invalid, or a resource parameter cannot be honoured.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2]})
                >>> ds.flat_map(lambda row: [{"x": row["x"]}, {"x": row["x"]}]).to_pydict()
                {'x': [1, 1, 2, 2]}
        """
        return self._map_rows(fn, True, locals())

    def _map_rows(self, fn: Callable | type, flat: bool, given: dict[str, Any]) -> Dataset:
        """The shared body of `map`/`flat_map`, over the caller's own arguments."""
        from batcher.api.dataset._udf import build_rows, resolve_placement
        from batcher.api.dataset._udf.build import BINDING_PARAMS, RAY_PARAMS, ROW_OPTIONS

        ray = {name: given[name] for name in RAY_PARAMS}
        return build_rows(
            self,
            fn,
            flat=flat,
            placement=resolve_placement("flat_map" if flat else "map", fn, **ray),
            bindings=tuple(given[name] for name in BINDING_PARAMS),
            **{name: given[name] for name in ROW_OPTIONS},
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
        from batcher.api.session.sql import current_session

        default = current_session()
        session = default if dialect is None else default._with_dialect(dialect)
        return session._run(query, {table_name: self})

    def drop(
        self,
        *names: str | Selector,
    ) -> Dataset:
        """Return a dataset without the named columns, preserving the rest in order.

        The complement of `select`: name the columns to remove rather than the ones
        to keep, either by name or with a column selector (``ds.drop(bt.temporal())``).
        Lazy. Raises `PlanError` on an unknown column name (with a suggestion) or if
        every column would be dropped.

        Args:
            *names: Names of the columns to remove, or column selectors matching them.
                A list is accepted in place of separate arguments.

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

                >>> ds.drop("b", "c").to_pydict()
                {'a': [1, 2]}

                >>> ds.drop(bt.matches("^[bc]$")).to_pydict()
                {'a': [1, 2]}
        """
        targets: tuple[str | Selector, ...] = flatten_varargs(names)
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
        available = self._plan.available_columns()
        if mapping is not None and not callable(mapping) and not hasattr(mapping, "items"):
            raise PlanError(
                f"rename(): mapping must be a dict of {{old: new}} or a callable, got "
                f"{type(mapping).__name__} {mapping!r}"
            )
        if callable(mapping):
            renamed = {c: mapping(c) for c in available}
            # One counting pass, not a `list.count()` per element: the latter is quadratic
            # in the column count, and `rename(str.lower)` over a wide relation is exactly
            # the call that would hit it hardest — thousands of columns, and the collision
            # check running longer than everything else the rename does.
            produced = Counter(renamed.values())
            collisions = sorted(name for name, n in produced.items() if n > 1)
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
        subset: str | list[str] | None = None,
        *,
        keep: str = "any",
        order_by: str | list[str] | list[tuple[str, bool]] | None = None,
    ) -> Dataset:
        """Remove duplicate rows.

        With no `subset`, DISTINCT over all columns. With `subset`, keep one row per
        distinct key combination, carrying every other column: `keep="first"`/`"last"`
        picks the row that is first/last in `order_by` order (required for those modes);
        `keep="any"` picks an arbitrary one. Either way this is one mergeable reduction —
        a single hash pass over the key, no sort and no rank column — so it stays bounded
        under spill and pre-reduces on each worker before a distributed shuffle.

        Args:
            subset: Columns defining the key; ``None`` deduplicates over all columns.
            keep: Which row to keep per key — ``"any"``, ``"first"``, or ``"last"``.
                ``"any"`` does not say *which* row, and the one you get may differ between
                runs and between a single-node and a distributed run; pass `order_by` with
                ``"first"``/``"last"`` when the surviving row's other columns matter.
            order_by: The order defining first/last (required for those `keep` modes).

        Returns:
            A new `Dataset` with duplicate rows removed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 1, 2, 2, 3]})
                >>> ds.distinct().sort("x").to_pydict()
                {'x': [1, 2, 3]}

                >>> events = bt.from_pydict(
                ...     {"user": ["a", "b", "a"], "ts": [3, 1, 1], "page": ["x", "y", "z"]}
                ... )
                >>> events.distinct(["user"], keep="first", order_by="ts").sort("user").to_pydict()
                {'user': ['a', 'b'], 'ts': [1, 1], 'page': ['z', 'y']}
        """
        if subset is None:
            return self._derive(Distinct(self._plan))
        return build_distinct(
            self, _as_opt_str_list(subset, self, "distinct(subset=...)"), keep, order_by
        )

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
        # A partition key nobody checked is a partition key that silently does nothing: the
        # write lays the data out by whatever `by` names, so a typo here produced one
        # unpartitioned output and no error at all.
        available = self.columns
        for key in by_cols:
            if key not in available:
                raise ColumnNotFoundError.of(key, sorted(available), where="in repartition()")
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
        label column and one Float64 column per input column. Numeric (integer, float,
        decimal) columns report count / null_count / mean / std / min / the requested
        `percentiles` (default quartiles) / max; other columns report count and
        null_count only. Composes the already-tested aggregates — no per-row work in
        Python. An input column named ``statistic`` would overwrite the labels, so it
        raises `PlanError`; `rename` it first.

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
        """A one-row dataset of each column's null (missing) value count.

        Lazy: lowers to a single global aggregate and a `select`, so it stays
        mergeable and identical single-node and distributed. It counts Arrow nulls
        only. A floating-point NaN is a value, not a null, so it is *not* counted,
        unlike pandas ``isnull().sum()``; count those with ``col(c).is_nan()``.

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

    def join_where(self, other: Dataset, *predicates: Expr, suffix: str = "_right") -> Dataset:
        """Inner-join on arbitrary predicates, such as inequalities (Polars ``join_where``).

        Keeps every pair of a left and a right row for which all `predicates` are true, the
        SQL ``JOIN ... ON a.t >= b.start AND a.t < b.end``. A predicate names left columns by
        name and right columns by name too, with `suffix` appended to a right column whose
        name a left column already has (``col("v_right")``). The output is the left columns,
        then the right ones under those names.

        One or two inequalities between a left and a right column run as a range join
        (IEJoin for two), which is output-sensitive rather than quadratic, and an equality
        runs as a hash join. Other predicates are checked on the pairs that survive. A null
        makes a predicate false, as in SQL.

        Args:
            other: The right-hand dataset.
            *predicates: Boolean expressions over both sides' columns, all of which must hold.
                A list of them is accepted too.
            suffix: Appended to right columns whose names collide with left ones.

        Returns:
            A new `Dataset` of the matching pairs.

        Raises:
            PlanError: If no predicate is given, or one is not an expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> events = bt.from_pydict({"t": [1, 5, 9]})
                >>> spans = bt.from_pydict({"lo": [0, 4], "hi": [6, 10], "span": ["a", "b"]})
                >>> events.join_where(
                ...     spans, bt.col("t") >= bt.col("lo"), bt.col("t") < bt.col("hi")
                ... ).sort("t", "span").select("t", "span").to_pydict()
                {'t': [1, 5, 5, 9], 'span': ['a', 'a', 'b', 'b']}
        """
        return build_join_where(self, other, predicates, suffix)

    def update(
        self,
        other: Dataset,
        on: str | list[str] | None = None,
        how: str = "left",
        *,
        left_on: str | list[str] | None = None,
        right_on: str | list[str] | None = None,
        include_nulls: bool = False,
    ) -> Dataset:
        """Overwrite values with `other`'s where the keys match (Polars ``update``).

        Every column the two datasets share, other than the keys, takes `other`'s value on
        a matched row. A null in `other` leaves the value alone unless ``include_nulls=True``.
        Columns only `other` has are ignored, and the result keeps this dataset's columns in
        their order.

        `how` picks the rows. ``"left"`` keeps every row of this dataset, ``"inner"`` only
        the matched ones, and ``"full"`` also adds `other`'s unmatched rows. A key is required:
        Polars pairs rows by position when it has none, and a relation has no row order to
        pair by, so number both sides with `with_row_index` where they are read and join on
        that. A null key matches nothing, and a key repeated in `other` repeats the row, as
        in any join.

        Args:
            other: The dataset supplying the new values.
            on: Key column(s) present on both sides.
            how: ``"left"``, ``"inner"`` or ``"full"``.
            left_on: This dataset's key column(s), when the names differ.
            right_on: `other`'s key column(s), when the names differ.
            include_nulls: Let a null in `other` overwrite a value.

        Returns:
            A new `Dataset` with the matched values replaced.

        Raises:
            PlanError: If `how` is unknown or no key is given.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> prices = bt.from_pydict({"id": [1, 2, 3], "price": [10, 20, 30]})
                >>> fixes = bt.from_pydict({"id": [2, 3], "price": [25, None]})
                >>> prices.update(fixes, on="id").sort("id").to_pydict()
                {'id': [1, 2, 3], 'price': [10, 25, 30]}
                >>> prices.update(fixes, on="id", include_nulls=True).sort("id").to_pydict()
                {'id': [1, 2, 3], 'price': [10, 25, None]}
        """
        return build_update(self, other, on, how, left_on, right_on, include_nulls)

    def zip(
        self,
        *others: Dataset,
        order_by: OrderSpec,
        descending: bool | Sequence[bool] = False,
    ) -> Dataset:
        """Pair rows by position, side by side, into one wider dataset (Ray Data ``zip``).

        The first row of each dataset is joined with the first row of every other, and so
        on. Position is taken under `order_by`, which every dataset must be able to
        evaluate, because a relation has no row order of its own. For data read in a known
        order, number each input where it is read with ``with_row_index("i")`` and pass
        ``order_by="i"``. Ties in `order_by` pair arbitrarily, so give it a key that
        identifies each row's position.

        A column name already taken gets the smallest free suffix of ``_1``, ``_2`` and so on,
        as Ray Data names it. The rows come out in position order.

        The datasets must have the same number of rows, which is checked by **executing a
        `count` of each eagerly**, so a mismatch fails here rather than as a short result.

        Args:
            *others: The datasets to zip onto this one, left to right.
            order_by: The ordering keys that define each row's position in every dataset.
            descending: Order every key, or each one, largest first.

        Returns:
            A new `Dataset` with this dataset's columns followed by each other's.

        Raises:
            PlanError: If no other dataset is given, the row counts differ, or `order_by`
                names no key.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> a = bt.from_pydict({"i": [0, 1, 2], "x": ["a", "b", "c"]})
                >>> b = bt.from_pydict({"i": [0, 1, 2], "x": [10, 20, 30]})
                >>> a.zip(b, order_by="i").to_pydict()
                {'i': [0, 1, 2], 'x': ['a', 'b', 'c'], 'i_1': [0, 1, 2], 'x_1': [10, 20, 30]}
        """
        return build_zip(self, others, order_by, descending)

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
        column = column_name(column, arg="column", api="explode")
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
        name = column_name(name, arg="name", api="with_row_index")
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

    def transform_with_state(
        self,
        fn: Callable[[tuple, Any, dict | None], tuple[Any, dict | None]],
        *,
        group_by: str | list[str],
        output_columns: list[str],
        state_ttl: str | None = None,
    ) -> Dataset:
        """Arbitrary keyed stateful processing over a stream (Spark ``transformWithState``).

        The escape hatch for the shapes the relational operators cannot express:
        sessionization with custom rules, a running fraud score, a per-device state
        machine, "alert when this key has been silent for ten minutes". `fn` owns one
        key's state; the engine owns when it is called, checkpointed, and expired.

        `fn(key, rows, state)` returns ``(rows_out, state_out)``, where `key` is the group
        key's values as a tuple, `rows` is that key's rows in this micro-batch as an Arrow
        `RecordBatch`, `state` is what the previous call returned for the key (None the
        first time), `rows_out` is what to emit (a `RecordBatch`, a column dict, or None),
        and `state_out` is the state to keep (None forgets the key).

        State must be a flat mapping of scalars, because the whole key space is
        checkpointed as one Arrow batch. Keep a large payload elsewhere and hold a
        reference to it.

        Args:
            fn: The per-key callback described above.
            group_by: The key column name(s) that partition the state.
            output_columns: The column names `fn` emits. Types come from what it returns.
            state_ttl: How long a key's state survives without new rows (``"10 minutes"``).
                ``None`` never expires, which is bounded only if the key space is — and is
                what the streaming state budget will eventually refuse.

        Returns:
            A new `Dataset` of whatever `fn` emitted.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> def running_total(key, rows, state):
                ...     total = (state or {"total": 0})["total"] + sum(rows.column("v").to_pylist())
                ...     return {"user": [key[0]], "total": [total]}, {"total": total}
                >>> events = bt.from_pydict({"user": ["a", "b", "a"], "v": [1, 2, 3]})
                >>> out = events.transform_with_state(
                ...     running_total,
                ...     group_by="user",
                ...     output_columns=["user", "total"],
                ...     state_ttl="1 hour",
                ... )
                >>> sorted(zip(*[out.to_pydict()[c] for c in ("user", "total")], strict=True))
                [('a', 4), ('b', 2)]
        """
        from batcher._internal.errors import PlanError
        from batcher.plan.functions.temporal import _duration_micros
        from batcher.plan.logical import TransformWithState

        keys = [group_by] if isinstance(group_by, str) else list(group_by)
        if not keys:
            raise PlanError("transform_with_state(): group_by must name at least one column")
        missing = [k for k in keys if k not in self.columns]
        if missing:
            raise PlanError(f"transform_with_state(): unknown group_by column(s) {missing}")
        if not output_columns:
            raise PlanError(
                "transform_with_state(): output_columns must name the columns fn emits — "
                "the engine cannot infer them from an opaque callback"
            )
        ttl = _duration_micros(state_ttl, arg="state_ttl") if state_ttl else 0
        node = TransformWithState(self._plan, fn, tuple(keys), tuple(output_columns), ttl)
        return Dataset(node, self._sources)

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

        Over a bounded source this composes from the window + group-by engine with no
        new operator, so it is differential-tested against DuckDB and runs single-node
        or distributed.

        Over a **stream** it has to wait, because a session's end is not knowable in
        advance: every event extends the session it lands in, and an event between two
        sessions merges them. So a session's rows are held until the watermark passes
        its last event plus `gap`, and only then aggregated and emitted — by the same
        code the bounded path runs. That bounds the state to sessions still open, and it
        means a late event cannot reopen a session already emitted; it is dropped, as it
        would be from a closed window. Use `with_watermark` to buy a straggler room.

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

        time_col = column_name(time_col, arg="time_col", api="session_window")
        return build_session_window(self, time_col, gap, partition_by or [], aggs)

    def unnest(self, *columns: str) -> Dataset:
        """Expand each struct `column` into its fields as top-level columns.

        Matches Polars ``unnest`` / Spark ``select("s.*")``. Each struct field becomes
        a column where the struct was; non-struct columns are unchanged. Raises
        `PlanError` if a column is not a struct or if an expanded field name would
        collide with an existing column.

        Args:
            *columns: The struct columns to expand. A list is accepted in place of
                separate arguments.

        Returns:
            A new `Dataset` with each struct's fields promoted to columns.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"s": [{"a": 1, "b": 2}]})
                >>> ds.unnest("s").to_pydict()
                {'a': [1], 'b': [2]}
        """
        return build_unnest(self, list(flatten_varargs(columns)))

    def sample(
        self,
        fraction: float | int | None = None,
        *,
        n: int | None = None,
        seed: int | None = None,
    ) -> Dataset:
        """Sample rows by a `fraction` (``0.0`` to ``1.0``) or a fixed count `n`.

        Deterministic and partition-independent: rows are kept by a stable seeded
        hash of their values, so the sampled set is identical single-node or
        distributed and reproducible for a given `seed`. `fraction` streams (no
        breaker, each row kept iff its hash is under `fraction`); `n` keeps exactly
        the `n` smallest-hash rows (a breaker). Pass exactly one of `fraction`/`n`.
        With `seed=None` a fresh seed is baked at plan-build.

        **Duplicate rows are sampled together, so the selection unit is the distinct
        row, not the row.** Hashing values is what buys partition-independence, and its
        price is that identical rows hash identically and are therefore all kept or all
        dropped. On a projection with few distinct values the result stops resembling a
        sample: 10,000 rows holding two distinct values return 0 rows at
        ``fraction=0.1`` and 5,000 at every fraction from ``0.25`` to ``0.9``, and
        ``n=1000`` returns a thousand copies of one value. Sample *before* narrowing to
        a low-cardinality projection, or keep a distinguishing column (a key) alongside
        the one you are sampling; both restore a row-level sample. This is a real
        limitation of the current sampler rather than a subtlety of the API.

        The positional argument reads the way both neighbouring libraries spell it:
        an `int` is a row count (``sample(100)``, as in Polars) and a `float` is a
        fraction (``sample(0.1)``).

        Args:
            fraction: A row count when an `int`, or a fraction in ``[0.0, 1.0]``
                when a `float`.
            n: An exact number of rows to keep (mutually exclusive with `fraction`).
            seed: Seeds the sampling; ``None`` bakes a fresh seed at plan-build.

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
        # A bare int positional is a row count, not a >100% fraction. bool is an int
        # subclass, so exclude it rather than reading `True` as "sample one row".
        if isinstance(fraction, int) and not isinstance(fraction, bool):
            n = _one_of(n, fraction, "n", "the positional row count")
            fraction = None
        if fraction is not None and n is not None:
            raise PlanError(
                f"sample() takes a row count or a fraction, not both; got n={n} "
                f"and fraction={fraction}"
            )
        return build_sample(self, fraction, seed, n)

    def pivot(
        self,
        *,
        index: str | list[str],
        on: str,
        values: str,
        aggregate: str = "sum",
        columns: list | None = None,
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

        Returns:
            A new `Dataset` reshaped from long to wide.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict(
                ...     {"idx": ["r", "r", "s"], "k": ["a", "b", "a"], "v": [1, 2, 3]}
                ... )
                >>> ds.pivot(index=["idx"], on="k", values="v").sort("idx").to_pydict()
                {'idx': ['r', 's'], 'a': [1, 3], 'b': [2, None]}
        """
        return build_pivot(self, _as_opt_str_list(index), on, values, aggregate, columns)

    def unpivot(
        self,
        *,
        index: str | list[str] | None = None,
        on: str | list[str] | None = None,
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
        index = _as_opt_str_list(index, self, "unpivot(index=...)")
        on = _as_opt_str_list(on, self, "unpivot(on=...)")
        return build_unpivot(self, index, on, variable_name, value_name)

    def transpose(
        self,
        *,
        column_names: str | list[str] | None = None,
        include_header: bool = False,
        header_name: str = "column",
        order_by: OrderSpec | None = None,
        descending: bool | Sequence[bool] = False,
    ) -> Dataset:
        """Turn rows into columns and columns into rows (Polars and Spark ``transpose``).

        Each input column becomes one output row, and each input row one output column. The
        output columns are named one of three ways. ``column_names="<column>"`` names them by
        that column's values, which is Spark's ``transpose(indexColumn)``; that column is
        not itself transposed, and must hold no null and no repeated value. A list names
        them explicitly, one per row, and with neither they are ``column_0``,
        ``column_1``, and so on.

        A list or positional names tie each name to a row by position, so they need
        `order_by`, because a relation has no row order of its own. Named by a column, the
        output columns follow `order_by` when it is given and ascend by name otherwise,
        which is Spark's order. ``include_header=True`` keeps a `header_name` column holding
        each input column's name. Spark's form is
        ``transpose(column_names=idx, include_header=True, header_name="key")``.

        The transposed values share one column type, so they are cast to their common
        supertype, or to ``string`` when there is none. The output is as wide as the input
        is long, so this **executes eagerly** to learn the names (a scan of the naming
        column, or a `count`). It is meant for small, summary-sized frames.

        Args:
            column_names: A column whose values name the output columns, or a list of names.
            include_header: Keep a column naming each transposed input column.
            header_name: The name of that column.
            order_by: The row order behind positional names and the output column order.
            descending: Order every `order_by` key, or each one, largest first.

        Returns:
            A new `Dataset` with one row per transposed input column.

        Raises:
            PlanError: If the input is empty, the naming column holds a null or a repeated
                value, names are positional with no `order_by`, or a list has the wrong length.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"name": ["p", "q"], "a": [1, 2], "b": [3, 4]})
                >>> ds.transpose(column_names="name", include_header=True).to_pydict()
                {'column': ['a', 'b'], 'p': [1, 3], 'q': [2, 4]}
        """
        return build_transpose(
            self, column_names, include_header, header_name, order_by, descending
        )

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

        A carrying fill with no `partition_by` is one global series, and that has **no
        distributed path**: a bucket cannot know the last non-null before it, so
        ``collect(distributed=True)`` raises rather than quietly running the whole relation
        on one node. Passing `partition_by` gives the shuffle a key and distributes it. The
        statistic strategies have no such limit -- they are ordinary aggregates.

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

    def drop_nulls(self, subset: str | list[str] | None = None, *, how: str = "any") -> Dataset:
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
        subset = _as_opt_str_list(subset, self, "drop_nulls(subset=...)")
        if subset == []:  # a selector that matched nothing: no column to test
            return self
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

    def drop_nans(self, subset: str | list[str] | None = None) -> Dataset:
        """Drop rows holding a NaN in any of `subset`'s floating-point columns.

        The NaN counterpart of `drop_nulls`, and a different test: a null is a missing value
        and a NaN is a float that is not a number, so a row with a null but no NaN survives.
        With `subset` omitted every floating-point column is checked. Lazy.

        Args:
            subset: The floating-point columns to check; ``None`` checks all of them.

        Returns:
            A new `Dataset` without the rows holding a NaN.

        Raises:
            PlanError: If a named column is unknown or not floating-point.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1.0, float("nan"), None], "s": ["a", "b", "c"]})
                >>> ds.drop_nans().to_pydict()
                {'x': [1.0, None], 's': ['a', 'c']}
        """
        return build_drop_nans(self, _as_opt_str_list(subset, self, "drop_nans(subset=...)"))

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

    def match_to_schema(
        self,
        schema: dict[str, Any] | pa.Schema,
        *,
        missing_columns: str | dict[str, str | Expr] = "raise",
        extra_columns: str = "raise",
    ) -> Dataset:
        """Conform to `schema`: its columns, in its order, with its types (Polars' spelling).

        Every schema column must already have the named type; a mismatch raises instead of
        casting, because a silent cast is how a pipeline stops failing on bad data. Cast
        first with `cast` when a conversion is intended. Narrow numeric types are widened at
        the engine boundary, so ``int32`` in `schema` matches an ``int64`` column. That is
        why Polars' ``integer_cast``/``float_cast`` upcast options have nothing to allow
        here.

        A schema column the input lacks raises by default. ``missing_columns="insert"`` adds
        it as nulls of its type, and a dict chooses per column, where an expression computes
        the column instead. An input column the schema lacks raises by default, and
        ``extra_columns="ignore"`` drops it. The check runs on the schema, before any data
        is read.

        Args:
            schema: Column name to dtype (a Batcher dtype name, Python type or pyarrow type),
                or a pyarrow schema.
            missing_columns: ``"raise"`` or ``"insert"``, or a per-column dict of either or
                of an expression computing the column.
            extra_columns: ``"raise"`` or ``"ignore"``.

        Returns:
            A new `Dataset` with exactly `schema`'s columns.

        Raises:
            PlanError: On a type mismatch, a missing or extra column the policy refuses, or
                an unknown policy.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"b": ["x"], "a": [1], "tmp": [0.5]})
                >>> ds.match_to_schema(
                ...     {"a": "int64", "b": "string", "c": "float64"},
                ...     missing_columns="insert",
                ...     extra_columns="ignore",
                ... ).to_pydict()
                {'a': [1], 'b': ['x'], 'c': [None]}
        """
        return build_match_to_schema(self, schema, missing_columns, extra_columns)

    def union(self, *others: Dataset, distinct: bool = False) -> Dataset:
        """Concatenate with other datasets (UNION ALL, or UNION if `distinct`).

        All datasets must have identical columns. Sources are merged so each
        side's scans resolve correctly.

        Args:
            *others: The datasets to concatenate; each must share this one's columns.
                A list of them is accepted in place of separate arguments.
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
        # Sources are concatenated, never merged by identity, and that is a measured trade
        # rather than an oversight. Merging them lets plan-level CSE see two branches over one
        # relation as the same computation (TPC-DS q77's plan goes from 0 repeated subtrees to
        # 75) -- but `stream::parallel::streaming_parallelizes` refuses to shard a plan whose
        # source is read more than once, so the merge *also* takes the whole query off the
        # parallel union path. Measured both ways on 2026-08-08: merging gained q80 1.6x and
        # q5 1.3x, and cost q22 2.0x, q18 2.9x and q14 2.0x -- a net loss, and a loss
        # concentrated in the queries the parallel union had just fixed. Making both work
        # wants CSE to weigh the parallelism it forfeits, which is a cost-model change.
        others = flatten_varargs(others)
        plans: list[LogicalPlan] = [self._plan]
        sources = list(self._sources)
        for other in others:
            plans.append(remap_sources(other._plan, len(sources)))
            sources.extend(other._sources)
        if self._watermark is not None or any(o._watermark is not None for o in others):
            _warn_watermark_dropped("union")
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
        whole thing is mergeable aggregation, so it distributes. Where NULLs provably
        cannot tell the two semantics apart, Kyber's `set_membership_to_join` swaps this
        shape for a semi or anti join, which costs a probe instead of an aggregate.

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
        from batcher.plan.logical._setops import (
            MEMBERSHIP_IN_LEFT,
            MEMBERSHIP_IN_RIGHT,
            MEMBERSHIP_LEFT_TAG,
            MEMBERSHIP_RIGHT_TAG,
        )

        keys = list(cols)
        left, right = self.select(*cols), other.select(*cols)
        if not distinct:
            ordinal = row_number().over(partition_by=cols, order_by=cols)
            left = left.with_columns(__bc_n__=ordinal)
            right = right.with_columns(__bc_n__=ordinal)
            keys = [*cols, "__bc_n__"]
        tag_l, tag_r = MEMBERSHIP_LEFT_TAG, MEMBERSHIP_RIGHT_TAG
        left = left.with_columns(**{tag_l: lit(True), tag_r: lit(False)})
        right = right.with_columns(**{tag_l: lit(False), tag_r: lit(True)})
        grouped = (
            left.union(right)
            .group_by(*keys)
            .agg(
                **{
                    MEMBERSHIP_IN_LEFT: col(tag_l).bool_or(),
                    MEMBERSHIP_IN_RIGHT: col(tag_r).bool_or(),
                }
            )
        )
        in_l, in_r = col(MEMBERSHIP_IN_LEFT), col(MEMBERSHIP_IN_RIGHT)
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

    def tail(self, n: int = 5) -> Dataset:
        """Keep the last `n` rows.

        Unlike `limit`, this needs to know how many rows there are, so it **executes a
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

    def split_at_indices(self, indices: list[int]) -> list[Dataset]:
        """Split into consecutive row ranges at `indices` (Ray Data ``split_at_indices``).

        ``ds.split_at_indices([2, 5])`` returns three datasets holding rows ``[0, 2)``,
        ``[2, 5)`` and ``[5, n)``. The boundaries are row *positions*, so put a `sort`
        first if they are to mean anything stable. An index past the end gives an empty
        part rather than an error, and a repeated index gives an empty part between the
        two — both matching ``numpy.split``.

        Unlike Ray Data's version this **materializes nothing**: every part is a row-index
        filter over the same plan and stays lazy until its own terminal op, so a pipeline
        that only ever consumes one part never pays for the rest. The cost is the mirror
        image, and worth knowing before collecting them all: each part reads the input
        again. Call `cache` first when the source is expensive and every part is wanted.

        Args:
            indices: Split positions, non-negative and non-decreasing.

        Returns:
            ``len(indices) + 1`` datasets, in row order.

        Raises:
            PlanError: If `indices` is empty, negative, or not non-decreasing.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> first, middle, last = bt.range(0, 10).split_at_indices([2, 5])
                >>> first.to_pydict()["value"]
                [0, 1]
                >>> middle.to_pydict()["value"]
                [2, 3, 4]
                >>> last.to_pydict()["value"]
                [5, 6, 7, 8, 9]
        """
        if not indices:
            raise PlanError(
                "split_at_indices(): indices must not be empty — one index splits into two parts"
            )
        cuts = [require_int(i, func="split_at_indices", arg="indices", minimum=0) for i in indices]
        if any(hi < lo for lo, hi in pairwise(cuts)):
            raise PlanError(
                f"split_at_indices(): indices must be non-decreasing, got {cuts} — "
                "each one starts the next part where the previous ended"
            )
        # Escaped against a real column of the same name, the way `cross_join` escapes its
        # key. `tail` and `gather_every` raise on that collision instead; it is worth avoiding
        # here because this method hands back several datasets, so the failure would surface
        # far from the call that caused it.
        idx = "__bc_split_idx"
        taken = set(self.columns)
        while idx in taken:
            idx += "_"
        parts: list[Dataset] = []
        for lo, hi in zip([0, *cuts], [*cuts, None], strict=True):
            keep = None if lo == 0 else Col(idx) >= lo
            if hi is not None:
                upper = Col(idx) < hi
                keep = upper if keep is None else keep & upper
            # The trailing part of a `[0]` split is the whole dataset, with no row to drop.
            parts.append(self if keep is None else self.with_row_index(idx).filter(keep).drop(idx))
        return parts

    def split_proportionately(self, proportions: list[float]) -> list[Dataset]:
        """Split into parts holding the given row fractions (Ray Data ``split_proportionately``).

        ``ds.split_proportionately([0.2, 0.5])`` returns three datasets holding 20%, 50%
        and the remaining 30% of the rows. The proportions name every part but the last,
        which takes the remainder — which is why they must sum to less than 1.

        Like `tail`, this needs to know how many rows there are, so it **executes a
        `count` eagerly** (often answered from metadata with no scan) before building the
        lazy plans. The parts themselves stay lazy, exactly as `split_at_indices` returns
        them.

        Every part is guaranteed at least one row: boundaries that would collide are
        nudged apart, as Ray Data does, and `PlanError` is raised when even that cannot
        give each part a row.

        This splits by *position*. For a train/test split prefer `ds.ml.train_test_split`,
        which assigns each row by a hash of its own values, so the split is reproducible
        and identical however the data is partitioned.

        Args:
            proportions: The fraction of rows in each part but the last. Each must be
                greater than 0, and together they must sum to less than 1.

        Returns:
            ``len(proportions) + 1`` datasets, in row order.

        Raises:
            PlanError: If `proportions` is empty, holds a non-positive value, sums to 1
                or more, or cannot give every part at least one row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> a, b, c = bt.range(0, 10).split_proportionately([0.2, 0.5])
                >>> [a.count(), b.count(), c.count()]
                [2, 5, 3]
        """
        if not proportions:
            raise PlanError(
                "split_proportionately(): proportions must not be empty — one proportion "
                "splits into two parts"
            )
        fractions = [
            require_float(p, func="split_proportionately", arg="proportions") for p in proportions
        ]
        if any(p <= 0 for p in fractions):
            raise PlanError(
                f"split_proportionately(): every proportion must be > 0, got {fractions}"
            )
        if sum(fractions) >= 1:
            raise PlanError(
                "split_proportionately(): proportions must sum to less than 1, because the "
                f"last part takes the remainder — got {fractions} summing to {sum(fractions)}"
            )
        total = self.count()
        cuts = [int(total * c) for c in accumulate(fractions)]
        # Walk backwards nudging colliding boundaries apart so no part comes out empty.
        # Backwards because a collision is resolved by moving the *earlier* cut down: moving
        # the later one up would push it into the part after it and cascade the other way.
        subtract = 0
        for i in range(len(cuts) - 2, -1, -1):
            cuts[i] -= subtract
            if cuts[i] == cuts[i + 1]:
                subtract += 1
                cuts[i] -= 1
        if any(c <= 0 for c in cuts):
            raise PlanError(
                f"split_proportionately(): {len(fractions) + 1} non-empty parts need at least "
                f"that many rows, and {fractions} over {total} row(s) cannot give each one — "
                "use fewer parts, or split_at_indices() to allow empty ones"
            )
        return self.split_at_indices(cuts)

    def split(
        self,
        n: int,
        *,
        order_by: OrderSpec,
        descending: bool | Sequence[bool] = False,
        equal: bool = False,
    ) -> list[Dataset]:
        """Split into `n` consecutive parts of near-equal size (Ray Data ``split``).

        Rows are numbered under `order_by` and dealt out in runs: with 10 rows and ``n=3``
        the parts hold positions ``1-4``, ``5-7`` and ``8-10``, the earlier parts taking the
        remainder one row each, as ``numpy.array_split`` does. ``equal=True`` gives every
        part exactly ``count // n`` rows and drops the remainder, which is Ray Data's
        ``equal=True``. Each part iterates in `order_by` order.

        `order_by` is required because a relation has no row order of its own: a parallel or
        distributed scan fixes none, so "the first four rows" is only defined under an
        explicit order. For data with no ordering column, number it where it is read with
        ``with_row_index("i")`` and pass ``order_by="i"``. Ties in `order_by` are broken
        arbitrarily, so give it a key that identifies each row's position.

        Like `split_proportionately`, this **executes a `count` eagerly** before building
        the parts, and each part stays lazy and re-reads the input when it runs. Unlike Ray
        Data, nothing is materialized; call `cache` first when every part will be consumed.

        Args:
            n: The number of parts, at least 1.
            order_by: The ordering keys that define each row's position.
            descending: Order every key, or each one, largest first.
            equal: Give every part the same size, dropping the remainder.

        Returns:
            `n` datasets, in position order.

        Raises:
            PlanError: If `n` is less than 1 or `order_by` names no key.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"i": list(range(10))})
                >>> [p.to_pydict()["i"] for p in ds.split(3, order_by="i")]
                [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
                >>> [p.count() for p in ds.split(3, order_by="i", equal=True)]
                [3, 3, 3]
        """
        return build_split(self, n, order_by, descending, equal)

    def partition_by(
        self, by: str | list[str], *more_by: str, include_key: bool = True
    ) -> dict[tuple, Dataset]:
        """Split into one dataset per distinct key value (Polars ``partition_by(as_dict=True)``).

        The distinct keys are found by an **eager** ``distinct`` over the key columns; each
        value of the returned dict is then a lazy filter of this dataset, so it re-reads the
        input when it runs. That makes this the right tool for a handful of keys, such as one
        output per region. For many keys, or to compute per group, `group_by` does the work
        in one pass and scales out, where this builds one query per key in the driver.

        Dict keys are always tuples, one element per key column, in ascending key order with
        nulls last. A null key is a group of its own, as in `group_by`. Polars' list form
        (``as_dict=False``) is ``list(ds.partition_by(...).values())``.

        Args:
            by: The key column, or a list of them.
            *more_by: Further key columns.
            include_key: Keep the key columns in each part.

        Returns:
            Key tuple to the dataset of that key's rows.

        Raises:
            PlanError: If a key is not a column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"k": ["a", "b", "a"], "v": [1, 2, 3]})
                >>> parts = ds.partition_by("k", include_key=False)
                >>> {key: part.to_pydict() for key, part in parts.items()}
                {('a',): {'v': [1, 3]}, ('b',): {'v': [2]}}
        """
        keys = [by] if isinstance(by, str) else list(by)
        return build_partition_by(self, [*keys, *more_by], include_key)

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
        if left.num_rows != right.num_rows:
            return False
        # Two multisets are equal iff sorting both by every column yields identical
        # relations, and Arrow sorts in compiled code over its own buffers. The row-wise
        # spelling this replaces (`sorted(map(repr, table.to_pylist()))`) built a Python
        # dict and a string per row on each side — a per-row touch in the control plane,
        # and 33x slower on a million rows.
        if _multiset_sortable(left.schema):
            keys = [(name, "ascending") for name in left.schema.names]
            return left.sort_by(keys).equals(right.sort_by(keys))
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

    # --- pandas-compatible spellings ------------------------------------------------
    # A data scientist arriving from pandas finds the operation under the name they
    # already type. Each delegates to the Batcher primary — same plan, same semantics.

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

    def round(self, decimals: int = 0) -> Dataset:
        """Round every numeric column to `decimals` places — the pandas ``round`` spelling.

        Non-numeric columns pass through untouched (the numeric selector picks the
        columns).

        A tie rounds **away from zero**, which is SQL's rule and the rule
        :meth:`~batcher.plan.expr_ir.core.Expr.round` follows, not NumPy's round-half-to-even
        that pandas inherits. So ``0.5`` becomes ``1.0`` here and ``0.0`` in pandas. The name
        is borrowed; the tie-breaking is the engine's, and it is the same in SQL and in the
        DataFrame API so one query cannot disagree with itself.

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
        return self.agg(**{name: Col(name).count_distinct() for name in self.columns})

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
        length = Col(column).str.len_chars()
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

        # Two sources rarely agree on a key's exact type — `decimal(10,2)` against
        # `decimal(12,4)`, `timestamp[ms]` against `timestamp[us]` — while the row encoder
        # the join builds needs them identical. Widen both sides to the pair's common
        # supertype first, so the join runs on the same pairs a union would reconcile.
        # Types with no common supertype are left alone and `Join` rejects them.
        left_plan, right_plan = align_join_key_types(
            self._plan, right_plan, tuple(left_keys), tuple(right_keys)
        )

        if self._watermark is not None or other._watermark is not None:
            _warn_watermark_dropped("join")
        node = Join(left_plan, right_plan, tuple(left_keys), tuple(right_keys), how, tuple(output))
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
        how: str = "inner",
    ) -> Dataset:
        """Watermark-bounded stream-stream interval join (Spark stream-stream join).

        Joins two streams on equality keys (`on` / `left_on`+`right_on`) **and** an
        event-time interval — a row pair matches only if
        ``|left_time - right_time| <= within``. That time bound is what lets buffered
        state be evicted once the watermark passes, keeping memory bounded over two
        unbounded streams. Over bounded sources it is a plain join plus the interval
        filter. Consume the streaming result with `iter_batches()`.

        An outer `how` emits a row that never matched, padded with nulls, at the moment
        the watermark guarantees no partner can still arrive — which is the only moment
        such a statement is decidable about an unbounded stream, and why the interval is
        required rather than optional.

        Args:
            other: The right-hand stream.
            on: Shared equality key column name(s).
            left_on: The left equality key(s), when the names differ.
            right_on: The right equality key(s), when the names differ.
            left_time: The left event-time column.
            right_time: The right event-time column.
            within: The maximum time difference for a pair to match (e.g. ``"1h"``).
            lateness: Extra grace before evicting buffered state; ``None`` for none.
            how: ``"inner"`` (default), ``"left"``, ``"right"``, or ``"full"``.

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
        from batcher._internal.errors import PlanError
        from batcher.io.source import is_bounded
        from batcher.plan.functions.temporal import _duration_micros
        from batcher.plan.logical import WatermarkStreamJoin

        if how not in ("inner", "left", "right", "full"):
            raise PlanError(
                f"join_stream(): unknown how={how!r}; use 'inner', 'left', 'right', or 'full'"
            )
        left_keys, right_keys = _resolve_join_keys(on, left_on, right_on)
        within_us = _duration_micros(within, arg="join within")
        lateness_us = _duration_micros(lateness, arg="join lateness") if lateness else 0
        offset = len(self._sources)
        combined = self._sources + other._sources

        if all(is_bounded(s) for s in combined):
            from batcher.api.dataset._build import _bounded_interval_join

            return _bounded_interval_join(
                self, other, left_keys, right_keys, left_time, right_time, within_us, how
            )

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
            how,
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
        tolerance: int | float | str | timedelta | None = None,
        allow_exact_matches: bool = True,
        suffix: str = "_right",
    ) -> Dataset:
        """ASOF (nearest-match) join — match each left row to the nearest right row.

        The match is on the right row whose `on` key is nearest (`direction`:
        ``"backward"`` ≤, ``"forward"`` ≥, ``"nearest"`` either way), within the same
        `by` group (exact). Left-style: every left row is kept (null right columns when
        unmatched). Both sides should be sorted on `on` within `by` for the intended
        semantics. Specify keys via `on`/`by` (shared) or `*_on`/`*_by`.

        Pass `allow_exact_matches=False` for the strict form: a right row stamped at
        exactly the left row's instant is then ignored. In a backtest that row is
        information the trade did not have, and matching it is look-ahead bias that
        inflates every result downstream without ever looking like a bug.

        Pass `tolerance` to cap how stale a match may be. Without it a trade at noon
        matches a quote from three days earlier without complaint, because that quote
        really is the nearest one preceding it; with it, the left row is left unmatched
        instead. Give a duration (``"5m"``, or a `datetime.timedelta`) for a timestamp or
        date key, and a plain number for a numeric key.

        Args:
            other: The right-hand dataset to match against.
            on: The shared nearest-match key column.
            left_on: The left match key, when the names differ.
            right_on: The right match key, when the names differ.
            by: Shared exact-match grouping column(s).
            left_by: The left grouping column(s), when the names differ.
            right_by: The right grouping column(s), when the names differ.
            direction: ``"backward"`` (≤), ``"forward"`` (≥), or ``"nearest"`` (the
                closer of the two, backward on an exact tie).
            tolerance: Largest allowed distance between the matched keys. A number is in
                the key's own units; a duration string or `datetime.timedelta` is for a
                temporal key. ``None`` (the default) never rejects a match.
            allow_exact_matches: Whether a right row whose key *equals* the left row's may
                be the match. Set it false for the strict form, where a backward join takes
                the last row strictly before the left key.
            suffix: Suffix appended to right columns whose names collide.

        Returns:
            A new `Dataset` with each left row matched to its nearest right row.

        Raises:
            PlanError: If no match key is given, `direction` is not one of the three, or
                `tolerance` is negative or unparseable.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> left = bt.from_pydict({"t": [1, 5, 10], "v": ["a", "b", "c"]})
                >>> right = bt.from_pydict({"t": [2, 6], "w": ["x", "y"]})
                >>> left.join_asof(right, on="t").to_pydict()
                {'t': [1, 5, 10], 'v': ['a', 'b', 'c'], 'w': [None, 'x', 'y']}

                >>> # Row `c` at t=10 is 4 away from the nearest quote, so a tolerance of
                >>> # 3 leaves it unmatched rather than carrying a stale value.
                >>> left.join_asof(right, on="t", tolerance=3).to_pydict()
                {'t': [1, 5, 10], 'v': ['a', 'b', 'c'], 'w': [None, 'x', None]}

                >>> # "nearest" may look forward: `a` at t=1 takes the quote at t=2, and
                >>> # `b` at t=5 takes the one at t=6 rather than the one at t=2.
                >>> left.join_asof(right, on="t", direction="nearest").to_pydict()
                {'t': [1, 5, 10], 'v': ['a', 'b', 'c'], 'w': ['x', 'y', 'y']}

                >>> # Strict matching ignores a right row stamped at the same instant —
                >>> # the difference between a backtest and look-ahead bias.
                >>> same = bt.from_pydict({"t": [5], "w": ["same"]})
                >>> left.join_asof(same, on="t").to_pydict()["w"]
                [None, 'same', 'same']
                >>> left.join_asof(same, on="t", allow_exact_matches=False).to_pydict()["w"]
                [None, None, 'same']
        """
        l_on, r_on = left_on or on, right_on or on
        if l_on is None or r_on is None:
            raise PlanError("join_asof() requires `on` (or both left_on and right_on)")
        l_by = _as_str_list(left_by if left_by is not None else by)
        r_by = _as_str_list(right_by if right_by is not None else by)
        output = _asof_output(self.columns, other.columns, r_on, r_by, suffix)
        right_plan = remap_sources(other._plan, len(self._sources))
        node = AsofJoin(
            self._plan,
            right_plan,
            l_on,
            r_on,
            tuple(l_by),
            tuple(r_by),
            direction,
            tuple(output),
            asof_tolerance(tolerance),
            bool(allow_exact_matches),
        )
        return Dataset(node, self._sources + other._sources)

    def lookup_join(
        self,
        source: str,
        *,
        on: str,
        schema: dict[str, str] | None = None,
        how: str = "left",
        prefix: str = "",
        cache_size: int = 100_000,
        cache_ttl: str | None = None,
        hash_values: bool = False,
        batch_size: int | None = None,
        num_workers: int | str = "auto",
    ) -> Dataset:
        """Enrich each row from a key-value store, by point lookup rather than by scan.

        The join to reach for when the dimension is far larger than what the data actually
        touches: a hundred-million-row customer store against a stream that sees ten
        thousand of them. A broadcast join has to move the whole store and a shuffle join
        has to sort it, where this asks only for the distinct keys each batch contains.
        It is Flink's lookup join, and it works unchanged single-node, distributed, and
        over an unbounded source, because the enrichment is per batch.

        Repeated keys are what make it fast, and they are the norm on a fact stream. Each
        worker keeps an LRU of what it has looked up, and — the part that matters on a
        dirty key column — remembers **absences** too, so a key the store does not hold is
        fetched once rather than once per batch.

        What it gives up is a consistent snapshot: the store is read as it stands when each
        batch arrives, and `cache_ttl` bounds how stale a cached row may be. Where a
        point-in-time answer is what you meant, read the dimension as a dataset and use
        :meth:`join`.

        Args:
            source: The store URI. ``redis://``, ``rediss://``, or ``unix://`` for Redis;
                ``rocksdb://<path>`` or a bare path for an embedded RocksDB database.
            on: The column to look up by. Its values are cast to strings, so any key type
                joins against a string keyspace.
            schema: The columns the lookup contributes, as ``{name: dtype}`` using the
                dtype names :meth:`cast` accepts. Required, because a join's output shape
                cannot depend on which keys the first batch happened to contain.
            how: ``"left"`` keeps every row and null-fills the misses; ``"inner"`` drops
                them.
            prefix: Prepended to every looked-up column name, for a dimension whose column
                names collide with this dataset's.
            cache_size: Entries each worker's lookup cache holds, hits and absences
                together. ``0`` disables it, which is how you measure what it is buying.
            cache_ttl: How long a cached entry stays usable (``"30s"``, ``"5m"``). ``None``
                keeps entries for the life of the worker, which is the right setting for a
                dimension that does not change during the run.
            hash_values: Read each Redis key as a hash whose fields are the columns, rather
                than as a string holding a JSON object. Match how the dimension was written.
            batch_size: Rows per lookup batch; ``None`` uses the engine default, which is
                the right choice unless you have measured otherwise. Larger batches mean
                fewer, bigger round trips **and** cheaper assembly: the per-batch cost is
                one unit of work per *distinct key in the batch*, so a batch smaller than
                the distinct-key count pays for the same keys over and over. Setting this
                to a small value is the one way to make a lookup join slow.
            num_workers: How many workers issue lookups concurrently. ``"auto"`` fans
                across local cores, which is what hides the store's latency. The cache is
                per worker, so this multiplies the round trips for the same distinct keys
                — the right trade against a store you are waiting on, the wrong one
                against a store you are close to rate-limiting.

        Returns:
            A new `Dataset` with the looked-up columns appended.

        Raises:
            PlanError: If `on` is not a column of this dataset, if `schema` is missing or
                names an unknown dtype, or if `how` is neither ``"left"`` nor ``"inner"``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> orders = bt.from_pydict({"customer": ["c1", "c9"], "total": [10, 20]})
                >>> enriched = orders.lookup_join(  # doctest: +SKIP
                ...     "redis://localhost:6379/0",
                ...     on="customer",
                ...     schema={"name": "string", "tier": "int64"},
                ...     prefix="cust_",
                ... )
                >>> orders.columns
                ['customer', 'total']
        """
        from batcher.io.lookup import LookupStage, lookup_schema

        if on not in self._plan.available_columns():
            raise PlanError(
                f"lookup_join(): unknown column {on!r}",
                available=self._plan.available_columns(),
            )
        if how not in ("left", "inner"):
            raise PlanError(
                f"lookup_join(how={how!r}) is not supported",
                hint=(
                    "A point-lookup store can answer 'left' (keep every row, null-fill "
                    "the misses) or 'inner' (drop the misses). A right or outer join "
                    "would have to enumerate the store, which is the scan a lookup join "
                    "exists to avoid."
                ),
            )
        # Resolved here to validate the dtype names and to name the output columns; the
        # *unresolved* mapping is what travels to the worker below. Round-tripping through
        # `str(field.type)` looked equivalent and is not: `resolve_dtype("timestamp(us)")`
        # renders as ``timestamp[us]``, which `resolve_dtype` does not parse back, so a
        # temporal lookup column resolved on the driver and then failed on the worker.
        resolved = lookup_schema(schema)
        added = [prefix + field.name for field in resolved]
        collision = sorted(set(added) & set(self._plan.available_columns()))
        if collision:
            raise PlanError(
                f"lookup_join(): the looked-up column(s) {', '.join(collision)} already "
                "exist in this dataset",
                hint="Pass prefix= to disambiguate them.",
            )
        return self.map_batches(
            LookupStage,
            batch_size=batch_size,
            num_workers=num_workers,
            output_columns=list(self._plan.available_columns()) + added,
            preserves_columns=list(self._plan.available_columns()),
            fn_constructor_kwargs={
                "source": source,
                "on": on,
                "schema": dict(schema or {}),
                "how": how,
                "prefix": prefix,
                "cache_size": cache_size,
                "cache_ttl": cache_ttl,
                "hash_values": hash_values,
            },
        )

    def group_by(self, *keys: str, maintain_order: bool = False, **named: Expr) -> GroupBy:
        """Begin a grouped aggregation over the given keys.

        Positional args are key columns by name; keyword args bind a derived key
        column to an expression (e.g. ``group_by("dept", decade=col("year") // 10)``).
        Follow with ``.agg(name=expr)``:
        ``ds.group_by("dept").agg(total=col("salary").sum(), n=count())``.
        Global aggregation (no keys) is ``ds.group_by().agg(...)``.

        Groups come out in no defined order unless `maintain_order` is set. With it, they come
        out in the order each group's first row appears in the input, identically under
        ``collect``, spilling, ``iter_batches`` and ``distributed=True``, because the order is
        computed rather than observed: each input row is numbered, each group keeps its
        smallest number, and the result is sorted on it. That costs a sort over the groups.
        A derived key cannot be named ``maintain_order``.

        Args:
            *keys: Key columns by name. A list is accepted in place of separate
                arguments.
            maintain_order: Emit groups in the order of their first appearance in the input.
            **named: Derived key columns bound to expressions.

        Returns:
            A `GroupBy` to finish with ``.agg(...)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"g": ["a", "b", "a"], "v": [1, 2, 3]})
                >>> ds.group_by("g").agg(s=bt.col("v").sum()).sort("g").to_pydict()
                {'g': ['a', 'b'], 's': [4, 2]}

                >>> ds = bt.from_pydict({"g": ["b", "a", "b"], "v": [1, 2, 3]})
                >>> ds.group_by("g", maintain_order=True).agg(s=bt.col("v").sum()).to_pydict()
                {'g': ['b', 'a'], 's': [4, 2]}
        """
        if not isinstance(maintain_order, bool):
            raise PlanError(f"group_by(maintain_order=...) must be a bool, got {maintain_order!r}")
        keys = _as_opt_str_list(list(flatten_varargs(keys)), self, "group_by()")
        available = set(self._plan.available_columns())
        for k in keys:
            if not isinstance(k, str):
                raise PlanError(
                    "positional group_by() keys must be column names; give a derived "
                    "key a name, e.g. group_by(bucket=col('x') % 10)"
                )
            if k not in available:
                raise ColumnNotFoundError.of(k, sorted(available), where="in group_by()")
        for alias, expr in named.items():
            if not isinstance(expr, Expr):
                raise PlanError(f"group_by() value for {alias!r} must be an expression")
            _reject_sliding_window_key(alias, expr)
        return GroupBy(self, keys, named, maintain_order=maintain_order)

    def rollup(self, *keys: str) -> MultiLevelGroupBy:
        """Aggregate at every prefix of `keys`, plus the grand total (SQL ``ROLLUP``).

        The subtotal report: ``ds.rollup("region", "city").agg(total=col("v").sum())``
        returns a row per (region, city), a row per region with `city` null, and one
        grand-total row with both null. An inactive key reads as NULL, which is how SQL
        marks a subtotal row.

        Args:
            *keys: The rollup key columns, most significant first. A list is accepted
                in place of separate arguments.

        Returns:
            A `MultiLevelGroupBy` to finish with ``.agg(...)``.

        Raises:
            PlanError: If a key is not a column of this dataset.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"r": ["e", "e", "w"], "v": [1, 2, 4]})
                >>> ds.rollup("r").agg(n=bt.col("v").sum()).sort("r").to_pydict()
                {'r': ['e', 'w', None], 'n': [3, 4, 7]}
        """
        keys = flatten_varargs(keys)
        self._check_group_keys(keys, "rollup")
        return MultiLevelGroupBy(self, keys, rollup_levels(keys))

    def cube(self, *keys: str) -> MultiLevelGroupBy:
        """Aggregate at every *subset* of `keys` (SQL ``CUBE``).

        The cross-tabulation: every combination of the keys, from all of them down to
        the grand total, so a two-key cube gives per-pair, per-first, per-second and
        overall rows. Costs 2ⁿ levels, so keep `n` small.

        Args:
            *keys: The cube key columns. A list is accepted in place of separate
                arguments.

        Returns:
            A `MultiLevelGroupBy` to finish with ``.agg(...)``.

        Raises:
            PlanError: If a key is not a column of this dataset.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": ["x"], "b": ["y"], "v": [2]})
                >>> len(ds.cube("a", "b").agg(n=bt.col("v").sum()).to_pydict()["n"])
                4
        """
        keys = flatten_varargs(keys)
        self._check_group_keys(keys, "cube")
        return MultiLevelGroupBy(self, keys, cube_levels(keys))

    def grouping_sets(self, *sets: Sequence[str]) -> MultiLevelGroupBy:
        """Aggregate at exactly the grouping levels given (SQL ``GROUPING SETS``).

        The explicit form the other two are shorthands for: each argument is one level's
        key list, and ``()`` is the grand total. Use it when the levels you want are not
        a prefix chain or a full cube.

        Args:
            *sets: One key-name sequence per grouping level.

        Returns:
            A `MultiLevelGroupBy` to finish with ``.agg(...)``.

        Raises:
            PlanError: If a key is not a column of this dataset.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"a": ["x", "x"], "b": ["y", "z"], "v": [1, 2]})
                >>> out = ds.grouping_sets(["a"], ["b"], []).agg(n=bt.col("v").sum())
                >>> sorted(out.to_pydict()["n"])
                [1, 2, 3, 3]
        """
        levels = [tuple(level) for level in sets]
        keys: list[str] = []
        for level in levels:
            keys.extend(k for k in level if k not in keys)
        self._check_group_keys(tuple(keys), "grouping_sets")
        return MultiLevelGroupBy(self, tuple(keys), levels)

    def _check_group_keys(self, keys: tuple[str, ...], what: str) -> None:
        """Reject a non-column key with the same message `group_by` uses."""
        available = set(self._plan.available_columns())
        for k in keys:
            if not isinstance(k, str) or k not in available:
                raise ColumnNotFoundError.of(k, sorted(available), where=f"in {what}()")

    def agg(self, *aggs: Expr, **aggregates: Expr) -> Dataset:
        """Aggregate over the whole dataset (no grouping).

        Shorthand for ``group_by().agg(...)``: ``ds.agg(total=col("x").sum())`` returns
        a single-row dataset. Positional args are self-naming aggregates —
        ``ds.agg(bt.sum("x"), bt.mean("y"))`` keeps each source column's name.

        Args:
            *aggs: Self-naming aggregate expressions (e.g. ``bt.sum("x")``), or ones
                named by ``.alias(...)``. A list is accepted in place of separate
                arguments.
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
        return self.group_by().agg(*flatten_varargs(aggs), **aggregates)

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
        ``format="json"`` returns the same profile as a machine-readable JSON *string*;
        parse it with ``json.loads`` to get a dict.

        Args:
            analyze: Execute the query and include measured per-operator metrics.
            format: ``"text"`` (or its alias ``"tree"``, as Polars and Spark spell it)
                for the rendered tree, ``"json"`` for the profile as a JSON string.

        Returns:
            The plan (and, when ``analyze``, the measured profile) as a text tree or a
            JSON string.

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
        """Execute the query and return its measured per-operator `RunStats`.

        Where `explain()` shows the *planned* shape with estimates, `stats()` runs
        the query and reports what the engine *measured* — rows in/out, wall time,
        peak bytes, spill, and backend per operator, plus a bottleneck call (the
        answer to "where is my time going"). It runs through the path `collect()`
        would take (single-node, spilling, or distributed under ``"auto"``), and a
        `map_batches`/ML pipeline is measured per stage rather than refused.

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
        return _count(self._plan, self._sources, self.columns, self._cache)

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
        return _is_empty(self._plan, self._sources, self.columns, self._cache)

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

    def _require_column(self, column: str, op: str) -> str:
        """Validate that `column` is an output column *name*, else raise `PlanError`.

        The type check is not redundant with the membership test below it — it is what
        makes the membership test reachable. ``column not in available`` evaluates
        ``Expr.__eq__`` against each name when handed an expression, which builds an
        expression and then asks it for a truth value, so every scalar terminal answered
        ``ds.sum(col("v"))`` with *the truth value of an Expr is ambiguous; use & | ~ to
        combine predicates* — a message about boolean operators, naming neither the method
        nor the argument, for a call that used none.
        """
        column = column_name(column, arg="column", api=op)
        available = self._plan.available_columns()
        if column not in available:
            raise PlanError(f"{op}(): unknown column {_unknown_cols({column}, available)}")
        return column

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

    def count_distinct(self, column: str) -> int:
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
                >>> bt.from_pydict({"x": [1, 1, 2, 3, 3]}).count_distinct("x")
                3
        """
        self._require_column(column, "count_distinct")
        from batcher.api.terminal.metadata_answer import metadata_n_unique

        answer = metadata_n_unique(self._plan, self._sources, column)
        return (
            answer if answer is not None else int(self._exec_scalar(Col(column).count_distinct()))
        )

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

    def product(self, column: str) -> Any:
        """The product of `column` (SQL ``PRODUCT``), ignoring nulls.

        A scalar terminal; runs one aggregate pass. An empty or all-null column yields
        ``None`` (matching `sum`), not ``1``. Reach for it for compounded growth — a
        chain of period returns multiplies rather than adds.

        Args:
            column: The column to reduce.

        Returns:
            The product, or ``None`` for an empty/all-null column.


        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0]}).product("x")
                24.0
        """
        self._require_column(column, "product")
        return self._exec_scalar(Col(column).product())

    def mode(self, column: str) -> Any:
        """The most frequent value in `column`, ignoring nulls.

        A scalar terminal; runs one aggregate pass. A tie is broken by the engine's
        grouping order, so treat the answer as *a* mode rather than *the* mode when the
        column has several.

        Args:
            column: The column to reduce.

        Returns:
            The most frequent value, or ``None`` for an empty/all-null column.


        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1, 2, 2, 3]}).mode("x")
                2
        """
        self._require_column(column, "mode")
        return self._exec_scalar(Col(column).mode())

    def skew(self, column: str) -> Any:
        """The sample skewness of `column` — how lopsided its distribution is.

        A scalar terminal; runs one aggregate pass. Positive means a long right tail,
        negative a long left one, and zero a symmetric distribution. Fewer than three
        non-null values leaves it undefined.

        Args:
            column: The column to reduce.

        Returns:
            The sample skewness, or ``None`` when undefined.


        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1.0, 2.0, 2.0, 3.0, 10.0]}).skew("x")
                2.0286991020803327
        """
        self._require_column(column, "skew")
        return self._exec_scalar(Col(column).skew())

    def kurtosis(self, column: str) -> Any:
        """The sample excess kurtosis of `column` — how heavy its tails are.

        A scalar terminal; runs one aggregate pass. Zero is the normal distribution's
        tail weight, positive means heavier tails (more outliers), negative lighter.
        Fewer than four non-null values leaves it undefined.

        Args:
            column: The column to reduce.

        Returns:
            The sample excess kurtosis, or ``None`` when undefined.


        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1.0, 2.0, 2.0, 3.0, 10.0]}).kurtosis("x")
                4.272146531742893
        """
        self._require_column(column, "kurtosis")
        return self._exec_scalar(Col(column).kurtosis())

    def mad(self, column: str) -> Any:
        """The mean absolute deviation of `column` from its mean, ignoring nulls.

        A scalar terminal; runs one aggregate pass. Unlike the standard deviation it
        does not square the deviations, so one far-out value moves it far less — which
        is why it is the spread to reach for when outliers are expected rather than
        exceptional.

        Args:
            column: The column to reduce.

        Returns:
            The mean absolute deviation, or ``None`` for an empty/all-null column.


        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"x": [1.0, 2.0, 2.0, 3.0]}).mad("x")
                0.5
        """
        self._require_column(column, "mad")
        return self._exec_scalar(Col(column).mad())

    def any(self, column: str) -> Any:
        """Whether any value in a boolean `column` is true (SQL ``BOOL_OR``), ignoring nulls.

        A scalar terminal; runs one aggregate pass. An empty or all-null column yields
        ``None`` rather than ``False``, because "no rows" is not evidence of absence.

        Args:
            column: The column to reduce.

        Returns:
            ``True`` if any value is true, ``False`` if none is, ``None`` when empty.


        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"flag": [False, True, False]}).any("flag")
                True
        """
        self._require_column(column, "any")
        return self._exec_scalar(Col(column).bool_or())

    def all(self, column: str) -> Any:
        """Whether every value in a boolean `column` is true (SQL ``BOOL_AND``), ignoring nulls.

        A scalar terminal; runs one aggregate pass. An empty or all-null column yields
        ``None`` rather than ``True``, so a vacuous truth never passes for a checked one.

        Args:
            column: The column to reduce.

        Returns:
            ``True`` if every value is true, ``False`` otherwise, ``None`` when empty.


        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"flag": [True, True, False]}).all("flag")
                False
        """
        self._require_column(column, "all")
        return self._exec_scalar(Col(column).bool_and())

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

    def corr_matrix(self, columns: str | list[str] | None = None) -> Dataset:
        """The pairwise Pearson correlation matrix over numeric columns.

        **Executes** and returns a small `Dataset`: a ``column`` label column plus one
        Float64 column per correlated column, forming a symmetric matrix (diagonal exactly
        ``1.0``, or ``None`` for a constant column). Every pair is computed in a **single**
        pass — not ``N**2`` separate scans — the standard first step of exploratory data
        analysis and feature selection. Numeric means integer, float, or decimal; other
        columns are skipped unless named explicitly (which errors). Naming a column twice,
        or correlating a column named ``column`` (the label), raises `PlanError`.

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

        return corr_matrix(self, _as_opt_str_list(columns))

    def cov_matrix(self, columns: str | list[str] | None = None) -> Dataset:
        """The pairwise sample covariance matrix over numeric columns.

        The covariance companion to `corr_matrix`: **executes** and returns a small
        symmetric `Dataset` (a ``column`` label plus one Float64 column per column), every
        pair computed in a **single** pass. The diagonal holds each column's variance. The
        input to PCA / whitening and multivariate-Gaussian modeling. Column selection and
        the ``column``-label collision rule are those of `corr_matrix`.

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

        return cov_matrix(self, _as_opt_str_list(columns))

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

    def approx_count_distinct(self, column: str) -> int | None:
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
                >>> ds.approx_count_distinct("x") is not None
                True
        """
        self._require_column(column, "approx_count_distinct")
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
        than the exact sort `quantile` would need. Returns ``None`` for an empty column
        or a non-numeric one (anything but integer, float, or decimal, so a timestamp,
        date, or boolean column too), decided from the schema before anything runs. Use
        the exact aggregate when precision matters.

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
        if not self.meta.schema.is_numeric(column):
            return None
        from batcher.api.terminal.metadata_answer import metadata_learned_quantile

        learned = metadata_learned_quantile(self._plan, column, q, self._sources)
        if learned is not None:
            return learned
        from batcher.api.orchestration import approx_quantile

        # Stream just the target column (projected, so only it crosses the boundary)
        # through the mergeable TDigest — the driver never holds the whole column. Routed
        # like every other terminal (`"auto"`, so `distributed.mode` reaches it); the
        # `iter_batches` default alone is single-node, which pinned this to the driver.
        batches = self.select(column).iter_batches(distributed="auto")
        return approx_quantile(batches, column, q)

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
        batch_format: str = "pyarrow",
        drop_last: bool = False,
        local_shuffle_buffer_size: int | None = None,
        local_shuffle_seed: int | None = None,
        prefetch_batches: int = 0,
        distributed: bool | str = False,
        num_workers: int | None = None,
        transport: str = "auto",
    ) -> Iterator[Any]:
        """Execute and yield the result batch by batch.

        The execution mode is automatic: a breaker-free pipeline (filter / project /
        map_batches over a single source) — and top-level aggregate / distinct /
        top-N over such an input — is consumed one source batch at a time in bounded
        memory, so a larger-than-memory or unbounded source streams incrementally.
        A top-level sort, join, or window over bounded sources streams too, from the
        out-of-core bucket pipeline: the input is consumed to disk, then the result is
        yielded one bounded bucket at a time. Anything else materializes first; if the
        source is unbounded and the plan cannot stream, a `PlanError` is raised
        rather than hanging. `batch_size` rebatches the output so every batch but the last
        holds exactly that many rows.

        `batch_format` converts each batch as it is yielded, with the same conversions
        `map_batches` uses. `drop_last` drops a final batch shorter than `batch_size`.
        `local_shuffle_buffer_size` shuffles rows within blocks of that many rows before
        batching, a streaming approximation of a global shuffle seeded by
        `local_shuffle_seed`. `prefetch_batches` prepares that many batches ahead on a
        background thread.

        With `distributed` (``True`` or ``"auto"`` on a multi-node cluster), a
        top-level breaker fans out across Ray workers and its result streams back one
        reducer bucket at a time, so the driver never holds the whole distributed
        result — the bounded-memory way to pull a large distributed output.

        On a dataset marked with :meth:`cache`, an already-cached result is streamed
        straight from the cache. Streaming does not *populate* it: filling the cache means
        materializing the whole result, which is the one thing a caller reaching for
        `iter_batches` has asked not to happen. Call a materializing terminal once to warm
        the cache, and every later stream is served from it.

        Args:
            batch_size: Rebatch the output to this many rows; ``None`` keeps engine batches.
            batch_format: The yielded batch type — ``"pyarrow"`` (a `RecordBatch`),
                ``"numpy"``, ``"pandas"``, ``"torch"``, ``"polars"`` or ``"jax"``.
            drop_last: Drop a final batch with fewer than `batch_size` rows. Requires
                `batch_size`.
            local_shuffle_buffer_size: Shuffle within blocks of this many rows; ``None``
                keeps the engine's order.
            local_shuffle_seed: Seed for the local shuffle; ``None`` uses seed 0, so a run
                is reproducible.
            prefetch_batches: Batches to prepare ahead on a background thread; 0 disables.
            distributed: Fan a top-level breaker across Ray workers (``True``/``"auto"``).
            num_workers: Worker fan-out for the distributed path.
            transport: The shuffle transport; ``"auto"`` selects one.

        Yields:
            The result batches, in `batch_format`.

        Raises:
            PlanError: If an option is invalid, or `drop_last` is set without `batch_size`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> sum(batch.num_rows for batch in ds.iter_batches())
                3
                >>> [len(b["x"]) for b in ds.iter_batches(2, batch_format="numpy", drop_last=True)]
                [2]
        """
        from batcher.api.dataset._export import shape_batches
        from batcher.api.terminal.core import _resolve_distributed, cached_batches
        from batcher.api.terminal.event_log import pipeline_signature, report_stream

        shape = shape_batches(
            batch_size=batch_size,
            batch_format=batch_format,
            drop_last=drop_last,
            local_shuffle_buffer_size=local_shuffle_buffer_size,
            local_shuffle_seed=local_shuffle_seed,
            prefetch_batches=prefetch_batches,
        )
        batches = cached_batches(self._plan, self._sources, self._cache, batch_size)
        if batches is None:
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
        yield from shape(
            report_stream(
                batches,
                label=type(self._plan).__name__.lower(),
                signature=pipeline_signature(self._plan),
            )
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
        return _collect(self._plan, self._sources, self.columns, cache=self._cache)

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
        return _to_pandas(self._plan, self._sources, self.columns, self._cache)

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
        return _to_polars(self._plan, self._sources, self.columns, self._cache)

    def to_numpy(self, columns: str | list[str] | None = None) -> dict[str, Any]:
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

        cols = _require_columns(self.columns, _as_opt_str_list(columns), where="in to_numpy()")
        return to_numpy(self, cols)

    def to_jax(self, columns: str | list[str] | None = None) -> dict[str, Any]:
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

        cols = _require_columns(self.columns, _as_opt_str_list(columns), where="in to_jax()")
        return to_jax(self, cols)

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
        return _to_pydict(self._plan, self._sources, self.columns, self._cache)

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
        return _to_pylist(self._plan, self._sources, self.columns, self._cache)

    def to_ray_dataset(
        self,
        *,
        batch_size: int | None = None,
        block_size_bytes: int | None = None,
        distributed: bool | str = False,
    ) -> Any:
        """Execute and hand the result to Ray Data as a ``ray.data.Dataset`` (needs `ray`).

        The return leg of :func:`batcher.from_ray_dataset`, so a Batcher query can feed a
        Ray Train, Tune, or Serve stage without staging the result through storage. Output
        batches are coalesced into blocks sized near Ray Data's own
        ``target_max_block_size`` and put into the object store one block at a time, so the
        driver holds one block rather than the whole result.

        The blocks are produced on the driver, which is the right shape for a result that
        has already been reduced (a model's training set, a scored table). For a result the
        size of the input, write Parquet and hand Ray the path instead: that keeps the data
        on the workers that produced it.

        Args:
            batch_size: Rows per engine batch before coalescing; ``None`` keeps engine batches.
            block_size_bytes: Target bytes per Ray block; ``None`` reads Ray Data's own target.
                Keep an override inside Ray Data's usable band of 1 MiB to 128 MiB.
            distributed: Fan a top-level breaker across Ray workers (``True``/``"auto"``).

        Returns:
            A ``ray.data.Dataset`` over the result's Arrow blocks.

        Raises:
            BackendError: If ``ray`` is not installed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.to_ray_dataset().count()  # doctest: +SKIP
                3
        """
        from batcher.api.dataset._export import to_ray_dataset

        return to_ray_dataset(
            self,
            batch_size=batch_size,
            block_size_bytes=block_size_bytes,
            distributed=distributed,
        )

    def to_daft(self) -> Any:
        """Execute and hand the result to Daft as a ``daft.DataFrame`` (needs `daft`).

        The return leg of :func:`batcher.from_daft`. The output batches are coalesced into
        tables of about 128 MiB and passed to ``daft.from_arrow``, which builds the frame
        from them, so the result is held in memory on this process by Daft. An empty result
        still carries its schema.

        Column types cross as Arrow. Batcher's ``string``/``binary`` become Daft's
        ``String``/``Binary``, and a dictionary-encoded source column arrives decoded, since
        the engine decodes dictionaries at its boundary.

        Returns:
            A ``daft.DataFrame`` over the result.

        Raises:
            BackendError: If ``daft`` is not installed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> frame = bt.from_pydict({"x": [1, 2, 3]}).to_daft()  # doctest: +SKIP
                >>> frame.to_pydict()  # doctest: +SKIP
                {'x': [1, 2, 3]}
        """
        from batcher.api.dataset._export import to_daft

        return to_daft(self)

    def to_spark(
        self,
        spark: Any,
        *,
        max_arrow_bytes: int | None = None,
        staging_path: str | None = None,
    ) -> Any:
        """Execute and hand the result to a Spark session as a ``pyspark.sql.DataFrame``.

        The return leg of :func:`batcher.from_spark`, taking the session explicitly so the
        caller decides which Spark application receives the data. Output batches are pulled
        until they pass `max_arrow_bytes`, 64 MiB by default. A result that stays under it
        goes to ``spark.createDataFrame`` as one Arrow table, which PySpark 4 accepts
        directly; PySpark 3 receives it through pandas.

        A larger result is written as Parquet to a new ``to_spark-<id>`` directory under
        `staging_path`, the batches already pulled first and then the rest of the stream,
        and Spark reads that directory with ``spark.read.parquet``. The driver then holds
        one engine batch at a time rather than the whole result. With no `staging_path` the
        directory goes under a fresh local temporary directory, which only a Spark whose
        executors share this machine's filesystem can read, so pass a shared path such as
        ``s3://<bucket>/<prefix>`` for a cluster. Spark reads the staged files lazily, so
        they are not removed. Delete the directory once the Spark frame is no longer used.
        Timestamps are staged at microsecond precision, Spark's own, and a value that would
        lose precision raises instead of being truncated.

        Args:
            spark: The ``pyspark.sql.SparkSession`` to create the frame in.
            max_arrow_bytes: The largest result, in retained Arrow bytes, handed over in
                memory. ``None`` uses 64 MiB. Pass ``0`` to always stage Parquet.
            staging_path: The directory or URI under which a large result is staged.
                ``None`` uses a new local temporary directory.

        Returns:
            A ``pyspark.sql.DataFrame`` over the result.

        Raises:
            BackendError: If ``pyspark`` is not installed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from pyspark.sql import SparkSession  # doctest: +SKIP
                >>> spark = SparkSession.builder.getOrCreate()  # doctest: +SKIP
                >>> sdf = bt.from_pydict({"x": [1, 2, 3]}).to_spark(spark)  # doctest: +SKIP
                >>> sdf.count()  # doctest: +SKIP
                3
        """
        from batcher.api.dataset._export import to_spark

        return to_spark(self, spark, max_arrow_bytes=max_arrow_bytes, staging_path=staging_path)

    def show(self, limit: int = 10) -> None:
        """Print a preview of the first `limit` result rows to stdout.

        A terminal operation for interactive use: it executes the plan (capped at
        `limit` rows) and prints the result, returning nothing. For programmatic
        access to the data use `to_pydict` / `to_pylist` / `collect`.

        The `limit` is pushed into the *plan*, so previewing a billion-row source reads
        only enough of it to fill the screen. The footer says "first N rows" only when the
        preview actually filled its limit, so a complete result does not read as a partial
        one. A value too long to fit, and a table too wide to fit, are both cut and marked
        rather than allowed to wrap.

        Args:
            limit: Maximum number of rows to print; must be non-negative.

        Raises:
            PlanError: If `limit` is negative.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.from_pydict({"city": ["oslo", "lima"], "temp": [-3.5, 21.0]}).show()
                +--------+--------+
                | city   | temp   |
                | string | double |
                +--------+--------+
                | oslo   | -3.5   |
                | lima   | 21.0   |
                +--------+--------+
                [2 rows x 2 columns]
        """
        limit = require_int(limit, func="show", arg="limit", minimum=0)
        _show(self._plan, self._sources, self.columns, limit)
