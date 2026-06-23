"""`Dataset` — the lazy, immutable, fluent entry point.

A `Dataset` is a handle to a `LogicalPlan` plus its bound input relations. Every
operation returns a new `Dataset` (nothing mutates); no work happens until a
terminal operation (`collect`, `to_pydict`, ...). At that point `api` orchestrates
the layers: Kyber optimizes, Carbonite checks feasibility, Core executes.

One obvious way to do each thing: expressions everywhere (no lambdas), `select`
for choosing/deriving the full output, `with_columns` for adding/replacing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from batcher._internal.errors import PlanError
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
    build_drop_nulls,
    build_explode,
    build_fill_null,
    build_fill_null_strategy,
    build_pivot,
    build_sample,
    build_unnest,
    build_unpivot,
    build_window,
    build_window_columns,
)
from batcher.api.dataset.dq import DatasetDQ
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
from batcher.plan.expr_ir.nodes import WindowExpr
from batcher.plan.logical import (
    AsofJoin,
    Distinct,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    Project,
    Projection,
    Sort,
    SortKeySpec,
    Union,
    remap_sources,
)
from batcher.plan.schema import suggest_columns

if TYPE_CHECKING:
    from batcher.api.io_namespace import Writer
    from batcher.api.stats import RunStats

__all__ = ["Dataset", "GroupBy"]


def _unknown_cols(missing: set[str], available: list[str]) -> str:
    """Render an unknown-column list with a 'did you mean' hint for the first miss."""
    ordered = sorted(missing)
    return f"{ordered}{suggest_columns(ordered[0], available)}" if ordered else "[]"


class Dataset:
    """A lazy relation. Construct via `batcher.from_arrow` / `from_pydict`."""

    __slots__ = ("_plan", "_repartition", "_sources")

    def __init__(
        self,
        plan: LogicalPlan,
        sources: list[Source],
        repartition: RepartitionSpec | None = None,
    ) -> None:
        self._plan = plan
        self._sources = sources
        # An optional output-layout hint consumed by `write` (set by `repartition`);
        # transformations drop it (it is a pre-write concern), so it never propagates.
        self._repartition = repartition

    # --- introspection -----------------------------------------------------
    @property
    def columns(self) -> list[str]:
        """The output column names of the current plan."""
        return self._plan.available_columns()

    @property
    def is_streaming(self) -> bool:
        """Whether any bound source is unbounded (e.g. Kafka, incremental files).

        A streaming dataset cannot be `collect()`-ed (it would never finish); consume
        it incrementally with `iter_batches()` or write it to a sink instead.
        """
        from batcher.io.source import is_bounded

        return any(not is_bounded(s) for s in self._sources)

    def __repr__(self) -> str:
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

    def __getitem__(self, key: str | list[str] | slice) -> Expr | Dataset:
        """Index sugar: ``ds["x"]`` → an `Expr`; ``ds[["a", "b"]]`` → a projected
        `Dataset`; ``ds[:n]`` / ``ds[i:j]`` → a row slice (like `limit`/`offset`).
        """
        if isinstance(key, str):
            return Col(key)
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
        raise PlanError("Dataset index must be a column name, list of names, or slice")

    def __len__(self) -> int:
        """Row count — ``len(ds)`` is sugar for `count()` (a terminal operation)."""
        return self.count()

    def _derive(self, plan: LogicalPlan) -> Dataset:
        return Dataset(plan, self._sources)

    # --- transformations ---------------------------------------------------
    def filter(self, predicate: Expr) -> Dataset:
        """Keep rows where `predicate` evaluates to true."""
        if not isinstance(predicate, Expr):
            raise PlanError("filter() requires an expression, e.g. col('x') > 0")
        return self._derive(Filter(self._plan, predicate))

    def select(self, *columns: str | Expr, **named: Expr | int | float | bool | str) -> Dataset:
        """Project to exactly the given columns.

        Positional args are column names (strings), bare ``col(...)`` references, or
        aliased expressions (``expr.alias("name")``); keyword args bind a new name to
        an expression: ``ds.select("id", total=col("price") * col("qty"))``.
        """
        items: list[Projection] = []
        for c in columns:
            if isinstance(c, str):
                items.append(Projection(c, Col(c)))
            elif isinstance(c, Aliased):
                items.append(Projection(c.name, c.inner))
            elif isinstance(c, Col):
                items.append(Projection(c.name, c))
            else:
                raise PlanError(
                    "positional select() arguments must be column names, col(...) "
                    "references, or aliased expressions; name other derived columns "
                    "via a keyword (select(total=expr)) or .alias('total')"
                )
        for alias, expr in named.items():
            items.append(Projection(alias, _as_expr(expr)))
        if not items:
            raise PlanError("select() requires at least one column")
        return self._derive(Project(self._plan, tuple(items)))

    def with_columns(self, **named: Expr | int | float | bool | str) -> Dataset:
        """Add or replace columns, keeping all existing ones.

        Values may be expressions, scalars, or window expressions from
        ``agg.over(...)`` (e.g. ``with_columns(total=col("x").sum().over(partition_by=["g"]))``),
        which append windowed columns. Mixing window and non-window values in one call
        is not supported — use separate calls.
        """
        if not named:
            raise PlanError("with_columns() requires at least one named column")
        windows = {a: e for a, e in named.items() if isinstance(e, WindowExpr)}
        if windows:
            if len(windows) != len(named):
                raise PlanError(
                    "with_columns(): mix of window (.over) and non-window columns; "
                    "add them in separate with_columns() calls"
                )
            return build_window_columns(self, windows)
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
        return self._derive(Project(self._plan, tuple(items)))

    def sort(
        self,
        *by: str | Expr,
        descending: bool | list[bool] = False,
        nulls_first: bool | list[bool] = False,
    ) -> Dataset:
        """Order rows by one or more keys (column names or expressions).

        `descending`/`nulls_first` are either a single bool applied to all keys or
        a list matching the number of keys.
        """
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
        """ML/multimodal accessor: batch `infer`/`embed`/`map_batches` with GPU and
        actor-pool scheduling (`ds.ml.infer(model, num_gpus=1, concurrency=4)`)."""
        return DatasetML(self)

    @property
    def dq(self) -> DatasetDQ:
        """Data-quality accessor: accumulate expectations
        (`not_null`/`unique`/`in_range`/`matches`/`accepted_values`/`check`) then
        `fail()` (raise), `drop()` (keep valid), `quarantine()` (split valid/rejected),
        or `validate()` (counts). E.g.
        ``ds.dq.not_null("id").unique(["id"]).in_range("age", 0, 120).quarantine()``."""
        return DatasetDQ(self)

    @property
    def scd(self) -> DatasetSCD:
        """Slowly-changing-dimension accessor: upsert this incoming snapshot into a
        target as `scd.type1` (overwrite), `scd.type2` (effective-dated history), or
        `scd.type3` (previous-value column)."""
        return DatasetSCD(self)

    def map_batches(
        self,
        fn: Callable | type,
        *,
        batch_size: int | None = None,
        output_columns: list[str] | None = None,
        num_workers: int = 1,
        num_gpus: float = 0.0,
        concurrency: int | None = None,
        batch_format: str = "pyarrow",
    ) -> Dataset:
        """Apply a Python function to each batch — `ds.ml.map_batches`, kept
        top-level for the familiar spelling (see `ds.ml` for the full ML surface)."""
        return self.ml.map_batches(
            fn,
            batch_size=batch_size,
            output_columns=output_columns,
            num_workers=num_workers,
            num_gpus=num_gpus,
            concurrency=concurrency,
            batch_format=batch_format,
        )

    def with_column(self, name: str, expr: Expr) -> Dataset:
        """Add or replace a single column (sugar for `with_columns`)."""
        return self.with_columns(**{name: expr})

    def drop(self, *columns: str) -> Dataset:
        """Return a dataset without the named columns."""
        to_drop = set(columns)
        available = self._plan.available_columns()
        missing = to_drop - set(available)
        if missing:
            raise PlanError(f"drop(): unknown column(s) {_unknown_cols(missing, available)}")
        keep = [c for c in available if c not in to_drop]
        if not keep:
            raise PlanError("drop() would remove all columns")
        return self.select(*keep)

    def rename(self, mapping: dict[str, str] | None = None, **renames: str) -> Dataset:
        """Rename columns, preserving order. Pass a ``{old: new}`` dict or kwargs
        (``rename(old="new")``); a dict and kwargs may be combined."""
        merged = {**(mapping or {}), **renames}
        available = self._plan.available_columns()
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
        """
        if subset is None:
            return self._derive(Distinct(self._plan))
        return build_distinct(self, subset, keep, order_by)

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
        """
        if num_files is not None and target_size_mb is not None:
            raise PlanError("repartition(): pass num_files or target_size_mb, not both")
        if num_files is not None and num_files < 1:
            raise PlanError(f"repartition(): num_files must be >= 1, got {num_files}")
        if target_size_mb is not None and target_size_mb <= 0:
            raise PlanError(f"repartition(): target_size_mb must be > 0, got {target_size_mb}")
        by_cols = () if by is None else ((by,) if isinstance(by, str) else tuple(by))
        if num_files is None and target_size_mb is None and not by_cols:
            raise PlanError("repartition(): provide num_files, target_size_mb, or by")
        spec = RepartitionSpec(num_files=num_files, by=by_cols, target_size_mb=target_size_mb)
        return Dataset(self._plan, self._sources, spec)

    def value_counts(self, column: str, *, name: str = "count", sort: bool = True) -> Dataset:
        """Count occurrences of each distinct value of `column` (pandas/Polars
        ``value_counts``). Returns ``[column, name]``, sorted by count descending
        unless `sort=False`. Sugar over ``group_by(column).agg(count())``."""
        from batcher.api.functions import count

        out = self.group_by(column).agg(**{name: count()})
        return out.sort(name, descending=True) if sort else out

    def describe(self, *, percentiles: tuple[float, ...] = (0.25, 0.5, 0.75)) -> Dataset:
        """Summary statistics per column (pandas/Polars ``describe``).

        **Executes** the query and returns a small `Dataset` with a ``statistic``
        label column and one Float64 column per input column. Numeric columns report
        count / null_count / mean / std / min / the requested `percentiles` (default
        quartiles) / max; non-numeric columns report count and null_count only.
        Composes the already-tested aggregates — no per-row work in Python.
        """
        from batcher.api.dataset._describe import describe

        return describe(self, percentiles)

    def null_count(self) -> Dataset:
        """A one-row dataset of each column's null count (pandas ``isnull().sum()``).

        Lazy: lowers to a single global aggregate and a `select`, so it stays
        mergeable and identical single-node and distributed.
        """
        from batcher.api.dataset._describe import null_count

        return null_count(self)

    def profile(self) -> Dataset:
        """A per-column data-quality profile (**executes**): one row per column with
        ``count``/``null_count``/``null_fraction``/``approx_distinct`` (HyperLogLog
        cardinality). The quick "what does this column look like" check before a load.
        """
        from batcher.api.dataset._describe import profile

        return profile(self)

    def top_k(self, k: int, by: str | list[str], *, descending: bool = True) -> Dataset:
        """The `k` rows ranked highest (or lowest, `descending=False`) by `by` — sugar
        for ``sort(by, descending).limit(k)`` (the engine fuses sort+limit to a top-N)."""
        keys = by if isinstance(by, list) else [by]
        return self.sort(*keys, descending=descending).limit(k)

    def cross_join(self, other: Dataset, *, suffix: str = "_right") -> Dataset:
        """Cartesian product — every left row paired with every right row.

        Lowered to an equi-join on a constant key, so it reuses the join engine; the
        temporary key is dropped from the output (colliding names get `suffix`).
        """
        from batcher.plan.expr_ir import lit

        key = "__cross_key__"
        left = self.with_columns(**{key: lit(1)})
        right = other.with_columns(**{key: lit(1)})
        return left.join(right, on=key, suffix=suffix).drop(key)

    def explode(self, column: str, *, alias: str | None = None) -> Dataset:
        """Explode a list/array column into one row per element (SQL ``UNNEST``).

        Other columns repeat per element; null/empty lists produce no rows. The
        exploded column replaces `column` in place (renamed to `alias` if given) and
        streams (no breaker). Raises `PlanError` if `column` is not a column.
        """
        return build_explode(self, column, alias)

    def unnest(self, *columns: str) -> Dataset:
        """Expand each struct `column` into its fields as top-level columns
        (Polars ``unnest``; Spark ``select("s.*")``).

        Each struct field becomes a column where the struct was; non-struct columns
        are unchanged. Raises `PlanError` if a column is not a struct or if an
        expanded field name would collide with an existing column.
        """
        return build_unnest(self, list(columns))

    def sample(
        self,
        fraction: float | None = None,
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
        """
        return build_sample(self, fraction, seed, n)

    def pivot(
        self,
        *,
        index: list[str],
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
        """
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
        """
        return build_unpivot(self, index, on, variable_name, value_name)

    def fill_null(
        self,
        value: Any | dict[str, Any] | None = None,
        *,
        strategy: str | None = None,
        subset: list[str] | None = None,
    ) -> Dataset:
        """Replace nulls with `value` (one for all columns, or a ``{col: value}`` dict).

        Pass `strategy` instead of `value` to fill from a statistic: ``"mean"``,
        ``"min"``, ``"max"`` (the column's whole-relation aggregate) or ``"zero"``.
        `subset` limits a strategy fill to specific columns.
        """
        if strategy is not None:
            if value is not None:
                raise PlanError("fill_null(): pass either `value` or `strategy`, not both")
            return build_fill_null_strategy(self, strategy, subset)
        if value is None:
            raise PlanError("fill_null(): provide a `value` or a `strategy`")
        return build_fill_null(self, value)

    def drop_nulls(self, subset: list[str] | None = None) -> Dataset:
        """Drop rows that are null in any of `subset` (default: any column)."""
        return build_drop_nulls(self, subset)

    def cast(self, dtypes: str | dict[str, str], *, strict: bool = True) -> Dataset:
        """Cast columns to `dtypes` — one dtype for all, or per-column via a dict.

        With `strict=False`, values that cannot be converted become NULL (DuckDB
        ``TRY_CAST``) instead of erroring the query — the safe-ingest spelling.
        """
        return build_cast(self, dtypes, strict=strict)

    def union(self, *others: Dataset, distinct: bool = False) -> Dataset:
        """Concatenate with other datasets (UNION ALL, or UNION if `distinct`).

        All datasets must have identical columns. Sources are merged so each
        side's scans resolve correctly.
        """
        plans: list[LogicalPlan] = [self._plan]
        sources = list(self._sources)
        for other in others:
            plans.append(remap_sources(other._plan, len(sources)))
            sources.extend(other._sources)
        return Dataset(Union(tuple(plans), distinct), sources)

    def intersect(self, other: Dataset) -> Dataset:
        """Distinct rows present in BOTH datasets (SQL INTERSECT).

        NULLs compare equal, matching SQL set semantics: a row that is identical —
        nulls included — in both inputs is in the result. Returns distinct rows
        (INTERSECT ALL multiplicity is not supported).
        """
        cols = self._same_columns(other, "intersect")
        return self._set_membership(other, cols, both=True)

    def except_(self, other: Dataset) -> Dataset:
        """Distinct rows in this dataset but NOT in `other` (SQL EXCEPT).

        NULLs compare equal (a wholly-null row in both inputs is excluded), matching
        SQL set semantics. Returns distinct rows (EXCEPT ALL is not supported).
        """
        cols = self._same_columns(other, "except")
        return self._set_membership(other, cols, both=False)

    def _set_membership(self, other: Dataset, cols: list[str], *, both: bool) -> Dataset:
        """INTERSECT/EXCEPT via group-by membership flags.

        Tag each side, union, then group by *all* columns. Grouping treats NULL as a
        single group, so NULLs compare equal — the SQL set-operation semantics a hash
        join cannot give (it drops NULL keys). `bool_or` records presence on each side
        per group; keep groups in both (INTERSECT) or only the left (EXCEPT). One row
        per distinct combination, so the result is DISTINCT by construction, and the
        whole thing is mergeable aggregation, so it distributes.
        """
        from batcher.plan.expr_ir import col, lit

        left = self.select(*cols).with_columns(__bc_l__=lit(True), __bc_r__=lit(False))
        right = other.select(*cols).with_columns(__bc_l__=lit(False), __bc_r__=lit(True))
        grouped = (
            left.union(right)
            .group_by(*cols)
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
        """Keep at most `n` rows after skipping `offset`."""
        if n < 0 or offset < 0:
            raise PlanError("limit() requires non-negative n and offset")
        return self._derive(Limit(self._plan, n, offset))

    def head(self, n: int = 5) -> Dataset:
        """Keep the first `n` rows (alias for `limit(n)`)."""
        return self.limit(n)

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
        """
        how = "full" if how == "outer" else how
        if how not in {"inner", "left", "right", "full", "semi", "anti"}:
            raise PlanError(
                f"unsupported join type {how!r} (inner|left|right|full|outer|semi|anti)"
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
        """ASOF (nearest-match) join — match each left row to the right row whose `on`
        key is nearest (``direction``: ``"backward"`` ≤, ``"forward"`` ≥), within the
        same `by` group (exact). Left-style: every left row is kept (null right columns
        when unmatched). Both sides should be sorted on `on` within `by` for the
        intended semantics. Specify keys via `on`/`by` (shared) or `*_on`/`*_by`.
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
        return GroupBy(self, keys, named)

    def agg(self, **aggregates: Expr) -> Dataset:
        """Aggregate over the whole dataset (no grouping).

        Shorthand for ``group_by().agg(...)``: ``ds.agg(total=col("x").sum())`` returns
        a single-row dataset.
        """
        return self.group_by().agg(**aggregates)

    # --- terminal operations ----------------------------------------------
    def collect(
        self,
        distributed: bool | str = "auto",
        num_workers: int | None = None,
        spill: bool = False,
        num_partitions: int | None = None,
        adaptive: bool = False,
        transport: str = "auto",
    ) -> pa.Table:
        """Execute the plan and materialize the result as a `pyarrow.Table`.

        Zero-config by default; every argument is an optional override.
        `distributed="auto"` uses Ray on a multi-node cluster, else single-node.
        Out-of-core spilling is automatic under memory pressure, with worker fan-out
        and partition count sized from the estimated data volume; `spill=True` forces
        it and `num_partitions` overrides the bucket count. The result is identical
        whichever way it runs. Raises `PlanError` if the dataset is unbounded (a
        streaming source) — use `iter_batches()` / `write()`.
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
        )

    def explain(self) -> str:
        """Return a human-readable optimized plan with cardinality estimates."""
        return _explain(self._plan, self._sources)

    def stats(self) -> RunStats:
        """Execute (single-node) and return measured per-operator `RunStats`.

        Where `explain()` shows the *planned* shape with estimates, `stats()` runs
        the query and reports what the engine *measured* — rows in/out, wall time,
        peak bytes, spill, and backend per operator, plus a bottleneck call (the
        answer to "where is my time going"). Not available for `map_batches`/ML
        pipelines (raises `BackendError`).

        Example:
            >>> print(ds.group_by("k").agg(s=col("v").sum()).stats())  # doctest: +SKIP
        """
        return _stats(self._plan, self._sources, self.columns)

    def count(self) -> int:
        """Return the number of result rows.

        Answered from metadata without execution whenever the count is provably
        exact — ``ds.limit(n).count()`` is ``min(n, ds.count())``, a global
        aggregate is ``1``, an empty source is ``0`` — and falls back to a full
        run otherwise. The result is always identical to executing.
        """
        return _count(self._plan, self._sources, self.columns)

    def is_empty(self) -> bool:
        """Whether the result has no rows.

        Answered from metadata when the row count is provably known; otherwise a
        single-row probe (which the streaming path reads without scanning the
        whole source).
        """
        return _is_empty(self._plan, self._sources, self.columns)

    @property
    def schema(self) -> pa.Schema:
        """The output Arrow schema (column names and types), without scanning rows.

        A scan returns its source schema directly; other plans resolve derived
        column types via a zero-row execution. Use `columns` for just the names
        (always free).
        """
        return _schema(self._plan, self._sources, self.columns)

    @property
    def dtypes(self) -> list[pa.DataType]:
        """The output column Arrow types, in order (see `schema`)."""
        return list(self.schema.types)

    def approx_quantile(self, column: str, q: float) -> float | None:
        """Approximate quantile `q` (in ``[0, 1]``) of a numeric `column`.

        Opt-in and explicitly approximate — a TDigest sketch, tail-accurate
        (p99/p999) and far cheaper than the exact sort `quantile` would need.
        Returns ``None`` for a non-numeric or empty column. Use the exact
        aggregate when precision matters.
        """
        from batcher.api.orchestration import approx_quantile

        return approx_quantile(_collect(self._plan, self._sources, self.columns), column, q)

    def iter_batches(self, batch_size: int | None = None):
        """Execute and yield the result as Arrow record batches.

        The execution mode is automatic: a breaker-free pipeline (filter / project /
        map_batches over a single source) — and top-level aggregate / distinct /
        top-N over such an input — is consumed one source batch at a time in bounded
        memory, so a larger-than-memory or unbounded source streams incrementally.
        Other plans (sort / join / window / multi-source) materialize first; if the
        source is unbounded and the plan cannot stream, a `PlanError` is raised
        rather than hanging. `batch_size` rebatches the output.
        """
        yield from _iter_batches(self._plan, self._sources, self.columns, batch_size=batch_size)

    @property
    def write(self) -> Writer:
        """The write namespace: ``ds.write(path)`` autodetects the sink format from the
        path; ``ds.write.<format>(...)`` (``parquet``/``delta``/…) is explicit. All
        accept `partition_by=`/`distributed=`/`num_workers=` and return a `WriteManifest`.
        """
        from batcher.api.io_namespace import Writer

        return Writer(self)

    def to_arrow(self) -> pa.Table:
        """Execute and return the result as a `pyarrow.Table` (the named form of `collect`)."""
        return _collect(self._plan, self._sources, self.columns)

    def to_pandas(self):
        """Execute and return as a pandas `DataFrame` (needs `batcher-engine[pandas]`)."""
        return _to_pandas(self._plan, self._sources, self.columns)

    def to_polars(self):
        """Execute and return as a Polars `DataFrame` (needs `batcher-engine[polars]`)."""
        return _to_polars(self._plan, self._sources, self.columns)

    def to_pydict(self) -> dict[str, list[Any]]:
        """Execute and return the result as a column-oriented dict (pyarrow-style)."""
        return _to_pydict(self._plan, self._sources, self.columns)

    def to_pylist(self) -> list[dict[str, Any]]:
        """Execute and return the result as a list of row dicts (pyarrow-style)."""
        return _to_pylist(self._plan, self._sources, self.columns)

    def to_torch(self, *, columns: list[str] | None = None, batch_size: int | None = None) -> Any:
        """A re-iterable ``torch.utils.data.IterableDataset`` of per-batch tensor dicts.

        Each item is a ``{column: torch.Tensor}`` for one engine batch (non-numeric
        columns are skipped). Re-iterating runs the query again, so it is safe for
        multi-epoch training and streams in bounded memory. Needs `torch`.
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
        """
        from batcher.api.dataset._export import to_torch_dataloader

        return to_torch_dataloader(self, columns, batch_size, **dl_kwargs)

    def to_tf(self, *, columns: list[str] | None = None, batch_size: int | None = None) -> Any:
        """A re-iterable ``tf.data.Dataset`` of per-batch tensor dicts (needs `tensorflow`).

        Each element is one engine batch's numeric columns as TensorFlow tensors;
        non-numeric columns are skipped.
        """
        from batcher.api.dataset._export import to_tf

        return to_tf(self, columns, batch_size)

    def show(self, limit: int = 10) -> None:
        """Print a preview of the result."""
        _show(self._plan, self._sources, self.columns, limit)
