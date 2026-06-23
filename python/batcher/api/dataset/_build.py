"""Plan-construction helpers behind the thinner `Dataset` methods.

`Dataset` stays a thin fluent builder (the v2 maintainability contract): its heavier
methods (`window`) and the frame-level convenience sugar (`fill_null`/`drop_nulls`/
`cast`) delegate their bodies here, mirroring how terminal ops live in `terminal.py`.
These functions take the `Dataset` and return a new one via its own public methods,
so they add no new IR — the sugar lowers to existing `select`/`with_columns`/`filter`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.api._join_helpers import _as_key_expr
from batcher.plan.expr_ir import Col, nullif, when
from batcher.plan.ir_tags import WINDOW_AGGREGATES
from batcher.plan.logical import (
    Sample,
    SortKeySpec,
    Unnest,
    Unpivot,
    Window,
    WindowFrame,
    WindowFuncSpec,
)

if TYPE_CHECKING:
    from batcher.api.dataset.frame import Dataset
    from batcher.plan.expr_ir import Expr


def build_window(
    ds: Dataset,
    *,
    partition_by: list[str | Expr],
    order_by: list[str | tuple[str, bool] | Expr],
    functions: dict[str, str | tuple[str, str]],
    frame: tuple[int | None, int | None] | None,
) -> Dataset:
    """Construct a `Window` node (see `Dataset.window` for the contract)."""
    if not functions:
        raise PlanError("window() requires at least one function")

    wframe = WindowFrame(*frame) if frame is not None else None
    part_keys = tuple(_as_key_expr(k) for k in partition_by)

    order_specs: list[SortKeySpec] = []
    for key in order_by:
        if isinstance(key, tuple):
            name, descending = key
            order_specs.append(SortKeySpec(_as_key_expr(name), descending=bool(descending)))
        else:
            order_specs.append(SortKeySpec(_as_key_expr(key)))

    specs: list[WindowFuncSpec] = []
    for alias, spec in functions.items():
        if isinstance(spec, str):
            specs.append(WindowFuncSpec(spec, None, alias))
        elif isinstance(spec, tuple):
            # (func, column) or, for lag/lead, (func, column, offset).
            if len(spec) == 2:
                func, column = spec
                offset = 1
            elif len(spec) == 3:
                func, column, offset = spec
            else:
                raise PlanError(
                    f"window function {alias!r} must be (func, column) or (func, column, offset)"
                )
            # `mean` is the canonical DataFrame spelling (matches Expr.mean()); the
            # window engine names the aggregate `avg`, so accept both here.
            if func == "mean":
                func = "avg"
            fn_frame = wframe if func in WINDOW_AGGREGATES else None
            specs.append(WindowFuncSpec(func, _as_key_expr(column), alias, int(offset), fn_frame))
        else:
            raise PlanError(
                f"window function {alias!r} must be a string or (func, column[, offset]) tuple"
            )

    return ds._derive(Window(ds._plan, part_keys, tuple(order_specs), tuple(specs)))


def build_fill_null(ds: Dataset, value: Any | dict[str, Any]) -> Dataset:
    """Replace nulls — one fill `value` for every column, or per-column via a dict."""
    cols = ds.columns
    if isinstance(value, dict):
        unknown = set(value) - set(cols)
        if unknown:
            raise PlanError(f"fill_null(): unknown column(s) {sorted(unknown)}")
        return ds.with_columns(**{c: Col(c).fill_null(value[c]) for c in value})
    return ds.with_columns(**{c: Col(c).fill_null(value) for c in cols})


# Strategies that lower to a whole-relation window aggregate broadcast into a coalesce.
_FILL_AGG_STRATEGIES = {"mean": "avg", "min": "min", "max": "max"}


def build_fill_null_strategy(
    ds: Dataset, strategy: str, subset: list[str] | None = None
) -> Dataset:
    """Replace nulls using a `strategy` rather than a constant.

    ``"zero"`` fills with 0; ``"mean"``/``"min"``/``"max"`` fill with the column's
    whole-relation aggregate (a single window-aggregate pass, distributed-safe).
    ``"forward"``/``"backward"``/``"median"`` are not supported — forward/backward
    fill require a defined row order (which a distributed relation does not carry),
    and median is not a window aggregate; raise an actionable error.
    """
    cols = subset if subset is not None else ds.columns
    unknown = set(cols) - set(ds.columns)
    if unknown:
        raise PlanError(f"fill_null(): unknown column(s) {sorted(unknown)}")
    if strategy == "zero":
        return ds.with_columns(**{c: Col(c).fill_null(0) for c in cols})
    if strategy not in _FILL_AGG_STRATEGIES:
        raise PlanError(
            f"fill_null(strategy={strategy!r}) is not supported; use one of "
            "'mean'/'min'/'max'/'zero', a constant value, or fill from another column. "
            "(forward/backward fill need a defined row order; median is not a window aggregate.)"
        )
    agg = _FILL_AGG_STRATEGIES[strategy]
    helpers = {f"__fill_{c}": (agg, c) for c in cols}
    filled = ds.window(partition_by=[], order_by=[], functions=helpers)
    filled = filled.with_columns(**{c: Col(c).fill_null(Col(f"__fill_{c}")) for c in cols})
    return filled.drop(*helpers.keys())


def build_drop_nulls(ds: Dataset, subset: list[str] | None) -> Dataset:
    """Drop rows that are null in any of `subset` (or any column when `subset` is None)."""
    cols = subset if subset is not None else ds.columns
    unknown = set(cols) - set(ds.columns)
    if unknown:
        raise PlanError(f"drop_nulls(): unknown column(s) {sorted(unknown)}")
    if not cols:
        return ds
    predicate = Col(cols[0]).is_not_null()
    for c in cols[1:]:
        predicate = predicate & Col(c).is_not_null()
    return ds.filter(predicate)


def build_cast(ds: Dataset, dtypes: str | dict[str, str], *, strict: bool = True) -> Dataset:
    """Cast columns — one dtype string for every column, or per-column via a dict.

    `strict=False` selects ``TRY_CAST`` (NULL on an unconvertible value).
    """

    def _cast(name: str, dtype: str) -> Expr:
        e = Col(name)
        return e.cast(dtype) if strict else e.try_cast(dtype)

    if isinstance(dtypes, dict):
        unknown = set(dtypes) - set(ds.columns)
        if unknown:
            raise PlanError(f"cast(): unknown column(s) {sorted(unknown)}")
        return ds.with_columns(**{c: _cast(c, t) for c, t in dtypes.items()})
    return ds.with_columns(**{c: _cast(c, dtypes) for c in ds.columns})


@dataclass(frozen=True, slots=True)
class RepartitionSpec:
    """How the next `write` should lay out its output files.

    - `num_files`: produce exactly this many files (rows split evenly).
    - `by`: Hive-partition the output by these column values (one subtree per value).
    - `target_size_mb`: coalesce into files of roughly this many megabytes.

    `num_files` and `target_size_mb` are resolved to a per-file row cap *after* the
    result materializes (so no extra counting pass), and may combine with `by`.
    """

    num_files: int | None = None
    by: tuple[str, ...] = ()
    target_size_mb: float | None = None


def build_distinct(
    ds: Dataset,
    subset: list[str],
    keep: str,
    order_by: str | list[str] | list[tuple[str, bool]] | None,
) -> Dataset:
    """Keep one row per `subset` key via ``row_number()`` over the partition.

    `keep="first"`/`"last"` order by `order_by` (ascending; `"last"` reverses);
    `keep="any"` orders by the subset keys themselves so the choice is deterministic
    and partition-independent. Lowers to window + filter + drop (existing IR).
    """
    unknown = set(subset) - set(ds.columns)
    if unknown:
        raise PlanError(f"distinct(): unknown subset column(s) {sorted(unknown)}")
    if keep not in ("first", "last", "any"):
        raise PlanError(f"distinct(): keep must be 'first'/'last'/'any', got {keep!r}")

    if keep == "any":
        order: list[tuple[str, bool]] = [(c, False) for c in subset]
    else:
        if order_by is None:
            raise PlanError(f"distinct(keep={keep!r}) requires order_by")
        keys = [order_by] if isinstance(order_by, str) else list(order_by)
        descending = keep == "last"
        order = [(k, descending) if isinstance(k, str) else (k[0], k[1] ^ descending) for k in keys]

    rn = "__bc_distinct_rn"
    ranked = build_window(
        ds, partition_by=list(subset), order_by=order, functions={rn: "row_number"}, frame=None
    )
    return ranked.filter(Col(rn) == 1).drop(rn)


def build_explode(ds: Dataset, column: str, alias: str | None) -> Dataset:
    """Construct an `Unnest` node (see `Dataset.explode` for the contract)."""
    if column not in ds.columns:
        raise PlanError(f"explode(): unknown column {column!r}")
    return ds._derive(Unnest(ds._plan, column, alias or column))


def build_unnest(ds: Dataset, columns: str | list[str]) -> Dataset:
    """Expand each struct `column` into its fields as top-level columns (Polars
    ``unnest``; Spark ``select("s.*")``). Composes ``struct.field`` extraction — no
    new IR. See `Dataset.unnest` for the contract."""
    import pyarrow as pa

    from batcher.plan.expr_ir import col

    names = [columns] if isinstance(columns, str) else list(columns)
    schema = ds.schema
    fields_of: dict[str, list[str]] = {}
    for name in names:
        if name not in ds.columns:
            raise PlanError(f"unnest(): unknown column {name!r}")
        ftype = schema.field(name).type
        if not pa.types.is_struct(ftype):
            raise PlanError(f"unnest(): column {name!r} is not a struct (got {ftype})")
        fields_of[name] = [ftype.field(i).name for i in range(ftype.num_fields)]

    # Output column order: each struct expands in place to its fields; others stay.
    final: list[str] = []
    for c in ds.columns:
        final.extend(fields_of[c]) if c in fields_of else final.append(c)
    if len(final) != len(set(final)):
        dup = sorted({n for n in final if final.count(n) > 1})
        raise PlanError(f"unnest(): output columns collide: {dup} (rename before unnesting)")

    derived = {
        fname: col(sname).struct.field(fname)
        for sname, fnames in fields_of.items()
        for fname in fnames
    }
    return ds.with_columns(**derived).select(*final)


def build_window_columns(ds: Dataset, items: dict[str, Any]) -> Dataset:
    """Append window-function columns from ``agg.over(...)`` expressions.

    Each `WindowExpr` becomes a `Window` node appending its aliased column (chained,
    so all input columns and earlier window columns are preserved) — the relational
    lowering of SQL ``<agg> OVER (PARTITION BY … ORDER BY …)``.
    """
    plan = ds._plan
    for alias, we in items.items():
        part_keys = tuple(_as_key_expr(k) for k in we.partition_by)
        order_specs: list[SortKeySpec] = []
        for key in we.order_by:
            if isinstance(key, tuple):
                name, descending = key
                order_specs.append(SortKeySpec(_as_key_expr(name), descending=bool(descending)))
            else:
                order_specs.append(SortKeySpec(_as_key_expr(key)))
        frame = WindowFrame(*we.frame) if we.frame is not None else None
        spec = WindowFuncSpec(we.func, we.input, alias, we.offset, frame)
        plan = Window(plan, part_keys, tuple(order_specs), (spec,))
    return ds._derive(plan)


_PIVOT_AGGS = ("sum", "mean", "min", "max", "count")


def build_pivot(
    ds: Dataset,
    index: list[str],
    on: str,
    values: str,
    aggregate: str,
    columns: list[Any] | None,
) -> Dataset:
    """Reshape long → wide (SQL ``PIVOT`` / pandas ``pivot_table``).

    Lowers to ``group_by(index).agg(...)`` with one conditional aggregate per pivot
    value: ``<agg>(values) WHERE on == v``, expressed as
    ``when(on == v).then(values).otherwise(<typed null>).<agg>()`` — so it reuses the
    tested grouping/aggregation engine with no new operator. The else-branch uses
    ``nullif(values, values)`` (a value-typed null) so non-matching rows are ignored
    by the aggregate. With `columns` omitted, the distinct pivot values are discovered
    by an eager pre-pass over `on` (like DuckDB's auto-`PIVOT`).
    """
    if aggregate not in _PIVOT_AGGS:
        raise PlanError(f"pivot(): aggregate must be one of {_PIVOT_AGGS}, got {aggregate!r}")
    for c in (*index, on, values):
        if c not in ds.columns:
            raise PlanError(f"pivot(): unknown column {c!r}")
    if columns is None:
        seen = ds.select(on).distinct().to_pydict()[on]
        cols = sorted(v for v in seen if v is not None)
    else:
        cols = list(columns)
    if not cols:
        raise PlanError("pivot(): no pivot column values to spread")
    typed_null = nullif(Col(values), Col(values))
    aggs: dict[str, Any] = {}
    for v in cols:
        masked = when(Col(on) == v).then(Col(values)).otherwise(typed_null)
        aggs[str(v)] = getattr(masked, aggregate)()
    return ds.group_by(*index).agg(**aggs)


def build_sample(
    ds: Dataset, fraction: float | None, seed: int | None, n: int | None = None
) -> Dataset:
    """Construct a `Sample` node — a fraction sample (`fraction`) or a fixed-count
    sample (`n`). Exactly one of `fraction`/`n` is set. `seed=None` bakes a fresh
    random seed at plan-build so the sample is reproducible within a run and
    consistent across workers."""
    if (fraction is None) == (n is None):
        raise PlanError("sample() takes exactly one of `fraction` or `n`")
    if seed is None:
        seed = random.randrange(2**63)
    # The fraction field is required by the node; for count mode it is unused (1.0).
    return ds._derive(Sample(ds._plan, 1.0 if n is not None else float(fraction), int(seed), n))


def build_unpivot(
    ds: Dataset,
    index: list[str] | None,
    on: list[str] | None,
    variable_name: str,
    value_name: str,
) -> Dataset:
    """Construct an `Unpivot` node (see `Dataset.unpivot` for the contract).

    With `on` omitted, every column not in `index` is melted; with `index` omitted,
    every column not in `on` becomes an identifier.
    """
    cols = ds.columns
    if index is None and on is None:
        raise PlanError("unpivot() requires `index` or `on`")
    idx = list(index) if index is not None else [c for c in cols if c not in set(on or ())]
    vals = list(on) if on is not None else [c for c in cols if c not in set(idx)]
    return ds._derive(Unpivot(ds._plan, tuple(idx), tuple(vals), variable_name, value_name))
