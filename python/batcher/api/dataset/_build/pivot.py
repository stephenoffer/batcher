"""Bodies of the `Dataset` verbs that turn long data wide and back: `pivot`, `unpivot`, `unnest`.

Each lowers onto existing operators, so none adds IR. `pivot` is one grouped aggregation with
a conditional aggregate per (value, aggregate, category) cell; `unpivot` is the `Unpivot` node,
plus a null filter when null cells are excluded; `unnest` is `struct.field` extraction worked
out from the schema alone, so the control plane never reads a row.
"""

from __future__ import annotations

import datetime
import decimal
from collections import Counter
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import Col, col, lit, nullif, when
from batcher.plan.ir_tags import RUNNING_AGGREGATES
from batcher.plan.logical import Unpivot

if TYPE_CHECKING:
    from batcher.api.dataset.frame import Dataset
    from batcher.plan.expr_ir import Expr

__all__ = ["build_pivot", "build_unnest", "build_unpivot"]


def build_unnest(
    ds: Dataset, columns: list[str], *, separator: str | None = None, max_depth: int = 1
) -> Dataset:
    """Expand each struct column into its fields as top-level columns (Polars ``unnest``).

    Composes ``struct.field`` extraction, so it needs no new IR. With `max_depth` above 1 a
    field that is itself a struct is expanded again, down to that many levels; the decision
    reads the schema only. A `separator` names each output by its path from the expanded
    column, starting with its name (``s.a``, as Polars names it); without one an output keeps
    its bare field name.

    Args:
        ds: The dataset to expand.
        columns: The struct columns to expand.
        separator: Joins the path segments of an output name; ``None`` keeps bare names.
        max_depth: How many struct levels to expand, at least 1.

    Returns:
        The dataset with each struct's fields in its place.

    Raises:
        PlanError: For an unknown or non-struct column, a bad `max_depth` or `separator`,
            or output names that collide.
    """
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1:
        raise PlanError(f"unnest(): max_depth must be an integer >= 1, got {max_depth!r}")
    if separator is not None and (not isinstance(separator, str) or not separator):
        raise PlanError(f"unnest(): separator must be a non-empty string, got {separator!r}")
    schema = ds.schema
    expanded: dict[str, list[tuple[str, Expr]]] = {}
    for name in columns:
        if name not in ds.columns:
            raise PlanError(f"unnest(): unknown column {name!r}")
        ftype = schema.field(name).type
        if not pa.types.is_struct(ftype):
            raise PlanError(f"unnest(): column {name!r} is not a struct (got {ftype})")
        expanded[name] = _struct_leaves(col(name), ftype, [name], separator, max_depth)

    # Output column order: each struct expands in place to its fields; others stay.
    final: list[str] = []
    for c in ds.columns:
        final.extend(out for out, _ in expanded[c]) if c in expanded else final.append(c)
    if len(final) != len(set(final)):
        # One counting pass rather than a `list.count()` per column: a struct-heavy
        # relation can expand to thousands of output names, and the error message is
        # not the place to spend quadratic time.
        dup = sorted(n for n, k in Counter(final).items() if k > 1)
        hint = "" if separator is not None else ", or pass separator='.' to keep each path"
        raise PlanError(f"unnest(): output columns collide: {dup} (rename before unnesting{hint})")
    derived = {out: expr for leaves in expanded.values() for out, expr in leaves}
    return ds.with_columns(**derived).select(*final)


def _struct_leaves(
    expr: Expr, ftype: pa.StructType, path: list[str], separator: str | None, depth: int
) -> list[tuple[str, Expr]]:
    """``(output name, extraction)`` for every field of a struct, recursing `depth` levels."""
    leaves: list[tuple[str, Expr]] = []
    for i in range(ftype.num_fields):
        field = ftype.field(i)
        child = expr.struct.field(field.name)
        child_path = [*path, field.name]
        if depth > 1 and pa.types.is_struct(field.type):
            leaves.extend(_struct_leaves(child, field.type, child_path, separator, depth - 1))
            continue
        name = field.name if separator is None else separator.join(child_path)
        leaves.append((name, child))
    return leaves


def build_pivot(
    ds: Dataset,
    index: list[str],
    on: str,
    values: list[str],
    aggregates: list[str],
    columns: list[Any] | None,
    fill_value: Any = None,
) -> Dataset:
    """Reshape long to wide (SQL ``PIVOT`` / pandas ``pivot_table``).

    Lowers to ``group_by(index).agg(...)`` with one conditional aggregate per output cell:
    ``when(on == v).then(value).otherwise(<typed null>).<agg>()``, so the grouping engine
    does the work in one pass and no new operator is involved. The else-branch is
    ``nullif(value, value)``, a null of the value's own type, which the aggregate ignores.
    With `columns` omitted the categories are discovered by an eager pass over `on`.

    A `fill_value` replaces a cell only when its (index, category) combination has **no**
    rows, which a per-category row count decides. A cell whose rows exist but aggregate to
    null stays null: the two mean different things and a blanket `fill_null` would merge them.

    Args:
        ds: The dataset to pivot.
        index: The output row key.
        on: The column whose values become output columns.
        values: The columns aggregated into the cells.
        aggregates: The aggregates applied to every value column.
        columns: The categories, or ``None`` to discover them.
        fill_value: The value of a cell with no rows; ``None`` leaves it null.

    Returns:
        The pivoted dataset.

    Raises:
        PlanError: For an unknown column or aggregate, a category of the wrong type, two
            categories naming the same output column, or no categories at all.
    """
    if not values or not aggregates:
        raise PlanError("pivot(): values and aggregate each need at least one entry")
    for agg in aggregates:
        if agg not in RUNNING_AGGREGATES:
            raise PlanError(
                f"pivot(): aggregate must be one of {sorted(RUNNING_AGGREGATES)}, got {agg!r}"
            )
    for c in (*index, on, *values):
        if c not in ds.columns:
            raise PlanError(f"pivot(): unknown column {c!r}")
    if columns is None:
        seen = ds.select(on).distinct().to_pydict()[on]
        cats = sorted(v for v in seen if v is not None)
    else:
        cats = list(columns)
        _check_categories(ds.schema.field(on).type, on, cats)
    if not cats:
        raise PlanError("pivot(): no pivot column values to spread")

    aggs: dict[str, Any] = {}
    for value in values:
        typed_null = nullif(Col(value), Col(value))
        for agg in aggregates:
            prefix = [value] * (len(values) > 1) + [agg] * (len(aggregates) > 1)
            for cat in cats:
                masked = when(Col(on) == cat).then(Col(value)).otherwise(typed_null)
                cell = getattr(masked, agg)()
                if fill_value is not None:
                    cell = when(_rows_in(on, cat) == 0).then(lit(fill_value)).otherwise(cell)
                aggs["_".join([*prefix, str(cat)])] = cell
    return ds.group_by(*index).agg(**aggs)


def _rows_in(on: str, cat: Any) -> Expr:
    """The number of rows of a group whose `on` equals `cat`, as an aggregate."""
    return when(Col(on) == cat).then(Col(on)).otherwise(nullif(Col(on), Col(on))).count()


#: The Python types a category may have for each kind of `on` column, checked at plan time
#: so a mismatch is named here rather than surfacing as an engine comparison error.
_CATEGORY_TYPES: tuple[tuple[Any, tuple[type, ...], str], ...] = (
    (pa.types.is_boolean, (bool,), "bool"),
    (
        lambda t: pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_decimal(t),
        (int, float, decimal.Decimal),
        "number",
    ),
    (lambda t: pa.types.is_string(t) or pa.types.is_large_string(t), (str,), "str"),
    (pa.types.is_timestamp, (datetime.datetime,), "datetime"),
    (pa.types.is_date, (datetime.date,), "date"),
)


def _check_categories(on_type: pa.DataType, on: str, cats: list[Any]) -> None:
    """Refuse explicit `columns=` values that cannot match `on`, or that name one column twice."""
    for matches, allowed, label in _CATEGORY_TYPES:
        if not matches(on_type):
            continue
        for v in cats:
            # `bool` is an `int` subclass; a number column must not accept `True`.
            wrong_bool = isinstance(v, bool) and bool not in allowed
            if v is None or wrong_bool or not isinstance(v, allowed):
                raise PlanError(
                    f"pivot(columns=...): {v!r} cannot match column {on!r} of type {on_type}; "
                    f"every category must be a {label}. A null category matches no row: "
                    "fill the nulls first if they should get a column."
                )
        break
    names = [str(v) for v in cats]
    dup_names = sorted(n for n, k in Counter(names).items() if k > 1)
    if dup_names or len(set(cats)) != len(cats):
        raise PlanError(
            f"pivot(columns=...): categories {cats!r} name the same output column or the "
            f"same rows twice ({dup_names or 'equal values'}); list each category once"
        )


def build_unpivot(
    ds: Dataset,
    index: list[str] | None,
    on: list[str] | None,
    variable_name: str,
    value_name: str,
    *,
    include_nulls: bool = True,
) -> Dataset:
    """Construct an `Unpivot` node (see `Dataset.unpivot` for the contract).

    With `on` omitted, every column not in `index` is melted; with `index` omitted,
    every column not in `on` becomes an identifier. `include_nulls=False` drops the rows
    whose melted value is null, which is SQL's default ``UNPIVOT`` (``EXCLUDE NULLS``).

    Args:
        ds: The dataset to melt.
        index: The identifier columns, or ``None`` for every column not in `on`.
        on: The melted columns, or ``None`` for every column not in `index`.
        variable_name: The column naming each melted column.
        value_name: The column holding each melted value.
        include_nulls: Keep rows whose melted value is null.

    Returns:
        The melted dataset.

    Raises:
        PlanError: If neither `index` nor `on` is given, or `include_nulls` is not a bool.
    """
    if not isinstance(include_nulls, bool):
        raise PlanError(f"unpivot(include_nulls=...) must be a bool, got {include_nulls!r}")
    cols = ds.columns
    if index is None and on is None:
        raise PlanError("unpivot() requires `index` or `on`")
    idx = list(index) if index is not None else [c for c in cols if c not in set(on or ())]
    vals = list(on) if on is not None else [c for c in cols if c not in set(idx)]
    out = ds._derive(Unpivot(ds._plan, tuple(idx), tuple(vals), variable_name, value_name))
    return out if include_nulls else out.filter(Col(value_name).is_not_null())
