"""Bodies of the `Dataset` verbs that cut or turn a relation: `transpose`, `partition_by`, `split`.

All three need something a lazy plan cannot know until it runs -- how many rows there are,
or which key values occur -- so each executes one small eager pre-pass (a `count`, or a
`distinct` over the key columns) and then returns ordinary lazy plans built from existing
operators: `unpivot` and `pivot` for `transpose`, a `filter` per key for `partition_by`, and
a `row_number` window plus a range filter per part for `split`. No new IR is involved.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from batcher._internal.errors import PlanError, require_int
from batcher.api.dataset._build.combine import OrderSpec, rank_rows, unused_name
from batcher.plan.expr_ir import Col, Expr, coalesce, lit, when
from batcher.plan.types import dtype_name, promote

if TYPE_CHECKING:
    from batcher.api.dataset.frame import Dataset

__all__ = ["build_partition_by", "build_split", "build_transpose"]


def _common_value_type(ds: Dataset, columns: list[str]) -> str:
    """The cast target every transposed value shares: the supertype, else ``string``."""
    schema = ds.schema
    common: pa.DataType | None = schema.field(columns[0]).type
    for c in columns[1:]:
        common = None if common is None else promote(common, schema.field(c).type)
    name = dtype_name(common) if common is not None else None
    return name if name is not None else "string"


def _labels(
    ds: Dataset,
    column_names: str | Sequence[str] | None,
    order_by: OrderSpec | None,
    descending: bool | Sequence[bool],
    rank: str,
) -> tuple[Dataset, str, list[tuple[Any, str]]]:
    """The relation to pivot, the column naming each row, and ``(row key, output name)`` pairs.

    The pairs are in output-column order. With a name column and no order they are sorted by
    name; with an order they follow it, which is the only way a positional name
    (``column_0``) or a list of names can be tied to a row.
    """
    named = isinstance(column_names, str)
    if named and column_names not in ds.columns:
        raise PlanError(f"transpose(): column_names={column_names!r} is not a column")
    if order_by is None:
        if not named:
            raise PlanError(
                "transpose(): naming output columns by row position needs an explicit order -- "
                "pass order_by=..., or column_names='<column>' to name them by a column's values"
            )
        values = ds.select(column_names).to_pydict()[column_names]
        _check_names(values, column_names)
        return ds, column_names, [(v, str(v)) for v in sorted(values)]
    ranked = rank_rows(ds, order_by, descending, name=rank, api="transpose")
    if named:
        rows = ranked.select(rank, column_names).sort(rank).to_pydict()
        _check_names(rows[column_names], column_names)
        return ranked, rank, list(zip(rows[rank], map(str, rows[column_names]), strict=True))
    total = ds.count()
    names = [f"column_{i}" for i in range(total)] if column_names is None else list(column_names)
    if len(names) != total:
        raise PlanError(
            f"transpose(): column_names gives {len(names)} name(s) for {total} row(s); "
            "give exactly one name per row"
        )
    return ranked, rank, list(zip(range(1, total + 1), names, strict=True))


def _check_names(values: list[Any], column: str) -> None:
    """Refuse a naming column that has a null or a repeated value, either of which loses data."""
    if any(v is None for v in values):
        raise PlanError(f"transpose(): column_names column {column!r} holds a null")
    if len(set(values)) != len(values):
        raise PlanError(
            f"transpose(): column_names column {column!r} repeats a value, and two rows cannot "
            "become one column"
        )


def build_transpose(
    ds: Dataset,
    column_names: str | Sequence[str] | None,
    include_header: bool,
    header_name: str,
    order_by: OrderSpec | None,
    descending: bool | Sequence[bool],
) -> Dataset:
    """Turn rows into columns (see `Dataset.transpose`).

    Args:
        ds: The relation to transpose.
        column_names: A column whose values name the output columns, explicit names, or
            ``None`` for ``column_0``, ``column_1``, ...
        include_header: Keep the column holding each input column's name.
        header_name: That column's name.
        order_by: The row order that orders the output columns.
        descending: Order every key, or each one, largest first.

    Returns:
        One row per transposed input column.

    Raises:
        PlanError: On an empty input, an unusable naming column, or a missing order.
    """
    rank = unused_name("__bc_transpose_rank", ds)
    source, key, labels = _labels(ds, column_names, order_by, descending, rank)
    value_cols = [
        c for c in ds.columns if not (isinstance(column_names, str) and c == column_names)
    ]
    if not labels or not value_cols:
        raise PlanError("transpose(): the input has no rows or no value columns to transpose")
    header = unused_name("__bc_transpose_header", ds)
    value = unused_name("__bc_transpose_value", ds)
    target = _common_value_type(ds, value_cols)
    long = source.select(Col(key), *(Col(c).cast(target).alias(c) for c in value_cols))
    long = long.unpivot(index=[key], on=value_cols, variable_name=header, value_name=value)
    wide = long.pivot(
        index=[header], on=key, values=value, aggregate="max", columns=[k for k, _ in labels]
    )
    # The pivot groups by the header, so its rows come back in no particular order. Each
    # row is one input column, and the input's column order is the order to restore.
    position = unused_name("__bc_transpose_pos", ds)
    order = when(Col(header) == value_cols[0]).then(lit(0))
    for i, c in enumerate(value_cols[1:], start=1):
        order = order.when(Col(header) == c).then(lit(i))
    outputs = [Col(str(k)).alias(name) for k, name in labels]
    head = [Col(header).alias(header_name)] if include_header else []
    positioned = wide.with_columns(**{position: order.otherwise(lit(len(value_cols)))})
    return positioned.sort(position).select(*head, *outputs)


def _key_predicate(column: str, value: Any) -> Expr:
    """The filter selecting one key value, with a null and a NaN matched as themselves."""
    if value is None:
        return Col(column).is_null()
    if isinstance(value, float) and math.isnan(value):
        return coalesce(Col(column).is_nan(), lit(False))
    return Col(column) == value


def _sort_key(row: tuple[Any, ...]) -> tuple[Any, ...]:
    """Order key tuples with nulls last and NaN after every number, as a sort does."""
    return tuple(
        (v is None, isinstance(v, float) and math.isnan(v), 0 if v is None else v) for v in row
    )


def build_partition_by(ds: Dataset, keys: list[str], include_key: bool) -> dict[tuple, Dataset]:
    """Split `ds` into one lazy dataset per distinct key (see `Dataset.partition_by`).

    Args:
        ds: The relation to split.
        keys: The key columns.
        include_key: Keep the key columns in each part.

    Returns:
        Key tuple to that key's rows, in ascending key order with nulls last.

    Raises:
        PlanError: If no key is given, or a key is not a column.
    """
    if not keys:
        raise PlanError("partition_by() needs at least one key column")
    unknown = [k for k in keys if k not in ds.columns]
    if unknown:
        raise PlanError(f"partition_by(): unknown column(s) {unknown}")
    found = ds.select(*keys).distinct().to_pydict()
    rows = sorted(zip(*(found[k] for k in keys), strict=True), key=_sort_key)
    kept = ds.columns if include_key else [c for c in ds.columns if c not in set(keys)]
    parts: dict[tuple, Dataset] = {}
    for row in rows:
        predicate = _key_predicate(keys[0], row[0])
        for k, v in zip(keys[1:], row[1:], strict=True):
            predicate = predicate & _key_predicate(k, v)
        parts[row] = ds.filter(predicate).select(*kept)
    return parts


def build_split(
    ds: Dataset, n: int, order_by: OrderSpec, descending: bool | Sequence[bool], equal: bool
) -> list[Dataset]:
    """Cut `ds` into `n` contiguous parts under `order_by` (see `Dataset.split`).

    Args:
        ds: The relation to split.
        n: How many parts.
        order_by: The ordering keys a row's position is taken under.
        descending: Order every key, or each one, largest first.
        equal: Give every part exactly ``count // n`` rows, dropping the remainder.

    Returns:
        `n` lazy datasets, in position order.

    Raises:
        PlanError: If `n` is not a positive integer or `order_by` is empty.
    """
    n = require_int(n, func="split", arg="n", minimum=1)
    pos = unused_name("__bc_split_pos", ds)
    ranked = rank_rows(ds, order_by, descending, name=pos, api="split")
    total = ds.count()
    base, extra = divmod(total, n)
    sizes = [base] * n if equal else [base + (1 if i < extra else 0) for i in range(n)]
    parts: list[Dataset] = []
    start = 0
    for size in sizes:
        keep = (Col(pos) > start) & (Col(pos) <= start + size)
        parts.append(ranked.filter(keep).sort(pos).drop(pos))
        start += size
    return parts
