"""Footer-derived statistics for columnar formats (Parquet, ORC, Arrow IPC).

These formats already carry per-column min/max/null-count in a footer the IO
layer opens anyway (for schema reads and split planning). This module mines that
footer into a `SourceStatistics` *without scanning a single row*, so the
estimator and the metadata-answer layer can prune predicates, skip files, and
answer `count()` / `min()` / `max()` for free.

Provenance discipline matters here. A numeric column's footer min/max is the
true extreme → `EXACT`. A string/binary min/max may be *byte-truncated* by the
writer (Parquet caps long values), so it is a valid bound for pruning but must
not answer an exact `max()` → tagged `DEFAULT`. Distinct counts are not additive
across row groups, so ndv is reported only when a single row group records it.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance

__all__ = ["parquet_statistics"]

# Arrow types whose footer min/max is the exact value (never truncated).
_EXACT_MINMAX_TYPES = (
    pa.types.is_integer,
    pa.types.is_floating,
    pa.types.is_decimal,
    pa.types.is_boolean,
    pa.types.is_date,
    pa.types.is_timestamp,
    pa.types.is_time,
)


def _minmax_is_exact(arrow_type: pa.DataType) -> bool:
    return any(pred(arrow_type) for pred in _EXACT_MINMAX_TYPES)


def parquet_statistics(fs: Any, files: list[str], schema: pa.Schema) -> SourceStatistics | None:
    """Aggregate footer statistics across one or more Parquet files.

    `fs` is a filesystem with an `open(path)` context manager; `files` are the
    paths to mine; `schema` is the dataset schema used to type column min/max
    provenance. Returns None if no footer could be read (the estimator then falls
    back to its defaults). Best-effort: any per-file failure is skipped rather
    than raised.
    """
    import pyarrow.parquet as pq

    total_rows = 0
    total_bytes = 0
    row_group_count = 0
    # Per column: accumulate global min, global max, summed null_count, and whether
    # every contributing chunk reported a null_count (else null_count is unknown).
    acc: dict[str, _ColAcc] = {}
    saw_any = False
    for path in files:
        try:
            with fs.open(path) as fh:
                meta = pq.ParquetFile(fh).metadata
        except Exception:
            continue
        saw_any = True
        total_rows += meta.num_rows
        names = meta.schema.names
        for rg in range(meta.num_row_groups):
            row_group_count += 1
            rgroup = meta.row_group(rg)
            total_bytes += rgroup.total_byte_size
            for ci in range(rgroup.num_columns):
                col = rgroup.column(ci)
                name = names[ci] if ci < len(names) else col.path_in_schema
                _accumulate(acc.setdefault(name, _ColAcc()), col)
    if not saw_any:
        return None
    columns = _finalize_columns(acc, schema, single_row_group=row_group_count == 1)
    return SourceStatistics(
        row_count=total_rows,
        byte_size=total_bytes or None,
        columns=columns,
        exact_rows=True,
    )


class _ColAcc:
    """Mutable per-column accumulator over row-group statistics."""

    __slots__ = ("count", "distinct", "has_stats", "max", "min", "null_count", "null_known")

    def __init__(self) -> None:
        self.min = None
        self.max = None
        self.null_count: int = 0
        self.null_known: bool = True
        self.count: int = 0
        self.distinct: int | None = None
        self.has_stats: bool = False


def _accumulate(acc: _ColAcc, column) -> None:
    stats = getattr(column, "statistics", None)
    acc.count += column.num_values or 0
    if stats is None:
        acc.null_known = False
        return
    acc.has_stats = True
    if getattr(stats, "has_null_count", False):
        acc.null_count += stats.null_count or 0
    else:
        acc.null_known = False
    if getattr(stats, "has_min_max", False):
        acc.min = stats.min if acc.min is None else min(acc.min, stats.min)
        acc.max = stats.max if acc.max is None else max(acc.max, stats.max)
    if getattr(stats, "distinct_count", None) is not None:
        acc.distinct = stats.distinct_count


def _finalize_columns(
    acc: dict[str, _ColAcc], schema: pa.Schema, *, single_row_group: bool
) -> dict[str, ColumnStat]:
    columns: dict[str, ColumnStat] = {}
    for name, a in acc.items():
        if not a.has_stats:
            continue
        arrow_type = schema.field(name).type if name in schema.names else None
        exact_minmax = arrow_type is not None and _minmax_is_exact(arrow_type)
        # ndv is only trustworthy when a single row group recorded it (not additive).
        ndv = float(a.distinct) if (single_row_group and a.distinct is not None) else None
        columns[name] = ColumnStat(
            min=a.min,
            max=a.max,
            null_count=float(a.null_count) if a.null_known else None,
            ndv=ndv,
            # Exact only for non-truncatable numeric/temporal min/max with known nulls.
            provenance=Provenance.EXACT if (exact_minmax and a.null_known) else Provenance.DEFAULT,
        )
    return columns
