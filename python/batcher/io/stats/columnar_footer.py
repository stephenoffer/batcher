"""Footer-derived statistics for columnar formats (Parquet, ORC, Arrow IPC).

These formats already carry per-column min/max/null-count in a footer the IO
layer opens anyway (for schema reads and split planning). This module mines that
footer into a `SourceStatistics` *without scanning a single row*, so the
estimator and the metadata-answer layer can prune predicates, skip files, and
answer `count()` / `min()` / `max()` / `null_count()` for free.

Provenance discipline is the load-bearing rule here — an `EXACT` footer stat must
be *provably* exact, because a downstream terminal answers from it without
executing:

  - Numeric/temporal/bool min/max is the true extreme aggregated across row
    groups (min of chunk mins, max of chunk maxs) → `EXACT`.
  - A string/binary min/max may be writer-*truncated* (Parquet caps long values),
    so it is a valid pruning bound but must never answer an exact `max()` →
    `DEFAULT`.
  - `null_count` is summed exactly across row groups (Parquet records it per
    chunk) → carried on the `EXACT` bundle.
  - A float column whose footer min/max is `NaN` is not soundly ordered, so its
    bounds are dropped (null_count stays exact); an exact `min()`/`max()` then
    falls back to execution rather than returning a wrong extreme.
  - Parquet's `distinct_count` is an *estimate* (the format does not guarantee
    exactness), so it is never placed on an `EXACT` column — that would let an
    approximate distinct wrongly answer `count_distinct`. It is kept only on
    already-inexact columns, where it can inform cost / `approx_count_distinct`.
"""

from __future__ import annotations

import math
from typing import Any

import pyarrow as pa

from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance

__all__ = ["is_exact_minmax_type", "orc_statistics", "parquet_statistics"]

# Arrow types whose footer/manifest min/max is the exact value (never truncated).
_EXACT_MINMAX_TYPES = (
    pa.types.is_integer,
    pa.types.is_floating,
    pa.types.is_decimal,
    pa.types.is_boolean,
    pa.types.is_date,
    pa.types.is_timestamp,
    pa.types.is_time,
    pa.types.is_duration,
)


def is_exact_minmax_type(arrow_type: pa.DataType) -> bool:
    """True iff a footer/manifest min/max for `arrow_type` is exact (untruncated).

    Numeric, decimal, boolean, and temporal columns record their true extremes;
    string/binary columns may be byte-truncated by the writer and are excluded.
    Shared by the Parquet footer and lakehouse-manifest extractors so both apply
    one provenance rule.
    """
    return any(pred(arrow_type) for pred in _EXACT_MINMAX_TYPES)


def _is_nan(value: Any) -> bool:
    """True iff `value` is a floating NaN (unordered → an unusable min/max bound)."""
    return isinstance(value, float) and math.isnan(value)


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


def orc_statistics(fs: Any, files: list[str]) -> SourceStatistics | None:
    """Exact row count across one or more ORC files from their footers, no scan.

    An ORC footer records `nrows` per file (the loader reads it anyway for split
    planning), so summing it answers `count()` for free. Column min/max are not
    surfaced here — pyarrow's ORC reader does not expose per-stripe column
    statistics through a stable API. Returns None if any footer is unreadable, so
    a partial sum is never reported as an exact count.
    """
    import pyarrow.orc as orc

    total = 0
    saw_any = False
    for path in files:
        try:
            with fs.open(path) as fh:
                total += orc.ORCFile(fh).nrows
        except Exception:
            return None
        saw_any = True
    if not saw_any:
        return None
    return SourceStatistics(row_count=total, exact_rows=True)


class _ColAcc:
    """Mutable per-column accumulator over row-group statistics."""

    __slots__ = (
        "count",
        "distinct",
        "has_stats",
        "max",
        "min",
        "nan_seen",
        "null_count",
        "null_known",
    )

    def __init__(self) -> None:
        self.min = None
        self.max = None
        self.null_count: int = 0
        self.null_known: bool = True
        self.count: int = 0
        self.distinct: int | None = None
        self.has_stats: bool = False
        self.nan_seen: bool = False


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
        cmin, cmax = stats.min, stats.max
        if _is_nan(cmin) or _is_nan(cmax):
            # A NaN bound is unordered; poison this column's min/max (kept null so
            # an exact min()/max() falls back rather than returning a wrong value).
            acc.nan_seen = True
        else:
            acc.min = cmin if acc.min is None else min(acc.min, cmin)
            acc.max = cmax if acc.max is None else max(acc.max, cmax)
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
        exact_minmax = arrow_type is not None and is_exact_minmax_type(arrow_type)
        # Exact only for non-truncatable numeric/temporal min/max with known nulls
        # and no NaN poisoning (which would make the bound unordered).
        is_exact = exact_minmax and a.null_known and not a.nan_seen
        cmin, cmax = (None, None) if a.nan_seen else (a.min, a.max)
        # Parquet's distinct_count is an estimate; never expose it on an EXACT
        # column (that would let it answer an exact count_distinct). Keep it only
        # on already-inexact columns for cost / approx_count_distinct, and only
        # from a single row group (per-chunk counts are not additive).
        ndv = (
            float(a.distinct)
            if (not is_exact and single_row_group and a.distinct is not None)
            else None
        )
        columns[name] = ColumnStat(
            min=cmin,
            max=cmax,
            null_count=float(a.null_count) if a.null_known else None,
            ndv=ndv,
            provenance=Provenance.EXACT if is_exact else Provenance.DEFAULT,
        )
    return columns
