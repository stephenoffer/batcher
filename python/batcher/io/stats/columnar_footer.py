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

from batcher._internal.hardware import available_cpu_count
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance

__all__ = [
    "is_exact_minmax_type",
    "orc_statistics",
    "parquet_file_manifest",
    "parquet_statistics",
]


def _read_footers(fs: Any, files: list[str], pq: Any) -> list[Any]:
    """Each file's Parquet metadata (footer), read concurrently, in file order.

    The footer read is one object-store round trip whose C++ parse releases the GIL, so
    a many-file dataset must not read them one at a time (that serial loop is the bulk of
    a small-many-files query's wall clock). A per-file failure maps to ``None`` (skipped
    by the caller) rather than failing the whole statistics pass.

    Concurrency uses pyarrow's **native** filesystem read (`read_metadata(path,
    filesystem=…)`) when the backend exposes one — that reads the footer C++-side without a
    Python file handle, which is both faster and thread-safe under fan-out (a buffered
    Python handle per thread is not). Backends with no native target fall back to the
    handle, read serially to stay safe.
    """
    target = getattr(fs, "native_read_target", None)

    def _native(path: str) -> Any:
        try:
            resolved = target(path) if target is not None else None
            if resolved is None:
                return _handle(path)
            native_fs, in_path = resolved
            return pq.read_metadata(in_path, filesystem=native_fs)
        except Exception:
            return None

    def _handle(path: str) -> Any:
        try:
            with fs.open(path) as fh:
                return pq.ParquetFile(fh).metadata
        except Exception:
            return None

    if len(files) <= 1:
        return [_native(files[0])] if files else []
    # No native target → the buffered-handle path is not safe to fan out, so read serially.
    if target is None or target(files[0]) is None:
        return [_handle(p) for p in files]
    from concurrent.futures import ThreadPoolExecutor

    # Footer reads are latency-bound (network round trips), not CPU-bound, so oversubscribe
    # the cores — many small concurrent GETs saturate object-store bandwidth where
    # one-per-core would stall on latency — but cap the pool so a huge dataset does not
    # open thousands of connections at once.
    workers = min(len(files), max(8, available_cpu_count() * 2), 64)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_native, files))  # order preserved


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
    # Reading each file's footer is one object-store round trip (~40 ms), and over a
    # many-file dataset a serial loop dominates a query's wall clock (100 small Parquet
    # files ≈ 4 s of pure footer I/O — more than the data read itself). The footer read
    # releases the GIL in the C++ layer, so fan it across a thread pool: the metadata
    # objects come back in file order, then the (cheap, CPU-only) accumulation stays
    # serial. Best-effort per file is preserved — an unreadable footer maps to None.
    for meta in _read_footers(fs, files, pq):
        if meta is None:
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

    from batcher.io._concurrent import read_each_file

    if not files:
        return None

    def _nrows(filesystem: Any, path: str) -> int:
        with filesystem.open(path) as fh:
            return orc.ORCFile(fh).nrows

    try:
        counts = read_each_file(fs, files, _nrows)  # concurrent footer reads, in order
    except Exception:
        return None  # any unreadable footer → not an exact count
    return SourceStatistics(row_count=sum(counts), exact_rows=True)


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


def parquet_file_manifest(fs: Any, files: list[str], columns: list[str]) -> pa.Table | None:
    """Per-file bounds for `columns`, in the add-action layout, scraped from the footers.

    The layout is the one `io.stats.file_skipping` defines and a lakehouse log already
    publishes — ``path | num_records | min.<col> | max.<col> | null_count.<col>`` — so a
    plain Parquet directory can be pruned by the same code that prunes a Delta table from
    its transaction log. That is what lets a copy-on-write ``MERGE`` skip files on an
    ordinary directory target, with no transaction log to consult
    (`io.stats.key_pruning`).

    Only the footer is read — one small metadata GET per file, fanned out concurrently and
    never a data page. A file whose footer is unreadable, or which records no statistic for
    a column, yields NULL bounds, which every consumer must treat as *unknown* (keep the
    file), never as *no match*.

    Args:
        fs: Filesystem exposing `open` (and optionally `native_read_target`).
        files: The data files to describe.
        columns: The columns whose bounds to record — for a merge, the join keys.

    Returns:
        The per-file manifest, or None if `files` is empty or no footer could be read.
    """
    import pyarrow.parquet as pq

    if not files:
        return None

    paths: list[str] = []
    rows: list[int | None] = []
    lows: dict[str, list[Any]] = {c: [] for c in columns}
    highs: dict[str, list[Any]] = {c: [] for c in columns}
    nulls: dict[str, list[int | None]] = {c: [] for c in columns}

    saw_any = False
    for path, meta in zip(files, _read_footers(fs, files, pq), strict=True):
        paths.append(path)
        if meta is None:
            # An unreadable footer proves nothing: NULL bounds keep the file.
            rows.append(None)
            for c in columns:
                lows[c].append(None)
                highs[c].append(None)
                nulls[c].append(None)
            continue
        saw_any = True
        rows.append(meta.num_rows)
        bounds = _file_bounds(meta, columns)
        for c in columns:
            low, high, null_count = bounds[c]
            lows[c].append(low)
            highs[c].append(high)
            nulls[c].append(null_count)

    if not saw_any:
        return None

    data: dict[str, Any] = {"path": paths, "num_records": rows}
    for c in columns:
        data[f"min.{c}"] = lows[c]
        data[f"max.{c}"] = highs[c]
        data[f"null_count.{c}"] = nulls[c]
    try:
        return pa.table(data)
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        return None  # bounds that will not unify into one column type prune nothing


def _file_bounds(meta: Any, columns: list[str]) -> dict[str, tuple[Any, Any, int | None]]:
    """One file's ``(min, max, null_count)`` per column, aggregated over its row groups.

    A column missing statistics in *any* row group has unknown bounds for the whole file
    (a partial min/max would be a bound over only part of the data — exactly the unsound
    prune this must never produce). A NaN bound is unordered, so it poisons the column's
    interval the same way it does in `parquet_statistics`.
    """
    out: dict[str, tuple[Any, Any, int | None]] = dict.fromkeys(columns, (None, None, None))
    names = meta.schema.names
    wanted = {c: names.index(c) for c in columns if c in names}
    if not wanted:
        return out

    for name, index in wanted.items():
        low = high = None
        null_count = 0
        known = True
        for rg in range(meta.num_row_groups):
            stats = meta.row_group(rg).column(index).statistics
            if stats is None or not getattr(stats, "has_min_max", False):
                known = False
                break
            if _is_nan(stats.min) or _is_nan(stats.max):
                known = False  # an unordered bound cannot prune
                break
            low = stats.min if low is None else min(low, stats.min)
            high = stats.max if high is None else max(high, stats.max)
            if getattr(stats, "has_null_count", False):
                null_count += stats.null_count or 0
        out[name] = (low, high, null_count) if known else (None, None, None)
    return out
