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

from typing import Any

import pyarrow as pa

from batcher._internal.mathx import is_nan
from batcher.io._concurrent import read_each_file
from batcher.io.stats.sortedness import proved_sorted_by
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance

__all__ = [
    "is_exact_minmax_type",
    "orc_statistics",
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
    Python file handle, which is both faster than a handle read.

    Backends with no native target — an fsspec scheme, or any backend with a read-through
    byte cache configured, since `native_read_target` withholds itself to avoid bypassing
    that cache — read through a handle, and **also concurrently**. This used to be a serial
    loop, justified by "a buffered Python handle is not thread-safe under fan-out". That is
    true of *sharing one handle* across threads and irrelevant here: each task calls
    `fs.open(path)` and gets its own. The effect of the serial loop was that turning on
    `file_cache_dir` silently converted every footer read in the dataset into a
    one-at-a-time walk — a config meant to make reads faster made the metadata pass N times
    slower, with nothing to indicate it.
    """
    target = getattr(fs, "native_read_target", None)

    def _handle(_fs: Any, path: str) -> Any:
        try:
            with _fs.open(path) as fh:
                return pq.ParquetFile(fh).metadata
        except Exception:
            return None

    def _native(_fs: Any, path: str) -> Any:
        try:
            resolved = target(path) if target is not None else None
            if resolved is None:
                return _handle(_fs, path)
            native_fs, in_path = resolved
            return pq.read_metadata(in_path, filesystem=native_fs)
        except Exception:
            return None

    if not files:
        return []
    # `read_each_file` owns the fan-out and its cap for every metadata extractor, so the
    # concurrency policy lives in one place rather than being restated per call site.
    return read_each_file(fs, files, _native)


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


def _native_accumulators(stats: Any) -> dict[str, _ColAcc]:
    """The native pass's per-column result, shaped as the Python accumulator's `_ColAcc`.

    Exists so both paths finish through the same `_finalize_columns`: provenance, NaN
    poisoning, and the distinct-count rule then have one implementation that neither path
    can drift away from.

    The bounds table is indexed by *position* through a name map built once. Both
    ``column_names`` and ``column(name)`` are linear in the table's width, so asking them
    per column made this loop quadratic — which is invisible on a ten-column table and the
    whole cost on a wide one: 512 columns spent 225 ms here against 18 ms in the native
    footer walk it was reading the results of.
    """
    bounds = stats.bounds
    position = {name: i for i, name in enumerate(bounds.column_names)}
    acc: dict[str, _ColAcc] = {}
    for name, has_stats, null_count, null_known, nan_seen, distinct in stats.columns:
        entry = _ColAcc()
        entry.has_stats = has_stats
        entry.null_count = null_count
        entry.null_known = null_known
        entry.nan_seen = nan_seen
        entry.distinct = distinct
        index = position.get(name)
        if index is not None:
            # Row 0 is the global min and row 1 the global max, each still carrying the
            # column's own Arrow type — so this is a typed read-out, not a parse.
            column = bounds.column(index)
            entry.min = column[0].as_py()
            entry.max = column[1].as_py()
        acc[name] = entry
    return acc


def _native_statistics(fs: Any, files: list[str], schema: pa.Schema) -> SourceStatistics | None:
    """`parquet_statistics` via the native footer walk, or None to use the Python path.

    The Python accumulator below constructs a pybind11 object per *column chunk*, so its
    cost is O(files x row_groups x columns) interpreter work on the driver before a single
    data page is read — on 200 files x 20 row-groups x 30 columns that measured ~750 ms on
    top of ~95 ms of actual footer I/O. The native pass does the same walk in Rust over
    footers the reader has usually already parsed and cached.

    It declines (returns None, caller falls through) in four cases, each of which keeps the
    slower path's behavior rather than approximating it:

    - the backend exposes no native read target (fsspec, or a read-through byte cache),
      so the Rust object store cannot address these files;
    - any footer was unreadable, so the row count would cover only part of the dataset;
    - the files declare `sorting_columns`, meaning a global-sortedness claim is *possible*
      and must be settled by the proof in `io.stats.sortedness` rather than assumed away;
    - anything at all went wrong natively (`footer_stats` returns None).

    Finalization deliberately routes back through `_finalize_columns` — the same function
    the Python path uses — so provenance, NaN poisoning, and the distinct-count rule have
    exactly one implementation and the two paths cannot drift apart.
    """
    if not files:
        return None
    target = getattr(fs, "native_read_target", None)
    if target is None or target(files[0]) is None:
        return None

    from batcher.io.formats.structured import _parquet_native

    stats = _parquet_native.footer_stats(files)
    if stats is None or stats.files_read != len(files):
        return None
    if stats.sort_declared:
        # Sortedness *may* be provable here. Proving it needs per-row-group bounds ordering
        # within and across files, and a wrong claim deletes a Sort and silently reorders
        # the user's rows — so hand these (rare) datasets to the implementation that does
        # the full proof instead of duplicating it.
        return None

    return SourceStatistics(
        row_count=stats.total_rows,
        byte_size=stats.total_bytes or None,
        columns=_finalize_columns(
            _native_accumulators(stats), schema, single_row_group=stats.row_group_count == 1
        ),
        exact_rows=True,
        row_group_count=stats.row_group_count or None,
        # Proved above to be unclaimable: every file was read and none declared a sort key.
        sorted_by=(),
    )


def parquet_statistics(fs: Any, files: list[str], schema: pa.Schema) -> SourceStatistics | None:
    """Aggregate footer statistics across one or more Parquet files.

    `fs` is a filesystem with an `open(path)` context manager; `files` are the
    paths to mine; `schema` is the dataset schema used to type column min/max
    provenance. Returns None if no footer could be read (the estimator then falls
    back to its defaults). Best-effort: any per-file failure is skipped rather
    than raised.
    """
    import pyarrow.parquet as pq

    native = _native_statistics(fs, files, schema)
    if native is not None:
        return native

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
    metadatas = list(_read_footers(fs, files, pq))
    for meta in metadatas:
        if meta is None:
            continue
        saw_any = True
        total_rows += meta.num_rows
        for rg in range(meta.num_row_groups):
            row_group_count += 1
            rgroup = meta.row_group(rg)
            total_bytes += rgroup.total_byte_size
            for ci in range(rgroup.num_columns):
                col = rgroup.column(ci)
                # `path_in_schema`, **not** `schema.names[ci]`. Parquet stores one column
                # chunk per *leaf*, and `ParquetSchema.names` reports each leaf's bare field
                # name while `path_in_schema` reports its full dotted path. For a flat table
                # the two agree; for a nested one they do not, and the bare name silently
                # merges a struct field's bounds into whatever top-level column shares its
                # name.
                #
                # Measured on a table with a top-level `a` of 1..3 beside a struct `s{a}` of
                # 1000..3000: this path reported `a` in [1, 3000]. Those bounds carry
                # `Provenance.EXACT` for a numeric column, which is the provenance that lets
                # Kyber answer `max(a)` from metadata without reading the data — so the
                # collision is a **wrong answer**, not a loose estimate. The native footer
                # walk keys by the path and reported [1, 3] for the same file, so the two
                # implementations of one statistic disagreed, and the Python one is the
                # fallback that runs whenever the native reader declines (an fsspec backend,
                # a read-through cache, a declared `sorting_columns`).
                _accumulate(acc.setdefault(col.path_in_schema, _ColAcc()), col)
    if not saw_any:
        return None
    columns = _finalize_columns(acc, schema, single_row_group=row_group_count == 1)
    return SourceStatistics(
        row_count=total_rows,
        byte_size=total_bytes or None,
        columns=columns,
        exact_rows=True,
        row_group_count=row_group_count or None,
        # Only ever set when the footers *prove* global order (see `io.stats.sortedness`).
        # Kyber deletes a redundant `Sort` on the strength of this, so a wrong claim is a
        # wrong output order rather than a slower plan.
        sorted_by=proved_sorted_by(metadatas),
        # Deliberately left at its default (False): the Parquet spec omits NaN from a
        # column's min/max, so a footer `max` is the largest *non-NaN* value while SQL
        # ranks NaN greatest. These bounds may prune and may answer `min`, never `max`.
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
        if is_nan(cmin) or is_nan(cmax):
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
    # `schema.names` builds a fresh list of every field name on each access, so consulting
    # it inside this loop is quadratic in the table's width — the same wide-table cost
    # `_native_accumulators` documents. Resolve the types once instead.
    field_types = {field.name: field.type for field in schema}
    for name, a in acc.items():
        if not a.has_stats:
            continue
        arrow_type = field_types.get(name)
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
            # Parquet records a null count per column chunk **for every type**, and summing them
            # across chunks is exact. That is independent of whether the *bounds* are trustworthy:
            # a string column's min/max may be writer-truncated (so the bundle is DEFAULT), but
            # its null count is not an estimate. Tagging it separately is what lets
            # `n_null("name")` / `null_count()` / `count(name)` / `dq.not_null("name")` be
            # answered from the footer on precisely the columns most real tables are made of —
            # before this, the exact answer was discarded because it sat next to an inexact one.
            null_count_provenance=Provenance.EXACT if a.null_known else Provenance.DEFAULT,
        )
    return columns
