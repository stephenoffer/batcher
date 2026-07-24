"""Per-**file** Parquet bounds, in the add-action layout, for file-level pruning.

The sibling of `columnar_footer`, split out along a responsibility seam: that module
collapses a dataset's footers into one `SourceStatistics` for costing and metadata answers,
while this one keeps a row per file so a consumer can decide *which files to open at all*.

The layout is the one `io.stats.file_skipping` defines and a lakehouse log already
publishes — ``path | num_records | min.<col> | max.<col> | null_count.<col>`` — so a plain
Parquet directory is pruned by exactly the code that prunes a Delta table from its
transaction log. That is what lets a copy-on-write ``MERGE`` skip files on an ordinary
directory target with no transaction log to consult (`io.stats.key_pruning`).

Only footers are read — one small metadata GET per file, never a data page. A file whose
footer is unreadable, or which records no statistic for a column, yields NULL bounds, which
every consumer MUST treat as *unknown* (keep the file), never as *no match*.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher._internal.mathx import is_nan
from batcher.io.stats.columnar_footer import _read_footers

__all__ = ["parquet_file_manifest"]


def _native_manifest(files: list[str], columns: list[str]) -> pa.Table | None:
    """The manifest via the native footer walk, or None to use the Python path below."""
    from batcher.io.formats.structured import _parquet_native

    return _parquet_native.file_manifest(files, columns)


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
    if not files:
        return None

    # Native first: the Python walk below builds a pybind11 object per row-group per column,
    # and — the larger cost — re-reads footers the statistics pass has already parsed and
    # cached on the Rust side. On 1,000 files that duplicate read was ~95% of the total
    # (1,835 ms -> 171 ms cold). Declines the same way `_native_statistics` does: no native
    # target, or any native failure, falls through to the walk below.
    target = getattr(fs, "native_read_target", None)
    if target is not None and target(files[0]) is not None:
        native = _native_manifest(files, columns)
        if native is not None:
            return native
    return _manifest_from_footers(fs, files, columns)


def _manifest_from_footers(fs: Any, files: list[str], columns: list[str]) -> pa.Table | None:
    """The manifest built by walking each footer from Python — the portable fallback."""
    import pyarrow.parquet as pq

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
            if is_nan(stats.min) or is_nan(stats.max):
                known = False  # an unordered bound cannot prune
                break
            low = stats.min if low is None else min(low, stats.min)
            high = stats.max if high is None else max(high, stats.max)
            if getattr(stats, "has_null_count", False):
                null_count += stats.null_count or 0
        out[name] = (low, high, null_count) if known else (None, None, None)
    return out
