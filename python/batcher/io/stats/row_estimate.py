"""Advisory row-count estimates from a sample, for datasets too large to count exactly.

A CSV or line-delimited JSON file carries no row count anywhere — the only exact answer is
a full scan, which is precisely what the estimator exists to avoid. But a cheap, *advisory*
estimate is far better than the planner's blind default: read a small sample from the first
file, measure its average bytes-per-row, and extrapolate against the dataset's total on-disk
size. DuckDB and Spark both size a CSV read this way.

The estimate is always `exact_rows=False` (a variable-width or quoted-newline file makes it
approximate), so it sizes joins and the worker fan-out but never answers an exact ``count()``.
It reads O(1) bytes (one sample, not the file), so it is a plan-time cost, not a scan. A
compressed file is skipped: its on-disk size is the *compressed* size, whose ratio to the row
count is the compression ratio, not a row width — extrapolating from it would be wildly wrong.

A **columnar** dataset has the opposite problem and the same answer. Every file states its
row count exactly, in its footer, and that is precisely why a large one cannot be counted:
the exact answer costs one metadata round trip per file, so a million-file table spends
twenty minutes of driver time before a task launches, and every scan path already refuses
that sweep. `estimate_rows_from_footer_sample` reads a *fixed* number of footers instead and
scales their rows-per-byte by the dataset's total size, so a table reports a usable size to
the planner at any file count for a price that does not grow with it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from batcher.io._concurrent import listed_sizes, read_each_file, total_file_bytes
from batcher.io.detect import compression_for_path

__all__ = ["estimate_delimited_rows", "estimate_rows_from_footer_sample"]

#: Footers read to estimate a large columnar dataset's row count. A constant, not a
#: fraction: the error of a mean is set by the *sample* size, not the population, so 64
#: footers describe a ten-million-file table as well as they describe a twenty-thousand-file
#: one — and for the same price, which is the entire point. Wide enough that one unusual
#: file cannot dominate, small enough to stay a sub-second plan-time cost on an object store.
_FOOTER_SAMPLE_FILES = 64

# Bytes sampled from the first file to measure average row width. 1 MiB spans enough rows
# that one long row does not skew the mean, while staying a single small read.
_SAMPLE_BYTES = 1 << 20


def estimate_delimited_rows(
    fs: Any,
    files: list[str],
    *,
    has_header: bool,
    total_bytes: int | None = None,
    sample_bytes: int = _SAMPLE_BYTES,
) -> int | None:
    """Estimate the total rows of a newline-delimited dataset from a byte sample, or None.

    Reads up to `sample_bytes` from the first file, counts its newlines to derive an average
    bytes-per-row, and scales by the dataset's total on-disk size. The header row is
    discounted once per file when `has_header` is set. Returns None when it cannot produce a
    meaningful estimate: no files, a compressed first file (whose size does not track row
    width), an empty sample, or a sample with no newline to measure a row from.

    Args:
        fs: A filesystem with `open(path)` and `size(path)`.
        files: The dataset's files, in order. The first is sampled.
        has_header: Whether each file's first line is a header (discounted from the estimate).
        total_bytes: The dataset's on-disk size, when the caller already computed it (as
            `FileSource.statistics` does). Passing it avoids a second O(files) `size` sweep
            of the same files — the whole point of threading it through. Computed here when
            omitted.
        sample_bytes: How many bytes to sample for the row-width measurement.

    Returns:
        An advisory total row count (>= 0), or None when no estimate is possible.
    """
    if not files or compression_for_path(files[0]) is not None:
        return None
    try:
        with fs.open(files[0]) as fh:
            sample = fh.read(sample_bytes)
    except Exception:
        return None
    if not sample:
        return None
    newlines = sample.count(b"\n")
    if newlines <= 0:
        return None
    avg_row_bytes = len(sample) / newlines
    if total_bytes is None:
        total_bytes = total_file_bytes(fs, files)
    if total_bytes is None or avg_row_bytes <= 0:
        return None
    estimate = int(total_bytes / avg_row_bytes)
    if has_header:
        estimate -= len(files)  # one header line per file is not a data row
    return max(estimate, 0)


def estimate_rows_from_footer_sample(
    fs: Any,
    files: list[str],
    count_rows: Callable[[str], int | None],
    *,
    total_bytes: int | None = None,
    sample_files: int = _FOOTER_SAMPLE_FILES,
) -> int | None:
    """Estimate a columnar dataset's rows from a fixed sample of its footers, or None.

    Reads `sample_files` footers spread evenly across `files`, measures their combined
    rows-per-byte, and scales it by the dataset's total on-disk size. The cost is constant
    in the file count, which is what makes it usable at the scale where the exact count is
    not: a table of a million Parquet files reports a row count to the planner for 64
    metadata round trips instead of a million.

    Evenly spread rather than taken from the front, because the front of a listing is one
    partition. A date-partitioned table's first sixty-four files are all the same day, and
    a day is exactly the thing whose size varies — sampling the head would estimate the
    whole table from its quietest morning.

    The result is always advisory. The caller marks it ``exact_rows=False`` so it sizes
    joins, spill budgets, and the worker fan-out but never answers a ``count()``.

    Args:
        fs: The filesystem the files were listed through.
        files: Every file in the dataset, in listing order.
        count_rows: Reads one file's exact row count from its footer, or returns None.
        total_bytes: The dataset's on-disk size when the caller already has it; computed
            here otherwise.
        sample_files: How many footers to read.

    Returns:
        An advisory total row count (>= 0), or None when no estimate is possible — too few
        files to be worth estimating, no readable footer, or no total size to scale by.
    """
    if len(files) <= sample_files:
        return None  # the caller can afford the exact count at this size, and prefers it
    sizes = listed_sizes(fs, files)
    if total_bytes is None:
        total_bytes = sum(sizes) if sizes is not None else total_file_bytes(fs, files)
    if not total_bytes:
        return None
    step = len(files) / sample_files
    picked = sorted({min(len(files) - 1, int(i * step)) for i in range(sample_files)})
    sampled = [files[i] for i in picked]
    try:
        counts = read_each_file(fs, sampled, lambda _fs, path: count_rows(path))
    except Exception:
        return None
    if sizes is not None:
        byte_sample = [sizes[i] for i in picked]
    else:
        try:
            byte_sample = read_each_file(fs, sampled, lambda filesystem, p: filesystem.size(p))
        except Exception:
            return None
    # A file whose footer would not read contributes neither rows nor bytes, so it widens
    # the confidence interval rather than dragging the ratio toward zero.
    pairs = [
        (rows, size)
        for rows, size in zip(counts, byte_sample, strict=True)
        if rows is not None and size
    ]
    if not pairs:
        return None
    rows_seen = sum(rows for rows, _ in pairs)
    bytes_seen = sum(size for _, size in pairs)
    if bytes_seen <= 0:
        return None
    return max(0, int(total_bytes * rows_seen / bytes_seen))
