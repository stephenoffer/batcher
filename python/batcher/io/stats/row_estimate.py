"""Advisory row-count estimates for footerless text formats, from a byte sample.

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
"""

from __future__ import annotations

from typing import Any

from batcher.io._concurrent import total_file_bytes
from batcher.io.detect import compression_for_path

__all__ = ["estimate_delimited_rows"]

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
