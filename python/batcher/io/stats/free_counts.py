"""Free row counts from file headers — metadata that costs one header read.

Some formats record their length in a header the loader parses anyway, then
discard it. A NumPy ``.npy`` header carries the array `shape`, whose leading axis
is the exact row count (Batcher maps the leading axis to rows). Reading the
header touches no array data, so `count()` over a ``.npy`` file becomes free.
"""

from __future__ import annotations

from typing import IO, Any

from batcher.plan.source_stats import SourceStatistics

__all__ = ["npy_header_rows", "numpy_statistics"]


def npy_header_rows(fh: IO[Any]) -> int | None:
    """Leading-axis length from a ``.npy`` header, without loading the array.

    Returns None for ``.npz`` archives (multiple arrays, no single shape here) or
    if the header can't be parsed. The file position is left after the header.
    """
    try:
        from numpy.lib import format as npfmt
    except Exception:
        return None
    try:
        major, _minor = npfmt.read_magic(fh)
        reader = {1: npfmt.read_array_header_1_0, 2: npfmt.read_array_header_2_0}.get(major)
        if reader is None:
            return None
        shape, _fortran, _dtype = reader(fh)
    except Exception:
        return None
    return int(shape[0]) if shape else 0


def numpy_statistics(fs: Any, files: list[str]) -> SourceStatistics | None:
    """Exact row count across ``.npy`` files by summing each header's leading axis.

    Returns None if any file's header is unreadable (so the count is not falsely
    reported as exact). ``.npz`` files yield None and are skipped by the caller.
    """
    if not files:
        return None
    from batcher.io._concurrent import read_each_file

    def _rows(filesystem: Any, path: str) -> int | None:
        with filesystem.open(path) as fh:
            return npy_header_rows(fh)

    try:
        counts = read_each_file(fs, files, _rows)  # concurrent header reads, in order
    except Exception:
        return None
    if any(c is None for c in counts):  # an unparseable header → not an exact count
        return None
    return SourceStatistics(row_count=sum(counts), exact_rows=True)
