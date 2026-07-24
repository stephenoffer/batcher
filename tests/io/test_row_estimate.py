"""Byte-sample row-count estimates for footerless text formats (CSV).

A CSV carries no row count, so a join against one was sized from the planner's blind
default. `io.stats.row_estimate` extrapolates a count from a small byte sample and the
dataset's on-disk size — advisory, never exact, and O(1) I/O at plan time. These tests pin
the accuracy on a uniform file, the header discount, the multi-file scaling, and every
decline path (compressed, empty, single-token) that must yield None rather than a wrong count.
"""

from __future__ import annotations

import gzip

import pytest

from batcher.io.filesystem import resolve_filesystem
from batcher.io.formats.structured.csv import CSVSource
from batcher.io.stats.row_estimate import estimate_delimited_rows

pytestmark = pytest.mark.unit


def _write_csv(path, n_rows: int, *, header: bool = True) -> None:
    with open(path, "w") as f:
        if header:
            f.write("id,name,value\n")
        for i in range(n_rows):
            f.write(f"{i},name_{i},{i * 3}\n")


def test_estimate_is_close_on_a_uniform_file(tmp_path) -> None:
    """A fixed-width file estimates within a few percent of the true row count."""
    path = tmp_path / "u.csv"
    _write_csv(path, 10_000)
    fs = resolve_filesystem(str(path))
    est = estimate_delimited_rows(fs, [str(path)], has_header=True)
    assert est is not None
    assert 9_500 <= est <= 10_500


def test_header_is_discounted(tmp_path) -> None:
    """The header line is not counted as a data row."""
    path = tmp_path / "small.csv"
    _write_csv(path, 3)  # tiny: the whole file is the sample, so the count is near-exact
    fs = resolve_filesystem(str(path))
    est = estimate_delimited_rows(fs, [str(path)], has_header=True)
    # 4 lines in the file (1 header + 3 data); the header discount lands it on 3.
    assert est == 3


def test_scales_across_multiple_files(tmp_path) -> None:
    """The estimate scales by the summed size of every file, not just the sampled one."""
    paths = []
    for i in range(4):
        p = tmp_path / f"p{i}.csv"
        _write_csv(p, 2_500)
        paths.append(str(p))
    fs = resolve_filesystem(paths[0])
    est = estimate_delimited_rows(fs, paths, has_header=True)
    assert est is not None
    assert 9_000 <= est <= 11_000  # ~10,000 rows across four files


def test_compressed_file_declines(tmp_path) -> None:
    """A compressed file's on-disk size does not track row width, so no estimate is made."""
    path = tmp_path / "c.csv.gz"
    with gzip.open(path, "wt") as f:
        f.write("id,v\n")
        for i in range(1000):
            f.write(f"{i},{i}\n")
    fs = resolve_filesystem(str(path))
    assert estimate_delimited_rows(fs, [str(path)], has_header=True) is None


def test_no_files_declines() -> None:
    """An empty file list yields no estimate."""
    assert estimate_delimited_rows(resolve_filesystem("."), [], has_header=True) is None


def test_single_line_no_newline_declines(tmp_path) -> None:
    """A sample with no newline gives no row width to measure from."""
    path = tmp_path / "oneline.csv"
    with open(path, "w") as f:
        f.write("no_trailing_newline_here")
    fs = resolve_filesystem(str(path))
    assert estimate_delimited_rows(fs, [str(path)], has_header=True) is None


def test_csv_source_statistics_carries_advisory_count(tmp_path) -> None:
    """`CSVSource.statistics()` surfaces the estimate as advisory (never exact)."""
    path = tmp_path / "s.csv"
    _write_csv(path, 5_000)
    stats = CSVSource(str(path)).statistics()
    assert stats is not None
    assert stats.exact_rows is False
    assert stats.row_count is not None
    assert 4_500 <= stats.row_count <= 5_500
    assert stats.byte_size == path.stat().st_size


def test_csv_headerless_source_is_not_discounted(tmp_path) -> None:
    """With `header=False` every line is a data row, so nothing is discounted."""
    path = tmp_path / "nh.csv"
    _write_csv(path, 100, header=False)
    stats = CSVSource(str(path), header=False).statistics()
    assert stats is not None
    assert stats.row_count is not None and stats.row_count >= 90
