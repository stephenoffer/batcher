"""One query must not re-derive a many-file source's row count several times.

Routing, partition sizing, the GPU backend decision, the metadata-answer path, and the
post-run learned-stats loop each ask a source for its row count, and none of them knows the
others did. On a 512-file directory of 400,000 rows that was **three** calls accounting for
0.104 s of a 0.157 s query — two thirds of the wall clock spent re-deriving one number.

The cost is not the footer I/O, which is cached per path. It is the walk: a fresh
`ThreadPoolExecutor` and a Python call per file, per call. At a million files that is three
full metadata passes before a single data page is read, which is the shape of the
"millions of small files" pattern rather than an edge case.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt

pytestmark = pytest.mark.io

_FILES = 24
_PER_FILE = 50


@pytest.fixture
def many_files(tmp_path):
    """A directory of `_FILES` small Parquet files with a known total row count."""
    rng = np.random.default_rng(0)
    for i in range(_FILES):
        table = pa.table({"k": pa.array(rng.integers(0, 100, _PER_FILE))})
        pq.write_table(table, tmp_path / f"p{i:05d}.parquet")
    return str(tmp_path)


def test_the_row_count_is_correct(many_files):
    """Memoizing must not change the answer — the property everything else rests on."""
    source = bt.read.parquet(many_files)._sources[0]
    assert source.row_count() == _FILES * _PER_FILE


def test_a_repeat_call_reads_no_footers(many_files):
    """The second caller gets the memo, not a second walk over every file."""
    source = bt.read.parquet(many_files)._sources[0]
    source.row_count()
    calls = 0
    original = source._file_row_count

    def counted(path):
        nonlocal calls
        calls += 1
        return original(path)

    # `FileSource` uses `__slots__`, so patch the bound method through the class.
    cls = type(source)
    saved = cls._file_row_count
    cls._file_row_count = lambda self, path: counted(path)  # type: ignore[assignment]
    try:
        assert source.row_count() == _FILES * _PER_FILE
    finally:
        cls._file_row_count = saved  # type: ignore[assignment]
    assert calls == 0


def test_a_fresh_source_still_computes_it(many_files):
    """The memo is per source, not global: a new handle re-derives from the files."""
    first = bt.read.parquet(many_files)._sources[0]
    second = bt.read.parquet(many_files)._sources[0]
    assert first is not second
    assert first.row_count() == second.row_count() == _FILES * _PER_FILE


def test_an_n_rows_cap_is_still_applied(many_files):
    """A capped source reports the cap, and reports it consistently across calls.

    The cap is applied *after* the sum, so memoizing the wrong side of it would report the
    uncapped total — which the optimizer would then size joins and worker fan-out from.
    """
    capped = bt.read.parquet(many_files, n_rows=10)._sources[0]
    assert capped.row_count() == 10
    assert capped.row_count() == 10


def test_the_query_result_is_unchanged(many_files):
    """The memo is an optimization; the rows it helps plan must be the same rows."""
    rows = bt.read.parquet(many_files).collect().num_rows
    assert rows == _FILES * _PER_FILE
