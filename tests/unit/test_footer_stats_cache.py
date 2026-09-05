"""The native footer-statistics aggregate is cached, and a changed file invalidates it.

`footer_stats` is a pure function of its file list and is called once per query with the same
list, so a repeated query re-derived it every time: 51 ms warm on a 100-file TPC-H read,
against a driver whose entire non-waiting time is ~180-250 ms
(`benchmarks/BENCHMARK_RESULTS.md`). `io/splits/parquet.py` already caches footers and planned
splits on `file_identity`; this was the member of that trio without a cache.

The correctness half is what these pin. `file_identity` is `(path, size, mtime_ns)`, so a
rewritten file must MISS -- serving a stale row count would answer `count()` wrongly without
executing, which is the one thing this layer must never do.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher.io.filesystem import resolve_filesystem
from batcher.io.stats import columnar_footer

pytestmark = pytest.mark.unit


def _write(path, values):
    pq.write_table(pa.table({"v": pa.array(values, pa.int64())}), path)


@pytest.fixture
def corpus(tmp_path):
    """Two small Parquet files and the filesystem/schema the statistics pass wants."""
    paths = []
    for i in range(2):
        p = tmp_path / f"part{i}.parquet"
        _write(p, np.arange(i * 100, (i + 1) * 100))
        paths.append(str(p))
    return paths, resolve_filesystem(paths[0]), pa.schema([pa.field("v", pa.int64())])


def _calls(monkeypatch):
    """Count the native aggregations actually performed."""
    from batcher.io.formats.structured import _parquet_native

    seen = []
    original = _parquet_native.footer_stats

    def counted(uris):
        seen.append(tuple(uris))
        return original(uris)

    monkeypatch.setattr(_parquet_native, "footer_stats", counted)
    return seen


def test_a_repeated_identical_read_aggregates_the_footers_once(corpus, monkeypatch):
    """The point: 51 ms of driver work per query, paid once instead of every time."""
    files, fs, schema = corpus
    monkeypatch.setattr(columnar_footer, "_NATIVE_STATS", columnar_footer.FileMetaCache(256))
    seen = _calls(monkeypatch)

    first = columnar_footer._native_statistics(fs, files, schema)
    second = columnar_footer._native_statistics(fs, files, schema)

    if first is None:  # the native path declined here (no native read target, say)
        pytest.skip("native footer statistics unavailable for this filesystem")
    assert second is not None
    assert first.row_count == second.row_count == 200
    assert len(seen) == 1, "the second read must be served from the cache"


def test_a_rewritten_file_is_not_served_from_the_cache(corpus, monkeypatch):
    """The correctness half. A stale row count answers `count()` without executing."""
    files, fs, schema = corpus
    monkeypatch.setattr(columnar_footer, "_NATIVE_STATS", columnar_footer.FileMetaCache(256))
    seen = _calls(monkeypatch)

    first = columnar_footer._native_statistics(fs, files, schema)
    if first is None:
        pytest.skip("native footer statistics unavailable for this filesystem")
    assert first.row_count == 200

    _write(files[1], np.arange(1000, 1300))  # 300 rows where there were 100
    again = columnar_footer._native_statistics(fs, files, schema)

    assert len(seen) == 2, "a changed file must miss, not serve the old aggregate"
    assert again is not None and again.row_count == 400


def test_an_unstatable_file_is_never_cached(corpus, monkeypatch):
    """`file_identity` returns None when it cannot detect a later change; then do not cache."""
    files, fs, schema = corpus
    monkeypatch.setattr(columnar_footer, "_NATIVE_STATS", columnar_footer.FileMetaCache(256))
    monkeypatch.setattr(columnar_footer, "file_identity", lambda path, fs=None: None)
    seen = _calls(monkeypatch)

    first = columnar_footer._native_statistics(fs, files, schema)
    if first is None:
        pytest.skip("native footer statistics unavailable for this filesystem")
    columnar_footer._native_statistics(fs, files, schema)

    assert len(seen) == 2, "with no usable identity every call must re-aggregate"
