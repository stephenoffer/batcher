"""A Parquet file's row count is read from its footer once, then cached per file *version*.

`learned_num_workers` → `total_source_rows` → `row_count` reads every source file's
footer to size the distributed worker fan-out, and it runs on EVERY collect. A footer is
a ~80 ms object-store round trip, so the count is cached process-wide; without it a warm
distributed groupby re-read 10 footers per collect (~0.9 s of driver time dwarfing the
shuffle it was sizing).

This file used to say "Parquet is write-once, so the count is immutable" and assert the
cache key was the bare path. That premise is false for what pipelines do: `FileSink`
writes deterministic names, so a re-run overwrites its own output, and the path is just as
often rewritten by the upstream job or a compaction. A path-keyed count then answered
`count()` with the *previous* file's total while `collect()` returned the new rows.

The key now carries the file's `(size, mtime)`. `test_stale_source_metadata.py` covers
that end to end; the assertions here are on the **behaviour** the cache exists for — the
right answer, and no second footer read — rather than on the shape of its key, plus one
case pinning the premise that changed.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher.io.formats.structured.parquet import source as pqmod

pytestmark = pytest.mark.unit


def _write(path: str, n: int) -> None:
    pq.write_table(pa.table({"v": pa.array(range(n))}), path)


def test_row_count_is_correct_and_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(pqmod, "_ROW_COUNT_CACHE", {})
    f = tmp_path / "a.parquet"
    _write(str(f), 1234)
    src = pqmod.ParquetSource(str(f))

    assert src.row_count() == 1234
    assert len(pqmod._ROW_COUNT_CACHE) == 1, "the footer read was not cached"

    calls = {"n": 0}
    real = pq.ParquetFile

    def _counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(pq, "ParquetFile", _counting)
    assert src.row_count() == 1234  # served from cache
    assert calls["n"] == 0


def test_a_rewritten_file_is_recounted(tmp_path, monkeypatch):
    """The premise the old key rested on — that the file cannot change. It can."""
    monkeypatch.setattr(pqmod, "_ROW_COUNT_CACHE", {})
    f = tmp_path / "part-00000.parquet"
    _write(str(f), 1234)
    assert pqmod.ParquetSource(str(f)).row_count() == 1234

    _write(str(f), 7)

    assert pqmod.ParquetSource(str(f)).row_count() == 7


def test_multi_file_row_count_sums(tmp_path, monkeypatch):
    monkeypatch.setattr(pqmod, "_ROW_COUNT_CACHE", {})
    for i, n in enumerate((100, 250, 7)):
        _write(str(tmp_path / f"part-{i}.parquet"), n)
    src = pqmod.ParquetSource(str(tmp_path / "*.parquet"))
    assert src.row_count() == 357
