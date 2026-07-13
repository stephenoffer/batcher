"""A Parquet file's row count is read from its footer once, then cached by path.

`learned_num_workers` → `total_source_rows` → `row_count` reads every source file's
footer to size the distributed worker fan-out, and it runs on EVERY collect. A footer is
a ~80 ms object-store round trip; Parquet is write-once, so the count is immutable and
cached process-wide. Without the cache a warm distributed groupby re-read 10 footers per
collect (~0.9 s of driver time dwarfing the shuffle it was sizing).
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
    # The footer read is cached by path — the second call never touches pyarrow again.
    assert pqmod._ROW_COUNT_CACHE[str(f)] == 1234

    calls = {"n": 0}
    real = pq.ParquetFile

    def _counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(pq, "ParquetFile", _counting)
    assert src.row_count() == 1234  # served from cache
    assert calls["n"] == 0


def test_multi_file_row_count_sums(tmp_path, monkeypatch):
    monkeypatch.setattr(pqmod, "_ROW_COUNT_CACHE", {})
    for i, n in enumerate((100, 250, 7)):
        _write(str(tmp_path / f"part-{i}.parquet"), n)
    src = pqmod.ParquetSource(str(tmp_path / "*.parquet"))
    assert src.row_count() == 357
