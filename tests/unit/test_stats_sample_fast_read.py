"""The driver-side column-stat sample uses the fast row-group reader for Parquet.

`collect_source_metadata` samples a base source on the driver (distributed/UDF paths).
The naive per-file read of even the bounded 262k-row sample is ~0.8 MB/s on high-latency
object storage — measured at ~45s for one sf10 sample, which blocks the query *return*
after the distributed work is already done. `_fast_sample` reads the sample through the
coalesced, multi-thread dataset scanner instead. These tests verify it (a) reads a
splittable Parquet source, (b) returns the same rows, (c) respects the row cap, and (d)
falls back to `iter_source` for a non-splittable (in-memory) source.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher.api.terminal import _metadata

pytestmark = pytest.mark.unit


def _write_parquet(path, n_rows, row_group_size):
    tbl = pa.table({"k": list(range(n_rows)), "v": [i % 7 for i in range(n_rows)]})
    pq.write_table(tbl, path, row_group_size=row_group_size)


def test_fast_sample_reads_parquet_rowgroups(tmp_path):
    path = str(tmp_path / "t.parquet")
    _write_parquet(path, 5000, row_group_size=1000)  # 5 row groups
    src = bt.read.parquet(path)._sources[0]

    batches = _metadata._fast_sample(src)
    assert batches is not None, "a row-group Parquet source must take the fast path"
    total = sum(b.num_rows for b in batches)
    assert total == 5000
    # Same data as a direct read (order-independent multiset of k).
    got = pa.Table.from_batches(batches).column("k").to_pylist()
    assert sorted(got) == list(range(5000))


def test_fast_sample_respects_row_cap(tmp_path, monkeypatch):
    # A source larger than the cap should stop after enough leading row-groups.
    monkeypatch.setattr(_metadata, "_STATS_SAMPLE_ROWS", 2000)
    path = str(tmp_path / "big.parquet")
    _write_parquet(path, 10000, row_group_size=1000)
    src = bt.read.parquet(path)._sources[0]

    batches = _metadata._fast_sample(src)
    assert batches is not None
    total = sum(b.num_rows for b in batches)
    assert 2000 <= total < 10000  # covered the cap without reading the whole source


def test_fast_sample_falls_back_for_in_memory_source():
    src = bt.from_pydict({"k": [1, 2, 3]})._sources[0]
    # An in-memory source is not row-group-splittable → fast path declines (None),
    # and the public sampler falls back to iter_source and still returns the rows.
    assert _metadata._fast_sample(src) is None
    sample = _metadata._stats_sample(src)
    assert sum(b.num_rows for b in sample) == 3


def test_stats_sample_prefers_fast_path(tmp_path, monkeypatch):
    path = str(tmp_path / "t.parquet")
    _write_parquet(path, 3000, row_group_size=1000)
    src = bt.read.parquet(path)._sources[0]

    called = {"iter_source": False}
    real_iter = _metadata.iter_source

    def spy(*a, **k):
        called["iter_source"] = True
        return real_iter(*a, **k)

    monkeypatch.setattr(_metadata, "iter_source", spy)
    sample = _metadata._stats_sample(src)
    assert sum(b.num_rows for b in sample) == 3000
    assert not called["iter_source"], "fast path must be used, not the naive iter_source"
