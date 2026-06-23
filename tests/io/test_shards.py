"""Sharded training dataset (F1/F6) — round-trip, global indexing, bounded cache.

The storage layer that lets the streaming loader feed a shuffled/sharded order
without materializing the corpus: equal-size Arrow-IPC shards + a JSON index, read
back by global row index through a bounded LRU shard cache.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.io.formats.ml.shards import ShardReader, read_shard_index, write_shards


def _table(n: int) -> pa.Table:
    return pa.table({"id": list(range(n)), "v": [float(i) for i in range(n)]})


def test_write_shards_repacks_to_fixed_size(tmp_path):
    # Input batched 10×10 must repack to shards of exactly rows_per_shard=30 (+rem).
    batches = [pa.record_batch({"id": list(range(i * 10, i * 10 + 10))}) for i in range(10)]
    idx = write_shards(batches, str(tmp_path), rows_per_shard=30)
    assert idx.total_rows == 100
    assert list(idx.shard_rows) == [30, 30, 30, 10]  # 4 shards, last is remainder
    # Reloading the index from disk reproduces it.
    idx2 = read_shard_index(str(tmp_path))
    assert idx2.shard_rows == idx.shard_rows
    assert idx2.total_rows == 100


def test_locate_maps_global_index_to_shard_offset(tmp_path):
    idx = write_shards(_table(100), str(tmp_path), rows_per_shard=30)
    assert idx.locate(0) == (0, 0)
    assert idx.locate(29) == (0, 29)
    assert idx.locate(30) == (1, 0)
    assert idx.locate(95) == (3, 5)
    with pytest.raises(IndexError):
        idx.locate(100)


def test_take_gathers_rows_in_requested_order(tmp_path):
    write_shards(_table(100), str(tmp_path), rows_per_shard=25)
    reader = ShardReader(str(tmp_path), cache_size=2)
    # A shuffled set of global indices spanning all shards.
    want = [5, 99, 30, 0, 76, 31, 24]
    out = reader.take(want)
    assert out.column("id").to_pylist() == want
    assert out.column("v").to_pylist() == [float(i) for i in want]


def test_cache_is_bounded(tmp_path):
    # Touching every shard with cache_size=2 must keep at most 2 shards resident.
    write_shards(_table(100), str(tmp_path), rows_per_shard=20)  # 5 shards
    reader = ShardReader(str(tmp_path), cache_size=2)
    # Read one row from each shard in turn (forces all 5 shards to load).
    for gi in (0, 20, 40, 60, 80):
        reader.take([gi])
        assert len(reader._cache) <= 2  # LRU never exceeds the bound


def test_take_empty_returns_typed_empty(tmp_path):
    write_shards(_table(10), str(tmp_path), rows_per_shard=5)
    reader = ShardReader(str(tmp_path))
    out = reader.take([])
    assert out.num_rows == 0
    assert out.column_names == ["id", "v"]


def test_full_scan_via_take_reconstructs_dataset(tmp_path):
    write_shards(_table(57), str(tmp_path), rows_per_shard=10)
    reader = ShardReader(str(tmp_path), cache_size=3)
    out = reader.take(list(range(57)))
    assert out.column("id").to_pylist() == list(range(57))


def test_rows_per_shard_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="rows_per_shard"):
        write_shards(_table(10), str(tmp_path), rows_per_shard=0)
