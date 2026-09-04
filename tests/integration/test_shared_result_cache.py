"""End-to-end shared result cache: a result computed once, served to a fresh reader.

The process cache cannot help the workload a scheduled job or a dashboard actually has —
the same query re-issued from a fresh interpreter — because its key is each input's
*object* identity. The shared cache is keyed by content instead, so a second reader of the
same versioned file hits an entry the first one wrote.

The store here is a dict standing in for Redis or RocksDB. What that leaves untested is the
two drivers' own `get`/`put`/`delete`, which is a handful of lines each; what it covers is
every line between `Dataset.cache()` and the store, which is where a caching bug becomes a
wrong answer rather than a slow one.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher.config import active_config, config_context

pytest.importorskip("batcher._native", reason="native engine not built")


class _DictCache:
    """An in-process `SharedCache`; see `tests/unit/test_shared_cache_contract.py`."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self.store.get(key)

    def put(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        self.store[key] = value

    def delete(self, key: str) -> None:
        self.store.pop(key, None)

    def close(self) -> None:
        pass


@pytest.fixture
def shared(monkeypatch):
    """A live shared cache backed by a dict, with the process cache emptied around it."""
    from batcher.carbonite.cache_shared import factory, store

    backend = _DictCache()
    live = store.SharedResultCache(backend)
    monkeypatch.setattr(factory, "_cache", live)
    monkeypatch.setattr(factory, "_cache_uri", "dict://test")
    cfg = active_config()
    memory = dataclasses.replace(cfg.memory, shared_cache_uri="dict://test")
    bt.clear_cache()
    with config_context(cfg.replace(memory=memory)):
        yield live
    bt.clear_cache()
    factory.reset_shared_cache()


@pytest.fixture
def table_path(tmp_path):
    path = str(tmp_path / "events.parquet")
    pq.write_table(pa.table({"k": [1, 1, 2, 2, 3], "v": [10, 20, 30, 40, 50]}), path)
    return path


def _query(path: str):
    return bt.read.parquet(path).group_by("k").agg(s=bt.col("v").sum()).cache()


def _rows(ds) -> list[tuple]:
    """Row tuples in a stable order — a group-by result has no guaranteed row order."""
    return sorted(zip(*ds.collect().to_pydict().values(), strict=True))


def test_a_second_reader_is_served_from_the_shared_store(shared, table_path):
    first = _rows(_query(table_path))
    assert shared.stats()["writes"] == 1

    # A fresh Dataset over the same file: new objects, so the process cache cannot match
    # it. This is the case the shared cache exists for.
    bt.clear_cache()
    before = shared.stats()["hits"]
    second = _rows(_query(table_path))
    assert shared.stats()["hits"] - before == 1
    assert second == first


def test_a_shared_hit_matches_an_uncached_run(shared, table_path):
    expected = sorted(
        zip(
            *bt.read.parquet(table_path)
            .group_by("k")
            .agg(s=bt.col("v").sum())
            .collect()
            .to_pydict()
            .values(),
            strict=True,
        )
    )
    _query(table_path)  # populate
    bt.clear_cache()
    assert _rows(_query(table_path)) == expected


def test_rewriting_the_input_retires_the_shared_entry(shared, table_path):
    assert _rows(_query(table_path)) == [(1, 30), (2, 70), (3, 50)]
    # The path is unchanged, so an identity-only key would serve the old rows here. The
    # content version in the key is what makes this the new answer instead.
    pq.write_table(pa.table({"k": [1, 2], "v": [100, 200]}), table_path)
    bt.clear_cache()
    assert _rows(_query(table_path)) == [(1, 100), (2, 200)]


def test_a_shared_hit_is_promoted_into_the_process_cache(shared, table_path):
    _query(table_path).collect()  # populate the shared store
    bt.clear_cache()
    ds = _query(table_path)
    ds.collect()  # served from the shared store, and stored locally on the way through
    before = bt.cache_stats()["hits"]
    ds.collect()  # this one must not go back to the shared store
    assert bt.cache_stats()["hits"] - before == 1
    assert shared.stats()["hits"] == 1


def test_in_memory_data_never_reaches_the_shared_store(shared):
    ds = (
        bt.from_pydict({"k": [1, 1, 2], "v": [10, 20, 30]})
        .group_by("k")
        .agg(s=bt.col("v").sum())
        .cache()
    )
    ds.collect()
    ds.collect()
    assert shared.stats()["writes"] == 0
    assert shared.stats()["hits"] == 0


def test_an_uncached_dataset_never_reaches_the_shared_store(shared, table_path):
    bt.read.parquet(table_path).group_by("k").agg(s=bt.col("v").sum()).collect()
    assert shared.stats()["writes"] == 0


def test_an_unreachable_store_degrades_to_recompute(monkeypatch, table_path):
    from batcher.carbonite.cache_shared import factory, store
    from batcher.carbonite.cache_shared.base import SharedCacheError

    class _Broken(_DictCache):
        def get(self, key):
            raise SharedCacheError("connection refused")

        def put(self, key, value, ttl_seconds=None):
            raise SharedCacheError("connection refused")

    live = store.SharedResultCache(_Broken())
    monkeypatch.setattr(factory, "_cache", live)
    monkeypatch.setattr(factory, "_cache_uri", "dict://broken")
    cfg = active_config()
    memory = dataclasses.replace(cfg.memory, shared_cache_uri="dict://broken")
    bt.clear_cache()
    try:
        with config_context(cfg.replace(memory=memory)):
            # The query must succeed and be right; only its cost changes.
            assert _rows(_query(table_path)) == [(1, 30), (2, 70), (3, 50)]
        assert live.stats()["errors"] > 0
    finally:
        bt.clear_cache()
        factory.reset_shared_cache()


def test_cache_stats_reports_the_shared_store(shared, table_path):
    _query(table_path).collect()
    stats = bt.cache_stats()
    assert stats["shared_writes"] == 1
    assert stats["shared_errors"] == 0
