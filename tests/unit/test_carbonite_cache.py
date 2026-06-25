"""The result cache: a memory-bounded LRU of materialized Arrow results.

Pins the storage-memory contract — bounded bytes, LRU eviction, the size guard, and
the pressure ladder that yields cache RAM back to execution — without the engine.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.carbonite.cache import CacheStore
from batcher.carbonite.memory.pressure import PressureLevel

pytestmark = pytest.mark.unit


def _table(n_rows: int, fill: int = 0) -> pa.Table:
    return pa.table({"v": pa.array([fill] * n_rows, pa.int64())})


def test_get_miss_then_hit_refreshes_recency():
    store = CacheStore(max_bytes=1 << 20)
    assert store.get("k") is None
    t = _table(10)
    store.put("k", t)
    got = store.get("k")
    assert got is not None and got.num_rows == 10


def test_lru_eviction_keeps_within_budget():
    one = _table(100)  # 100 * 8 bytes = 800 B
    store = CacheStore(max_bytes=2 * one.nbytes + 1)  # room for ~2 entries
    store.put("a", _table(100))
    store.put("b", _table(100))
    store.get("a")  # touch "a" → "b" is now least-recently-used
    store.put("c", _table(100))  # over budget → evict the LRU ("b")
    assert store.get("a") is not None
    assert store.get("c") is not None
    assert store.get("b") is None  # evicted
    assert store.used_bytes <= store.max_bytes


def test_oversized_entry_is_not_cached():
    big = _table(1000)
    store = CacheStore(max_bytes=big.nbytes // 2)  # the table alone exceeds the budget
    store.put("big", big)
    assert store.get("big") is None  # skipped, not thrashed in
    assert store.used_bytes == 0


def test_put_same_key_replaces_and_reaccounts():
    store = CacheStore(max_bytes=1 << 20)
    store.put("k", _table(10))
    before = store.used_bytes
    store.put("k", _table(50))  # replace with a larger result
    assert store.get("k").num_rows == 50
    assert store.used_bytes > before


def test_invalidate_and_clear_free_bytes():
    store = CacheStore(max_bytes=1 << 20)
    store.put("a", _table(10))
    store.put("b", _table(10))
    store.invalidate("a")
    assert store.get("a") is None
    assert store.used_bytes > 0
    store.clear()
    assert store.used_bytes == 0
    assert store.get("b") is None


def test_on_pressure_ladder_yields_storage_to_execution():
    one = _table(100).nbytes
    store = CacheStore(max_bytes=8 * one)
    for i in range(8):
        store.put(f"k{i}", _table(100))
    full = store.used_bytes
    store.on_pressure(PressureLevel.ELEVATED)  # trim toward 3/4
    assert store.used_bytes <= store.max_bytes * 3 // 4 < full
    store.on_pressure(PressureLevel.SPILL)  # halve
    assert store.used_bytes <= store.max_bytes // 2
    store.on_pressure(PressureLevel.CRITICAL)  # evict everything
    assert store.used_bytes == 0


def test_cost_aware_eviction_keeps_expensive_result_over_cheap():
    # Two equal-size results; the budget holds only one more on the next insert. The
    # cheap one must be evicted, the expensive (slow-to-recompute) one kept.
    one = _table(100).nbytes
    store = CacheStore(max_bytes=2 * one + 1)
    store.put("cheap", _table(100), cost=0.001)
    store.put("expensive", _table(100), cost=10.0)
    store.put("filler", _table(100), cost=1.0)  # over budget → evict the lowest value
    assert store.get("expensive") is not None  # high recompute cost → survives
    assert store.get("cheap") is None  # cheap → evicted first


def test_zero_budget_caches_nothing():
    store = CacheStore(max_bytes=0)
    store.put("k", _table(10))
    assert store.get("k") is None


def test_evict_to_free_releases_at_least_the_requested_bytes():
    one = _table(100).nbytes
    store = CacheStore(max_bytes=8 * one)
    for i in range(8):
        store.put(f"k{i}", _table(100))
    freed = store.evict_to_free(3 * one)
    assert freed >= 3 * one  # at least the requested deficit
    assert store.used_bytes <= 8 * one - 3 * one
    assert store.evict_to_free(0) == 0  # nothing requested → no-op


def test_reserve_reclaims_cache_when_pool_is_tight(monkeypatch):
    """`ResourceManager.reserve` reclaims storage precisely: when the pool can't grant
    the request, exactly the deficit is dropped from the cache so its RAM goes to the
    running query (execution-evicts-storage)."""
    from batcher.carbonite import cache as cache_mod
    from batcher.carbonite.manager import ResourceManager
    from batcher.config import Config, MemoryConfig, config_context

    store = CacheStore(max_bytes=1 << 20)
    for i in range(8):
        store.put(f"k{i}", _table(100))
    monkeypatch.setattr(cache_mod, "current_result_cache", lambda: store)

    # A tiny envelope → small pool; holding one reservation makes the next one tight.
    cfg = Config().replace(memory=MemoryConfig(max_memory_bytes=10_000))
    with config_context(cfg):
        rm = ResourceManager()
        with rm.reserve(6000):  # holds 6000 of the ~9000-byte hard budget
            before = store.used_bytes
            with rm.reserve(5000):  # available ~3000 < 5000 → deficit forces reclaim
                pass
            assert store.used_bytes < before  # storage yielded RAM to execution
