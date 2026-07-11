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


def test_bulk_eviction_drops_lowest_value_first_and_stays_ordered():
    # A bulk eviction (budget shrinks to half) must drop the lowest keep-value entries
    # first, exactly as a per-victim min-scan would — the batched sort is only a speed
    # change, never a policy change. Equal-value entries evict oldest-first (stable).
    store = CacheStore(max_bytes=1 << 30)
    n = 200
    # Same size, distinct cost so keep-value = (cost + eps) / size is strictly ordered.
    for i in range(n):
        store.put(f"k{i}", _table(50), cost=float(i))
    total = store.used_bytes
    store._evict_to(total // 2)  # force a bulk eviction
    survivors = set(store._entries)
    # The half with the *highest* cost (value) must survive; the cheap half is gone.
    assert survivors == {f"k{i}" for i in range(n // 2, n)}
    assert store.used_bytes <= total // 2


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


def test_reserve_invokes_the_pressure_ladder(monkeypatch):
    """The ladder existed but had no production caller — the two budgets were disjoint.

    `CacheStore` bytes are not accounted against the buffer pool, so a cache sitting at its
    full `result_cache_max_bytes` silently eats the headroom every other Carbonite decision
    assumes. `ResourceManager.reserve` — the moment execution asks for memory — now applies
    the ladder before checking the deficit. Evicting a cache only costs a recompute, so this
    is result-invariant.
    """
    from batcher.carbonite import cache as cache_module
    from batcher.carbonite.manager import ResourceManager

    one = _table(100).nbytes
    store = CacheStore(max_bytes=8 * one)
    monkeypatch.setattr(cache_module, "_result_cache", store)
    manager = ResourceManager()

    def _refill() -> int:
        store.clear()
        for i in range(8):
            store.put(f"k{i}", _table(100))
        return store.used_bytes

    # NORMAL must not touch the cache: the common path pays nothing.
    full = _refill()
    monkeypatch.setattr(manager._pressure, "classify", lambda: PressureLevel.NORMAL)
    with manager.reserve(1024):
        pass
    assert store.used_bytes == full

    for level, ceiling in (
        (PressureLevel.ELEVATED, store.max_bytes * 3 // 4),
        (PressureLevel.SPILL, store.max_bytes // 2),
        (PressureLevel.CRITICAL, 0),
    ):
        _refill()
        monkeypatch.setattr(manager._pressure, "classify", lambda level=level: level)
        with manager.reserve(1024):
            pass
        assert store.used_bytes <= ceiling, f"cache did not yield at {level.name}"


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


def test_cache_store_stats_hit_rate():
    import pyarrow as pa

    from batcher.carbonite.cache import CacheStore

    c = CacheStore(max_bytes=10_000_000)
    t = pa.table({"x": list(range(100))})
    c.put("k", t)
    assert c.get("k") is not None  # hit
    assert c.get("missing") is None  # miss
    s = c.stats()
    assert s["hits"] == 1 and s["misses"] == 1
    assert s["hit_rate"] == 0.5
    assert CacheStore(1).stats()["hit_rate"] == 0.0  # cold
