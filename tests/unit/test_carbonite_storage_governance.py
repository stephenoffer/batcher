"""The result cache's budget and eviction ranking, and the manager's aggregate reading.

The cache is the *storage* half of the memory envelope. Its failures are quiet by
construction: over-retaining shrinks the headroom every execution decision assumes, and
over-evicting only ever shows up as a query that recomputes something it should not have.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.carbonite import ResourceManager
from batcher.carbonite.cache import CacheStore, reset_result_cache, result_cache
from batcher.carbonite.memory.pool import reset_process_pool
from batcher.carbonite.memory.pressure import PressureLevel
from batcher.config import Config, MemoryConfig, config_context

pytestmark = pytest.mark.unit


def _table(n_bytes: int) -> pa.Table:
    """A table of roughly `n_bytes` (int64 column)."""
    return pa.table({"v": pa.array(range(max(1, n_bytes // 8)), type=pa.int64())})


@pytest.fixture(autouse=True)
def _fresh():
    reset_result_cache()
    reset_process_pool()
    yield
    reset_result_cache()
    reset_process_pool()


def test_evictions_are_counted_and_explain_a_poor_hit_rate() -> None:
    """A low hit rate means nothing until you know whether entries survived to be hit."""
    store = CacheStore(2400)
    store.put("a", _table(1000))
    store.put("b", _table(1000))
    store.put("c", _table(1000))  # forces at least one eviction

    stats = store.stats()
    assert stats["evictions"] >= 1
    assert stats["entries"] == len(store)
    assert stats["used_bytes"] <= stats["max_bytes"]
    assert 0.0 <= stats["fill"] <= 1.0


def test_eviction_prefers_cheap_cold_large_entries() -> None:
    """Greedy-Dual-Size-Frequency: an expensive, hot, small result outlives a cheap bulky one."""
    store = CacheStore(3000)
    store.put("expensive", _table(800), cost=100.0)
    store.get("expensive")
    store.put("cheap_big", _table(2000), cost=0.0)
    store.put("newcomer", _table(800), cost=1.0)

    assert "expensive" in store
    assert "cheap_big" not in store


def test_membership_does_not_count_as_an_access() -> None:
    """Probing with `get` would silently promote an entry every time anything asked."""
    store = CacheStore(10_000)
    store.put("a", _table(100))
    assert "a" in store
    assert store.stats()["hits"] == 0
    assert store.stats()["misses"] == 0


def test_set_budget_shrinks_and_evicts_at_once() -> None:
    store = CacheStore(10_000)
    store.put("a", _table(2000))
    store.put("b", _table(2000))
    store.set_budget(1500)
    assert store.max_bytes == 1500
    assert store.used_bytes <= 1500
    store.set_budget(-5)
    assert store.max_bytes == 0
    assert len(store) == 0


def test_the_pressure_ladder_only_ever_shrinks_storage() -> None:
    store = CacheStore(8000)
    for k in "abcdef":
        store.put(k, _table(1000))
    full = store.used_bytes

    store.on_pressure(PressureLevel.NORMAL)
    assert store.used_bytes == full  # storage is untouched when execution is not pressured

    store.on_pressure(PressureLevel.ELEVATED)
    elevated = store.used_bytes
    assert elevated <= int(8000 * 0.75)

    store.on_pressure(PressureLevel.SPILL)
    assert store.used_bytes <= int(8000 * 0.5) <= elevated

    store.on_pressure(PressureLevel.CRITICAL)
    assert store.used_bytes == 0 and len(store) == 0


def test_evict_to_free_frees_at_least_the_deficit() -> None:
    store = CacheStore(10_000)
    for k in "abcd":
        store.put(k, _table(1000))
    before = store.used_bytes
    freed = store.evict_to_free(2000)
    assert freed >= 2000
    assert store.used_bytes == before - freed
    assert store.evict_to_free(0) == 0


def test_an_oversized_result_is_skipped_rather_than_thrashing_the_cache() -> None:
    store = CacheStore(1000)
    store.put("small", _table(500))
    store.put("huge", _table(100_000))
    assert "small" in store and "huge" not in store


def test_the_process_cache_reconciles_its_budget_through_the_store() -> None:
    with config_context(Config().replace(memory=MemoryConfig(result_cache_max_bytes=5000))):
        cache = result_cache()
        cache.put("a", _table(3000))
        assert cache.max_bytes == 5000
    with config_context(Config().replace(memory=MemoryConfig(result_cache_max_bytes=1000))):
        again = result_cache()
        assert again is cache, "one store per process"
        assert again.max_bytes == 1000
        assert again.used_bytes <= 1000


# --- the manager's aggregate reading -----------------------------------------


def _plan(peak: int):
    from batcher.plan.physical import OpId, PhysicalOp, PhysicalPlan
    from batcher.plan.resource import ResourceBounds

    op = PhysicalOp(
        op_id=OpId(1),
        kind="Aggregate",
        backend="native",
        algorithm="hash",
        bounds=ResourceBounds(m_max_bytes=peak, c_max_credits=4, n_max_parallelism=4),
        inputs=(),
    )
    return PhysicalPlan(ir={}, output_schema=None, ops=(op,))


def test_the_spill_reason_names_the_plan_when_the_estimate_is_the_cause() -> None:
    with config_context(Config().replace(memory=MemoryConfig(max_memory_bytes=1_000))):
        rm = ResourceManager()
        reason = rm.spill_reason(_plan(10_000_000))
        assert reason is not None
        assert "estimated peak" in reason
        assert rm.should_spill(_plan(10_000_000)) is True


def test_a_fitting_plan_has_no_spill_reason() -> None:
    with config_context(Config().replace(memory=MemoryConfig(max_memory_bytes=1 << 40))):
        rm = ResourceManager()
        assert rm.spill_reason(_plan(1024)) is None
        assert rm.should_spill(_plan(1024)) is False


def test_manager_stats_reads_every_corner_it_governs() -> None:
    with config_context(Config().replace(memory=MemoryConfig(max_memory_bytes=1 << 30))):
        rm = ResourceManager()
        with rm.reserve(1024):
            stats = rm.stats()
        assert stats["pressure_level"] in {lvl.name for lvl in PressureLevel}
        assert stats["hard_budget_bytes"] > 0
        assert stats["soft_budget_bytes"] > 0
        assert "pool" in stats and stats["pool"]["limit_bytes"] > 0
        assert stats["headroom_bytes"] >= 0
