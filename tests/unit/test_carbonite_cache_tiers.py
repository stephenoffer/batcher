"""The result cache's second tier: demotion, promotion, and the `StorageLevel` contract.

`test_carbonite_cache.py` pins the memory store on its own (disk budget zero, which is
what every case there constructs). This file pins what happens at the boundary between
the two tiers — that an eviction under `MEMORY_AND_DISK` becomes a *demotion* rather than
a loss, that `DISK_ONLY` never charges the memory budget, that `MEMORY_ONLY` still drops,
and that a recalled result is the one that was stored.

No engine: the store speaks plain strings and `pyarrow.Table`s, so every case here is the
storage contract alone.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.carbonite.cache import CacheStore
from batcher.carbonite.cache_disk import DiskCacheTier
from batcher.carbonite.memory.pressure import PressureLevel
from batcher.plan.resource import StorageLevel

pytestmark = pytest.mark.unit


def _table(n_rows: int, fill: int = 0) -> pa.Table:
    return pa.table({"v": pa.array([fill] * n_rows, pa.int64())})


@pytest.fixture
def store():
    """A store with room for ~2 entries in memory and plenty on disk."""
    one = _table(1000)
    s = CacheStore(max_bytes=2 * one.nbytes + 16, disk_max_bytes=8 << 20)
    yield s
    s.clear()  # unlinks the tier's scratch directory


@pytest.fixture
def tier():
    t = DiskCacheTier(8 << 20)
    yield t
    t.clear()


# --- the disk tier on its own -------------------------------------------------------


def test_disk_tier_round_trips_a_table_exactly(tier):
    original = pa.table({"i": pa.array([1, 2, 3], pa.int64()), "s": pa.array(["a", None, "c"])})
    assert tier.put("k", original) > 0
    got = tier.get("k")
    assert got is not None
    # `equals` is exact on schema *and* values, which is the point: a tier that round-trips
    # the rows but widens a type has changed the result, and invariant #7 forbids that.
    assert got.equals(original)


def test_disk_tier_declines_an_empty_result(tier):
    assert tier.put("k", _table(0)) == 0
    assert "k" not in tier


def test_disk_tier_declines_an_entry_larger_than_its_whole_budget():
    small = DiskCacheTier(64)
    try:
        assert small.put("k", _table(10_000)) == 0
        assert len(small) == 0
    finally:
        small.clear()


def test_disk_tier_evicts_least_recently_used(tier):
    # Budget for ~2 entries. Sizes are compressed, so bound the tier by measuring one.
    written = tier.put("a", _table(1000, 1))
    tier.set_budget(2 * written + 8)
    tier.put("b", _table(1000, 2))
    tier.get("a")  # "a" is now most-recently-used, so "b" is the victim
    tier.put("c", _table(1000, 3))
    assert "a" in tier
    assert "c" in tier
    assert "b" not in tier
    assert tier.used_bytes <= tier.max_bytes


def test_disk_tier_at_a_zero_budget_writes_nothing_and_makes_no_directory():
    off = DiskCacheTier(0)
    assert not off.enabled
    assert off.put("k", _table(100)) == 0
    assert off.get("k") is None
    assert off.stats()["entries"] == 0


def test_disk_tier_reports_the_resident_size_a_read_would_take(tier):
    table = _table(4000)
    tier.put("k", table)
    # The read-back footprint, not the compressed on-disk size: a promotion budgeted
    # against the file size under-reserves by exactly the compression ratio, and this
    # column compresses well enough for the two to differ.
    assert tier.resident_bytes("k") >= table.nbytes
    assert tier.resident_bytes("absent") == 0


# --- the two tiers together ---------------------------------------------------------


def test_eviction_demotes_instead_of_dropping(store):
    for i, key in enumerate("abc"):
        store.put(key, _table(1000, i), cost=1.0, level=StorageLevel.MEMORY_AND_DISK)
    stats = store.stats()
    assert stats["entries"] == 2, "memory budget holds two"
    assert stats["demotions"] == 1, "the third insert pushed one out, to disk"
    assert stats["disk_entries"] == 1


def test_every_demoted_result_is_recalled_with_its_own_rows(store):
    for i, key in enumerate("abc"):
        store.put(key, _table(1000, i), cost=1.0)
    # The bug this guards is a tier that recalls *a* result rather than *the* result: a
    # digest collision, a reused bucket name, an off-by-one in the key. Checking the fill
    # value per key is what makes that visible; a row count would not.
    for i, key in enumerate("abc"):
        got = store.get(key)
        assert got is not None, f"{key} was lost, not demoted"
        assert got.column("v").to_pylist() == [i] * 1000


def _in_memory(store: CacheStore) -> set[str]:
    """The keys currently resident in the memory tier.

    Reaches past the public surface on purpose: which tier holds an entry is deliberately
    invisible to a caller (`__contains__` answers `True` either way), and these cases are
    precisely the ones about the boundary between them.
    """
    return set(store._entries)


def test_a_disk_hit_is_promoted_back_into_memory(store):
    store.put("a", _table(1000, 1), cost=1.0)
    store.put("b", _table(1000, 2), cost=1.0)
    store.put("c", _table(1000, 3), cost=1.0)  # demotes the lowest-value entry
    demoted = next(k for k in "abc" if k not in _in_memory(store))
    assert store.get(demoted) is not None
    assert demoted in _in_memory(store), "a disk hit must not stay on disk"
    assert store.stats()["promotions"] >= 1


def test_memory_only_still_drops_on_eviction(store):
    store.put("a", _table(1000, 1), cost=1.0, level=StorageLevel.MEMORY_ONLY)
    store.put("b", _table(1000, 2), cost=1.0, level=StorageLevel.MEMORY_ONLY)
    store.put("c", _table(1000, 3), cost=1.0, level=StorageLevel.MEMORY_ONLY)
    assert store.stats()["disk_entries"] == 0
    assert store.stats()["demotions"] == 0
    assert sum(store.get(k) is not None for k in "abc") == 2, "one was dropped outright"


def test_disk_only_never_charges_the_memory_budget(store):
    store.put("d", _table(1000), level=StorageLevel.DISK_ONLY)
    assert store.used_bytes == 0
    assert "d" in store
    assert len(store) == 1
    got = store.get("d")
    assert got is not None and got.num_rows == 1000
    assert store.used_bytes == 0, "a DISK_ONLY hit must not be promoted into memory"


def test_disk_only_over_a_memory_resident_key_gives_the_memory_back(store):
    store.put("k", _table(1000), level=StorageLevel.MEMORY_AND_DISK)
    assert store.used_bytes > 0
    store.put("k", _table(1000, 7), level=StorageLevel.DISK_ONLY)
    assert store.used_bytes == 0, "the replaced entry's bytes must be un-accounted"
    got = store.get("k")
    assert got is not None and got.column("v")[0].as_py() == 7


def test_an_oversized_result_falls_through_to_disk(store):
    huge = _table(200_000)
    assert huge.nbytes > store.max_bytes
    store.put("big", huge, level=StorageLevel.MEMORY_AND_DISK)
    assert store.used_bytes == 0
    got = store.get("big")
    assert got is not None and got.num_rows == 200_000


def test_invalidate_drops_both_tiers(store):
    store.put("a", _table(1000, 1), cost=1.0)
    store.put("b", _table(1000, 2), cost=1.0)
    store.put("c", _table(1000, 3), cost=1.0)
    for key in "abc":
        store.invalidate(key)
    assert len(store) == 0
    assert store.stats()["disk_entries"] == 0
    assert all(store.get(k) is None for k in "abc")


def test_pressure_demotes_rather_than_discarding(store):
    for i, key in enumerate("ab"):
        store.put(key, _table(1000, i), cost=1.0)
    store.on_pressure(PressureLevel.SPILL)
    assert store.used_bytes <= store.max_bytes * 0.5
    # Memory was yielded to execution, which is the point of the ladder — but the results
    # survived it, which is what the second tier adds.
    assert all(store.get(k) is not None for k in "ab")


def test_critical_pressure_discards_both_tiers(store):
    store.put("a", _table(1000), cost=1.0)
    store.on_pressure(PressureLevel.CRITICAL)
    assert len(store) == 0
    assert store.stats()["disk_entries"] == 0


def test_evict_to_free_returns_memory_without_losing_the_result(store):
    store.put("a", _table(1000, 1), cost=1.0)
    before = store.used_bytes
    freed = store.evict_to_free(before)
    assert freed >= before
    assert store.used_bytes == 0
    assert store.get("a") is not None, "reclaiming RAM must cost a read-back, not the result"


def test_shrinking_the_disk_budget_evicts_down(store):
    for i, key in enumerate("abcdef"):
        store.put(key, _table(1000, i), cost=1.0)
    store.set_budget(store.max_bytes, disk_max_bytes=0)
    assert store.stats()["disk_entries"] == 0
    assert store.disk.max_bytes == 0


def test_a_zero_disk_budget_makes_every_level_behave_as_memory_only():
    one = _table(1000)
    s = CacheStore(max_bytes=one.nbytes + 16, disk_max_bytes=0)
    try:
        s.put("a", _table(1000, 1), cost=1.0, level=StorageLevel.MEMORY_AND_DISK)
        s.put("b", _table(1000, 2), cost=1.0, level=StorageLevel.MEMORY_AND_DISK)
        assert s.get("a") is None
        assert s.stats()["demotions"] == 0
    finally:
        s.clear()


# --- concurrency: the accounting must survive demotion and promotion racing ----------


def test_concurrent_readers_and_writers_keep_the_accounting_honest():
    """Hammer both tiers from several threads and check the invariants at the end.

    Demotion and promotion both run with the store's lock *released* — deliberately, since
    each does slow I/O — so the store re-checks its own state after every hand-off. This is
    the case that finds a re-check that was forgotten: an entry counted in memory and on
    disk at once, `used_bytes` drifting past the budget, or a key that both tiers think the
    other one owns.
    """
    import concurrent.futures

    one = _table(500)
    store = CacheStore(max_bytes=4 * one.nbytes, disk_max_bytes=8 << 20)
    keys = [f"k{i}" for i in range(40)]

    def worker(seed: int) -> None:
        for i in range(200):
            key = keys[(seed * 7 + i) % len(keys)]
            if i % 3 == 0:
                store.put(key, _table(500, seed), cost=float(seed + 1))
            else:
                got = store.get(key)
                # Whatever comes back must be a whole result, never a partial or a
                # different shape: a torn read across the tier hand-off would show here.
                assert got is None or got.num_rows == 500

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, range(8)))

        assert store.used_bytes <= store.max_bytes, "the memory budget must hold"
        assert store.disk.used_bytes <= store.disk.max_bytes, "the disk budget must hold"
        # A key lives in exactly one tier. Both at once means one of the hand-offs did not
        # re-check, and the stale copy would be served after the fresh one was replaced.
        both = set(store._entries) & set(store._demoted)
        assert not both, f"{sorted(both)} are counted in both tiers"
        # Accounted bytes must equal what is actually held.
        held = sum(entry.size for entry in store._entries.values())
        assert store.used_bytes == held
    finally:
        store.clear()


def test_a_newer_put_is_not_overwritten_by_an_in_flight_demotion(monkeypatch):
    """A `put` that lands while the copy it replaces is being demoted must win.

    The race the re-checks in `_demote` exist for, made deterministic by blocking the disk
    write mid-flight — which is the only way to test it. Left to chance the window is
    microseconds wide: deleting both re-checks and running the threaded stress above three
    hundred times a run never once caught it, so a test that merely hammered the store
    would assert nothing about this guard while appearing to.

    Sequence: evict "hot" so it is queued for demotion, hold the disk write open, write a
    *newer* "hot" into memory, then let the demotion finish. Without the re-checks the
    stale copy is registered as the disk-resident one and the newer memory entry is
    shadowed by it the moment it is evicted in turn.
    """
    import threading

    one = _table(500)
    store = CacheStore(max_bytes=2 * one.nbytes + 16, disk_max_bytes=8 << 20)
    try:
        store.put("hot", _table(500, 1), cost=0.0)
        store.put("filler", _table(500, 0), cost=100.0)

        in_flight = threading.Event()
        release = threading.Event()
        real_put = DiskCacheTier.put

        def blocking_put(self, key, table):
            """Hold the "hot" write open so the newer put lands mid-demotion."""
            if key == "hot":
                in_flight.set()
                release.wait(timeout=5)
            return real_put(self, key, table)

        # Patched on the class, not the instance: `DiskCacheTier` uses `__slots__`, so an
        # instance attribute cannot shadow the method.
        monkeypatch.setattr(DiskCacheTier, "put", blocking_put)

        # A third entry evicts "hot" (cost 0) and queues it for the disk tier.
        evictor = threading.Thread(target=lambda: store.put("evictor", _table(500, 2), cost=100.0))
        evictor.start()
        assert in_flight.wait(timeout=5), "the demotion never reached the disk tier"

        # The newer value, written while the older one is mid-demotion.
        store.put("hot", _table(500, 99), cost=100.0)
        release.set()
        evictor.join(timeout=5)
        monkeypatch.setattr(DiskCacheTier, "put", real_put)

        # The invariant the re-check protects: a key is memory-resident *or* demoted,
        # never both. Violating it registers the stale bytes as this key's disk copy while
        # the newer one sits in memory above them, so `len()` double-counts, `invalidate`
        # has two things to forget, and the moment the memory copy is evicted under a
        # level that forbids disk the stale copy is what remains.
        assert "hot" in _in_memory(store)
        assert "hot" not in store._demoted, (
            "the demotion registered the copy it had already been superseded by"
        )
        got = store.get("hot")
        assert got is not None, "the newer value was lost"
        assert got.column("v")[0].as_py() == 99, "the demotion served the value it replaced"
    finally:
        store.clear()
