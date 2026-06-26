"""The result cache — a memory-bounded LRU of materialized query results.

Carbonite owns the engine's *storage* memory the way it owns execution memory: a
single process-wide [`CacheStore`] holds the Arrow results of `Dataset.cache()`d
plans, keyed by an opaque string (the conductor builds it from the plan signature
plus the inputs' identities, so a changed input misses). The store is bounded by a
byte budget and evicts least-recently-used entries — a cached result never grows the
process without bound, and it yields that RAM back to execution under memory pressure
(`on_pressure`), the storage-vs-execution split Spark's `UnifiedMemoryManager` makes.

This module is the storage half of `.claude/rules/architecture.md`'s resource
manager: it accounts and evicts, it never optimizes or executes. It speaks only
plain strings and `pyarrow.Table`s, so it imports no other subsystem (the conductor
in `api` computes the key — Carbonite cannot import `kyber`).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import pyarrow as pa

from batcher.carbonite.memory.pressure import PressureLevel
from batcher.config import active_config

__all__ = ["CacheStore", "current_result_cache", "result_cache"]


@dataclass(slots=True)
class _Entry:
    """One cached result and the inputs that determine its eviction value."""

    table: pa.Table
    keepalive: object
    cost: float  # wall-clock seconds the result took to compute (recompute cost)
    hits: int  # times served since insertion (access frequency)

    def value(self) -> float:
        """Greedy-Dual-Size-Frequency keep-value: recompute-cost x frequency / size.

        Higher means more worth keeping. Expensive, frequently-served, *small* results
        score high; cheap, cold, *large* ones score low and are evicted first — far
        better than plain LRU when cached results vary by orders of magnitude in both
        recompute cost and size. The `+`-ones keep a zero-cost or never-hit entry
        comparable (ordered by size), and the size floor avoids divide-by-zero.
        """
        size = max(1, self.table.nbytes)
        return (self.cost + 1e-9) * (self.hits + 1) / size


class CacheStore:
    """A thread-safe, byte-bounded LRU cache of `pyarrow.Table` results.

    Bounded by `max_bytes`: an insert evicts least-recently-used entries until the
    total fits, and a single result larger than the whole budget is **not** cached
    (caching it would evict everything else for one entry — Spark's `MemoryStore`
    rule). A `get` hit refreshes recency. All operations are guarded by one lock; the
    store is shared process-wide, so concurrent queries see one consistent budget.

    The bytes counted are `Table.nbytes` (the Arrow buffers). Shared/dictionary
    buffers can make that an under-count, so keep the budget conservative relative to
    real RAM.
    """

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max(0, max_bytes)
        # key -> _Entry. The keep-alive pins whatever the caller derived the key from
        # (the input source objects) for the entry's lifetime, so an identity-based key
        # (e.g. `id(source)`) can never collide with a *different* object that reused
        # the id — an evicted entry simply misses. Only the table's bytes count against
        # the budget; eviction is cost-aware (see `_Entry.value`).
        self._entries: dict[str, _Entry] = {}
        self._used = 0
        self._lock = threading.Lock()

    @property
    def max_bytes(self) -> int:
        """The cache's byte budget."""
        return self._max_bytes

    @property
    def used_bytes(self) -> int:
        """Bytes currently held by cached results."""
        return self._used

    def get(self, key: str) -> pa.Table | None:
        """Return the cached result for `key` (counting the hit), or `None`."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.hits += 1  # access frequency feeds the keep-value
                return entry.table
            return None

    def put(self, key: str, table: pa.Table, keepalive: object = None, cost: float = 0.0) -> None:
        """Cache `table` under `key`, evicting low-value entries to stay within budget.

        `keepalive` is pinned for the entry's lifetime — pass whatever the key was
        derived from (the input source objects) so an identity-based key stays valid.
        `cost` is the wall-clock seconds the result took to compute; with size and
        access frequency it drives cost-aware eviction (`_Entry.value`), so an
        expensive result outlives a cheap one. A no-op when the budget is zero or the
        table alone exceeds it (an entry too big to cache is skipped rather than
        thrashing out everything else).
        """
        size = table.nbytes
        with self._lock:
            if self._max_bytes == 0 or size > self._max_bytes:
                return
            existing = self._entries.pop(key, None)
            if existing is not None:
                self._used -= existing.table.nbytes
            self._entries[key] = _Entry(table=table, keepalive=keepalive, cost=cost, hits=0)
            self._used += size
            self._evict_to(self._max_bytes)

    def invalidate(self, key: str) -> None:
        """Drop `key` from the cache if present (e.g. its input changed)."""
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None:
                self._used -= entry.table.nbytes

    def clear(self) -> None:
        """Evict everything, returning all storage memory."""
        with self._lock:
            self._entries.clear()
            self._used = 0

    def evict_to_free(self, n_bytes: int) -> int:
        """Drop the lowest-value entries until at least `n_bytes` are freed, returning
        the bytes actually freed.

        The precise execution-reclaims-storage primitive: when a query needs memory the
        pool can't grant, it frees *exactly* the deficit from the cache (cheapest, then
        coldest/largest) so total RSS stays bounded without dropping the whole cache.
        """
        if n_bytes <= 0:
            return 0
        with self._lock:
            before = self._used
            self._evict_to(max(0, self._used - n_bytes))
            return before - self._used

    def on_pressure(self, level: PressureLevel) -> None:
        """Yield storage memory to execution as memory pressure rises.

        The ladder mirrors the execution side: at `ELEVATED` trim the cache to most
        of its budget (drop the coldest entries), at `SPILL` halve it, and at
        `CRITICAL` evict everything — storage always yields to execution, never the
        reverse, so the cache can never starve a running query.
        """
        if level >= PressureLevel.CRITICAL:
            self.clear()
        elif level >= PressureLevel.SPILL:
            with self._lock:
                self._evict_to(self._max_bytes // 2)
        elif level >= PressureLevel.ELEVATED:
            with self._lock:
                self._evict_to(self._max_bytes * 3 // 4)

    def _evict_to(self, target_bytes: int) -> None:
        """Evict the lowest-value entries until `used <= target_bytes`.

        Caller holds the lock. Each round drops the entry with the smallest
        `_Entry.value` (cheap, cold, large → goes first); ties break by insertion order
        (the oldest), so a never-hit zero-cost set degrades to size-then-FIFO.
        """
        while self._used > target_bytes and self._entries:
            victim = min(self._entries, key=lambda k: self._entries[k].value())
            self._used -= self._entries.pop(victim).table.nbytes


_result_cache: CacheStore | None = None
_result_cache_lock = threading.Lock()


def result_cache() -> CacheStore:
    """The process-wide result cache, created once from the active config budget.

    One store per process so every query draws on (and evicts against) the same
    storage envelope. The budget is `MemoryConfig.result_cache_max_bytes`; later
    calls reconcile the budget if the config changed, evicting down if it shrank.
    """
    global _result_cache
    budget = active_config().memory.result_cache_max_bytes
    if _result_cache is None:
        with _result_cache_lock:
            if _result_cache is None:
                _result_cache = CacheStore(budget)
                return _result_cache
    if _result_cache.max_bytes != budget:
        with _result_cache._lock:
            _result_cache._max_bytes = max(0, budget)
            _result_cache._evict_to(_result_cache._max_bytes)
    return _result_cache


def current_result_cache() -> CacheStore | None:
    """The process-wide result cache if one has been created, else `None`."""
    return _result_cache
