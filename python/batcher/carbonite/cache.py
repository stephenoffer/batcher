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
from typing import ClassVar

import pyarrow as pa

from batcher._internal.mathx import safe_div
from batcher.carbonite.memory.pressure import PressureLevel
from batcher.config import active_config

__all__ = ["CacheStore", "current_result_cache", "reset_result_cache", "result_cache"]


#: How many times a table's retained footprint may exceed its logical size before the
#: store compacts it on insert. Any slice retains its parent's buffers, so some excess is
#: normal and copying for it would cost more than it saves; a 4x gap means most of what the
#: entry pins is not what the entry *is*.
_COMPACT_RATIO = 4.0
#: Below this, the excess is not worth a copy whatever the ratio says — a 200-byte entry
#: pinning 4 KiB is not a memory problem, and the copy is pure overhead.
_COMPACT_FLOOR_BYTES = 1 << 20


def _retained_bytes(table: pa.Table) -> int:
    """Bytes the process cannot reclaim while `table` is referenced.

    **Not** `Table.nbytes`, which is the size of the *rows the table addresses*. A sliced
    or zero-copy-derived table addresses a window of a parent buffer and keeps the whole
    parent alive, so the two figures can differ by orders of magnitude: a 10-row slice of a
    4M-row column reports 80 bytes from `nbytes` and pins 32 MB. A cache budgeted on
    `nbytes` would then hold hundreds of times its budget in real memory, and the entry
    that does it is the *cheapest-looking* one, so eviction never chooses it.

    Measured on this engine: `bt.from_pydict(2M rows).limit(10).collect()` reports 160
    bytes and retains 262,144 — a morsel's buffers, a 1,638x under-count. The engine's
    morselization is what bounds the ratio per entry; a table handed in from a source that
    does not morselize has no such bound.

    `get_total_buffer_size` is the retained figure. It over-counts when two columns share a
    buffer (a dictionary encoded twice), which is the safe direction: over-counting evicts
    an entry sooner and costs a recompute, where under-counting costs the process.

    Args:
        table: The result to measure.

    Returns:
        The retained byte count, falling back to `nbytes` if the table cannot report one.
    """
    try:
        return max(int(table.get_total_buffer_size()), int(table.nbytes))
    except (AttributeError, TypeError):  # pragma: no cover - a table-like without the API
        return int(table.nbytes)


def _compacted(table: pa.Table, retained: int) -> tuple[pa.Table, int]:
    """`table` copied free of any parent buffers it merely windows, and its new footprint.

    Caching a slice is the common shape here: `head`, `limit`, and a selective filter all
    produce one, and all three are exactly the cheap results a user caches. Refusing to
    cache them (the alternative) throws away the useful case to avoid the footprint; taking
    a compacting copy keeps the entry *and* makes its accounted size true.

    `take` rather than `combine_chunks`, which does not compact a single already-contiguous
    chunk and so leaves the parent pinned. The copy runs in Arrow's C++ kernels over whole
    columns, not per row.

    Args:
        table: The result to compact.
        retained: Its footprint before compaction, returned unchanged if the copy is
            skipped or fails.

    Returns:
        The table to store and the bytes to account for it.
    """
    if retained < _COMPACT_FLOOR_BYTES or retained < table.nbytes * _COMPACT_RATIO:
        return table, retained
    try:
        compact = table.take(pa.array(range(table.num_rows), type=pa.int64()))
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowMemoryError):
        return table, retained  # an exotic column type; account it honestly instead
    after = _retained_bytes(compact)
    return (compact, after) if after < retained else (table, retained)


@dataclass(slots=True)
class _Entry:
    """One cached result and the inputs that determine its eviction value."""

    table: pa.Table
    keepalive: object
    cost: float  # wall-clock seconds the result took to compute (recompute cost)
    hits: int  # times served since insertion (access frequency)
    # The retained footprint measured at insertion, and the single number the budget is
    # kept against. Stored rather than re-derived at eviction: a store that measures one
    # way on the way in and another on the way out leaks its accounting a little on every
    # entry, and the leak is invisible until the budget stops meaning anything.
    size: int

    def value(self) -> float:
        """Greedy-Dual-Size-Frequency keep-value: recompute-cost x frequency / size.

        Higher means more worth keeping. Expensive, frequently-served, *small* results
        score high; cheap, cold, *large* ones score low and are evicted first — far
        better than plain LRU when cached results vary by orders of magnitude in both
        recompute cost and size. The `+`-ones keep a zero-cost or never-hit entry
        comparable (ordered by size), and the size floor avoids divide-by-zero.

        Size is the *retained* footprint, so an entry that pins a large parent buffer
        ranks as the large entry it is rather than as the small window it addresses.
        """
        size = max(1, self.size)
        return (self.cost + 1e-9) * (self.hits + 1) / size


class CacheStore:
    """A thread-safe, byte-bounded LRU cache of `pyarrow.Table` results.

    Bounded by `max_bytes`: an insert evicts least-recently-used entries until the
    total fits, and a single result larger than the whole budget is **not** cached
    (caching it would evict everything else for one entry — Spark's `MemoryStore`
    rule). A `get` hit refreshes recency. All operations are guarded by one lock; the
    store is shared process-wide, so concurrent queries see one consistent budget.

    The bytes counted are what an entry keeps *resident* (`_retained_bytes`), not the
    size of the rows it addresses, so a slice cannot enter the cache reporting a fraction
    of what it pins. Where the two differ enough to matter the store takes a compacting
    copy on insert, making the entry as small as it claims to be.
    """

    # Fractions of the budget the cache is trimmed to at each pressure level. Storage always
    # yields to execution, never the reverse, so these only ever shrink the cache.
    _PRESSURE_RETAIN: ClassVar[dict[PressureLevel, float]] = {
        PressureLevel.ELEVATED: 0.75,
        PressureLevel.SPILL: 0.5,
    }

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max(0, max_bytes)
        # key -> _Entry. The keep-alive pins whatever the caller derived the key from
        # (the input source objects) for the entry's lifetime, so an identity-based key
        # (e.g. `id(source)`) can never collide with a *different* object that reused
        # the id — an evicted entry simply misses. Only the table's bytes count against
        # the budget; eviction is cost-aware (see `_Entry.value`).
        self._entries: dict[str, _Entry] = {}
        self._used = 0
        # Store-level hit/miss counters (distinct from per-entry `hits`, which drives keep
        # value): the aggregate hit-rate tells whether the result cache is *earning its RAM*.
        self._hits = 0
        self._misses = 0
        # Entries dropped to stay within budget over this store's life. A hit rate alone
        # cannot distinguish "nobody asked for it again" from "it was evicted before they
        # could" — the first says the cache is not useful here, the second says it is too
        # small — and those call for opposite responses.
        self._evictions = 0
        self._lock = threading.Lock()

    @property
    def max_bytes(self) -> int:
        """The cache's byte budget."""
        return self._max_bytes

    def set_budget(self, max_bytes: int) -> None:
        """Resize the storage envelope, evicting down at once if it shrank.

        The reconcile `result_cache()` performs. It is a method rather than the module
        function reaching in through `_lock` and `_max_bytes`, because reaching in put the
        store's two invariants — the budget and the accounted bytes — outside the class
        that maintains them, which is exactly where a later edit stops keeping them
        together.

        Args:
            max_bytes: The new byte budget. Negative is treated as zero.
        """
        with self._lock:
            self._max_bytes = max(0, max_bytes)
            self._evict_to(self._max_bytes)

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
                self._hits += 1
                return entry.table
            self._misses += 1
            return None

    def stats(self) -> dict[str, int | float]:
        """Result-cache effectiveness: hits, misses, evictions, and how full it is.

        Returns:
            The hit/miss counts and aggregate hit-rate (`0.0` before any get), the byte
            budget and what is held against it, the entry count, and how many entries were
            evicted. Evictions are what disambiguate a poor hit rate: many of them means
            the budget is too small, none of them means the cache is not useful here.
        """
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": safe_div(self._hits, total),
                "evictions": self._evictions,
                "entries": len(self._entries),
                "used_bytes": self._used,
                "max_bytes": self._max_bytes,
                "fill": safe_div(self._used, self._max_bytes),
            }

    def put(self, key: str, table: pa.Table, keepalive: object = None, cost: float = 0.0) -> None:
        """Cache `table` under `key`, evicting low-value entries to stay within budget.

        `keepalive` is pinned for the entry's lifetime — pass whatever the key was
        derived from (the input source objects) so an identity-based key stays valid.
        `cost` is the wall-clock seconds the result took to compute; with size and
        access frequency it drives cost-aware eviction (`_Entry.value`), so an
        expensive result outlives a cheap one. A no-op when the budget is zero or the
        table alone exceeds it (an entry too big to cache is skipped rather than
        thrashing out everything else).

        The table is charged its *retained* footprint, and compacted first when that
        greatly exceeds what it addresses — so what the store holds is a copy the entry
        owns outright, rather than a window pinning a parent it will never serve.
        """
        # Measured and compacted outside the lock: both are pure functions of `table`, and
        # the copy is the one genuinely slow step here. Holding the store's lock across it
        # would stall every concurrent `get` behind one insert's memcpy.
        table, size = _compacted(table, _retained_bytes(table))
        with self._lock:
            if self._max_bytes == 0 or size > self._max_bytes:
                return
            existing = self._entries.pop(key, None)
            if existing is not None:
                self._used -= existing.size
            self._entries[key] = _Entry(
                table=table, keepalive=keepalive, cost=cost, hits=0, size=size
            )
            self._used += size
            self._evict_to(self._max_bytes)

    def invalidate(self, key: str) -> None:
        """Drop `key` from the cache if present (e.g. its input changed)."""
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None:
                self._used -= entry.size

    def __len__(self) -> int:
        """How many results are cached right now."""
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        """Whether `key` is cached, **without** counting a hit or refreshing its value.

        A membership test is not an access, so it must not move the eviction ranking. The
        alternative — probing with `get` — silently promotes an entry every time anything
        merely asks whether it exists.
        """
        return key in self._entries

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
            return
        retain = self._PRESSURE_RETAIN.get(
            PressureLevel.SPILL if level >= PressureLevel.SPILL else PressureLevel.ELEVATED
        )
        if level < PressureLevel.ELEVATED or retain is None:
            return
        with self._lock:
            self._evict_to(int(self._max_bytes * retain))

    def _evict_to(self, target_bytes: int) -> None:
        """Evict the lowest-value entries until `used <= target_bytes`.

        Caller holds the lock. Entries are dropped smallest-`_Entry.value` first (cheap,
        cold, large → goes first); ties break by insertion order (the oldest), so a
        never-hit zero-cost set degrades to size-then-FIFO.

        An entry's keep-value is independent of which *other* entries remain, so the
        eviction order is a single stable sort — not a fresh O(n) min-scan per victim.
        That makes a bulk eviction (`on_pressure` halving the cache, a large insert pushing
        out many small entries) O(n log n) instead of O(n²), and computes each `value()`
        once instead of once per comparison per round. Stable sort keeps insertion order
        among equal values, so a never-hit zero-cost set degrades to size-then-FIFO.

        A heap was tried here for the common single-victim case, on the reasoning that
        `heapify` is O(n) where the sort is O(n log n). Measured over 2k / 10k / 40k
        entries it was 1.10x / 0.86x / 1.00x — a wash, because what dominates is computing
        `value()` once per entry and materializing the key sequence, which both approaches
        pay identically, while Timsort's extra comparisons run in C over precomputed keys.
        The sort stays: same cost, less machinery. Recorded so the next reader does not
        re-derive the same idea and re-measure it.
        """
        if self._used <= target_bytes or not self._entries:
            return
        victims = sorted(self._entries.items(), key=lambda kv: kv[1].value())
        for key, entry in victims:
            if self._used <= target_bytes:
                break
            del self._entries[key]
            self._used -= entry.size
            self._evictions += 1


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
    cache = _result_cache
    if cache is None:
        with _result_cache_lock:
            if _result_cache is None:
                _result_cache = CacheStore(budget)
                return _result_cache
            cache = _result_cache
    if cache.max_bytes != budget:
        cache.set_budget(budget)
    return cache


def current_result_cache() -> CacheStore | None:
    """The process-wide result cache if one has been created, else `None`."""
    return _result_cache


def reset_result_cache() -> None:
    """Drop the process-wide result cache so the next call builds a fresh one.

    For tests, which otherwise inherit whatever entries and hit/miss counters an earlier
    test left behind — the same reason `reset_process_pool` exists for the buffer pool.
    """
    global _result_cache
    with _result_cache_lock:
        _result_cache = None
