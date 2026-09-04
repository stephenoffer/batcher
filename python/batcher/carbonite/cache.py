"""The result cache — a memory-bounded, cost-aware store of materialized query results.

Carbonite owns the engine's *storage* memory the way it owns execution memory: a
single process-wide [`CacheStore`] holds the Arrow results of `Dataset.cache()`d
plans, keyed by an opaque string (the conductor builds it from the plan signature
plus the inputs' identities, so a changed input misses). The store is bounded by a
byte budget and evicts by **Greedy-Dual-Size-Frequency** — recompute cost and access
frequency against retained size, over a rising inflation floor that ages out what has
stopped being asked for (`_Entry.value`). A cached result never grows the process
without bound, and it yields that RAM back to execution under memory pressure
(`on_pressure`), the storage-vs-execution split Spark's `UnifiedMemoryManager` makes.

Not LRU, though it was described as such for long enough to be worth saying plainly:
recency is one of three inputs here, not the ranking.

Eviction is not the end of an entry. Under `StorageLevel.MEMORY_AND_DISK` — the default —
what the memory budget sheds is **demoted** to `cache_disk.DiskCacheTier` rather than
dropped, and a later `get` reads it back; under `DISK_ONLY` a result never occupies the
memory envelope at all. That is what makes the cache useful to a working set larger than
RAM, which is the working set that asked for a cache: a memory-only store answers a full
budget by forgetting, so every recall becomes a full recompute exactly when recall matters
most.

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
from batcher.carbonite.cache_disk import DiskCacheTier
from batcher.carbonite.memory.pressure import PressureLevel
from batcher.config import active_config
from batcher.plan.resource import StorageLevel
from batcher.plan.types import logical_bytes
from batcher.plan.types import retained_bytes as _retained_bytes

__all__ = ["CacheStore", "current_result_cache", "reset_result_cache", "result_cache"]


# `_retained_bytes` is `plan.types.retained_bytes`: what the table keeps resident, not the
# size of the rows it addresses. Measured on this engine, `limit(10)` over 2M rows reports
# 160 bytes from `nbytes` and retains 262,144 — morselization bounds the per-entry ratio at
# one morsel, and a table from a source that does not morselize carries no bound at all.


#: How many times a table's retained footprint may exceed its logical size before the
#: store compacts it on insert. Any slice retains its parent's buffers, so some excess is
#: normal and copying for it would cost more than it saves; a 4x gap means most of what the
#: entry pins is not what the entry *is*.
_COMPACT_RATIO = 4.0
#: Below this, the excess is not worth a copy whatever the ratio says — a 200-byte entry
#: pinning 4 KiB is not a memory problem, and the copy is pure overhead.
_COMPACT_FLOOR_BYTES = 1 << 20


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
    if retained < _COMPACT_FLOOR_BYTES or retained < logical_bytes(table) * _COMPACT_RATIO:
        return table, retained
    try:
        compact = table.take(pa.array(range(table.num_rows), type=pa.int64()))
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowMemoryError):
        return table, retained  # an exotic column type; account it honestly instead
    after = _retained_bytes(compact)
    return (compact, after) if after < retained else (table, retained)


@dataclass(slots=True)
class _Entry:
    """One memory-resident cached result and the inputs that determine its eviction value."""

    table: pa.Table
    keepalive: object
    cost: float  # wall-clock seconds the result took to compute (recompute cost)
    hits: int  # times served since insertion (access frequency)
    # The retained footprint measured at insertion, and the single number the budget is
    # kept against. Stored rather than re-derived at eviction: a store that measures one
    # way on the way in and another on the way out leaks its accounting a little on every
    # entry, and the leak is invisible until the budget stops meaning anything.
    size: int
    # Which media this result may occupy. Recorded per entry rather than per store because
    # it is the *caller's* declaration about one result, and two datasets cached in one
    # process routinely want different answers — a small hot lookup table wants memory, a
    # re-scanned fact table wants disk.
    level: StorageLevel = StorageLevel.MEMORY_AND_DISK
    # The store's eviction clock as of this entry's last *access* (its admission counts
    # as one) — GDSF's `L`. Carried per entry rather than read from the store at ranking
    # time so an entry's keep-value stays a pure function of itself, which is what lets
    # `_evict_to` sort once instead of re-ranking after every victim.
    base: float = 0.0

    def value(self) -> float:
        """Greedy-Dual-Size-Frequency keep-value: `L` + recompute-cost x frequency / size.

        Higher means more worth keeping. Expensive, frequently-served, *small* results
        score high; cheap, cold, *large* ones score low and are evicted first — far
        better than plain LRU when cached results vary by orders of magnitude in both
        recompute cost and size. The `+`-ones keep a zero-cost or never-hit entry
        comparable (ordered by size), and the size floor avoids divide-by-zero.

        Size is the *retained* footprint, so an entry that pins a large parent buffer
        ranks as the large entry it is rather than as the small window it addresses.

        `base` is Cherkasova's inflation term, and without it the ranking is not GDSF but
        the frequency half of it — which freezes. A result hit a thousand times during
        warmup outranks every later arrival forever, because a fresh entry starts at zero
        hits and is therefore always the cheapest victim: it is evicted by the `put` that
        admitted it, and the store serves whatever got hot first for the rest of the
        process's life. Admitting each entry at the keep-value of the last thing evicted
        prices it against the current floor rather than against an unbounded history, so a
        working set that moves can actually displace one that has stopped being asked for.
        """
        size = max(1, self.size)
        return self.base + (self.cost + 1e-9) * (self.hits + 1) / size


@dataclass(slots=True)
class _Demoted:
    """A result the memory budget shed that now lives on the disk tier.

    Held separately from `_Entry` rather than as an `_Entry` with a `None` table, so the
    memory accounting has exactly one kind of thing to iterate and eviction cannot pick a
    victim that occupies no memory. What survives the demotion is everything needed to
    promote the result back: the keep-alive that keeps the identity-based key valid, the
    recompute cost that ranks it once it is resident again, and the level that says
    whether promotion is allowed at all (`DISK_ONLY` says it is not).
    """

    keepalive: object
    cost: float
    level: StorageLevel


class CacheStore:
    """A thread-safe, byte-bounded, cost-aware cache of `pyarrow.Table` results.

    Bounded by `max_bytes`: an insert evicts the **lowest-keep-value** entries until the
    total fits, and a single result larger than the whole budget is **not** cached
    (caching it would evict everything else for one entry — Spark's `MemoryStore`
    rule). The ranking is Greedy-Dual-Size-Frequency, not recency — see `_Entry.value`
    and `_evict_to`. A `get` counts as an access and raises the entry's *frequency*; it
    does not reorder anything by time, and nothing here does, so an entry that is
    expensive, small and often served outlives a recently-touched cheap large one. All
    operations are guarded by one lock; the store is shared process-wide, so concurrent
    queries see one consistent budget.

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

    def __init__(self, max_bytes: int, disk_max_bytes: int = 0) -> None:
        self._max_bytes = max(0, max_bytes)
        # The second tier. Constructed unconditionally but inert at a zero budget, and it
        # touches no disk until something is actually demoted — so a deployment that never
        # fills its memory budget pays nothing for the tier existing.
        self._disk = DiskCacheTier(disk_max_bytes)
        # key -> _Demoted, for results that live only on the disk tier. Disjoint from
        # `_entries` by construction: a key is memory-resident or demoted, never both, so
        # the two dicts can never disagree about where an entry's bytes are.
        self._demoted: dict[str, _Demoted] = {}
        # Demotions and promotions over this store's life. Together with the disk tier's
        # own hit rate they answer the question a single hit rate cannot: whether the disk
        # tier is *converting* evictions into recalls, or merely writing bytes nothing
        # reads back — the second means the working set is churning faster than it repeats
        # and the disk budget is being spent for nothing.
        self._demotions = 0
        self._promotions = 0
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
        # GDSF's inflation clock `L`: the highest keep-value this store has ever evicted,
        # and the base every newly admitted entry starts from. It only rises, so it acts
        # as an aging floor — the accumulated frequency of an entry nothing asks for any
        # more is eventually passed by the arrivals that are being asked for. See
        # `_Entry.value` for what goes wrong without it.
        self._clock = 0.0
        self._lock = threading.Lock()

    @property
    def max_bytes(self) -> int:
        """The cache's byte budget."""
        return self._max_bytes

    def set_budget(self, max_bytes: int, disk_max_bytes: int | None = None) -> None:
        """Resize the storage envelope, evicting down at once if it shrank.

        The reconcile `result_cache()` performs. It is a method rather than the module
        function reaching in through `_lock` and `_max_bytes`, because reaching in put the
        store's two invariants — the budget and the accounted bytes — outside the class
        that maintains them, which is exactly where a later edit stops keeping them
        together.

        Args:
            max_bytes: The new byte budget. Negative is treated as zero.
            disk_max_bytes: The new disk-tier budget, or `None` to leave it alone.
        """
        with self._lock:
            self._max_bytes = max(0, max_bytes)
            victims = self._evict_to(self._max_bytes)
        self._demote(victims)
        if disk_max_bytes is not None:
            self._disk.set_budget(disk_max_bytes)

    @property
    def used_bytes(self) -> int:
        """Bytes currently held by cached results in memory."""
        return self._used

    @property
    def disk(self) -> DiskCacheTier:
        """The disk tier backing this store, for its own budget and statistics."""
        return self._disk

    def get(self, key: str) -> pa.Table | None:
        """Return the cached result for `key` (counting the hit), or `None`.

        A memory miss falls through to the disk tier, and a disk hit is promoted back
        into memory when its level allows it — so the entry that is being read repeatedly
        stops paying the read-back on every recall. Promotion can itself evict, which is
        correct: the promoted entry has just proved it is being used, and whatever it
        displaces has not.

        The disk read happens with the store's lock **released**. Holding it would make
        every concurrent `get` of a memory-resident result queue behind one entry's
        decompress, which is precisely the cost the memory tier exists to avoid.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.hits += 1  # access frequency feeds the keep-value
                # Re-base on the *access*, not just the admission. This is what makes the
                # ranking a recency/frequency hybrid rather than frequency-with-aging: an
                # entry still being read carries the current floor, while one that has gone
                # cold keeps the floor from whenever it was last wanted, and the rising
                # clock overtakes it. Basing only at admission drops that half of GDSF.
                entry.base = self._clock
                self._hits += 1
                return entry.table
            if key not in self._demoted:
                self._misses += 1
                return None

        table = self._disk.get(key)
        with self._lock:
            demoted = self._demoted.get(key)
            if table is None or demoted is None:
                # The tier evicted it (or the read failed) between the two lock windows.
                # Self-healing: drop the record and report the miss the caller would have
                # got had the entry never been demoted.
                self._demoted.pop(key, None)
                self._misses += 1
                return None
            self._hits += 1
            if not demoted.level.uses_memory:
                return table
            resident, victims = self._promote(key, table, demoted)
        self._demote(victims)
        if resident:
            # The memory copy is authoritative now, so the disk copy is a duplicate
            # holding disk against the tier's budget for a result already in RAM.
            self._disk.release(key)
        return table

    def stats(self) -> dict[str, int | float]:
        """Result-cache effectiveness: hits, misses, evictions, and how full it is.

        Returns:
            The hit/miss counts and aggregate hit-rate (`0.0` before any get), the byte
            budget and what is held against it, the entry count, and how many entries were
            evicted. Evictions are what disambiguate a poor hit rate: many of them means
            the budget is too small, none of them means the cache is not useful here.
            `demotions`/`promotions` and the `disk_*` figures say whether the second tier
            is converting those evictions back into hits or merely writing bytes nothing
            reads.
        """
        # The tier's reading is taken *before* this store's lock, not under it: it is the
        # only place the two locks would nest, and a snapshot of two counters does not need
        # them to be consistent with each other to be useful.
        disk = self._disk.stats()
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
                "demotions": self._demotions,
                "promotions": self._promotions,
                "disk_entries": disk["entries"],
                "disk_used_bytes": disk["used_bytes"],
                "disk_max_bytes": disk["max_bytes"],
                "disk_hits": disk["hits"],
                "disk_misses": disk["misses"],
                "disk_evictions": disk["evictions"],
            }

    def put(
        self,
        key: str,
        table: pa.Table,
        keepalive: object = None,
        cost: float = 0.0,
        level: StorageLevel = StorageLevel.MEMORY_AND_DISK,
    ) -> None:
        """Cache `table` under `key`, evicting low-value entries to stay within budget.

        `keepalive` is pinned for the entry's lifetime — pass whatever the key was
        derived from (the input source objects) so an identity-based key stays valid.
        `cost` is the wall-clock seconds the result took to compute; with size and
        access frequency it drives cost-aware eviction (`_Entry.value`), so an
        expensive result outlives a cheap one. `level` says which media the result may
        occupy: `DISK_ONLY` goes straight to the disk tier without ever charging the
        memory budget, and `MEMORY_AND_DISK` (the default) demotes there on eviction
        rather than being dropped.

        A no-op when the memory budget is zero or the table alone exceeds it *and* the
        level forbids disk — an entry too big to cache is skipped rather than thrashing
        out everything else. With disk allowed, that same oversized entry is written down
        instead, which is the case the disk tier exists for.

        The table is charged its *retained* footprint, and compacted first when that
        greatly exceeds what it addresses — so what the store holds is a copy the entry
        owns outright, rather than a window pinning a parent it will never serve.
        """
        # Measured and compacted outside the lock: both are pure functions of `table`, and
        # the copy is the one genuinely slow step here. Holding the store's lock across it
        # would stall every concurrent `get` behind one insert's memcpy.
        table, size = _compacted(table, _retained_bytes(table))
        if not level.uses_memory or self._max_bytes == 0 or size > self._max_bytes:
            if level.uses_disk:
                self._write_through(key, table, keepalive, cost, level)
            return
        with self._lock:
            self._forget(key)
            self._entries[key] = _Entry(
                table=table,
                keepalive=keepalive,
                cost=cost,
                hits=0,
                size=size,
                level=level,
                base=self._clock,
            )
            self._used += size
            victims = self._evict_to(self._max_bytes)
        self._demote(victims)

    def invalidate(self, key: str) -> None:
        """Drop `key` from **both** tiers if present (e.g. its input changed).

        Args:
            key: The cache key to forget. An unknown key is ignored.
        """
        with self._lock:
            self._forget(key)
        self._disk.release(key)

    def __len__(self) -> int:
        """How many results are cached right now, across both tiers."""
        return len(self._entries) + len(self._demoted)

    def __contains__(self, key: str) -> bool:
        """Whether `key` is cached, **without** counting a hit or refreshing its value.

        A membership test is not an access, so it must not move the eviction ranking. The
        alternative — probing with `get` — silently promotes an entry every time anything
        merely asks whether it exists. True for a demoted entry as well: where the bytes
        live is the cache's business, not the caller's.
        """
        return key in self._entries or key in self._demoted

    def clear(self) -> None:
        """Evict everything from both tiers, returning all storage memory and disk.

        Counted as evictions, because it is what `on_pressure(CRITICAL)` does and that is
        the store yielding its RAM rather than a reset. `stats` offers the eviction count
        precisely to separate "nobody asked again" from "it was dropped before they could",
        and a whole cache discarded under pressure is the strongest instance of the second —
        it read as zero.
        """
        with self._lock:
            self._evictions += len(self._entries)
            self._entries.clear()
            self._demoted.clear()
            self._used = 0
            # Nothing is left to age against, and carrying the floor forward would price
            # the next admission against a generation that no longer exists.
            self._clock = 0.0
        self._disk.clear()

    def evict_to_free(self, n_bytes: int) -> int:
        """Drop the lowest-value entries until at least `n_bytes` are freed, returning
        the bytes actually freed.

        The precise execution-reclaims-storage primitive: when a query needs memory the
        pool can't grant, it frees *exactly* the deficit from the cache (cheapest, then
        coldest/largest) so total RSS stays bounded without dropping the whole cache.
        Freed memory is not necessarily lost work — an entry whose level allows disk is
        written down on the way out, so reclaiming RAM from the cache costs a read-back
        rather than a recompute.
        """
        if n_bytes <= 0:
            return 0
        with self._lock:
            before = self._used
            victims = self._evict_to(max(0, self._used - n_bytes))
            freed = before - self._used
        self._demote(victims)
        return freed

    def on_pressure(self, level: PressureLevel) -> None:
        """Yield storage memory to execution as memory pressure rises.

        The ladder mirrors the execution side: at `ELEVATED` trim the cache to most
        of its budget (drop the coldest entries), at `SPILL` halve it, and at
        `CRITICAL` evict everything — storage always yields to execution, never the
        reverse, so the cache can never starve a running query.

        Memory pressure is not disk pressure, so what is shed here is demoted rather than
        discarded wherever the level allows: the query that triggered the ladder gets its
        RAM back either way, and the cached results survive it. `CRITICAL` is the one rung
        that does not demote — the process is close enough to the wall that writing several
        hundred megabytes of Arrow through the IPC encoder is the wrong thing to do with
        the memory it would need to do it.
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
            victims = self._evict_to(int(self._max_bytes * retain))
        self._demote(victims)

    # --- internals ---------------------------------------------------------

    def _forget(self, key: str) -> None:
        """Un-account `key` in memory, wherever it currently lives. Caller holds the lock.

        Does not touch the disk tier: the two callers want opposite things from it —
        `put` is about to overwrite the bucket anyway, `invalidate` deletes it explicitly —
        and a tier call under this lock would be the one slow step in an otherwise
        memory-only critical section.
        """
        existing = self._entries.pop(key, None)
        if existing is not None:
            self._used -= existing.size
        self._demoted.pop(key, None)

    def _promote(
        self, key: str, table: pa.Table, demoted: _Demoted
    ) -> tuple[bool, list[tuple[str, _Entry]]]:
        """Move a disk hit back into memory. Caller holds the lock.

        Returns whether the entry is now memory-resident, and any entries its arrival
        evicted. The two are reported separately because "no victims" and "did not
        promote" are different outcomes with opposite consequences for the disk copy: the
        first makes it redundant, the second makes it the only copy there is.

        The entry keeps the recompute cost it was demoted with, and starts at one hit —
        the one that just promoted it — so it is ranked as the used entry it has just
        proved itself to be rather than re-entering as cold.
        """
        size = _retained_bytes(table)
        if self._max_bytes == 0 or size > self._max_bytes:
            return False, []
        self._demoted.pop(key, None)
        self._entries[key] = _Entry(
            table=table,
            keepalive=demoted.keepalive,
            cost=demoted.cost,
            hits=1,
            size=size,
            level=demoted.level,
            base=self._clock,
        )
        self._used += size
        self._promotions += 1
        victims = self._evict_to(self._max_bytes)
        # A promotion can evict itself straight back out, against a cache full of
        # higher-value entries. It is then one of `victims` and `_demote` rewrites it, so
        # the disk copy must not be dropped — hence reporting residency, not just victims.
        return key in self._entries, victims

    def _write_through(
        self, key: str, table: pa.Table, keepalive: object, cost: float, level: StorageLevel
    ) -> None:
        """Store a result on the disk tier only, without charging the memory budget.

        The `DISK_ONLY` path, and the fallback for a result too large for the memory
        envelope. Called with the lock released because the write is slow I/O.
        """
        if self._disk.put(key, table) <= 0:
            return
        with self._lock:
            # `_forget`, not a bare `pop`: a `DISK_ONLY` put over a key that was already
            # memory-resident must give its bytes back to the budget, and a `pop` that
            # leaves `_used` alone leaks exactly that entry's size on every replacement.
            self._forget(key)
            self._demoted[key] = _Demoted(keepalive=keepalive, cost=cost, level=level)

    def _demote(self, victims: list[tuple[str, _Entry]]) -> None:
        """Write evicted entries to the disk tier. Called with the lock **released**.

        The lock is released for the whole of this because a demotion is an IPC encode
        plus a write per entry, and a bulk eviction (`on_pressure` halving the cache) can
        hand over dozens at once. Doing that inside the critical section would stall every
        concurrent reader for as long as the slowest disk.

        A victim that was re-inserted or invalidated in the meantime is skipped: the newer
        write is authoritative, and demoting the stale copy over it would serve the older
        result on the next recall.
        """
        for key, entry in victims:
            if not entry.level.uses_disk or not self._disk.enabled:
                continue
            with self._lock:
                if key in self._entries:
                    continue  # superseded by a newer put; that copy owns the key now
            if self._disk.put(key, entry.table) <= 0:
                continue
            with self._lock:
                if key in self._entries:
                    self._disk.release(key)
                    continue
                self._demoted[key] = _Demoted(
                    keepalive=entry.keepalive, cost=entry.cost, level=entry.level
                )
                self._demotions += 1

    def _evict_to(self, target_bytes: int) -> list[tuple[str, _Entry]]:
        """Evict the lowest-value entries until `used <= target_bytes`, returning them.

        Caller holds the lock. Entries are dropped smallest-`_Entry.value` first (cheap,
        cold, large → goes first); ties break by insertion order (the oldest), so a
        never-hit zero-cost set degrades to size-then-FIFO.

        The victims are *returned* rather than written to disk here, because writing them
        is slow I/O and this runs under the store's lock. The caller demotes them once it
        has released it — see `_demote`.

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
            return []
        evicted: list[tuple[str, _Entry]] = []
        victims = sorted(self._entries.items(), key=lambda kv: kv[1].value())
        for key, entry in victims:
            if self._used <= target_bytes:
                break
            del self._entries[key]
            self._used -= entry.size
            self._evictions += 1
            # The clock rises to what it just discarded, and is monotone without needing
            # a `max` to make it so. Inductively: after an eviction every survivor ranks
            # at or above the victim, which is the new clock; and every subsequent entry
            # is based on the clock at its last access plus a strictly positive term. So
            # no resident entry can ever rank below the clock, and the next victim — the
            # lowest-ranked resident — cannot lower it. A guard was written here first and
            # a randomized probe over 240k operations never once reached it.
            self._clock = entry.value()
            evicted.append((key, entry))
        return evicted


_result_cache: CacheStore | None = None
_result_cache_lock = threading.Lock()


def result_cache() -> CacheStore:
    """The process-wide result cache, created once from the active config budgets.

    One store per process so every query draws on (and evicts against) the same
    storage envelope. The budgets are `MemoryConfig.result_cache_max_bytes` and
    `MemoryConfig.result_cache_disk_max_bytes`; later calls reconcile either if the
    config changed, evicting down if one shrank.
    """
    global _result_cache
    mem = active_config().memory
    budget = mem.result_cache_max_bytes
    disk_budget = mem.result_cache_disk_max_bytes
    cache = _result_cache
    if cache is None:
        with _result_cache_lock:
            if _result_cache is None:
                _result_cache = CacheStore(budget, disk_budget)
                return _result_cache
            cache = _result_cache
    if cache.max_bytes != budget or cache.disk.max_bytes != disk_budget:
        cache.set_budget(budget, disk_budget)
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
        stale, _result_cache = _result_cache, None
    if stale is not None:
        # Unlink the disk tier's scratch directory. Dropping the reference alone leaks it:
        # nothing else knows the path, so the files survive until the process exits and,
        # across a test session that resets between cases, accumulate a directory per reset.
        stale.clear()
