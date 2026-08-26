"""The result cache's disk tier — where an evicted result goes instead of nowhere.

A memory-only cache answers a full budget by forgetting, so the workload it helps
least is the one that asked for it: a working set larger than RAM evicts every entry
before anything reads it again, and every recall becomes a full recompute. This tier
is the other half. Under `StorageLevel.MEMORY_AND_DISK` an entry the memory store
evicts is *demoted* here rather than dropped; a later `get` reads it back. Under
`DISK_ONLY` it never enters the memory budget at all.

Storage is the `TieredSpillStore` the out-of-core operators already use, which brings
three things worth not re-implementing: Arrow IPC framing with the configured codec,
a local byte budget clamped to the volume that actually exists, and overflow to
`memory.spill_remote_uri` once local disk fills — so a cache on a small-NVMe node
degrades to object storage instead of failing. The scratch directory is resolved by
`carbonite.spill.scratch`, the same way a spilling aggregate resolves its own.

What this tier is **not** is durable. It is process-scoped scratch: the directory is
removed on `clear()` and on interpreter exit, and a cold start re-computes. Sharing a
cache across processes or nodes is a different contract with different correctness
obligations (key stability, invalidation, versioning) and belongs to a shared cache
backend, not to a scratch tier.

Recall is cheaper than recompute here by roughly the ratio of decode bandwidth to
query cost, so the tier earns its disk on expensive results and loses on trivial
ones. That is exactly the ranking `_Entry.value` already computes, so the memory
store's eviction order chooses demotion candidates correctly for free: the entries it
sheds first are the cheap ones, which are also the ones least worth writing down.
"""

from __future__ import annotations

import contextlib
import hashlib
import shutil
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from batcher._internal.logging import note_suppressed
from batcher._internal.mathx import safe_div

if TYPE_CHECKING:
    from batcher.carbonite.spill.handle import SpillHandle
    from batcher.carbonite.spill.store import TieredSpillStore

__all__ = ["DiskCacheTier"]


@dataclass(slots=True)
class _OnDisk:
    """One demoted result: where its bytes are, and what it costs to read them back."""

    handle: SpillHandle
    schema: pa.Schema  # kept so an all-empty read still reconstructs the right table
    num_rows: int
    # The on-disk (compressed) size, which is what the tier's byte budget is kept against.
    # `handle.logical_nbytes` is the resident size a read will occupy — a different figure
    # by the compression ratio, and the one the memory store must budget on promotion.
    nbytes: int


def _bucket_name(key: str) -> str:
    """A filesystem- and object-store-safe bucket name for a cache key.

    Cache keys carry plan signatures, source identities, and a tenant/viewer scope, so
    they contain separators and glob metacharacters that `TieredSpillStore` rejects (it
    validates bucket names precisely so a caller-supplied identifier cannot write outside
    the scratch directory). Hashing sidesteps that without weakening it: the digest is the
    *name*, never the identity — the tier's own dict is keyed by the full string, so a
    digest collision would have to survive an exact key comparison to matter.
    """
    return "c" + hashlib.sha256(key.encode("utf-8")).hexdigest()


class DiskCacheTier:
    """A byte-bounded LRU of cached Arrow results on local disk, overflowing to remote.

    Thread-safe. Every store operation is serialized on one lock because
    `TieredSpillStore` is not itself concurrent; the memory store above calls in from
    *outside* its own lock precisely so a demotion's write does not stall concurrent
    reads of the results still resident.

    Eviction here is plain LRU rather than the memory store's cost-aware ranking. The
    two are ordered, not parallel: an entry only reaches disk after the cost-aware
    ranking already decided it was the least worth keeping, so re-applying that ranking
    would double-count it. What disk adds is a second chance ordered by recency.
    """

    __slots__ = (
        "_entries",
        "_evictions",
        "_hits",
        "_lock",
        "_max_bytes",
        "_misses",
        "_store",
        "_used",
        "_work_dir",
        "_writes",
    )

    def __init__(self, max_bytes: int) -> None:
        """Create the tier with a byte budget; nothing touches disk until the first write.

        Args:
            max_bytes: On-disk byte budget. Zero (or negative) disables the tier
                entirely, which is what makes `MEMORY_AND_DISK` degrade cleanly to
                `MEMORY_ONLY` on a node with no scratch volume to spare.
        """
        self._max_bytes = max(0, int(max_bytes))
        self._lock = threading.Lock()
        # Insertion order is recency order: `get` moves an entry to the end, and eviction
        # takes from the front. A plain dict is an ordered dict since 3.7, and `move_to_end`
        # is spelled as a pop-and-reinsert below rather than importing OrderedDict for it.
        self._entries: dict[str, _OnDisk] = {}
        self._used = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._writes = 0
        self._store: TieredSpillStore | None = None
        self._work_dir: str | None = None

    @property
    def max_bytes(self) -> int:
        """The tier's on-disk byte budget."""
        return self._max_bytes

    @property
    def used_bytes(self) -> int:
        """Bytes currently held on disk by demoted results."""
        return self._used

    @property
    def enabled(self) -> bool:
        """Whether the tier can accept a write at all (a positive budget)."""
        return self._max_bytes > 0

    def set_budget(self, max_bytes: int) -> None:
        """Resize the on-disk envelope, evicting down at once if it shrank.

        Args:
            max_bytes: The new budget. Negative is treated as zero.
        """
        with self._lock:
            self._max_bytes = max(0, int(max_bytes))
            self._evict_to(self._max_bytes)

    def __contains__(self, key: str) -> bool:
        """Whether `key` is on disk, without counting a hit or refreshing its recency."""
        return key in self._entries

    def __len__(self) -> int:
        """How many results are held on disk right now."""
        return len(self._entries)

    def put(self, key: str, table: pa.Table) -> int:
        """Write `table` to disk under `key`, evicting to stay within budget.

        A no-op returning `0` when the tier is disabled, when the table holds no rows
        (an empty result costs nothing to recompute and cannot round-trip through a
        bucket, which is empty by definition), or when the write fails — a cache tier
        that cannot write is a slower cache, never a wrong answer, so a failure is
        recorded and swallowed rather than raised into a query that has its result.

        Args:
            key: The cache key, used verbatim for identity and hashed for the filename.
            table: The result to write.

        Returns:
            The on-disk bytes the entry now occupies, or `0` if nothing was written.
        """
        if not self.enabled or table.num_rows == 0:
            return 0
        with self._lock:
            store = self._ensure_store()
            if store is None:
                return 0
            self._release_locked(key)
            try:
                handle = store.spill(table.to_batches(), name=_bucket_name(key))
            except Exception as exc:  # a full disk, a revoked mount, an exotic column type
                note_suppressed("carbonite", "demote a cached result to disk", exc)
                return 0
            if handle is None:
                return 0
            if handle.nbytes > self._max_bytes:
                # One entry larger than the whole tier would evict everything else to hold
                # itself — the same rule the memory store applies, for the same reason.
                store.release(handle)
                return 0
            self._entries[key] = _OnDisk(
                handle=handle,
                schema=table.schema,
                num_rows=table.num_rows,
                nbytes=handle.nbytes,
            )
            self._used += handle.nbytes
            self._writes += 1
            self._evict_to(self._max_bytes)
            # An entry can be evicted by its own insert only if it no longer fits, which
            # the size guard above already excluded — so a successful put is readable.
            return self._entries[key].nbytes if key in self._entries else 0

    def get(self, key: str) -> pa.Table | None:
        """Read `key` back from disk, counting the hit and refreshing its recency.

        A read that fails (the scratch volume reclaimed under a spot node, a truncated
        file) drops the entry and reports a miss: the result is recomputable by
        construction, so an unreadable cache entry must never surface as a query error.

        Args:
            key: The cache key to look up.

        Returns:
            The cached result, or `None` on a miss.
        """
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None:
                self._misses += 1
                return None
            store = self._store
            if store is None:  # pragma: no cover - an entry cannot exist without a store
                self._used -= entry.nbytes
                self._misses += 1
                return None
            try:
                batches = store.read(entry.handle)
            except Exception as exc:
                note_suppressed("carbonite", "read a cached result back from disk", exc)
                self._drop_locked(entry)
                self._misses += 1
                return None
            self._entries[key] = entry  # reinsert at the end: most recently used
            self._hits += 1
            return pa.Table.from_batches(batches, schema=entry.schema)

    def resident_bytes(self, key: str) -> int:
        """What reading `key` back would occupy in memory, or `0` if it is not held.

        The *uncompressed* size, which is what the memory store must budget against when
        it decides whether to promote a disk hit back into RAM. Budgeting against the
        on-disk size instead under-reserves by exactly the compression ratio, which on a
        well-compressing column is most of the entry.

        Args:
            key: The cache key to size.

        Returns:
            The resident footprint a read would take, falling back to the on-disk size
            when the handle recorded no logical size.
        """
        entry = self._entries.get(key)
        if entry is None:
            return 0
        return entry.handle.logical_nbytes or entry.nbytes

    def release(self, key: str) -> None:
        """Delete `key`'s bytes from disk if present.

        Args:
            key: The cache key to drop. An unknown key is ignored.
        """
        with self._lock:
            self._release_locked(key)

    def clear(self) -> None:
        """Delete every demoted result and the scratch directory holding them.

        Counted as evictions for the same reason the memory store counts its own: a tier
        emptied under pressure is the strongest case of "it was dropped before anyone
        could read it", and reporting that as zero is what makes a poor hit rate
        unreadable.
        """
        with self._lock:
            self._evictions += len(self._entries)
            self._entries.clear()
            self._used = 0
            store, work_dir = self._store, self._work_dir
            self._store = None
            self._work_dir = None
        # Outside the lock: `cleanup` unlinks every bucket and the rmtree walks the tree,
        # neither of which anything else here needs to wait behind.
        if store is not None:
            with contextlib.suppress(Exception):
                store.cleanup()
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)

    def stats(self) -> dict[str, int | float]:
        """Disk-tier effectiveness: hits, misses, writes, evictions, and how full it is.

        Returns:
            The hit and miss counts and the aggregate hit-rate (`0.0` before any read),
            how many results were written down and how many were evicted, the entry
            count, and the on-disk bytes held against the budget.
        """
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": safe_div(self._hits, total),
                "writes": self._writes,
                "evictions": self._evictions,
                "entries": len(self._entries),
                "used_bytes": self._used,
                "max_bytes": self._max_bytes,
                "fill": safe_div(self._used, self._max_bytes),
            }

    # --- internals (caller holds `_lock`) ----------------------------------

    def _ensure_store(self) -> TieredSpillStore | None:
        """The backing store, created on first use, or `None` if it cannot be created.

        Deferred so that enabling the tier in config costs nothing until something is
        actually demoted — a query that never fills the memory budget never makes a
        directory. A creation failure disables the tier rather than raising: no scratch
        volume is a reason to have a smaller cache, not to fail a query.
        """
        if self._store is not None:
            return self._store
        from batcher.carbonite.spill.scratch import make_store, scratch_dir

        try:
            work_dir, _owns = scratch_dir(None, "batcher_cache_")
            self._store = make_store(work_dir)
        except Exception as exc:
            note_suppressed("carbonite", "open the result cache's disk tier", exc)
            self._max_bytes = 0  # stop retrying a directory that cannot be made
            return None
        self._work_dir = work_dir
        return self._store

    def _release_locked(self, key: str) -> None:
        """Drop `key` and delete its bucket. Caller holds the lock."""
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._drop_locked(entry)

    def _drop_locked(self, entry: _OnDisk) -> None:
        """Un-account `entry` (already popped) and unlink its bucket. Caller holds the lock."""
        self._used = max(0, self._used - entry.nbytes)
        if self._store is not None:
            with contextlib.suppress(Exception):
                self._store.release(entry.handle)

    def _evict_to(self, target_bytes: int) -> None:
        """Evict least-recently-used entries until `used <= target_bytes`.

        Caller holds the lock. Insertion order is recency order (`get` reinserts), so
        the eviction order is the dict's own iteration order — no sort, no scan.
        """
        if self._used <= target_bytes:
            return
        for key in list(self._entries):
            if self._used <= target_bytes:
                break
            entry = self._entries.pop(key)
            self._drop_locked(entry)
            self._evictions += 1
