"""The shared result cache as the engine uses it: serialize, store, count, never fail.

`base` says what a shared cache *is* and when a result may go in one; the backends speak
to Redis and RocksDB. This is the piece between them, and it holds the three behaviours
that must be identical whichever backend is configured:

- **Serialization.** Results cross as Arrow IPC (`plan.types.ipc`), so the schema a
  process reads back is exactly the one another process wrote.
- **Failure is a miss.** Every backend call is wrapped. A shared store that is down, slow,
  or holding a value this version cannot decode must make queries *slower*, never make
  them fail — the result is recomputable by construction, so there is no failure mode here
  worth surfacing to a query that could simply run.
- **Measurement.** Hits, misses, writes and errors, because a shared cache is remote and
  its value is entirely an empirical question: a hit saves a whole query, a miss costs a
  round trip, and only the ratio says which is happening.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed
from batcher._internal.mathx import safe_div
from batcher.carbonite.cache_shared.base import SharedCache, SharedCacheError
from batcher.plan.types import logical_bytes, table_from_ipc, table_to_ipc

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["SharedResultCache"]


#: The largest result that may be written to a shared store, measured before serializing.
#:
#: The local `cache.CacheStore` declines an entry larger than its whole budget; this store
#: had no equivalent, so a multi-gigabyte result was serialized into a second full copy in
#: the driver and pushed at the network on every run of the query. Both costs land on the
#: query that was *already* the expensive one, and neither is visible: the write fails
#: inside `SharedCacheError` containment, is counted, and is retried identically next time.
#:
#: 256 MiB. Two things bound it from above and they agree closely: Redis refuses a string
#: value over 512 MiB outright, so anything past that is guaranteed waste; and the entry
#: has to be worth a round trip at *both* ends, since every reader pays the fetch and the
#: decode before it can use a row. Below the cap the trade is clearly right — a cached
#: result skips a whole query — and the cap only has to keep the clearly-wrong case out.
_MAX_SHARED_BYTES = 256 << 20


class SharedResultCache:
    """A `SharedCache` backend plus serialization, error containment, and counters."""

    __slots__ = (
        "_backend",
        "_bytes_read",
        "_bytes_written",
        "_declined",
        "_errors",
        "_hits",
        "_lock",
        "_lookups",
        "_misses",
        "_ttl",
        "_writes",
    )

    def __init__(self, backend: SharedCache, ttl_seconds: int | None = None) -> None:
        """Wrap `backend` as the engine's shared result cache.

        Args:
            backend: The keyed blob store to read and write.
            ttl_seconds: Expiry applied to every write, or `None` for no expiry. A TTL is
                a second line of defence behind the content version in the key: the key
                already changes when an input changes, so this bounds staleness only for
                a source whose version token is coarser than its data, and bounds the
                store's growth when nobody configured an eviction policy.
        """
        self._backend = backend
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        # Every `get` that reached this store, including the ones a contained backend
        # error ended. `hits / (hits + misses)` counts only the lookups that got an
        # *answer*, so a store failing half its reads reported a perfect hit rate.
        self._lookups = 0
        # Writes refused by the size guard. Distinct from `errors`: nothing went wrong,
        # the result is simply not worth shipping, and a workload whose results are all
        # too large needs to know the shared cache is doing nothing for it.
        self._declined = 0
        self._writes = 0
        self._errors = 0
        self._bytes_read = 0
        self._bytes_written = 0

    def get(self, key: str) -> pa.Table | None:
        """Read and deserialize the result under `key`, or `None` on a miss.

        Args:
            key: The content-addressed cache key from `shareable_key`.

        Returns:
            The cached table, or `None` when the key is absent, the store is unreachable,
            or the stored bytes cannot be decoded.
        """
        with self._lock:
            self._lookups += 1
        try:
            raw = self._backend.get(key)
        except SharedCacheError as exc:
            self._record_error("read from the shared result cache", exc)
            return None
        if raw is None:
            with self._lock:
                self._misses += 1
            return None
        try:
            table = table_from_ipc(raw)
        except Exception as exc:
            # A truncated or foreign value. Drop it rather than leaving something no
            # version of the engine can read to be re-fetched on every query.
            self._record_error("decode a shared result-cache entry", exc)
            self.invalidate(key)
            return None
        with self._lock:
            self._hits += 1
            self._bytes_read += len(raw)
        return table

    def put(self, key: str, table: pa.Table) -> None:
        """Serialize and store `table` under `key`.

        A result larger than `_MAX_SHARED_BYTES` is declined and counted rather than
        written. The check is made on the table, *before* `table_to_ipc`, because the
        serialization is itself half the cost being avoided: it builds a second full copy
        of the result in this process, and does so on the query that was already the
        expensive one.

        Args:
            key: The content-addressed cache key from `shareable_key`.
            table: The result to store.
        """
        if logical_bytes(table) > _MAX_SHARED_BYTES:
            with self._lock:
                self._declined += 1
            return
        try:
            raw = table_to_ipc(table)
        except Exception as exc:  # an exotic column type Arrow IPC cannot frame
            self._record_error("serialize a result for the shared cache", exc)
            return
        try:
            self._backend.put(key, raw, self._ttl)
        except SharedCacheError as exc:
            self._record_error("write to the shared result cache", exc)
            return
        with self._lock:
            self._writes += 1
            self._bytes_written += len(raw)

    def invalidate(self, key: str) -> None:
        """Remove `key` from the shared store.

        Args:
            key: The content-addressed cache key to drop.
        """
        try:
            self._backend.delete(key)
        except SharedCacheError as exc:
            self._record_error("delete a shared result-cache entry", exc)

    def stats(self) -> dict[str, int | float]:
        """Shared-cache effectiveness: hits, misses, writes, errors, and bytes moved.

        Returns:
            The counts, the aggregate hit-rate (`0.0` before any read), and the bytes read
            and written. `errors` is the figure to watch first: a shared cache degrades
            silently by design, so a store that is unreachable looks exactly like a store
            that is cold until this is non-zero. `declined` is the second: it is not a
            failure, but a workload whose results all exceed the size guard is paying for
            a shared store that can never answer it.
        """
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                # Over *lookups*, not over hits plus misses. A read the backend failed is
                # a lookup that cost a round trip and returned nothing, so excluding it
                # let an unreachable store report the hit rate of a healthy one.
                "hit_rate": safe_div(self._hits, self._lookups),
                "writes": self._writes,
                "declined": self._declined,
                "errors": self._errors,
                "bytes_read": self._bytes_read,
                "bytes_written": self._bytes_written,
            }

    def close(self) -> None:
        """Release the backend's connection or database handle."""
        try:
            self._backend.close()
        except Exception as exc:
            note_suppressed("carbonite", "close the shared result cache", exc)

    def _record_error(self, step: str, exc: BaseException) -> None:
        """Count and log a contained failure."""
        with self._lock:
            self._errors += 1
        note_suppressed("carbonite", step, exc)
