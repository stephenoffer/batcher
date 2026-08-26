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
from batcher.plan.types import table_from_ipc, table_to_ipc

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["SharedResultCache"]


class SharedResultCache:
    """A `SharedCache` backend plus serialization, error containment, and counters."""

    __slots__ = (
        "_backend",
        "_bytes_read",
        "_bytes_written",
        "_errors",
        "_hits",
        "_lock",
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

        Args:
            key: The content-addressed cache key from `shareable_key`.
            table: The result to store.
        """
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
            that is cold until this is non-zero.
        """
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": safe_div(self._hits, total),
                "writes": self._writes,
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
