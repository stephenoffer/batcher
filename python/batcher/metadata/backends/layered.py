"""Layered backend — a fast in-process cache over a durable shared store.

Reads hit the local cache first and fall through to the shared store (object
storage / Redis) on a miss; writes go through to *both*, so this driver sees its own
updates instantly while every other driver sees them durably. `refresh()` drops the
cache so the next read re-pulls the shared store — the cross-driver freshness hook
the optimizer calls between runs to pick up statistics other drivers have learned.
This keeps the common read path off the network without giving up cluster sharing.
"""

from __future__ import annotations

from collections.abc import Iterator

from batcher.metadata.backends.in_process import InProcessBackend
from batcher.metadata.store import Key

__all__ = ["LayeredBackend"]


def _durable_from_uri(uri: str | None):
    """Build the durable shared store from a uri scheme (`redis://` → Redis, else
    object storage)."""
    if uri and uri.startswith(("redis://", "rediss://", "unix://")):
        from batcher.metadata.backends.redis import RedisBackend

        return RedisBackend(uri)
    from batcher.metadata.backends.object_storage import ObjectStorageBackend

    return ObjectStorageBackend(uri)


class LayeredBackend:
    """A `MetadataBackend` caching a durable shared backend behind a local dict."""

    def __init__(self, shared, cache=None) -> None:
        self._shared = shared
        self._cache = cache if cache is not None else InProcessBackend()

    @classmethod
    def from_uri(cls, uri: str | None) -> LayeredBackend:
        """Build a layered cache over the shared store the `uri` scheme selects."""
        return cls(_durable_from_uri(uri))

    def get(self, table: str, key: Key) -> bytes | None:
        cached = self._cache.get(table, key)
        if cached is not None:
            return cached
        value = self._shared.get(table, key)
        if value is not None:
            self._cache.put(table, key, value)
        return value

    def put(self, table: str, key: Key, value: bytes) -> None:
        self._shared.put(table, key, value)  # durable first, so a crash never loses it
        self._cache.put(table, key, value)

    def scan(self, table: str, prefix: Key = ()) -> Iterator[tuple[Key, bytes]]:
        # The shared store is authoritative for a scan (it sees every driver's writes);
        # warm the cache with what we read so subsequent point gets stay local.
        for key, value in self._shared.scan(table, prefix):
            self._cache.put(table, key, value)
            yield key, value

    def batch_put(self, table: str, items: list[tuple[Key, bytes]]) -> None:
        self._shared.batch_put(table, items)
        self._cache.batch_put(table, items)

    def refresh(self) -> None:
        """Drop the local cache so the next read re-pulls the shared store.

        Called between runs so this driver picks up statistics other drivers have
        written since the cache was warmed (cross-driver freshness)."""
        self._cache = InProcessBackend()
