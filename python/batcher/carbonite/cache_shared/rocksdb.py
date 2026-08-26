"""RocksDB-backed shared result cache — durable reuse across runs on one node.

The embedded counterpart to `RedisSharedCache`, for the case that has no server and does
not need one: a single machine whose scheduled jobs and notebooks re-issue the same
queries, where the useful sharing is across *time* rather than across nodes. A cached
result survives the interpreter, so the second run of a report reads it back instead of
rescanning the table it came from.

It is not a substitute for Redis on a cluster. RocksDB holds an exclusive lock on its
directory, so exactly one process may have it open — a second one gets a clear error at
construction rather than a store that silently disagrees. Reach for this on one node, and
for Redis when more than one process needs the same entries.

TTL is enforced on read rather than by a compaction filter. `rocksdict` does not expose
one, and a read-side check is honest about what it costs: an expired entry occupies disk
until something asks for it or `evict_expired` sweeps. Both are cheap because the stamp is
an 8-byte header on a value that is otherwise megabytes.

``rocksdict`` is an optional dependency (the ``rocksdb`` extra).
"""

from __future__ import annotations

import struct
import time
from typing import Any

from batcher._internal.errors import ConfigError
from batcher._internal.optional import require
from batcher._internal.paths import private_dir
from batcher.carbonite.cache_shared.base import SharedCacheError

__all__ = ["RocksDBSharedCache"]

#: Big-endian unsigned 64-bit expiry stamp, prepended to every value. Zero means "never
#: expires", which keeps one encoding for both cases instead of two value shapes. Fixed
#: width so the payload starts at a known offset and the slice is free.
_STAMP = struct.Struct(">Q")


def _stamped(value: bytes, ttl_seconds: int | None, now: float) -> bytes:
    """`value` with its expiry stamp prepended."""
    expires = int(now) + int(ttl_seconds) if ttl_seconds and ttl_seconds > 0 else 0
    return _STAMP.pack(expires) + value


def _unstamped(raw: bytes, now: float) -> bytes | None:
    """The payload in `raw`, or `None` if it has expired or the header is unreadable.

    A short or malformed record answers `None` rather than raising: these bytes live in a
    file other processes and earlier versions have written, and an unreadable entry is a
    miss by definition.
    """
    if len(raw) < _STAMP.size:
        return None
    (expires,) = _STAMP.unpack_from(raw, 0)
    if expires and expires <= now:
        return None
    return raw[_STAMP.size :]


class RocksDBSharedCache:
    """A `SharedCache` backed by an embedded RocksDB database directory."""

    __slots__ = ("_db", "_path")

    def __init__(self, path: str) -> None:
        """Open (or create) the cache database at `path`.

        Args:
            path: A filesystem *directory*. RocksDB owns the whole directory and locks
                it, so only one process may hold it open.

        Raises:
            ConfigError: If the directory cannot be opened, naming the lock case
                explicitly — a second process gets an error that reads as a permissions
                problem otherwise.
            MissingDependencyError: If the ``rocksdict`` package is not installed.
        """
        rocksdict = require(
            "rocksdict",
            feature="the rocksdb shared result cache",
            provides="RocksDB",
            extra="rocksdb",
        )
        # The entries are the user's actual query results, on a path that may sit under a
        # shared volume. Owner-only, before RocksDB creates anything inside.
        private_dir(path)
        try:
            self._db: Any = rocksdict.Rdict(path)
        except Exception as exc:
            raise ConfigError(
                f"Cannot open the RocksDB shared result cache at {path!r}: {exc}.",
                hint=(
                    "A RocksDB database is locked by the one process that has it open. "
                    "Point memory.shared_cache_uri at a per-process directory, or use a "
                    "redis:// URI to share entries between processes."
                ),
            ) from exc
        self._path = path

    def get(self, key: str) -> bytes | None:
        """Return the serialized result under `key`, or `None` on a miss or an expiry.

        Args:
            key: The content-addressed cache key.

        Returns:
            The stored bytes, or `None`.

        Raises:
            SharedCacheError: If the database read fails.
        """
        try:
            raw = self._db.get(key.encode("utf-8"))
        except Exception as exc:
            raise SharedCacheError(f"rocksdb get failed: {exc}") from exc
        if raw is None:
            return None
        payload = _unstamped(bytes(raw), time.time())
        if payload is None:
            self.delete(key)  # reclaim the disk the moment the expiry is observed
            return None
        return payload

    def put(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        """Store `value` under `key`, expiring after `ttl_seconds` if given.

        Args:
            key: The content-addressed cache key.
            value: The serialized result.
            ttl_seconds: Seconds until the entry expires; `None` or a non-positive value
                stores it without an expiry.

        Raises:
            SharedCacheError: If the write fails.
        """
        try:
            self._db[key.encode("utf-8")] = _stamped(value, ttl_seconds, time.time())
        except Exception as exc:
            raise SharedCacheError(f"rocksdb put failed: {exc}") from exc

    def delete(self, key: str) -> None:
        """Remove `key` if present.

        Args:
            key: The content-addressed cache key.

        Raises:
            SharedCacheError: If the delete fails.
        """
        try:
            del self._db[key.encode("utf-8")]
        except KeyError:
            return
        except Exception as exc:
            raise SharedCacheError(f"rocksdb delete failed: {exc}") from exc

    def evict_expired(self) -> int:
        """Delete every expired entry, returning how many were removed.

        The sweep the read-side expiry check does not do. Expired entries are invisible to
        `get` but still occupy disk, so a long-lived store wants this called occasionally
        — from a maintenance job, not from a query, since it walks the whole database.

        Returns:
            The number of entries removed.

        Raises:
            SharedCacheError: If the scan fails.
        """
        now = time.time()
        try:
            stale = [
                bytes(key)
                for key, value in self._db.items()
                if _unstamped(bytes(value), now) is None
            ]
            for key in stale:
                del self._db[key]
        except Exception as exc:
            raise SharedCacheError(f"rocksdb sweep failed: {exc}") from exc
        return len(stale)

    def close(self) -> None:
        """Flush and close the database, releasing the directory lock. Idempotent."""
        db, self._db = self._db, None
        if db is not None:
            db.close()
