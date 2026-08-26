"""Redis-backed shared result cache — low-latency reuse across processes and nodes.

One Redis string per cached result, under a configurable key prefix, holding the Arrow
IPC bytes. That is deliberately the whole data model: a result is read and written whole,
so a hash or a list would add structure nothing reads, and `SETEX`/`GET` are one round
trip each.

Eviction is Redis's, not this client's. A shared store is shared with other tenants and
often with other applications, so its memory policy belongs to whoever runs it
(`maxmemory` + `maxmemory-policy allkeys-lru` is the configuration this expects). What
this client contributes is a TTL, which bounds staleness for a source whose version token
is coarse and bounds the store's growth when nobody set a policy at all.

Every failure is a miss. A shared cache that is down must make queries slower, never make
them fail, so `SharedCacheError` is raised into the caller's miss path rather than out of
it, and a corrupt or truncated value is dropped rather than decoded.

The ``redis`` package is an optional dependency (the ``redis`` extra).
"""

from __future__ import annotations

from typing import Any

from batcher._internal.errors import ConfigError
from batcher._internal.optional import require
from batcher.carbonite.cache_shared.base import SharedCacheError

__all__ = ["RedisSharedCache"]

#: URI schemes `redis.Redis.from_url` accepts. Checked up front because `from_url` answers
#: a wrong scheme with an opaque failure, and because the mistake a user actually makes —
#: a bare ``host:port`` — is one this can name precisely.
_SCHEMES = ("redis://", "rediss://", "unix://")


class RedisSharedCache:
    """A `SharedCache` backed by a Redis server (one string per cached result)."""

    __slots__ = ("_client", "_prefix")

    def __init__(self, uri: str, *, prefix: str = "batcher:cache") -> None:
        """Connect to a Redis server.

        Args:
            uri: A ``redis://``, ``rediss://``, or ``unix://`` URL.
            prefix: Key prefix for every entry this cache owns, so it can share a
                database with the learned-stats store and with other applications.

        Raises:
            ConfigError: If `uri` is not a Redis URL.
            MissingDependencyError: If the ``redis`` package is not installed.
        """
        if not isinstance(uri, str) or not uri.startswith(_SCHEMES):
            raise ConfigError(
                f"The redis shared result cache needs a Redis URL, but got {uri!r}.",
                hint=(
                    "Set memory.shared_cache_uri to a URL such as "
                    "'redis://localhost:6379/0'. A bare host:port is not one."
                ),
            )
        redis = require(
            "redis",
            feature="the redis shared result cache",
            provides="Redis",
            extra="redis",
        )
        # `decode_responses` stays off: the values are Arrow IPC frames, and decoding them
        # as text would corrupt every one of them.
        self._client: Any = redis.Redis.from_url(uri)
        self._prefix = prefix

    def _full(self, key: str) -> str:
        """The stored key for `key`."""
        return f"{self._prefix}:{key}"

    def get(self, key: str) -> bytes | None:
        """Return the serialized result stored under `key`, or `None` on a miss.

        Args:
            key: The content-addressed cache key.

        Returns:
            The stored bytes, or `None` if absent.

        Raises:
            SharedCacheError: If the server is unreachable or answers unusably. The
                caller treats this as a miss.
        """
        try:
            value = self._client.get(self._full(key))
        except Exception as exc:
            raise SharedCacheError(f"redis GET failed: {exc}") from exc
        return bytes(value) if value is not None else None

    def put(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        """Store `value` under `key`, expiring after `ttl_seconds` if given.

        Args:
            key: The content-addressed cache key.
            value: The serialized result.
            ttl_seconds: Seconds until the entry expires; `None` or a non-positive
                value stores it without one.

        Raises:
            SharedCacheError: If the write fails.
        """
        try:
            if ttl_seconds and ttl_seconds > 0:
                self._client.setex(self._full(key), int(ttl_seconds), value)
            else:
                self._client.set(self._full(key), value)
        except Exception as exc:
            raise SharedCacheError(f"redis SET failed: {exc}") from exc

    def delete(self, key: str) -> None:
        """Remove `key` if present.

        Args:
            key: The content-addressed cache key.

        Raises:
            SharedCacheError: If the delete fails.
        """
        try:
            self._client.delete(self._full(key))
        except Exception as exc:
            raise SharedCacheError(f"redis DEL failed: {exc}") from exc

    def close(self) -> None:
        """Close the connection pool. Idempotent."""
        client, self._client = self._client, None
        if client is not None:
            client.close()
