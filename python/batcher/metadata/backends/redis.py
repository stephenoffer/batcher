"""Redis backend — low-latency, cluster-shared learned statistics.

Each metadata table is one Redis hash (`{namespace}:{table}`) whose fields are the
JSON-encoded keys, so a `(table, key) -> value` write is a single `HSET` field and
concurrent drivers updating *different* keys never collide (the per-key write
granularity the Hub's keyed-param model relies on). `scan` uses `HSCAN` so a large
table doesn't block the server. Redis is an optional dependency; importing this
module without `redis` installed raises a clear error.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from batcher._internal.errors import ConfigError, MissingDependencyError
from batcher.metadata.store import Key, require_uri

__all__ = ["RedisBackend"]

#: URI schemes `redis.Redis.from_url` accepts. Checked up front because `from_url`
#: answers a wrong scheme with an opaque failure, and because the mistake a user
#: actually makes — a bare ``host:port`` — is one this can name precisely.
_SCHEMES = ("redis://", "rediss://", "unix://")


def _encode_key(key: Key) -> str:
    return json.dumps(list(key), separators=(",", ":"))


class RedisBackend:
    """A `MetadataBackend` backed by a Redis server (one hash per table)."""

    def __init__(self, uri: str | None, *, namespace: str = "batcher:meta") -> None:
        """Connect to a Redis server.

        Args:
            uri: A ``redis://``, ``rediss://``, or ``unix://`` URL.
            namespace: Key prefix for every hash this backend owns.

        Raises:
            ConfigError: If `uri` is missing or is not a Redis URL.
            MissingDependencyError: If the ``redis`` package is not installed.
        """
        uri = require_uri("redis", uri, example="redis://localhost:6379/0")
        if not uri.startswith(_SCHEMES):
            raise ConfigError(
                f"{uri!r} is not a Redis URL.",
                available=_SCHEMES,
                available_label="Supported schemes",
                hint="A bare host:port needs the scheme, e.g. 'redis://localhost:6379/0'.",
            )
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - exercised only without redis
            raise MissingDependencyError.of(
                feature="The redis metadata backend", provides="redis-py", extra="redis"
            ) from exc

        self._uri = uri
        self._redis = redis.Redis.from_url(uri)
        self._ns = namespace

    def __repr__(self) -> str:
        """Name the server and namespace, so a misrouted store is visible when printed."""
        return f"RedisBackend(uri={self._uri!r}, namespace={self._ns!r})"

    def _hash(self, table: str) -> str:
        return f"{self._ns}:{table}"

    def get(self, table: str, key: Key) -> bytes | None:
        return self._redis.hget(self._hash(table), _encode_key(key))

    def put(self, table: str, key: Key, value: bytes) -> None:
        self._redis.hset(self._hash(table), _encode_key(key), value)

    def scan(self, table: str, prefix: Key = ()) -> Iterator[tuple[Key, bytes]]:
        plen = len(prefix)
        for field, value in self._redis.hscan_iter(self._hash(table)):
            key = tuple(json.loads(field))
            if key[:plen] == prefix:
                yield key, value

    def batch_put(self, table: str, items: list[tuple[Key, bytes]]) -> None:
        if not items:
            return
        mapping = {_encode_key(k): v for k, v in items}
        self._redis.hset(self._hash(table), mapping=mapping)
