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

from batcher.metadata.store import Key

__all__ = ["RedisBackend"]


def _encode_key(key: Key) -> str:
    return json.dumps(list(key), separators=(",", ":"))


class RedisBackend:
    """A `MetadataBackend` backed by a Redis server (one hash per table)."""

    def __init__(self, uri: str | None, *, namespace: str = "batcher:meta") -> None:
        if not uri:
            raise ValueError("redis metadata backend requires a uri (e.g. redis://host:6379/0)")
        try:
            import redis
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without redis
            raise ValueError(
                "the redis metadata backend requires the 'redis' package (pip install redis)"
            ) from exc

        self._redis = redis.Redis.from_url(uri)
        self._ns = namespace

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
