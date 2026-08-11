"""Redis backend — low-latency, cluster-shared learned statistics.

Each metadata table is one Redis hash (`{namespace}:{table}`) whose fields are the
JSON-encoded keys, so a `(table, key) -> value` write is a single `HSET` field and
concurrent drivers updating *different* keys never collide (the per-key write
granularity the Hub's keyed-param model relies on). `scan` uses `HSCAN` so a large
table doesn't block the server. Redis is an optional dependency; importing this
module without `redis` installed raises a clear error.
"""

from __future__ import annotations

from collections.abc import Iterator

from batcher._internal.errors import ConfigError
from batcher._internal.optional import require
from batcher.metadata.store import Key, decode_key, encode_key, require_uri

__all__ = ["RedisBackend"]

#: URI schemes `redis.Redis.from_url` accepts. Checked up front because `from_url`
#: answers a wrong scheme with an opaque failure, and because the mistake a user
#: actually makes — a bare ``host:port`` — is one this can name precisely.
_SCHEMES = ("redis://", "rediss://", "unix://")

#: Characters Redis reads as glob metacharacters in an `HSCAN MATCH` pattern. A namespace is
#: user-supplied (a file path, a model name), so any of these can appear in one literally and
#: must be escaped — an unescaped `[` in a path would otherwise turn the pattern into a
#: character class and silently match nothing, reading as "this namespace has never been
#: written" rather than as an error.
_GLOB_META = "\\*?[]^-"


def _match(prefix: Key) -> str | None:
    """An `HSCAN MATCH` glob covering every encoded key under `prefix`, or `None` for all.

    Built from `encode_key(prefix)` with its closing `]` dropped, so it covers both the
    prefix key itself (which continues `]`) and every key extending it (which continues `,`).
    """
    if not prefix:
        return None
    literal = encode_key(prefix)[:-1]
    escaped = "".join("\\" + c if c in _GLOB_META else c for c in literal)
    return escaped + "*"


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
        redis = require(
            "redis", feature="The redis metadata backend", provides="redis-py", extra="redis"
        )

        self._uri = uri
        self._redis = redis.Redis.from_url(uri)
        self._ns = namespace

    def __repr__(self) -> str:
        """Name the server and namespace, so a misrouted store is visible when printed."""
        return f"RedisBackend(uri={self._uri!r}, namespace={self._ns!r})"

    def _hash(self, table: str) -> str:
        return f"{self._ns}:{table}"

    def get(self, table: str, key: Key) -> bytes | None:
        return self._redis.hget(self._hash(table), encode_key(key))

    def put(self, table: str, key: Key, value: bytes) -> None:
        self._redis.hset(self._hash(table), encode_key(key), value)

    def scan(self, table: str, prefix: Key = ()) -> Iterator[tuple[Key, bytes]]:
        """Every `(key, value)` under `prefix`, matched server-side where possible.

        The prefix becomes an `HSCAN MATCH` pattern rather than a filter applied after the
        whole hash has crossed the network. That matters most for the table it matters most
        on: `learned_params` holds every namespace at once, and source statistics take one
        namespace per source path, so reading one tuning loop's parameters used to pull every
        persisted source's statistics blob back to the driver and `decode_key` all of them —
        per query, over the network.

        The tuple comparison stays as the authority. `MATCH` is a glob over the encoded key,
        which is a superset for a numeric prefix (the pattern for `(5,)` also spans `[50]`),
        and this way a key written by another encoding can never be mis-attributed.
        """
        plen = len(prefix)
        for field, value in self._redis.hscan_iter(self._hash(table), match=_match(prefix)):
            key = decode_key(field)
            if key[:plen] == prefix:
                yield key, value

    def batch_put(self, table: str, items: list[tuple[Key, bytes]]) -> None:
        if not items:
            return
        mapping = {encode_key(k): v for k, v in items}
        self._redis.hset(self._hash(table), mapping=mapping)

    def delete(self, table: str, keys: list[Key]) -> None:
        """Drop `keys` from `table`; absent fields are ignored.

        Optional beyond the four `MetadataBackend` methods. Offered because a cluster-shared
        store is the one every driver writes to, so it is where the unbounded feedback table
        grows fastest — and without a `delete` the hub's prune silently does nothing there.
        """
        if not keys:
            return
        self._redis.hdel(self._hash(table), *(encode_key(k) for k in keys))
