"""The point-lookup stores: Redis, RocksDB, and a plain in-memory table.

Each is a `KeyValueLookup` — one batched fetch and a fixed schema — because everything
that makes a lookup join fast (deduplication, the LRU, negative caching, the Arrow
assembly) lives above them and is shared. A new store is `multi_get` and a schema.

The batching is the point in every case, and it is what a naive implementation gets wrong:

- **Redis** pipelines the whole key list into one round trip. `MGET` for a string-valued
  keyspace, one pipelined `HGETALL` per key for a hash-valued one. The difference between
  that and a key-at-a-time loop is the difference between one network round trip per batch
  and ten thousand.
- **RocksDB** uses the driver's own `multi_get`, which reads the keys in sorted order so
  the SST blocks each one needs are touched once rather than per key.
- **In memory** exists because the shape is genuinely useful without a server — a
  dimension small enough to hold, joined the same way — and because it is what makes every
  layer above testable on a machine with no driver installed.
"""

from __future__ import annotations

import json
from typing import Any

import pyarrow as pa

from batcher._internal.errors import ConfigError
from batcher._internal.optional import require

__all__ = ["InMemoryLookup", "RedisLookup", "RocksDBLookup"]

#: How many keys to send in one pipeline or `multi_get`. Large enough that the per-call
#: overhead disappears, small enough that one batch's worth of replies does not become the
#: memory problem the lookup join exists to avoid. A batch with more distinct keys than
#: this is split, so the bound holds whatever `batch_size` the caller chose.
_CHUNK = 4096


def _chunks(keys: list[str]) -> list[list[str]]:
    """`keys` split into `_CHUNK`-sized pieces."""
    return [keys[i : i + _CHUNK] for i in range(0, len(keys), _CHUNK)]


class InMemoryLookup:
    """A `KeyValueLookup` over an in-process Arrow table, keyed by one column.

    For a dimension small enough to hold in memory, and the reference implementation every
    other backend is checked against: it has no network, no driver, and no failure mode, so
    a difference between it and a real store is the store's.
    """

    __slots__ = ("_rows", "_schema")

    def __init__(self, table: pa.Table, key_column: str) -> None:
        """Index `table` by `key_column`.

        Args:
            table: The dimension rows.
            key_column: The column to key on. Its values are read as strings, so the join
                key matches whatever the other backends use.

        Raises:
            ConfigError: If `key_column` is not a column of `table`. Discovered here rather
                than on the first batch, where it would surface mid-query.
        """
        if key_column not in table.column_names:
            raise ConfigError(
                f"lookup key column {key_column!r} is not in the table "
                f"({', '.join(table.column_names)}).",
                hint="Pass the column the dimension is keyed by.",
            )
        value_names = [name for name in table.column_names if name != key_column]
        self._schema = pa.schema([table.schema.field(name) for name in value_names])
        keys = table.column(key_column).to_pylist()
        values = table.select(value_names).to_pylist()
        # Last write wins on a duplicate key, which is what a key-value store would do.
        self._rows = {
            str(key): row for key, row in zip(keys, values, strict=True) if key is not None
        }

    def multi_get(self, keys: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch `keys` from the indexed table."""
        return {key: self._rows[key] for key in keys if key in self._rows}

    def value_schema(self) -> pa.Schema:
        """The dimension's columns, without the key."""
        return self._schema

    def close(self) -> None:
        """Nothing to release."""


class RedisLookup:
    """A `KeyValueLookup` over a Redis keyspace, one pipeline per batch."""

    __slots__ = ("_client", "_hash", "_prefix", "_schema")

    def __init__(
        self,
        uri: str,
        schema: pa.Schema,
        *,
        prefix: str = "",
        hash_values: bool = False,
    ) -> None:
        """Connect to a Redis server.

        Args:
            uri: A ``redis://``, ``rediss://``, or ``unix://`` URL.
            schema: The columns this lookup contributes. Required rather than inferred:
                the output schema of a join cannot depend on which keys the first batch
                happened to contain.
            prefix: Prepended to every key, for a keyspace that namespaces its dimensions.
            hash_values: Read each key as a Redis **hash** whose fields are the columns,
                rather than as a string holding a JSON object. Set it to match how the
                dimension was written.

        Raises:
            ConfigError: If `uri` is not a Redis URL.
            MissingDependencyError: If the ``redis`` package is not installed.
        """
        if not isinstance(uri, str) or not uri.startswith(("redis://", "rediss://", "unix://")):
            raise ConfigError(
                f"a Redis lookup needs a Redis URL, but got {uri!r}.",
                hint="Pass a URL such as 'redis://localhost:6379/0'. A bare host:port is not one.",
            )
        redis = require("redis", feature="a Redis lookup join", provides="Redis", extra="redis")
        self._client: Any = redis.Redis.from_url(uri, decode_responses=True)
        self._schema = schema
        self._prefix = prefix
        self._hash = hash_values

    def multi_get(self, keys: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch `keys` in one pipeline per chunk.

        Args:
            keys: Distinct keys to fetch.

        Returns:
            The rows found, keyed by the *unprefixed* key the caller asked for.
        """
        out: dict[str, dict[str, Any]] = {}
        for chunk in _chunks(keys):
            full = [self._prefix + key for key in chunk]
            replies = self._fetch(full)
            for key, reply in zip(chunk, replies, strict=True):
                row = self._row(reply)
                if row is not None:
                    out[key] = row
        return out

    def _fetch(self, full_keys: list[str]) -> list[Any]:
        """One round trip for a chunk of already-prefixed keys."""
        if not self._hash:
            return list(self._client.mget(full_keys))
        pipe = self._client.pipeline(transaction=False)
        for key in full_keys:
            pipe.hgetall(key)
        return list(pipe.execute())

    def _row(self, reply: Any) -> dict[str, Any] | None:
        """One reply as a field mapping, or `None` for a key the store does not hold.

        A hash reply is already a mapping and an empty one means the key is absent, which
        is Redis's own way of saying so. A string reply is parsed as JSON; anything that
        is not a JSON object is stored under this lookup's *first* column, so a plain
        single-value keyspace works without wrapping every value in an object.
        """
        if reply is None:
            return None
        if self._hash:
            return dict(reply) if reply else None
        try:
            parsed = json.loads(reply)
        except (TypeError, ValueError):
            parsed = reply
        if isinstance(parsed, dict):
            return parsed
        return {self._schema.field(0).name: parsed} if len(self._schema) else None

    def value_schema(self) -> pa.Schema:
        """The columns this lookup contributes."""
        return self._schema

    def close(self) -> None:
        """Close the connection pool. Idempotent."""
        client, self._client = self._client, None
        if client is not None:
            client.close()


class RocksDBLookup:
    """A `KeyValueLookup` over an embedded RocksDB database, one `multi_get` per batch."""

    __slots__ = ("_db", "_schema")

    def __init__(self, path: str, schema: pa.Schema) -> None:
        """Open the database at `path` for reading.

        Args:
            path: The RocksDB directory. It is locked by the process that opens it, so a
                lookup join distributed across workers on one node needs Redis or a
                read-only copy per worker instead.
            schema: The columns this lookup contributes.

        Raises:
            ConfigError: If the database cannot be opened.
            MissingDependencyError: If the ``rocksdict`` package is not installed.
        """
        rocksdict = require(
            "rocksdict", feature="a RocksDB lookup join", provides="RocksDB", extra="rocksdb"
        )
        try:
            self._db: Any = rocksdict.Rdict(path)
        except Exception as exc:
            raise ConfigError(
                f"Cannot open the RocksDB lookup database at {path!r}: {exc}.",
                hint=(
                    "A RocksDB database is locked by the one process that has it open. "
                    "Use a Redis lookup to serve several workers from one store."
                ),
            ) from exc
        self._schema = schema

    def multi_get(self, keys: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch `keys` through the driver's batched read.

        Args:
            keys: Distinct keys to fetch.

        Returns:
            The rows found, keyed as asked.
        """
        out: dict[str, dict[str, Any]] = {}
        for chunk in _chunks(keys):
            encoded = [key.encode("utf-8") for key in chunk]
            for key, raw in zip(chunk, self._db.get(encoded), strict=True):
                row = _decode(raw)
                if row is not None:
                    out[key] = row
        return out

    def value_schema(self) -> pa.Schema:
        """The columns this lookup contributes."""
        return self._schema

    def close(self) -> None:
        """Close the database, releasing the directory lock. Idempotent."""
        db, self._db = self._db, None
        if db is not None:
            db.close()


def _decode(raw: Any) -> dict[str, Any] | None:
    """A stored RocksDB value as a field mapping, or `None` when there is nothing usable.

    Values are JSON objects. A value that is not one is skipped rather than raising: the
    database is written by something outside this engine, and an unreadable row is a row
    that does not match, not a query failure.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw if isinstance(raw, str) else bytes(raw).decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None
