"""Redis connector — slot-partitioned SCAN to Arrow.

Redis Cluster partitions the key space into 16,384 hash slots. `RedisSource`
splits that space into contiguous slot ranges (one `Split` per range) and uses
``SCAN`` to walk the keys in each range, fetching each key's value. The slot
ranges cover ``[0, 16384)`` disjointly, so the splits cover every key exactly
once on a cluster; on a single node the same ranges still partition the cursor
work. Keys and values are assembled into Arrow at batch granularity.

`RedisSink` is the write half, and it writes the same ``(key, value)`` shape the
source reads: one string value per key, or one hash per key when the frame has more
columns than that. Both go through a pipeline, so a batch of ten thousand rows is one
round trip rather than ten thousand.

``overwrite`` is declined. Emptying a Redis keyspace is ``FLUSHDB``, which discards
every key in the database rather than only the ones this write would replace, and a
destructive operation of that reach should not be reachable by passing a string to a
``mode`` argument.

The ``redis`` import is deferred; a missing driver raises `BackendError` with the
``redis`` extra hint. Connection kwargs (host, password) are stored verbatim and
never logged.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.nosql.base import (
    BulkSink,
    PartitionSpec,
    ScanSource,
    require_driver,
    rows_to_batches,
)

__all__ = ["RedisSink", "RedisSource"]

# Redis Cluster has a fixed number of hash slots (a protocol constant — it only
# coincidentally equals the engine morsel size, and must not be tied to it).
_NUM_SLOTS = 16_384

# A slot-range locator: a half-open ``[start_slot, end_slot)`` interval.
_SlotRange = tuple[int, int]

# The Arrow shape Redis exposes: each row is a (key, value) pair.
_REDIS_SCHEMA = pa.schema([("key", pa.string()), ("value", pa.string())])


@SOURCES.register("redis")
class RedisSource(ScanSource):
    """A Redis keyspace read as ``(key, value)`` rows, partitioned by slot range.

    Args:
        host: The Redis host; never logged.
        port: The Redis port (default 6379).
        db: The logical database index (default 0).
        password: Optional auth password; never logged.
        match: Optional ``SCAN MATCH`` glob to restrict keys (default ``"*"``).
        partition_spec: Optional parallelism hint; ``segments`` sets the number of
            slot ranges (default 1).
    """

    format_name = "redis"

    __slots__ = ()

    def __init__(
        self,
        *,
        host: str,
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        match: str = "*",
        partition_spec: PartitionSpec | None = None,
    ) -> None:
        super().__init__(
            partition_spec=partition_spec,
            host=host,
            port=port,
            db=db,
            password=password,
            match=match,
        )

    def _client(self) -> Any:
        redis = require_driver("redis", "redis")
        kw = self._conn_kwargs
        return redis.Redis(
            host=kw["host"],
            port=kw["port"],
            db=kw["db"],
            password=self._secret("password"),
            decode_responses=True,
        )

    def _identity_suffix(self) -> str:
        kw = self._conn_kwargs
        return f"{kw['host']}:{kw['port']}/{kw['db']}"

    def _infer_schema(self) -> pa.Schema:
        return _REDIS_SCHEMA

    def _enumerate_partitions(self) -> list[_SlotRange]:
        segments = max(1, self._partition_spec.segments)
        span = _NUM_SLOTS // segments
        ranges: list[_SlotRange] = []
        start = 0
        for i in range(segments):
            end = _NUM_SLOTS if i == segments - 1 else start + span
            ranges.append((start, end))
            start = end
        return ranges

    def _read_partition(
        self,
        partition: _SlotRange,
        projection: list[str] | None,
        predicate: dict | None = None,  # noqa: ARG002 (a key/value scan has no server-side filter)
    ) -> Iterator[pa.RecordBatch]:
        client = self._client()
        try:
            rows = _scan_range(client, partition, self._conn_kwargs["match"])
            for batch in rows_to_batches(rows, schema=_REDIS_SCHEMA):
                yield batch.select(projection) if projection else batch
        finally:
            # There was no cleanup here at all — not even a GC-time one, since nothing
            # closed the client on the normal path either. Every partition read leaked a
            # connection pool, so a wide scan exhausted Redis's `maxclients` rather than
            # merely holding sockets a little too long.
            client.close()


def _scan_range(client: Any, slot_range: _SlotRange, match: str) -> Iterator[dict[str, Any]]:
    """Yield ``{"key", "value"}`` rows for keys whose slot is in `slot_range`.

    Walks the full keyspace with ``SCAN`` and keeps only keys whose cluster hash
    slot falls in the half-open range, so concurrently-run ranges form a disjoint
    cover. ``CLUSTER KEYSLOT`` computes the slot the same way the server does.
    """
    start, end = slot_range
    cursor = 0
    while True:
        cursor, keys = client.scan(cursor=cursor, match=match, count=1000)
        for key in keys:
            slot = client.cluster("KEYSLOT", key) if _is_cluster(client) else _crc16_slot(key)
            if start <= slot < end:
                yield {"key": key, "value": client.get(key)}
        if cursor == 0:
            return


def _is_cluster(client: Any) -> bool:
    """Whether `client` is a Redis Cluster client (has a ``cluster`` command)."""
    return hasattr(client, "cluster")


def _crc16_slot(key: str) -> int:
    """Compute a key's hash slot on a non-cluster client (CRC16 mod 16384).

    Honors hashtags: only the substring between the first ``{`` and ``}`` is
    hashed when present, matching Redis Cluster's slot assignment.
    """
    start = key.find("{")
    if start != -1:
        end = key.find("}", start + 1)
        if end > start + 1:
            key = key[start + 1 : end]
    return _crc16(key.encode("utf-8")) % _NUM_SLOTS


# CCITT CRC16 (XMODEM) — the polynomial Redis Cluster uses for slot assignment.
def _crc16(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = (crc << 1) ^ 0x1021 if crc & 0x8000 else crc << 1
            crc &= 0xFFFF
    return crc


@SINKS.register("redis")
class RedisSink(BulkSink):
    """Write rows into a Redis keyspace, one pipeline per batch.

    The value written depends on the frame's shape, and the rule is the one that makes a
    round trip through `RedisSource` return what was written:

    * two columns, ``key`` and ``value`` — the source's own shape — writes each value as a
      plain string under its key;
    * anything wider writes each row as a **hash** under its key, one field per remaining
      column, which is how a record is represented in Redis.

    ``append`` is declined because Redis has no such operation on a key: ``SET`` replaces,
    which is an upsert. ``overwrite`` is declined because emptying the keyspace is
    ``FLUSHDB``, which discards keys this write knows nothing about.

    Args:
        host: The Redis host; never logged.
        port: The Redis port.
        db: The database index.
        password: The password, as a literal or an ``env:``/``file:`` reference.
        key_field: The column holding each row's key (default ``"key"``).
        prefix: Prepended to every key, so one keyspace can hold several relations.
        ttl_seconds: Expiry set on every key written, or None to leave keys permanent.
        mode: ``"upsert"`` (default) or ``"delete"``.

    Raises:
        BackendError: If `key_field` is absent from the rows being written.
    """

    format_name = "redis"
    supported_modes = ("upsert", "delete")

    __slots__ = ("prefix", "ttl_seconds")

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        key_field: str = "key",
        prefix: str = "",
        ttl_seconds: int | None = None,
        mode: str = "upsert",
    ) -> None:
        super().__init__(
            key_field=key_field,
            mode=mode,
            host=host,
            port=port,
            db=db,
            password=password,
        )
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds

    def _client(self) -> Any:
        """A Redis client, built here so the credential is resolved on the worker."""
        redis = require_driver("redis", "redis")
        kw = self._conn_kwargs
        return redis.Redis(
            host=kw["host"],
            port=kw["port"],
            db=kw["db"],
            password=self._secret("password"),
            decode_responses=True,
        )

    def _key(self, row: dict[str, Any], path: str) -> str:
        """The Redis key for `row`, prefixed by `prefix` or by the write's destination."""
        from batcher._internal.errors import BackendError

        if self.key_field not in row:
            raise BackendError(
                f"redis write needs a {self.key_field!r} column to key each row; this row "
                f"has {sorted(row)}. Name the key column with key_field=."
            )
        prefix = self.prefix if self.prefix else (f"{path}:" if path else "")
        return f"{prefix}{row[self.key_field]}"

    def _apply(self, rows: list[dict[str, Any]], path: str) -> None:
        """Queue every row onto one pipeline and execute it in a single round trip."""
        client = self._client()
        try:
            pipe = client.pipeline(transaction=False)
            for row in rows:
                key = self._key(row, path)
                if self.mode == "delete":
                    pipe.delete(key)
                    continue
                fields = {k: v for k, v in row.items() if k != self.key_field}
                if list(fields) == ["value"]:
                    pipe.set(key, _as_text(fields["value"]))
                else:
                    pipe.hset(key, mapping={k: _as_text(v) for k, v in fields.items()})
                if self.ttl_seconds is not None:
                    pipe.expire(key, self.ttl_seconds)
            pipe.execute()
        finally:
            client.close()


def _as_text(value: Any) -> str:
    """Render a value for Redis, which stores bytes and strings and nothing else.

    A null becomes the empty string rather than the four characters ``None``, which is what
    `str()` would have produced and what a reader would then have to know to undo.
    """
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "replace")
    return value if isinstance(value, str) else str(value)
