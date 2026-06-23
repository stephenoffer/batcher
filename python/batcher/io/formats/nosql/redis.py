"""Redis connector — slot-partitioned SCAN to Arrow.

Redis Cluster partitions the key space into 16,384 hash slots. `RedisSource`
splits that space into contiguous slot ranges (one `Split` per range) and uses
``SCAN`` to walk the keys in each range, fetching each key's value. The slot
ranges cover ``[0, 16384)`` disjointly, so the splits cover every key exactly
once on a cluster; on a single node the same ranges still partition the cursor
work. Keys and values are assembled into Arrow at batch granularity.

The ``redis`` import is deferred; a missing driver raises `BackendError` with the
``redis`` extra hint. Connection kwargs (host, password) are stored verbatim and
never logged.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from batcher.io.formats.base import SOURCES
from batcher.io.formats.nosql.base import (
    PartitionSpec,
    ScanSource,
    require_driver,
    rows_to_batches,
)

__all__ = ["RedisSource"]

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
            password=kw["password"],
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
    ) -> Iterator[pa.RecordBatch]:
        client = self._client()
        rows = _scan_range(client, partition, self._conn_kwargs["match"])
        for batch in rows_to_batches(rows, schema=_REDIS_SCHEMA):
            yield batch.select(projection) if projection else batch


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
