"""HBase connector — region-range partitioned scan to Arrow via happybase.

HBase tables are sorted by row key and physically sharded into *regions* on key
boundaries. `HBaseSource` enumerates the region start keys and emits one `Split`
per region range, each issuing a bounded ``Scan`` over its half-open
``[start_key, stop_key)`` key range. The region ranges are a disjoint, exhaustive
cover of the key space, so concatenating the splits equals a full scan. Each
region's rows are assembled into Arrow at batch granularity.

The ``happybase`` import is deferred; a missing driver raises `BackendError` with
the ``hbase`` extra hint. Connection kwargs (host, port) are stored verbatim and
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

__all__ = ["HBaseSource"]

# A region-range locator: half-open ``[start_key, stop_key)`` byte ranges; an
# empty bound means unbounded on that side. Stored as hex strings to stay
# picklable and human-readable in identities.
_KeyRange = tuple[str, str]


@SOURCES.register("hbase")
class HBaseSource(ScanSource):
    """An HBase table scanned in parallel by region key range.

    Rows are emitted as ``{"row_key": ..., "<family:qualifier>": ...}`` dicts —
    one column per cell, values decoded as UTF-8 strings.

    Args:
        host: The HBase Thrift host; never logged.
        table: The table name.
        port: The Thrift port (default 9090).
        partition_spec: Optional parallelism hint; honored only as a fallback when
            region boundaries cannot be read (then the key space is split evenly).
    """

    format_name = "hbase"

    __slots__ = ()

    def __init__(
        self,
        *,
        host: str,
        table: str,
        port: int = 9090,
        partition_spec: PartitionSpec | None = None,
    ) -> None:
        super().__init__(
            partition_spec=partition_spec,
            host=host,
            table=table,
            port=port,
        )

    def _connection(self) -> Any:
        happybase = require_driver("happybase", "hbase")
        kw = self._conn_kwargs
        return happybase.Connection(host=kw["host"], port=kw["port"])

    def _identity_suffix(self) -> str:
        kw = self._conn_kwargs
        return f"{kw['host']}:{kw['port']}/{kw['table']}"

    def _infer_schema(self) -> pa.Schema:
        conn = self._connection()
        try:
            table = conn.table(self._conn_kwargs["table"])
            rows = [_decode_row(key, data) for key, data in table.scan(limit=1)]
        finally:
            conn.close()
        if not rows:
            return pa.schema([("row_key", pa.string())])
        return pa.RecordBatch.from_pylist(rows).schema

    def _enumerate_partitions(self) -> list[_KeyRange]:
        conn = self._connection()
        try:
            regions = conn.table(self._conn_kwargs["table"]).regions()
        finally:
            conn.close()
        starts = sorted(_to_hex(r["start_key"]) for r in regions)
        if not starts:
            return [("", "")]
        bounds = ["", *(s for s in starts if s), ""]
        # De-duplicate while preserving order (the first region's start is empty).
        seen: list[str] = []
        for b in bounds:
            if not seen or seen[-1] != b:
                seen.append(b)
        return [(seen[i], seen[i + 1]) for i in range(len(seen) - 1)]

    def _read_partition(
        self, partition: _KeyRange, projection: list[str] | None
    ) -> Iterator[pa.RecordBatch]:
        start_hex, stop_hex = partition
        conn = self._connection()
        schema = self.schema() if not projection else None
        try:
            table = conn.table(self._conn_kwargs["table"])
            scan = table.scan(
                row_start=_from_hex(start_hex) or None,
                row_stop=_from_hex(stop_hex) or None,
            )
            rows = (_decode_row(key, data) for key, data in scan)
            for batch in rows_to_batches(rows, schema=schema):
                yield batch.select(projection) if projection else batch
        finally:
            conn.close()


def _decode_row(row_key: bytes, data: dict[bytes, bytes]) -> dict[str, Any]:
    """Flatten one HBase row to a ``{column: value}`` dict (UTF-8 decoded)."""
    out: dict[str, Any] = {"row_key": row_key.decode("utf-8", "replace")}
    for col, val in data.items():
        out[col.decode("utf-8", "replace")] = val.decode("utf-8", "replace")
    return out


def _to_hex(key: bytes) -> str:
    return key.hex()


def _from_hex(key: str) -> bytes:
    return bytes.fromhex(key)
