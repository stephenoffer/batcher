"""HBase connector — region-range partitioned scan to Arrow via happybase.

HBase tables are sorted by row key and physically sharded into *regions* on key
boundaries. `HBaseSource` enumerates the region start keys and emits one `Split`
per region range, each issuing a bounded ``Scan`` over its half-open
``[start_key, stop_key)`` key range. The region ranges are a disjoint, exhaustive
cover of the key space, so concatenating the splits equals a full scan. Each
region's rows are assembled into Arrow at batch granularity.

`HBaseSink` is the write half, over happybase's `Batch`: one round trip per batch of
puts or deletes. An HBase ``Put`` replaces the cells it names and leaves the rest of the
row alone, which is an upsert, so that is the mode this sink implements — along with
``delete``, which removes whole rows by key.

The ``happybase`` import is deferred; a missing driver raises `BackendError` with
the ``hbase`` extra hint. Connection kwargs (host, port) are stored verbatim and
never logged.
"""

from __future__ import annotations

import contextlib
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

__all__ = ["HBaseSink", "HBaseSource"]

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
        self,
        partition: _KeyRange,
        projection: list[str] | None,
        predicate: dict | None = None,  # noqa: ARG002 (no server-side filter; the engine's Filter re-checks)
    ) -> Iterator[pa.RecordBatch]:
        start_hex, stop_hex = partition
        # Resolved *before* the scan connection is opened. `self.schema()` opens a
        # connection of its own, so calling it here held two at once — and a raise from it
        # landed between `_connection()` and the `try`, stranding the outer one with
        # nothing left holding a reference to close it.
        schema = self.schema() if not projection else None
        conn = self._connection()
        try:
            table = conn.table(self._conn_kwargs["table"])
            scan = table.scan(
                row_start=_from_hex(start_hex) or None,
                row_stop=_from_hex(stop_hex) or None,
            )
            # `closing` on the scanner, not just on the connection: a consumer that stops
            # early (a LIMIT downstream) abandons this generator, and the server-side
            # scanner would otherwise stay open until the collector happened to run.
            with contextlib.closing(scan):
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


@SINKS.register("hbase")
class HBaseSink(BulkSink):
    """Write rows into an HBase table, one happybase `Batch` per Arrow batch.

    Every column but `key_field` becomes a cell, and its name is taken as the fully
    qualified ``family:qualifier`` HBase expects. A column with no colon is placed in
    `column_family`, so a frame read back from `HBaseSource` — whose column names already
    carry the family — round-trips unchanged, and a plain relational frame does not have
    to be renamed first.

    ``append`` and ``overwrite`` are declined. A ``Put`` replaces the cells it names, so an
    append would silently be an upsert; and emptying an HBase table means disabling and
    truncating it through the admin API, which is a schema operation rather than a write.

    Args:
        host: The HBase Thrift host; never logged.
        table: The target table, when it is not the write's destination name.
        port: The Thrift port.
        key_field: The column holding each row's row key (default ``"row_key"``, which is
            what `HBaseSource` emits).
        column_family: The family for a column whose name carries none.
        mode: ``"upsert"`` (default) or ``"delete"``.
    """

    format_name = "hbase"
    supported_modes = ("upsert", "delete")

    __slots__ = ("column_family", "table")

    def __init__(
        self,
        *,
        host: str,
        table: str | None = None,
        port: int = 9090,
        key_field: str = "row_key",
        column_family: str = "cf",
        mode: str = "upsert",
    ) -> None:
        super().__init__(key_field=key_field, mode=mode, host=host, port=port)
        self.table = table
        self.column_family = column_family

    def _connection(self) -> Any:
        """A happybase connection, opened here so it is never pickled onto a worker."""
        happybase = require_driver("happybase", "hbase")
        kw = self._conn_kwargs
        return happybase.Connection(host=kw["host"], port=kw["port"])

    def _cells(self, row: dict[str, Any]) -> dict[bytes, bytes]:
        """The ``{family:qualifier: value}`` cells `row` writes, as bytes."""
        cells: dict[bytes, bytes] = {}
        for name, value in row.items():
            if name == self.key_field or value is None:
                continue
            qualified = name if ":" in name else f"{self.column_family}:{name}"
            cells[qualified.encode()] = _encode(value)
        return cells

    def _apply(self, rows: list[dict[str, Any]], path: str) -> None:
        """Send every row as one put or delete inside a single happybase batch."""
        from batcher._internal.errors import BackendError

        connection = self._connection()
        try:
            table = connection.table(self.table or path)
            with table.batch() as batch:
                for row in rows:
                    if self.key_field not in row:
                        raise BackendError(
                            f"hbase write needs a {self.key_field!r} column to key each "
                            f"row; this row has {sorted(row)}. Name it with key_field=."
                        )
                    key = _encode(row[self.key_field])
                    if self.mode == "delete":
                        batch.delete(key)
                    else:
                        batch.put(key, self._cells(row))
        finally:
            connection.close()


def _encode(value: Any) -> bytes:
    """Render a value as the bytes HBase stores; every cell is bytes and nothing else."""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return str(value).encode("utf-8")
