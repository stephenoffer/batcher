"""MessagePack format — row-oriented read + write via `ormsgpack`, to Arrow.

`MsgpackSource` reads a stream of MessagePack-encoded records (one map per row)
and assembles them into Arrow at *batch* granularity — the unavoidable
deserialization for a row-oriented, non-Arrow source. The on-disk shape is a
length-delimited sequence of msgpack documents (a 4-byte big-endian length prefix
per record), which `MsgpackSink` writes and `MsgpackSource` reads. One file is one
`Split`.

All `ormsgpack` imports are deferred — importing this module never requires the
optional dependency. A missing dependency raises `BackendError` with a
``pip install 'batcher-engine[msgpack]'`` hint.
"""

from __future__ import annotations

import struct
from typing import IO, Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.config import active_config
from batcher.io.base import FileSink, FileSource
from batcher.io.formats.base import SINKS, SOURCES

__all__ = ["MsgpackSink", "MsgpackSource"]

_LEN = struct.Struct(">I")  # 4-byte big-endian length prefix per record.


def _require_ormsgpack() -> Any:
    """Import and return the `ormsgpack` module or raise `BackendError`."""
    try:
        import ormsgpack
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise BackendError(
            "MessagePack support requires ormsgpack: pip install 'batcher-engine[msgpack]'"
        ) from exc
    return ormsgpack


def _iter_records(fh: IO[bytes], ormsgpack: Any) -> Any:
    """Yield decoded records from a length-delimited msgpack stream."""
    while True:
        header = fh.read(_LEN.size)
        if not header:
            return
        if len(header) != _LEN.size:
            raise BackendError("truncated MessagePack record (length prefix incomplete)")
        (length,) = _LEN.unpack(header)
        payload = fh.read(length)
        if len(payload) != length:
            raise BackendError("truncated MessagePack record (payload shorter than prefix)")
        yield ormsgpack.unpackb(payload)


@SOURCES.register("msgpack")
class MsgpackSource(FileSource):
    """A length-delimited MessagePack record stream read to Arrow."""

    suffix = ".msgpack"
    format_name = "msgpack"

    __slots__ = ()

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        batches = self._read_file(fh, None)
        return batches[0].schema if batches else pa.schema([])

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        ormsgpack = _require_ormsgpack()
        batch_rows = active_config().execution.morsel_rows
        out: list[pa.RecordBatch] = []
        rows: list[dict[str, Any]] = []
        for record in _iter_records(fh, ormsgpack):
            rows.append(record)
            if len(rows) >= batch_rows:
                out.append(self._to_batch(rows, projection))
                rows = []
        if rows:
            out.append(self._to_batch(rows, projection))
        return out

    @staticmethod
    def _to_batch(rows: list[dict[str, Any]], projection: list[str] | None) -> pa.RecordBatch:
        batch = pa.RecordBatch.from_pylist(rows)
        return batch.select(projection) if projection is not None else batch


@SINKS.register("msgpack")
class MsgpackSink(FileSink):
    """Write a length-delimited MessagePack record stream (one map per row)."""

    suffix = ".msgpack"
    format_name = "msgpack"

    __slots__ = ()

    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        ormsgpack = _require_ormsgpack()
        for row in table.to_pylist():
            payload = ormsgpack.packb(row)
            fh.write(_LEN.pack(len(payload)))
            fh.write(payload)
