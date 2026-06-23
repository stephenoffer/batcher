"""Protobuf format — length-delimited message stream → Arrow via `protarrow`.

`ProtobufSource` reads a length-delimited stream of a single protobuf message type
(the standard ``writeDelimitedTo`` framing: a varint length prefix per message)
and converts the decoded messages to Arrow via `protarrow`, which maps a protobuf
descriptor to an Arrow schema. Messages are decoded then assembled at *batch*
granularity. The caller supplies the generated message class. One file is one
`Split`.

All `protarrow` / `google.protobuf` imports are deferred — importing this module
never requires the optional dependency. A missing dependency raises `BackendError`
with a ``pip install 'batcher[protobuf]'`` hint.
"""

from __future__ import annotations

from typing import IO, Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.config import active_config
from batcher.io.base import FileSource
from batcher.io.formats.base import SOURCES

__all__ = ["ProtobufSource"]


def _require_protarrow() -> Any:
    """Import and return the `protarrow` module or raise `BackendError`."""
    try:
        import protarrow
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise BackendError(
            "Protobuf support requires protarrow + protobuf: pip install 'batcher[protobuf]'"
        ) from exc
    return protarrow


def _read_varint(fh: IO[bytes]) -> int | None:
    """Read one base-128 varint length prefix; return None at clean EOF."""
    shift = 0
    result = 0
    while True:
        chunk = fh.read(1)
        if not chunk:
            return None if shift == 0 else result
        byte = chunk[0]
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result
        shift += 7


def _iter_messages(fh: IO[bytes], message_cls: Any) -> Any:
    """Yield parsed messages from a length-delimited protobuf stream."""
    while True:
        length = _read_varint(fh)
        if length is None:
            return
        payload = fh.read(length)
        if len(payload) != length:
            raise BackendError("truncated protobuf message (length prefix exceeds data)")
        message = message_cls()
        message.ParseFromString(payload)
        yield message


@SOURCES.register("protobuf")
class ProtobufSource(FileSource):
    """A length-delimited protobuf stream read to Arrow via protarrow.

    Args:
        path: The file (single file, directory, or glob).
        message_cls: The generated protobuf message class for the stream.
    """

    suffix = ".pb"
    format_name = "protobuf"

    __slots__ = ("_message_cls",)

    def __init__(self, path: str, *, message_cls: Any) -> None:
        super().__init__(path)
        self._message_cls = message_cls

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:  # noqa: ARG002 (from descriptor)
        protarrow = _require_protarrow()
        return protarrow.message_type_to_schema(self._message_cls.DESCRIPTOR)

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        protarrow = _require_protarrow()
        batch_rows = active_config().execution.morsel_rows
        out: list[pa.RecordBatch] = []
        buffer: list[Any] = []
        for message in _iter_messages(fh, self._message_cls):
            buffer.append(message)
            if len(buffer) >= batch_rows:
                out.append(self._to_batch(protarrow, buffer, projection))
                buffer = []
        if buffer:
            out.append(self._to_batch(protarrow, buffer, projection))
        return out

    def _to_batch(
        self, protarrow: Any, messages: list[Any], projection: list[str] | None
    ) -> pa.RecordBatch:
        table = protarrow.messages_to_table(messages, self._message_cls)
        if projection is not None:
            table = table.select(projection)
        return table.combine_chunks().to_batches()[0]
