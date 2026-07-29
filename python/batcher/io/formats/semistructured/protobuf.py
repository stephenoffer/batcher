"""Protobuf format — length-delimited message stream → Arrow via `protarrow`.

`ProtobufSource` reads a length-delimited stream of a single protobuf message type
(the standard ``writeDelimitedTo`` framing: a varint length prefix per message)
and converts the decoded messages to Arrow via `protarrow`, which maps a protobuf
descriptor to an Arrow schema. Messages are decoded then assembled at *batch*
granularity. The caller supplies the generated message class. One file is one
`Split`.

All `protarrow` / `google.protobuf` imports are deferred — importing this module
never requires the optional dependency. A missing dependency raises `BackendError`
with a ``pip install 'batcher-engine[protobuf]'`` hint.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import IO, Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher._internal.optional import require
from batcher.config import active_config
from batcher.io.base import FileSource
from batcher.io.formats.base import SOURCES

__all__ = ["ProtobufSource"]


def _require_protarrow() -> Any:
    """Import and return the `protarrow` module or raise `BackendError`."""
    return require(
        "protarrow", feature="Protobuf support", provides="protarrow + protobuf", extra="protobuf"
    )


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

    def __init__(self, path: str, *, message_cls: Any, **kwargs: Any) -> None:
        # Forward the base options; dropping them made `on_error="skip"` a silent no-op.
        super().__init__(path, **kwargs)
        self._message_cls = message_cls

    def _reader_kwargs(self) -> dict[str, object]:
        # `message_cls` is required — a worker rebuilding the reader without it raises. The
        # generated protobuf class is picklable (module-qualified), so it ships to the worker.
        return {**super()._reader_kwargs(), "message_cls": self._message_cls}

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:  # noqa: ARG002 (from descriptor)
        protarrow = _require_protarrow()
        return protarrow.message_type_to_schema(self._message_cls.DESCRIPTOR)

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        """Every batch of one protobuf stream, materialized — the `read()` contract."""
        return list(self._batches_from(fh, projection))

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        """Stream one protobuf file a morsel at a time rather than decoding it whole.

        `_read_file` batched internally but accumulated every batch before returning, so a
        large length-delimited stream was fully resident as Arrow before its first batch
        reached the consumer.

        Streaming is sound here for a reason the neighbouring MessagePack reader does not
        have: this schema comes from the message **descriptor** (`_read_schema`), not from
        inferring over the records, so where the morsel boundary falls cannot change a
        batch's types. MessagePack must see every record before it can type any of them,
        which is why it materializes on purpose.

        Args:
            path: The protobuf stream to read.
            projection: Columns the scan must produce. All columns when omitted.

        Yields:
            One `RecordBatch` per morsel of messages, in stream order.
        """
        with self._open(path) as fh:
            yield from self._batches_from(fh, projection)

    def _batches_from(self, fh: IO[Any], projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        """Yield morsel-sized batches of `fh`'s messages — the one loop both paths use."""
        protarrow = _require_protarrow()
        batch_rows = active_config().execution.morsel_rows
        buffer: list[Any] = []
        for message in _iter_messages(fh, self._message_cls):
            buffer.append(message)
            if len(buffer) >= batch_rows:
                yield self._to_batch(protarrow, buffer, projection)
                buffer = []
        if buffer:
            yield self._to_batch(protarrow, buffer, projection)

    def _to_batch(
        self, protarrow: Any, messages: list[Any], projection: list[str] | None
    ) -> pa.RecordBatch:
        table = protarrow.messages_to_table(messages, self._message_cls)
        if projection is not None:
            table = table.select(projection)
        return table.combine_chunks().to_batches()[0]
