"""TFRecord format — TensorFlow record stream → Arrow via manual framing.

A TFRecord file is a sequence of length-prefixed, CRC-checked records (each a
serialized ``tf.train.Example`` protobuf). `TFRecordSource` reads the framing with
the standard layout — ``uint64 length``, ``uint32 masked-crc32c(length)``, payload,
``uint32 masked-crc32c(payload)`` — and emits the raw record payloads as Arrow
``binary`` rows at *batch* granularity. Decoding the protobuf into typed feature
columns is left to a downstream Rust expression; the source's job is framing only.
CRC verification uses `crc32c` when present. One file is one `Split`.

The `crc32c` import is deferred — importing this module never requires it (CRCs are
simply not verified without it). A missing-but-required dependency raises
`BackendError` with a ``pip install 'batcher[tfrecord]'`` hint.
"""

from __future__ import annotations

import struct
from typing import IO, Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.config import active_config
from batcher.io.base import FileSource
from batcher.io.formats.base import SOURCES

__all__ = ["TFRecordSource"]

_U64 = struct.Struct("<Q")
_U32 = struct.Struct("<I")
_TFRECORD_SCHEMA = pa.schema([("record", pa.binary())])


def _crc32c() -> Any | None:
    """Return the `crc32c` module if installed, else None (CRCs unverified)."""
    try:
        import crc32c
    except ImportError:  # pragma: no cover - optional integrity check
        return None
    return crc32c


def _masked_crc(crc: int) -> int:
    """Apply TensorFlow's CRC mask used in the TFRecord framing."""
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _iter_records(fh: IO[bytes], crc: Any | None) -> Any:
    """Yield raw record payloads from a TFRecord stream, verifying CRCs if able."""
    while True:
        length_bytes = fh.read(_U64.size)
        if not length_bytes:
            return
        if len(length_bytes) != _U64.size:
            raise BackendError("truncated TFRecord (length header incomplete)")
        (length,) = _U64.unpack(length_bytes)
        length_crc = _U32.unpack(fh.read(_U32.size))[0]
        if crc is not None and _masked_crc(crc.crc32c(length_bytes)) != length_crc:
            raise BackendError("TFRecord length CRC mismatch (corrupt file)")
        payload = fh.read(length)
        if len(payload) != length:
            raise BackendError("truncated TFRecord (payload shorter than length)")
        payload_crc = _U32.unpack(fh.read(_U32.size))[0]
        if crc is not None and _masked_crc(crc.crc32c(payload)) != payload_crc:
            raise BackendError("TFRecord payload CRC mismatch (corrupt file)")
        yield payload


@SOURCES.register("tfrecord")
class TFRecordSource(FileSource):
    """One or more TFRecord files, emitting raw record bytes as Arrow ``binary``.

    The schema is fixed to ``{record: binary}``; decoding ``tf.train.Example``
    into feature columns is a downstream Rust expression, not Python hot-path work.
    """

    suffix = ".tfrecord"
    format_name = "tfrecord"

    __slots__ = ()

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:  # noqa: ARG002 (fixed schema)
        return _TFRECORD_SCHEMA

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        crc = _crc32c()
        batch_rows = active_config().execution.morsel_rows
        out: list[pa.RecordBatch] = []
        records: list[bytes] = []
        for payload in _iter_records(fh, crc):
            records.append(payload)
            if len(records) >= batch_rows:
                out.append(self._to_batch(records, projection))
                records = []
        if records or not out:
            out.append(self._to_batch(records, projection))
        return out

    @staticmethod
    def _to_batch(records: list[bytes], projection: list[str] | None) -> pa.RecordBatch:
        array = pa.array(records, pa.binary())
        batch = pa.RecordBatch.from_arrays([array], schema=_TFRECORD_SCHEMA)
        return batch.select(projection) if projection is not None else batch
