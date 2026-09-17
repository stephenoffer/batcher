"""TFRecord format — TensorFlow record stream ↔ Arrow via manual framing, read and write.

A TFRecord file is a sequence of length-prefixed, CRC-checked records (each a
serialized ``tf.train.Example`` protobuf). `TFRecordSource` reads the framing with
the standard layout — ``uint64 length``, ``uint32 masked-crc32c(length)``, payload,
``uint32 masked-crc32c(payload)`` — and emits the raw record payloads as Arrow
``binary`` rows at *batch* granularity. Decoding the protobuf into typed feature
columns is left to a downstream Rust expression; the source's job is framing only.
CRC verification uses `crc32c` when present. One file is one `Split`.

The `crc32c` import is deferred — importing this module never requires it (CRCs are
simply not verified without it). A missing-but-required dependency raises
`BackendError` with a ``pip install 'batcher-engine[tfrecord]'`` hint.

`TFRecordSink` writes the same framing, and a writer cannot skip the checksum the way a
reader can: TensorFlow verifies it on read. It encodes each row as a ``tf.train.Example``
(Ray Data's ``write_tfrecords`` layout), or passes a single binary column through as the
raw record payloads with ``record_format="raw"``, the inverse of this source. The Example
protobuf is small and fixed, so it is encoded here by hand rather than through TensorFlow.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from typing import IO, Any

import pyarrow as pa

from batcher._internal.errors import BackendError, FormatError, SchemaError
from batcher.config import active_config
from batcher.io.base import FileSink, FileSource
from batcher.io.formats.base import SINKS, SOURCES

__all__ = ["TFRecordSink", "TFRecordSource"]

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

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        """Stream a TFRecord file's batches instead of collecting them all first.

        `_read_file` already batches at the morsel size, but accumulates every batch into
        one list before returning — so the whole file's decoded records are resident, and
        a TFRecord shard of images is exactly where that is expensive. The record framing
        is a forward-only length-prefixed scan and the schema is fixed, so yielding as we
        go changes nothing about what is produced.
        """
        crc = _crc32c()
        batch_rows = active_config().execution.morsel_rows
        records: list[bytes] = []
        emitted = False
        with self._fs.open(path) as fh:
            for payload in _iter_records(fh, crc):
                records.append(payload)
                if len(records) >= batch_rows:
                    yield self._to_batch(records, projection)
                    emitted = True
                    records = []
        if records or not emitted:
            # An empty file still yields one empty batch, so the schema is observable —
            # matching `_read_file`.
            yield self._to_batch(records, projection)

    @staticmethod
    def _to_batch(records: list[bytes], projection: list[str] | None) -> pa.RecordBatch:
        array = pa.array(records, pa.binary())
        batch = pa.RecordBatch.from_arrays([array], schema=_TFRECORD_SCHEMA)
        return batch.select(projection) if projection is not None else batch


# --- writing -------------------------------------------------------------------------
# `tf.train.Feature` kinds, by protobuf field number: BytesList, FloatList, Int64List.
_BYTES_LIST, _FLOAT_LIST, _INT64_LIST = 1, 2, 3


def _crc_function() -> Any:
    """A ``bytes -> int`` CRC32C, from whichever implementation is installed.

    Writing needs one where reading does not: TensorFlow checks every record's checksum, so
    a file written without real CRCs is unreadable by the one consumer the format exists
    for. Missing both raises the install hint rather than writing such a file.
    """
    crc = _crc32c()
    if crc is not None:
        return crc.crc32c
    from batcher._internal.optional import require

    google_crc32c = require(
        "google_crc32c", feature="Writing TFRecord", provides="google-crc32c", extra="tfrecord"
    )
    return google_crc32c.value


def _varint(value: int) -> bytes:
    """A protobuf base-128 varint; a negative int64 is its 64-bit two's complement."""
    value &= 0xFFFFFFFFFFFFFFFF
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _field(number: int, payload: bytes) -> bytes:
    """One length-delimited protobuf field (wire type 2)."""
    return _varint((number << 3) | 2) + _varint(len(payload)) + payload


def _feature(kind: int, values: list[Any]) -> bytes:
    """One serialized ``tf.train.Feature`` holding `values` as `kind`."""
    if kind == _BYTES_LIST:
        inner = b"".join(_field(1, v) for v in values)
    elif not values:
        inner = b""
    elif kind == _FLOAT_LIST:
        inner = _field(1, struct.pack(f"<{len(values)}f", *values))
    else:
        inner = _field(1, b"".join(_varint(v) for v in values))
    return _field(kind, inner)


def _feature_column(name: str, column: pa.Array) -> tuple[int, list[list[Any]]]:
    """A column as its Feature kind and each row's value list (empty for a null)."""
    dtype = column.type
    listed = pa.types.is_list(dtype) or pa.types.is_large_list(dtype)
    if listed or pa.types.is_fixed_size_list(dtype):
        dtype = dtype.value_type
    if pa.types.is_integer(dtype) or pa.types.is_boolean(dtype):
        kind, target = _INT64_LIST, pa.int64()
    elif pa.types.is_floating(dtype):
        # `tf.train.FloatList` is float32 by definition, so a float64 narrows here exactly
        # as it does when TensorFlow builds the Example itself.
        kind, target = _FLOAT_LIST, pa.float32()
    elif any(
        check(dtype)
        for check in (
            pa.types.is_string,
            pa.types.is_large_string,
            pa.types.is_binary,
            pa.types.is_large_binary,
        )
    ):
        kind, target = _BYTES_LIST, pa.large_binary()
    else:
        raise SchemaError(
            f"write.tfrecord cannot write column {name!r} of type {column.type}: a "
            "tf.train.Example feature holds integers, floats, or bytes, or a list of one of them"
        )
    item = pa.field("item", target)
    if listed or pa.types.is_fixed_size_list(column.type):
        cast = column.cast(pa.large_list(item), safe=False)
        rows = [v if v is not None else [] for v in cast.to_pylist()]
        if any(x is None for row in rows for x in row):
            raise SchemaError(f"write.tfrecord: column {name!r} holds a null list item")
        return kind, rows
    values = column.cast(target, safe=False).to_pylist()
    return kind, [[] if v is None else [v] for v in values]


@SINKS.register("tfrecord")
class TFRecordSink(FileSink):
    """Write rows as TFRecord files, one record per row.

    With ``record_format="example"`` (the default) each row is a ``tf.train.Example`` with
    one feature per column: an integer or boolean column is an ``Int64List``, a float
    column a ``FloatList`` (float32, as the format defines it), a string or binary column a
    ``BytesList``, and a list column the same kinds with one value per item. A null is an
    empty feature list, which is how an Example leaves a value out.

    With ``record_format="raw"`` the dataset must be one binary column, and each value is
    written as the record payload unchanged. That is the inverse of `TFRecordSource`, which
    reads the payloads back as ``record``.

    Every record carries the masked CRC32C the format requires, so writing needs
    ``google-crc32c`` (or ``crc32c``) installed.

    Args:
        record_format: ``"example"`` to encode rows as ``tf.train.Example``, or ``"raw"``
            to write a single binary column's values as the records.
    """

    suffix = ".tfrecord"
    format_name = "tfrecord"

    __slots__ = ("_crc", "_record_format")

    def __init__(self, *, record_format: str = "example", **kwargs: Any) -> None:
        super().__init__(**kwargs)  # carries filesystem= / storage_options=
        if record_format not in ("example", "raw"):
            raise FormatError(
                f"write.tfrecord record_format must be 'example' or 'raw', got {record_format!r}"
            )
        self._record_format = record_format
        self._crc: Any = None

    def _check_schema(self, schema: pa.Schema) -> None:
        if self._record_format != "raw":
            return
        if len(schema) != 1 or not (
            pa.types.is_binary(schema[0].type) or pa.types.is_large_binary(schema[0].type)
        ):
            raise SchemaError(
                "write.tfrecord(record_format='raw') needs exactly one binary column, got "
                f"{[f'{f.name}: {f.type}' for f in schema]}"
            )

    def _payloads(self, batch: pa.RecordBatch) -> list[bytes]:
        if self._record_format == "raw":
            column = batch.column(0)
            if column.null_count:
                raise SchemaError("write.tfrecord(record_format='raw'): a record cannot be null")
            return column.to_pylist()
        features = [
            (_field(1, name.encode("utf-8")), *_feature_column(name, batch.column(name)))
            for name in batch.schema.names
        ]
        return [
            _field(
                1,
                b"".join(
                    _field(1, key + _field(2, _feature(kind, rows[row])))
                    for key, kind, rows in features
                ),
            )
            for row in range(batch.num_rows)
        ]

    def _write_records(self, fh: IO[Any], batch: pa.RecordBatch) -> None:
        if self._crc is None:
            self._crc = _crc_function()
        crc = self._crc
        out = bytearray()
        for payload in self._payloads(batch):
            length = _U64.pack(len(payload))
            out += length
            out += _U32.pack(_masked_crc(crc(length)))
            out += payload
            out += _U32.pack(_masked_crc(crc(payload)))
        fh.write(bytes(out))

    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        self._check_schema(table.schema)
        for batch in table.to_batches():
            self._write_records(fh, batch)

    def _open_stream_writer(self, fh: IO[Any], schema: pa.Schema) -> Any:
        self._check_schema(schema)
        return fh

    def _write_batch(self, writer: Any, batch: pa.RecordBatch) -> None:
        self._write_records(writer, batch)

    def _close_stream_writer(self, writer: Any) -> None:
        """Nothing to flush: the handle belongs to the atomic writer, which closes it."""
