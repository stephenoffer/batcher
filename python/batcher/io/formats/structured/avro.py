"""Avro format — row-oriented read + write via `fastavro`, assembled to Arrow.

pyarrow has no Avro reader, so this format bridges through `fastavro`: the reader
yields Python dicts which are buffered and converted to Arrow at *batch*
granularity (default 16,384 rows) — never per-row query logic, just the unavoidable
deserialization an Arrow-less source requires. The Avro schema maps to an Arrow
schema for `schema()`; one whole file is one `Split`.

All `fastavro` imports are deferred — importing this module never requires the
optional dependency. A missing dependency raises `BackendError` with a
``pip install 'batcher[avro]'`` hint.
"""

from __future__ import annotations

from typing import IO, Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.config import active_config
from batcher.io.base import FileSink, FileSource
from batcher.io.formats.base import SINKS, SOURCES

__all__ = ["AvroSink", "AvroSource"]

# Avro primitive type → Arrow type (logical types fall back to their base).
_AVRO_TO_ARROW: dict[str, pa.DataType] = {
    "null": pa.null(),
    "boolean": pa.bool_(),
    "int": pa.int32(),
    "long": pa.int64(),
    "float": pa.float32(),
    "double": pa.float64(),
    "bytes": pa.binary(),
    "string": pa.string(),
}


def _require_fastavro() -> Any:
    """Import and return the `fastavro` module or raise `BackendError`."""
    try:
        import fastavro
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise BackendError("Avro support requires fastavro: pip install 'batcher[avro]'") from exc
    return fastavro


def _arrow_type(avro_type: Any) -> pa.DataType:
    """Map one Avro field type (possibly a union) to an Arrow type."""
    if isinstance(avro_type, list):  # union: pick the first non-null branch
        branches = [t for t in avro_type if t != "null"]
        return _arrow_type(branches[0]) if branches else pa.null()
    if isinstance(avro_type, dict):  # logical/complex type: use its base
        return _AVRO_TO_ARROW.get(avro_type.get("type", "string"), pa.string())
    return _AVRO_TO_ARROW.get(avro_type, pa.string())


def _avro_schema_to_arrow(avro_schema: dict[str, Any]) -> pa.Schema:
    """Translate an Avro record schema into an Arrow schema."""
    return pa.schema([(f["name"], _arrow_type(f["type"])) for f in avro_schema.get("fields", [])])


@SOURCES.register("avro")
class AvroSource(FileSource):
    """One or more Avro files (single file, directory, or glob).

    Records are deserialized by fastavro and assembled into Arrow batches of the
    configured morsel size; projection is applied to the assembled batch.
    """

    suffix = ".avro"
    format_name = "avro"

    __slots__ = ()

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        fastavro = _require_fastavro()
        reader = fastavro.reader(fh)
        return _avro_schema_to_arrow(reader.writer_schema)

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        fastavro = _require_fastavro()
        reader = fastavro.reader(fh)
        schema = _avro_schema_to_arrow(reader.writer_schema)
        batch_rows = active_config().execution.morsel_rows
        out: list[pa.RecordBatch] = []
        rows: list[dict[str, Any]] = []
        for record in reader:
            rows.append(record)
            if len(rows) >= batch_rows:
                out.append(self._to_batch(rows, schema, projection))
                rows = []
        if rows:
            out.append(self._to_batch(rows, schema, projection))
        return out

    @staticmethod
    def _to_batch(
        rows: list[dict[str, Any]], schema: pa.Schema, projection: list[str] | None
    ) -> pa.RecordBatch:
        batch = pa.RecordBatch.from_pylist(rows, schema=schema)
        return batch.select(projection) if projection is not None else batch


@SINKS.register("avro")
class AvroSink(FileSink):
    """Write an Avro file (Arrow schema → Avro schema, rows via fastavro)."""

    suffix = ".avro"
    format_name = "avro"

    __slots__ = ()

    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        fastavro = _require_fastavro()
        avro_schema = {
            "type": "record",
            "name": "batcher",
            "fields": [{"name": n, "type": ["null", _avro_branch(t)]} for n, t in _fields(table)],
        }
        parsed = fastavro.parse_schema(avro_schema)
        fastavro.writer(fh, parsed, table.to_pylist())


def _fields(table: pa.Table) -> list[tuple[str, pa.DataType]]:
    return [(f.name, f.type) for f in table.schema]


def _avro_branch(arrow_type: pa.DataType) -> str:
    """Map an Arrow type to the nearest Avro primitive name for writing."""
    if pa.types.is_boolean(arrow_type):
        return "boolean"
    if pa.types.is_integer(arrow_type):
        return "long"
    if pa.types.is_floating(arrow_type):
        return "double"
    if pa.types.is_binary(arrow_type):
        return "bytes"
    return "string"
