"""Avro format — row-oriented read + write via `fastavro`, assembled to Arrow.

pyarrow has no Avro reader, so this format bridges through `fastavro`: the reader
yields Python dicts which are buffered and converted to Arrow at *batch*
granularity (default 16,384 rows) — never per-row query logic, just the unavoidable
deserialization an Arrow-less source requires. The Avro schema maps to an Arrow
schema for `schema()`; one whole file is one `Split`.

All `fastavro` imports are deferred — importing this module never requires the
optional dependency. A missing dependency raises `BackendError` with a
``pip install 'batcher-engine[avro]'`` hint.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import IO, Any

import pyarrow as pa

from batcher._internal.native import engine
from batcher._internal.optional import require
from batcher.config import active_config
from batcher.io.base import FileSink, FileSource
from batcher.io.formats.base import SINKS, SOURCES

__all__ = ["AvroSink", "AvroSource"]

# Avro primitive type → Arrow type (logical types are handled by `_AVRO_LOGICAL_TO_ARROW`).
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

# Avro logicalType → Arrow type, kept in lockstep with what the native `arrow-avro`
# reader decodes (verified against it): the reader emits the Arrow *logical* type, so the
# advertised schema must too. Mapping these to the underlying int/long instead — as the
# base map alone would — makes `schema()` disagree with the decoded batches AND makes the
# `fastavro` fallback fail, since it cannot coerce the datetime/date/time values fastavro
# yields into an int column. `decimal` is handled separately (it needs precision/scale).
_AVRO_LOGICAL_TO_ARROW: dict[str, pa.DataType] = {
    "date": pa.date32(),
    "time-millis": pa.time32("ms"),
    "time-micros": pa.time64("us"),
    "timestamp-millis": pa.timestamp("ms", tz="+00:00"),
    "timestamp-micros": pa.timestamp("us", tz="+00:00"),
    "local-timestamp-millis": pa.timestamp("ms"),
    "local-timestamp-micros": pa.timestamp("us"),
}


def _read_native(data: bytes, batch_rows: int) -> list[pa.RecordBatch] | None:
    """Decode Avro bytes with the native `arrow-avro` reader (via `bc_io`), or ``None``.

    Returns ``None`` — signalling the caller to fall back to the row-by-row `fastavro`
    path — if the native engine is unavailable or the decode errors (an Avro feature
    ``arrow-avro`` does not yet cover), so the result is identical either way.
    """
    try:
        _native = engine()
        return _native.read_avro(data, batch_rows)
    except Exception:
        return None


def _require_fastavro() -> Any:
    """Import and return the `fastavro` module or raise `BackendError`."""
    return require("fastavro", feature="Avro", provides="fastavro", extra="avro")


def _arrow_type(avro_type: Any) -> pa.DataType:
    """Map one Avro field type (possibly a union or logical type) to an Arrow type."""
    if isinstance(avro_type, list):
        branches = [t for t in avro_type if t != "null"]
        if not branches:
            return pa.null()
        if len(branches) == 1:  # the nullable-scalar idiom: `["null", T]` is just T
            return _arrow_type(branches[0])
        return pa.struct([pa.field(f"member{i}", _arrow_type(b)) for i, b in enumerate(branches)])
    if isinstance(avro_type, dict):  # logical/complex type
        logical = avro_type.get("logicalType")
        if logical == "decimal":
            return pa.decimal128(avro_type.get("precision", 38), avro_type.get("scale", 0))
        if logical in _AVRO_LOGICAL_TO_ARROW:
            return _AVRO_LOGICAL_TO_ARROW[logical]
        return _AVRO_TO_ARROW.get(avro_type.get("type", "string"), pa.string())
    return _AVRO_TO_ARROW.get(avro_type, pa.string())


# A union of two or more real branches (`["null", "long", "string"]`) has no primitive
# Arrow equivalent, and picking `branches[0]` — as this did — is not a lossy approximation
# but a broken contract: `schema()` advertised `int64`, and then the read *failed* with a
# pyarrow conversion error from deep inside `from_pylist`, on a valid Avro file.
#
# Arrow's own answer is a union type, but a dense/sparse union does not survive the rest of
# the engine (no operator consumes one, and it does not cross the IR). So this follows the
# same choice Spark's Avro reader makes: a struct with one nullable `memberN` field per
# branch, exactly one of which is set per row. Lossless, and every operator already handles
# structs.
#
# fastavro yields the *bare* value for a primitive union rather than a tagged one, so the
# branch has to be recovered from the value's Python type. Avro forbids duplicate primitive
# branches within one union, so at most one member can match — the mapping is unambiguous.
_MEMBER_PREDICATES: tuple[tuple[type, Any], ...] = (
    (bool, pa.types.is_boolean),  # before int: bool is a subclass of int
    (int, pa.types.is_integer),
    (float, pa.types.is_floating),
    (str, pa.types.is_string),
    (bytes, pa.types.is_binary),
    (list, pa.types.is_list),
    (dict, pa.types.is_struct),
)


def _union_columns(schema: pa.Schema, avro_schema: dict[str, Any]) -> tuple[str, ...]:
    """The fields this Avro schema maps to a multi-branch union struct."""
    unions = {
        f["name"]
        for f in avro_schema.get("fields", [])
        if isinstance(f["type"], list) and len([t for t in f["type"] if t != "null"]) > 1
    }
    return tuple(n for n in schema.names if n in unions)


def _as_union_member(value: Any, struct_type: pa.StructType) -> dict[str, Any] | None:
    """Place one bare Avro union value into the `memberN` field whose type accepts it."""
    if value is None:
        return None
    out: dict[str, Any] = {f.name: None for f in struct_type}
    for py_type, accepts in _MEMBER_PREDICATES:
        if not isinstance(value, py_type):
            continue
        for field in struct_type:
            if accepts(field.type):
                out[field.name] = value
                return out
    # A logical type (date/decimal/...) or a branch whose Python value defies the table:
    # fall back to the first member that is not already excluded, so the value survives.
    out[struct_type.field(0).name] = value
    return out


def _avro_field_nullable(avro_type: Any) -> bool:
    """True iff the Avro field type is a union that admits ``"null"`` (a nullable field).

    A non-union Avro field cannot be null, so it maps to a non-nullable Arrow field —
    matching what the native `arrow-avro` reader produces, so `schema()` equals the
    decoded batches for a nullable/non-nullable field alike.
    """
    return isinstance(avro_type, list) and "null" in avro_type


def _avro_schema_to_arrow(avro_schema: dict[str, Any]) -> pa.Schema:
    """Translate an Avro record schema into an Arrow schema.

    Logical types (date/time/timestamp/decimal) map to their Arrow logical type — not the
    underlying int/long — and a field's nullability follows its Avro union, so the schema
    matches the batches the native reader decodes and the `fastavro` fallback assembles.
    """
    return pa.schema(
        [
            pa.field(f["name"], _arrow_type(f["type"]), nullable=_avro_field_nullable(f["type"]))
            for f in avro_schema.get("fields", [])
        ]
    )


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
        batch_rows = active_config().execution.morsel_rows
        data = fh.read()
        native = _read_native(data, batch_rows)
        if native is not None:
            return [b.select(projection) for b in native] if projection is not None else native
        return self._read_fastavro(data, batch_rows, projection)

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        """Stream one Avro file rather than holding the whole compressed file plus its decode.

        `_read_file` does `fh.read()` on the entire compressed file and then decodes all of
        it, so peak memory is the compressed size *plus* the (usually several times larger)
        decoded size. fastavro reads the container incrementally straight off the handle, so
        peak memory is one batch.

        This uses the fastavro decoder rather than the native one `_read_file` prefers,
        because only fastavro can decode incrementally. The two agree by construction — both
        map the file's writer schema through `_avro_schema_to_arrow` — and
        `tests/unit/test_io_avro_streaming.py` pins that agreement, since a silent
        divergence would make `read()` and `iter_batches()` return different data.
        """
        batch_rows = active_config().execution.morsel_rows
        fastavro = _require_fastavro()
        with self._fs.open(path) as fh:
            reader = fastavro.reader(fh)
            schema = _avro_schema_to_arrow(reader.writer_schema)
            unions = _union_columns(schema, reader.writer_schema)
            rows: list[dict[str, Any]] = []
            for record in reader:
                rows.append(record)
                if len(rows) >= batch_rows:
                    yield self._to_batch(rows, schema, projection, unions)
                    rows = []
            if rows:
                yield self._to_batch(rows, schema, projection, unions)

    def _read_fastavro(
        self, data: bytes, batch_rows: int, projection: list[str] | None
    ) -> list[pa.RecordBatch]:
        """Row-by-row fallback for anything the native reader cannot decode."""
        import io

        fastavro = _require_fastavro()
        reader = fastavro.reader(io.BytesIO(data))
        schema = _avro_schema_to_arrow(reader.writer_schema)
        unions = _union_columns(schema, reader.writer_schema)
        out: list[pa.RecordBatch] = []
        rows: list[dict[str, Any]] = []
        for record in reader:
            rows.append(record)
            if len(rows) >= batch_rows:
                out.append(self._to_batch(rows, schema, projection, unions))
                rows = []
        if rows:
            out.append(self._to_batch(rows, schema, projection, unions))
        return out

    @staticmethod
    def _to_batch(
        rows: list[dict[str, Any]],
        schema: pa.Schema,
        projection: list[str] | None,
        unions: tuple[str, ...] = (),
    ) -> pa.RecordBatch:
        for name in unions:
            struct_type = schema.field(name).type
            for row in rows:
                row[name] = _as_union_member(row.get(name), struct_type)
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


# Arrow temporal/decimal → the Avro **logical** type that carries it. Falling through to
# `"string"` (as this did) does not merely lose the type: `to_pylist()` yields `date` /
# `datetime` / `Decimal` objects, which fastavro refuses against a string branch, so
# writing *any* temporal or decimal column raised. The reader has always mapped these
# logical types back (`_AVRO_LOGICAL_TO_ARROW`); this is the write side of that map, and
# the two are inverses so a round trip preserves the type.
_ARROW_TO_AVRO_LOGICAL: tuple[tuple[Any, dict[str, Any]], ...] = (
    (pa.date32(), {"type": "int", "logicalType": "date"}),
    (pa.date64(), {"type": "long", "logicalType": "timestamp-millis"}),
    (pa.time32("ms"), {"type": "int", "logicalType": "time-millis"}),
    (pa.time64("us"), {"type": "long", "logicalType": "time-micros"}),
)


def _avro_branch(arrow_type: pa.DataType) -> str | dict[str, Any]:
    """Map an Arrow type to the Avro type name (or logical type) used for writing."""
    if pa.types.is_boolean(arrow_type):
        return "boolean"
    if pa.types.is_integer(arrow_type):
        return "long"
    if pa.types.is_floating(arrow_type):
        return "double"
    if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
        return "bytes"
    if pa.types.is_decimal(arrow_type):
        return {
            "type": "bytes",
            "logicalType": "decimal",
            "precision": arrow_type.precision,
            "scale": arrow_type.scale,
        }
    if pa.types.is_timestamp(arrow_type):
        # Avro distinguishes instant (`timestamp-*`, tz-aware) from wall clock
        # (`local-timestamp-*`), exactly as the reader's map does.
        unit = "millis" if arrow_type.unit in ("s", "ms") else "micros"
        prefix = "timestamp" if arrow_type.tz else "local-timestamp"
        return {"type": "long", "logicalType": f"{prefix}-{unit}"}
    for candidate, logical in _ARROW_TO_AVRO_LOGICAL:
        if arrow_type.equals(candidate):
            return logical
    return "string"
