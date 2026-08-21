"""Avro payload codec — bare Avro records, or Confluent Schema Registry framing.

Avro is the default wire format on Kafka, and until this existed a Batcher pipeline reading
one had to hand-roll the decode in `map_batches`, which put the payload schema outside the
plan entirely.

Two shapes are handled, and the difference matters more than it looks. A *bare* payload is
a single Avro record written with no header, so writer and reader schema are whatever the
caller supplies. A *Confluent-framed* payload carries a schema id, and the writer schema is
therefore per-message: a producer that evolves its schema changes the id without changing
the topic, and messages written under both versions sit next to each other in one poll.
Avro's own schema resolution is what makes that decodable — every record is read with *its*
writer schema against *one* reader schema — and it is why the reader schema is resolved
once at construction and pinned. Decoding each message against its own schema and hoping
the results line up would produce a column whose type changes between micro-batches.

The Avro-to-Arrow type mapping is not restated here: it is the one
`io.formats.structured.avro` already uses for files, imported directly, so a record decodes
to the same Arrow types whether it arrived in a file or on a topic.
"""

from __future__ import annotations

import io
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError, PlanError
from batcher._internal.optional import require
from batcher.io.formats.streaming.codecs.base import CODECS, null_mask_from
from batcher.io.formats.streaming.codecs.wire import (
    SchemaRegistry,
    frame_confluent,
    unframe_confluent,
)

__all__ = ["AvroCodec"]


def _require_fastavro() -> Any:
    """Import `fastavro`, or raise the shared install hint."""
    return require("fastavro", feature="Avro payload decoding", provides="fastavro", extra="avro")


def _parse_schema_text(schema: Any) -> dict[str, Any]:
    """Accept an Avro schema as a dict, a JSON string, or a path to a ``.avsc`` file."""
    import json as _json

    if isinstance(schema, dict):
        return schema
    if isinstance(schema, str):
        text = schema.strip()
        if not text.startswith(("{", "[", '"')):
            # A path is the form a user reaches for first, and reading it here saves every
            # caller the same three lines. A file that does not exist is reported as such
            # rather than as a JSON parse error on its own name.
            try:
                with open(text, encoding="utf-8") as handle:
                    text = handle.read()
            except OSError as exc:
                raise PlanError(
                    f"avro schema {schema!r} is neither JSON nor a readable .avsc file: {exc}"
                ) from exc
        try:
            parsed = _json.loads(text)
        except ValueError as exc:
            raise PlanError(f"avro schema is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise PlanError(
                f"avro schema must be a record schema object, got {type(parsed).__name__}"
            )
        return parsed
    raise PlanError(
        f"avro schema must be a dict, a JSON string, or a .avsc path, not {type(schema).__name__}"
    )


@CODECS.register("avro")
class AvroCodec:
    """Decode and encode Avro message payloads.

    Give it either an explicit `schema` (a bare-Avro topic) or a `registry` (a
    Confluent-framed topic), or both — an explicit schema alongside a registry pins the
    *reader* schema while writer schemas continue to be resolved per message, which is how a
    consumer stays on a stable column set across a producer's schema evolution.
    """

    name = "avro"

    __slots__ = (
        "_arrow_type",
        "_mode",
        "_reader_schema",
        "_registry",
        "_schema_id",
        "_subject",
        "_union_fields",
        "_writer_cache",
    )

    def __init__(
        self,
        *,
        schema: Any = None,
        registry: SchemaRegistry | None = None,
        subject: str | None = None,
        mode: str = "fail",
        **_: Any,
    ) -> None:
        """Resolve the reader schema now, so the column's Arrow type is known before any poll.

        Args:
            schema: The reader schema, as a dict, JSON text, or a ``.avsc`` path.
            registry: A `SchemaRegistry` when payloads carry Confluent framing.
            subject: The subject to read the reader schema from when `schema` is omitted.
            mode: ``"fail"`` to raise on an undecodable record, ``"permissive"`` to null it.
            _: Ignored passthrough, so an unrelated source option is not a construction error.

        Raises:
            PlanError: If neither a schema nor a registry subject was given, since the
                stream's Arrow type would then be unknowable until the first message —
                and a schema that is discovered rather than declared cannot be reported by
                `Dataset.schema` before the query runs.
        """
        fastavro = _require_fastavro()
        self._registry = registry
        self._subject = subject
        self._mode = mode
        self._schema_id: int | None = None
        if schema is not None:
            reader = _parse_schema_text(schema)
        elif registry is not None and subject:
            self._schema_id, text = registry.latest(subject)
            reader = _parse_schema_text(text)
        else:
            raise PlanError(
                "value_format='avro' needs a reader schema: pass value_schema=<dict | JSON | "
                "path.avsc>, or schema_registry=<url> so the subject's latest version can be "
                "resolved."
            )
        self._reader_schema = fastavro.parse_schema(reader)
        from batcher.io.formats.structured.avro import _avro_schema_to_arrow, _union_columns

        arrow_schema = _avro_schema_to_arrow(reader)
        self._arrow_type = pa.struct(list(arrow_schema))
        self._union_fields = _union_columns(arrow_schema, reader)
        # Parsed writer schemas, keyed by registry id. `parse_schema` walks and normalizes
        # the whole schema, so doing it per message on a high-rate topic costs more than the
        # decode it prepares for.
        self._writer_cache: dict[int, Any] = {}

    def arrow_type(self) -> pa.DataType:
        """The struct type one decoded Avro record becomes.

        Returns:
            A `pa.struct` with one field per Avro field, carrying Avro's logical types.
        """
        return self._arrow_type

    def _writer_for(self, schema_id: int) -> Any:
        """The parsed writer schema for `schema_id`, fetched from the registry once."""
        cached = self._writer_cache.get(schema_id)
        if cached is not None:
            return cached
        fastavro = _require_fastavro()
        if self._registry is None:  # pragma: no cover - unreachable without framing
            raise BackendError("a Confluent-framed payload needs schema_registry= to decode")
        parsed = fastavro.parse_schema(_parse_schema_text(self._registry.schema_by_id(schema_id)))
        self._writer_cache[schema_id] = parsed
        return parsed

    def decode(self, column: pa.Array) -> pa.Array:
        """Decode a binary payload column into one struct column.

        Args:
            column: The raw `value` (or `key`) column of one poll.

        Returns:
            A struct array of `arrow_type()`, null wherever the payload was null (a Kafka
            tombstone) or, under ``mode="permissive"``, undecodable.

        Raises:
            BackendError: Under ``mode="fail"``, on the first record that will not decode,
                naming the row so the offending offset can be found.
        """
        fastavro = _require_fastavro()
        payloads = column.to_pylist()
        nulls = null_mask_from(column)
        rows: list[dict[str, Any] | None] = []
        for index, payload in enumerate(payloads):
            if nulls[index] or payload is None:
                rows.append(None)
                continue
            try:
                if self._registry is not None:
                    framed = unframe_confluent(payload)
                    writer = self._writer_for(framed.schema_id)
                    body = framed.body
                else:
                    writer = self._reader_schema
                    body = payload
                rows.append(
                    fastavro.schemaless_reader(io.BytesIO(body), writer, self._reader_schema)
                )
            except Exception as exc:
                if self._mode == "permissive":
                    rows.append(None)
                    continue
                raise BackendError(
                    f"Avro decode failed on row {index} of the batch ({len(payload)} bytes): "
                    f"{exc}. Pass value_decode_mode='permissive' to null undecodable records "
                    "instead of failing the query."
                ) from exc
        if self._union_fields:
            self._place_unions(rows)
        return pa.array(rows, type=self._arrow_type)

    def _place_unions(self, rows: list[dict[str, Any] | None]) -> None:
        """Rewrite multi-branch union values into the `memberN` struct Arrow holds them as.

        fastavro yields the bare value for a primitive union, which has no single Arrow
        type. The file reader already solved this; the placement helper is imported rather
        than restated so a union decodes identically from a topic and from a file.
        """
        from batcher.io.formats.structured.avro import _as_union_member

        for row in rows:
            if row is None:
                continue
            for name in self._union_fields:
                if name in row:
                    row[name] = _as_union_member(row[name], self._arrow_type.field(name).type)

    def encode(self, column: pa.Array) -> pa.Array:
        """Encode a struct column back into Avro payloads.

        Frames the output for the Schema Registry when this codec was built with one, so a
        topic written by Batcher is readable by a Confluent deserializer without a shim.

        Args:
            column: A struct column matching `arrow_type()`.

        Returns:
            A binary column of serialized payloads, null where the input row was null.

        Raises:
            PlanError: If the codec was built against a registry but has no schema id to
                frame with — an explicit reader schema that was never registered.
        """
        fastavro = _require_fastavro()
        if self._registry is not None and self._schema_id is None:
            raise PlanError(
                "encoding Confluent-framed Avro needs the writer schema's registry id; build "
                "the codec from a subject (schema_registry=<url>) rather than an inline "
                "value_schema, or drop schema_registry to write bare Avro."
            )
        out: list[bytes | None] = []
        for row in column.to_pylist():
            if row is None:
                out.append(None)
                continue
            buffer = io.BytesIO()
            fastavro.schemaless_writer(buffer, self._reader_schema, row)
            body = buffer.getvalue()
            out.append(body if self._schema_id is None else frame_confluent(self._schema_id, body))
        return pa.array(out, type=pa.binary())
