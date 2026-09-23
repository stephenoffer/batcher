"""JSON payload codec — the most common Kafka wire format, decoded to a real struct column.

A JSON topic is usually read today by casting `value` to a string and pulling fields out
one at a time with `.json.extract_*`. That works and it is what the Kafka guide shows, but
it re-parses the whole document once per extracted field, and it leaves the payload's shape
outside the plan — so the engine cannot report the stream's schema, and a projection cannot
be pushed into the parse.

Decoding once per batch into a struct column fixes both. The parse itself is pyarrow's own
JSON reader over the concatenated payloads, which is the same C++ path
`io.formats.semistructured.json` uses for files — so a document decodes to the same Arrow
types whether it arrived in a file or on a topic, and the per-row work stays out of Python.

The reader schema must be **declared** (`value_schema=`, or a registry subject). Inferring it
from the first batch is not an option a broker source can offer, and the reason is worth
stating because it looks like a convenience: the plan is built — and every expression over
the payload type-checked — before a single message is polled, so a type discovered on the
first poll arrives after everything that needed it. What that produced when it was tried was
not an error but an empty struct: the plan carried `struct<>`, the decode produced real
fields, and the batch was coerced back to the plan's type on the way out. Decoded data,
silently dropped. Declaring the schema is what makes the feature work at all.

**Decoding preserves rows.** One message is one row, always: a whitespace-only message is a
null like an empty one, and a message holding two documents is malformed rather than two
rows. The batch parse is checked against the number of messages it was given, and on any
disagreement — or any parse error — the batch is re-parsed one message at a time, so a bad
message costs its own row (a null under ``permissive``, a named-row error under ``fail``)
and never shifts or nulls its neighbours.

With `schema_registry=` the payloads are taken to be what a Confluent JSON Schema serializer
writes: the five-byte magic-byte-and-schema-id header, then the document. The header is
checked and stripped on decode and written on encode, as the Avro codec does.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError, PlanError
from batcher.io.formats.streaming.codecs.base import CODECS, null_mask_from, payloads_of, scatter
from batcher.io.formats.streaming.codecs.wire import frame_confluent, unframe_confluent

__all__ = ["JsonCodec"]

#: Newline byte used to join payloads into the NDJSON document pyarrow's reader consumes.
_NL = b"\n"


def _as_arrow_schema(schema: Any) -> pa.Schema:
    """Accept a reader schema as a `pa.Schema`, a struct type, or a ``{name: type}`` map."""
    if isinstance(schema, pa.Schema):
        return schema
    if isinstance(schema, pa.DataType):
        if not pa.types.is_struct(schema):
            raise PlanError(f"a JSON value_schema given as a type must be a struct, got {schema}")
        return pa.schema(list(schema))
    if isinstance(schema, dict):
        return pa.schema(
            [pa.field(name, _field_type(name, dtype)) for name, dtype in schema.items()]
        )
    raise PlanError(
        f"JSON value_schema must be a pyarrow Schema, a struct type, or a "
        f"{{name: type}} mapping, not {type(schema).__name__}"
    )


def _field_type(name: str, dtype: Any) -> pa.DataType:
    """One `{name: type}` entry's Arrow type, accepting a `pa.DataType` or a dtype name.

    The name is resolved through `plan.types.resolve_dtype`, the engine's single dtype
    vocabulary, so ``"int64"`` means here exactly what it means in `cast` and in a schema
    declared anywhere else.
    """
    if isinstance(dtype, pa.DataType):
        return dtype
    from batcher.plan.types import resolve_dtype

    resolved = resolve_dtype(str(dtype))
    if resolved is None:
        raise PlanError(
            f"JSON value_schema field {name!r} has unknown type {dtype!r}; use a dtype name "
            "such as 'int64' / 'string' / 'timestamp[us]', or a pyarrow DataType."
        )
    return resolved


@CODECS.register("json")
class JsonCodec:
    """Decode and encode JSON message payloads as one struct column.

    One document per message, as every JSON producer on a message broker writes them.
    Embedded newlines inside a document are fine — they are escaped in valid JSON — which
    is what lets the batch be joined into one NDJSON buffer and parsed in a single call.
    """

    name = "json"

    __slots__ = ("_arrow_type", "_mode", "_registry", "_schema", "_schema_id")

    def __init__(
        self,
        *,
        schema: Any = None,
        registry: Any = None,
        subject: str | None = None,
        mode: str = "fail",
        **_: Any,
    ) -> None:
        """Pin the reader schema, resolving it from a registry subject when one is given.

        Args:
            schema: The reader schema: a `pa.Schema`, a struct type, or a ``{name: type}``
                mapping.
            registry: A `SchemaRegistry`, when the topic is written by a Confluent JSON
                Schema serializer: payloads carry its framing, and — when `schema` is
                omitted — the subject's latest version supplies the reader schema.
            subject: The subject to resolve the reader schema from.
            mode: ``"fail"`` to raise on the first unparseable message, naming its row;
                ``"permissive"`` to null that row and keep every other.
            _: Ignored passthrough.

        Raises:
            PlanError: If neither a schema nor a registry subject was given. See the module
                docstring for why this cannot be inferred instead.
        """
        self._mode = mode
        self._registry = registry
        self._schema_id: int | None = None
        if schema is None and registry is not None and subject:
            self._schema_id, text = registry.latest(subject)
            schema = _json_schema_to_arrow(text)
        if schema is None:
            raise PlanError(
                "value_format='json' needs a reader schema: pass value_schema={'col': "
                "'type', ...} (or a pyarrow Schema), or schema_registry=<url> so the "
                "subject's registered JSON Schema can be resolved. It cannot be inferred "
                "from the data: the plan is built before the first message is polled, so a "
                "type discovered later arrives too late for every expression that needed it."
            )
        self._schema: pa.Schema = _as_arrow_schema(schema)
        self._arrow_type = pa.struct(list(self._schema))

    def arrow_type(self) -> pa.DataType:
        """The struct type one decoded document becomes.

        Returns:
            The declared struct type.
        """
        return self._arrow_type

    def decode(self, column: pa.Array) -> pa.Array:
        """Parse a binary payload column into one struct column, one row per message.

        Args:
            column: The raw `value` (or `key`) column of one poll.

        Returns:
            A struct array with exactly one row per message: null wherever the payload was
            null, empty or whitespace, and, under ``mode="permissive"``, wherever the
            message would not parse as exactly one document.

        Raises:
            BackendError: Under ``mode="fail"``, naming the first message that would not
                parse as exactly one document.
        """
        nulls = null_mask_from(column)
        bodies: list[bytes | None] = []
        for index, payload in enumerate(payloads_of(column)):
            if nulls[index] or payload is None:
                bodies.append(None)
                continue
            if self._registry is not None:
                try:
                    payload = unframe_confluent(payload).body
                except BackendError as exc:
                    bodies.append(self._malformed(index, exc))
                    continue
            # An empty or whitespace-only payload is not a document. Left in, pyarrow's
            # reader skips the line, which shifted every later row onto the wrong message.
            bodies.append(payload if payload.strip() else None)
        keep = [index for index, body in enumerate(bodies) if body is not None]
        if not keep:
            return pa.nulls(len(column), type=self._arrow_type)
        # The leading newline keeps a byte-order mark off the buffer's first byte, where the
        # reader would strip it: then every kept message yields at least one row or an
        # error, so the count check below cannot be satisfied by two faults cancelling out.
        buffer = _NL + _NL.join(_one_line(bodies[i]) for i in keep)
        try:
            table = self._parse(buffer)
        except Exception:
            table = None
        if table is None or table.num_rows != len(keep):
            # A bad message, or one holding more (or fewer) than one document: find it by
            # parsing one message at a time, so only its own row is affected.
            return self._decode_each(bodies, keep, len(column))
        decoded = _struct_from_table(table, self._arrow_type)
        if len(keep) == len(column):
            return decoded
        return scatter(decoded, keep, len(column), self._arrow_type)

    def _parse(self, buffer: bytes) -> pa.Table:
        import pyarrow.json as pj

        return pj.read_json(
            pa.BufferReader(buffer),
            parse_options=pj.ParseOptions(
                explicit_schema=self._schema, unexpected_field_behavior="ignore"
            ),
        )

    def _decode_each(self, bodies: list[bytes | None], keep: list[int], total: int) -> pa.Array:
        """The slow path: parse each message alone and hold each to exactly one document."""
        parts: list[pa.Array] = []
        good: list[int] = []
        for index in keep:
            try:
                table = self._parse(_one_line(bodies[index]))  # type: ignore[arg-type]
                if table.num_rows != 1:
                    raise BackendError(
                        f"the message holds {table.num_rows} JSON documents; a message must "
                        "carry exactly one"
                    )
            except Exception as exc:
                self._malformed(index, exc)
                continue
            parts.append(_struct_from_table(table, self._arrow_type))
            good.append(index)
        if not good:
            return pa.nulls(total, type=self._arrow_type)
        return scatter(pa.concat_arrays(parts), good, total, self._arrow_type)

    def _malformed(self, index: int, exc: Exception) -> None:
        """Answer one bad message: None under ``permissive``, raise naming it under ``fail``."""
        if self._mode == "permissive":
            return None
        raise BackendError(
            f"JSON decode failed on row {index} of the batch: {exc}. Check that value_schema= "
            "matches the documents on the topic, or pass value_decode_mode='permissive' to "
            "null the messages that do not parse."
        ) from exc

    def encode(self, column: pa.Array) -> pa.Array:
        """Serialize a struct column back into one JSON document per row.

        Framed for the Schema Registry when the codec was built from a registry subject, so
        a Confluent JSON Schema deserializer reads it without a shim.

        Args:
            column: A struct column.

        Returns:
            A binary column of UTF-8 JSON documents, null where the row was null.

        Raises:
            PlanError: If the codec has a registry but no schema id to frame with — an
                explicit `value_schema` alongside `schema_registry`.
        """
        if self._registry is not None and self._schema_id is None:
            raise PlanError(
                "encoding Confluent-framed JSON needs the schema's registry id; build the codec "
                "from a subject (schema_registry=<url>) rather than an inline value_schema, or "
                "drop schema_registry to write bare JSON."
            )
        out: list[bytes | None] = []
        for row in column.to_pylist():
            if row is None:
                out.append(None)
                continue
            body = _dump(row)
            out.append(body if self._schema_id is None else frame_confluent(self._schema_id, body))
        return pa.array(out, type=pa.binary())


def _dump(row: dict) -> bytes:
    """One decoded row as a compact UTF-8 JSON document."""
    import json as _json

    return _json.dumps(row, default=str, separators=(",", ":")).encode()


def _one_line(payload: bytes) -> bytes:
    """One payload as a single NDJSON line.

    A pretty-printed document spans several physical lines, which the NDJSON reader would
    read as several broken records. Compacting it is not free, so it is done only when the
    payload actually contains a newline — the overwhelmingly common single-line case pays a
    membership test and nothing else.
    """
    if _NL not in payload:
        return payload
    import json as _json

    try:
        return _json.dumps(_json.loads(payload), separators=(",", ":")).encode()
    except ValueError:
        # Not parseable here either; hand it to the batch parser, whose error names the
        # payload and honours the decode mode rather than failing inside this helper.
        return payload.replace(_NL, b" ")


def _struct_from_table(table: pa.Table, struct_type: pa.DataType) -> pa.Array:
    """One struct array holding every row of `table`, in `struct_type`'s field order."""
    combined = table.combine_chunks()
    arrays = []
    for field in struct_type:
        if field.name not in combined.column_names:
            arrays.append(pa.nulls(combined.num_rows, type=field.type))
        elif combined.num_rows:
            arrays.append(combined.column(field.name).chunk(0))
        else:
            arrays.append(pa.array([], type=field.type))
    arrays = [
        array if array.type.equals(field.type) else array.cast(field.type)
        for array, field in zip(arrays, struct_type, strict=True)
    ]
    return pa.StructArray.from_arrays(arrays, fields=list(struct_type))


def _json_schema_to_arrow(text: str) -> pa.Schema:
    """Map a JSON Schema document's top-level ``properties`` onto an Arrow schema.

    Only the object-with-properties shape a message payload actually uses is translated;
    anything richer — ``$ref``, ``oneOf``/``anyOf``/``allOf``/``not``, or a ``type`` list
    naming two non-null types — is refused with the property named rather than approximated,
    because a silently-wrong column type is worse than an explicit ``value_schema=``. (The
    previous version said so here and then mapped all of them to ``string``.)
    """
    import json as _json

    document = _json.loads(text)
    properties = document.get("properties")
    if not isinstance(properties, dict):
        raise PlanError(
            "the registered JSON Schema has no top-level 'properties' object, so it does "
            "not describe a record; pass value_schema= explicitly."
        )
    return pa.schema([pa.field(name, _json_type(spec, name)) for name, spec in properties.items()])


#: JSON Schema type (plus optional `format`) → Arrow type.
_JSON_TYPES: dict[str, pa.DataType] = {
    "string": pa.string(),
    "integer": pa.int64(),
    "number": pa.float64(),
    "boolean": pa.bool_(),
}

_JSON_FORMATS: dict[str, pa.DataType] = {
    "date": pa.date32(),
    "date-time": pa.timestamp("us", tz="UTC"),
    "time": pa.time64("us"),
}


#: JSON Schema keywords whose meaning is a choice between, or a reference to, other schemas.
#: None has one Arrow type, so a property using any of them is refused.
_UNTRANSLATABLE = ("$ref", "oneOf", "anyOf", "allOf", "not")


def _json_type(spec: Any, path: str) -> pa.DataType:
    """One JSON Schema property spec as an Arrow type; `path` names it in an error."""
    if not isinstance(spec, dict):
        return pa.string()
    refused = [k for k in _UNTRANSLATABLE if k in spec]
    kind = spec.get("type")
    if isinstance(kind, list):  # `["null", "string"]` — the nullable idiom
        named = [k for k in kind if k != "null"]
        if len(named) > 1:
            refused.append(f"type {kind}")
        kind = named[0] if named else "string"
    if refused:
        raise PlanError(
            f"the registered JSON Schema's property {path!r} uses {', '.join(refused)}, which "
            "has no single Arrow type; pass value_schema= with the column types you want."
        )
    fmt = spec.get("format")
    if isinstance(fmt, str) and fmt in _JSON_FORMATS:
        return _JSON_FORMATS[fmt]
    if kind == "object":
        properties = spec.get("properties")
        if isinstance(properties, dict):
            return pa.struct(
                [pa.field(n, _json_type(s, f"{path}.{n}")) for n, s in properties.items()]
            )
        return pa.struct([])
    if kind == "array":
        return pa.list_(_json_type(spec.get("items", {}), f"{path}[]"))
    return _JSON_TYPES.get(kind or "string", pa.string())
