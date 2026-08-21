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
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError, PlanError
from batcher.io.formats.streaming.codecs.base import CODECS, null_mask_from, scatter

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

    __slots__ = ("_arrow_type", "_mode", "_schema")

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
            registry: A `SchemaRegistry`, when the topic registers a JSON Schema for the
                subject; its latest version supplies the reader schema.
            subject: The subject to resolve the reader schema from.
            mode: ``"fail"`` to raise on an unparseable batch, ``"permissive"`` to null the
                rows it could not parse.
            _: Ignored passthrough.

        Raises:
            PlanError: If neither a schema nor a registry subject was given. See the module
                docstring for why this cannot be inferred instead.
        """
        self._mode = mode
        if schema is None and registry is not None and subject:
            schema = _json_schema_to_arrow(registry.latest(subject)[1])
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
        """Parse a binary payload column into one struct column.

        Args:
            column: The raw `value` (or `key`) column of one poll.

        Returns:
            A struct array, null wherever the payload was null or empty.

        Raises:
            BackendError: Under ``mode="fail"``, when the batch will not parse.
        """
        import pyarrow.json as pj

        nulls = null_mask_from(column)
        payloads = column.to_pylist()
        # An empty payload is not a document. Left in, pyarrow's reader either skips the
        # line (shifting every subsequent row onto the wrong message) or refuses the whole
        # buffer, so blanks are tracked as nulls and excluded from the parse instead.
        keep = [
            index
            for index, payload in enumerate(payloads)
            if not nulls[index] and payload not in (None, b"")
        ]
        if not keep:
            return pa.nulls(len(column), type=self._arrow_type)
        buffer = _NL.join(_one_line(payloads[i]) for i in keep)
        try:
            table = pj.read_json(
                pa.BufferReader(buffer),
                parse_options=pj.ParseOptions(
                    explicit_schema=self._schema, unexpected_field_behavior="ignore"
                ),
            )
        except Exception as exc:
            if self._mode == "permissive":
                return pa.nulls(len(column), type=self._arrow_type)
            raise BackendError(
                f"JSON decode failed for a batch of {len(keep)} payloads: {exc}. Check that "
                "value_schema= matches the documents on the topic, or pass "
                "value_decode_mode='permissive' to null unparseable batches."
            ) from exc
        decoded = _struct_from_table(table, self._arrow_type)
        if len(keep) == len(column):
            return decoded
        return scatter(decoded, keep, len(column), self._arrow_type)

    def encode(self, column: pa.Array) -> pa.Array:
        """Serialize a struct column back into one JSON document per row.

        Args:
            column: A struct column.

        Returns:
            A binary column of UTF-8 JSON documents, null where the row was null.
        """

        return pa.array(
            [None if row is None else _dump(row) for row in column.to_pylist()],
            type=pa.binary(),
        )


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
    anything richer (``oneOf``, ``$ref`` across documents) is refused rather than
    approximated, because a silently-wrong column type is worse than an explicit
    ``value_schema=``.
    """
    import json as _json

    document = _json.loads(text)
    properties = document.get("properties")
    if not isinstance(properties, dict):
        raise PlanError(
            "the registered JSON Schema has no top-level 'properties' object, so it does "
            "not describe a record; pass value_schema= explicitly."
        )
    return pa.schema([pa.field(name, _json_type(spec)) for name, spec in properties.items()])


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


def _json_type(spec: Any) -> pa.DataType:
    """One JSON Schema property spec as an Arrow type."""
    if not isinstance(spec, dict):
        return pa.string()
    kind = spec.get("type")
    if isinstance(kind, list):  # `["null", "string"]` — the nullable idiom
        kind = next((k for k in kind if k != "null"), "string")
    fmt = spec.get("format")
    if isinstance(fmt, str) and fmt in _JSON_FORMATS:
        return _JSON_FORMATS[fmt]
    if kind == "object":
        properties = spec.get("properties")
        if isinstance(properties, dict):
            return pa.struct([pa.field(n, _json_type(s)) for n, s in properties.items()])
        return pa.struct([])
    if kind == "array":
        return pa.list_(_json_type(spec.get("items", {})))
    return _JSON_TYPES.get(kind or "string", pa.string())
