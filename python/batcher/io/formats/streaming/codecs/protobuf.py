"""Protobuf payload codec — a generated message class, or a registry-framed descriptor.

Protobuf on a broker differs from Protobuf in a file in exactly one way, and it is the way
that breaks a naive decoder: a Confluent-framed payload carries a *message-index array*
after the schema id, naming which message inside a multi-message ``.proto`` was written.
Skipping only the five-byte header leaves that array at the front of the body, where it
parses as field data — so the decode succeeds and returns wrong values. `wire.unframe_confluent`
consumes it; this codec supplies the descriptor to interpret it against.

The message-to-Arrow conversion is `protarrow`, the same library
`io.formats.semistructured.protobuf` uses for files, so a message decodes to the same Arrow
types from either side.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError, PlanError
from batcher._internal.optional import require
from batcher.io.formats.streaming.codecs.base import CODECS, null_mask_from, scatter
from batcher.io.formats.streaming.codecs.wire import (
    SchemaRegistry,
    frame_confluent,
    unframe_confluent,
)

__all__ = ["ProtobufCodec"]


def _require_protarrow() -> Any:
    """Import `protarrow`, or raise the shared install hint."""
    return require(
        "protarrow",
        feature="Protobuf payload decoding",
        provides="protarrow + protobuf",
        extra="protobuf",
    )


@CODECS.register("protobuf")
class ProtobufCodec:
    """Decode and encode Protobuf message payloads.

    `schema` is the generated message *class* (``my_pb2.Order``). A descriptor cannot be
    derived from a registry-hosted ``.proto`` without compiling it, so a registry is used
    here for framing and for validating the message index — never as a substitute for the
    generated class.
    """

    name = "protobuf"

    __slots__ = ("_arrow_type", "_indexes", "_message", "_mode", "_registry", "_schema_id")

    def __init__(
        self,
        *,
        schema: Any = None,
        registry: SchemaRegistry | None = None,
        subject: str | None = None,
        mode: str = "fail",
        message_indexes: tuple[int, ...] | None = None,
        **_: Any,
    ) -> None:
        """Bind the message class and derive the Arrow type it maps to.

        Args:
            schema: The generated protobuf message class.
            registry: A `SchemaRegistry` when payloads carry Confluent framing.
            subject: The subject whose id frames encoded payloads.
            mode: ``"fail"`` or ``"permissive"``.
            message_indexes: The message-index path to write when encoding. Defaults to
                ``(0,)``, the first message in the descriptor, which is what a
                single-message ``.proto`` always is.
            _: Ignored passthrough.

        Raises:
            PlanError: If no message class was given. It cannot be inferred: the registry
                stores ``.proto`` text, and turning that into a descriptor means running
                ``protoc`` at query time.
        """
        protarrow = _require_protarrow()
        if schema is None or not hasattr(schema, "DESCRIPTOR"):
            raise PlanError(
                "value_format='protobuf' needs the generated message class, e.g. "
                "value_schema=orders_pb2.Order. A Schema Registry stores .proto text, which "
                "cannot be turned into a descriptor without compiling it."
            )
        self._message = schema
        self._mode = mode
        self._registry = registry
        self._indexes = tuple(message_indexes) if message_indexes else (0,)
        self._schema_id: int | None = None
        if registry is not None and subject:
            self._schema_id = registry.latest(subject)[0]
        self._arrow_type = pa.struct(list(protarrow.message_type_to_schema(schema)))

    def arrow_type(self) -> pa.DataType:
        """The struct type one decoded message becomes.

        Returns:
            A `pa.struct` with one field per protobuf field.
        """
        return self._arrow_type

    def decode(self, column: pa.Array) -> pa.Array:
        """Decode a binary payload column into one struct column.

        Args:
            column: The raw payload column of one poll.

        Returns:
            A struct array of `arrow_type()`, null where the payload was null or, under
            ``mode="permissive"``, undecodable.

        Raises:
            BackendError: Under ``mode="fail"``, on the first record that will not parse.
        """
        protarrow = _require_protarrow()
        nulls = null_mask_from(column)
        messages: list[Any] = []
        keep: list[int] = []
        for index, payload in enumerate(column.to_pylist()):
            if nulls[index] or payload is None:
                continue
            try:
                body = (
                    unframe_confluent(payload, protobuf=True).body
                    if self._registry is not None
                    else payload
                )
                message = self._message()
                message.ParseFromString(body)
            except Exception as exc:
                if self._mode == "permissive":
                    continue
                raise BackendError(
                    f"Protobuf decode failed on row {index} of the batch: {exc}. Pass "
                    "value_decode_mode='permissive' to null undecodable records."
                ) from exc
            messages.append(message)
            keep.append(index)
        if not keep:
            return pa.nulls(len(column), type=self._arrow_type)
        table = protarrow.messages_to_table(messages, self._message)
        decoded = pa.StructArray.from_arrays(
            [table.column(f.name).combine_chunks() for f in self._arrow_type],
            fields=list(self._arrow_type),
        )
        if len(keep) == len(column):
            return decoded
        return scatter(decoded, keep, len(column), self._arrow_type)

    def encode(self, column: pa.Array) -> pa.Array:
        """Encode a struct column back into Protobuf payloads.

        Args:
            column: A struct column matching `arrow_type()`.

        Returns:
            A binary column, framed for the Schema Registry when this codec has an id.
        """
        protarrow = _require_protarrow()
        out: list[bytes | None] = []
        rows = column.to_pylist()
        present = [row for row in rows if row is not None]
        encoded: list[bytes] = []
        if present:
            table = pa.Table.from_pylist(present, schema=pa.schema(list(self._arrow_type)))
            encoded = [
                message.SerializeToString()
                for message in protarrow.table_to_messages(table, self._message)
            ]
        cursor = 0
        for row in rows:
            if row is None:
                out.append(None)
                continue
            body = encoded[cursor]
            cursor += 1
            out.append(
                body
                if self._schema_id is None
                else frame_confluent(self._schema_id, body, message_indexes=self._indexes)
            )
        return pa.array(out, type=pa.binary())
