"""The two payload codecs with no schema: raw bytes, and UTF-8 text.

``value_format="string"`` is the one every quick-look pipeline starts with — a topic of
newline-delimited text, a CSV line, a log record — and writing it as a codec rather than as
a `cast` has one property that matters: the source's declared schema then says `string`, so
`Dataset.schema` is right before a single message is polled, and an expression over the
column type-checks at plan time instead of at the first micro-batch.

``value_format="bytes"`` is the identity codec. It exists so "leave the payload alone" is
sayable, which makes the option total: every source can name a format, and the default is
not a special case in the wiring.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.formats.streaming.codecs.base import CODECS

__all__ = ["BytesCodec", "StringCodec"]


@CODECS.register("string")
class StringCodec:
    """Decode payloads as UTF-8 text (Spark's ``CAST(value AS STRING)``)."""

    name = "string"

    __slots__ = ("_encoding", "_mode")

    def __init__(self, *, mode: str = "fail", encoding: str = "utf-8", **_: Any) -> None:
        """Create the codec.

        Args:
            mode: ``"fail"`` to raise on a payload that is not valid text in `encoding`,
                ``"permissive"`` to null it.
            encoding: The text encoding. UTF-8 by default, which is what every broker
                client and every Spark job assumes.
            _: Ignored passthrough.
        """
        self._mode = mode
        self._encoding = encoding

    def arrow_type(self) -> pa.DataType:
        """The Arrow type of a decoded payload.

        Returns:
            `pa.string()`.
        """
        return pa.string()

    def decode(self, column: pa.Array) -> pa.Array:
        """Decode the payload column as text.

        UTF-8 is a zero-copy reinterpretation of the same buffers, so the common case costs
        a validation pass and no allocation. Any other encoding needs a real transcode.

        Args:
            column: The raw payload column.

        Returns:
            A string column.

        Raises:
            BackendError: Under ``mode="fail"``, when a payload is not valid text.
        """
        if self._encoding.lower().replace("-", "") == "utf8":
            try:
                return column.cast(pa.string())
            except pa.ArrowInvalid as exc:
                if self._mode == "permissive":
                    return self._lenient(column)
                raise BackendError(
                    f"payload is not valid UTF-8: {exc}. Pass "
                    "value_decode_mode='permissive' to null undecodable records, or "
                    "value_format='bytes' to keep the raw payload."
                ) from exc
        return self._lenient(column)

    def _lenient(self, column: pa.Array) -> pa.Array:
        """Decode row by row, nulling (or raising on) what will not decode."""
        out: list[str | None] = []
        for index, payload in enumerate(column.to_pylist()):
            if payload is None:
                out.append(None)
                continue
            try:
                out.append(payload.decode(self._encoding))
            except (UnicodeDecodeError, LookupError) as exc:
                if self._mode == "permissive":
                    out.append(None)
                    continue
                raise BackendError(
                    f"payload on row {index} is not valid {self._encoding}: {exc}"
                ) from exc
        return pa.array(out, type=pa.string())

    def encode(self, column: pa.Array) -> pa.Array:
        """Encode a text column back into payload bytes.

        Args:
            column: A string column.

        Returns:
            A binary column.
        """
        if self._encoding.lower().replace("-", "") == "utf8":
            return column.cast(pa.binary())
        return pa.array(
            [None if v is None else v.encode(self._encoding) for v in column.to_pylist()],
            type=pa.binary(),
        )


@CODECS.register("bytes")
class BytesCodec:
    """Leave the payload as raw bytes — the identity codec, and the default."""

    name = "bytes"

    __slots__ = ()

    def __init__(self, **_: Any) -> None:
        """Create the codec. Every option is accepted and ignored."""

    def arrow_type(self) -> pa.DataType:
        """The Arrow type of a decoded payload.

        Returns:
            `pa.binary()`.
        """
        return pa.binary()

    def decode(self, column: pa.Array) -> pa.Array:
        """Return the payload column unchanged.

        Args:
            column: The raw payload column.

        Returns:
            The same column.
        """
        return column if pa.types.is_binary(column.type) else column.cast(pa.binary())

    def encode(self, column: pa.Array) -> pa.Array:
        """Return the payload column unchanged.

        Args:
            column: A binary column.

        Returns:
            The same column.
        """
        return self.decode(column)
