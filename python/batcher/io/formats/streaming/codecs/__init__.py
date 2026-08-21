"""`io.formats.streaming.codecs` — wire formats for broker message payloads.

A broker delivers opaque bytes; a pipeline needs columns. These codecs are that step, done
once per micro-batch rather than once per row, and known to the plan rather than buried in
a `map_batches`:

* `avro` — bare Avro, or Confluent Schema Registry framing with per-message writer schemas;
* `json` — one document per message, parsed by pyarrow's own JSON reader;
* `protobuf` — a generated message class, with the Confluent message-index array handled;
* `string` / `bytes` — UTF-8 text, and the identity codec.

`SchemaRegistry` and the `frame_confluent` / `unframe_confluent` pair in `wire` are shared
by the framed formats, and are usable on their own by a source Batcher does not ship.
"""

from __future__ import annotations

from batcher.io.formats.streaming.codecs.base import (
    CODECS,
    DecodeMode,
    PayloadCodec,
    resolve_codec,
)
from batcher.io.formats.streaming.codecs.wire import (
    CONFLUENT_MAGIC,
    FramedPayload,
    SchemaRegistry,
    frame_confluent,
    unframe_confluent,
)

__all__ = [
    "CODECS",
    "CONFLUENT_MAGIC",
    "DecodeMode",
    "FramedPayload",
    "PayloadCodec",
    "SchemaRegistry",
    "frame_confluent",
    "resolve_codec",
    "unframe_confluent",
]
