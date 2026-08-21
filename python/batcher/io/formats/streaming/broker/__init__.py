"""Shared base for row/message-based streaming brokers (Kafka, Kinesis, ...).

A *broker* source models an unbounded stream of raw messages. Unlike file formats, the
payload is opaque: each message is delivered as raw ``bytes`` plus its coordinates
(partition/shard, offset, timestamp, topic) -- decoding and schema-registry handling are
downstream concerns expressed as ordinary expressions over the ``value`` column.

The contract:

* the schema is **fixed** --
  ``{key: binary, value: binary, partition: int64, offset: int64,
  timestamp: int64, topic: string}``;
* ``row_count()`` is ``None`` (the stream is unbounded);
* ``iter_batches()`` is an (infinite) generator that polls ``poll_size`` messages at a time
  and assembles each poll into **one** ``RecordBatch`` -- batch-granularity assembly, never
  per-row Python in the hot path;
* ``splits()`` returns one picklable :class:`BrokerSplit` per partition/shard so a
  distributed reader consumes partitions in parallel.

Concrete brokers subclass :class:`BrokerSource` and implement two primitives: discover the
partitions/shards (``_discover_partitions``) and poll a batch of messages (``_poll``).

The package splits along the seam that matters for imports: `schema` holds the
client-free value types and redaction, `source` the abstract source and its poll loop,
and `split` the per-partition epoch reader and its warm-consumer cache.
"""

from __future__ import annotations

from batcher.io.formats.streaming.broker.schema import (
    BrokerMessage,
    as_header_pairs,
    broker_schema,
    opaque_offset,
    redact_broker_options,
)
from batcher.io.formats.streaming.broker.source import BrokerSource
from batcher.io.formats.streaming.broker.split import BrokerSplit

__all__ = [
    "BrokerMessage",
    "BrokerSource",
    "BrokerSplit",
    "as_header_pairs",
    "broker_schema",
    "opaque_offset",
    "redact_broker_options",
]
