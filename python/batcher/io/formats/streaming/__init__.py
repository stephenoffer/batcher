"""`io.formats.streaming` — broker + incremental-file sources, behind the registry.

Importing this package imports every streaming source module, so each registers
itself into the ``SOURCES`` registry as a side effect (``"kafka"``,
``"kinesis"``, ``"eventhubs"``, ``"pubsub"``, ``"pulsar"``,
``"files_incremental"``). Broker sources deliver raw message ``bytes`` plus
coordinates at batch granularity; the incremental file source replicates
Databricks Auto Loader (``cloudFiles``). Each broker's client dependency is an
optional extra, deferred until construction.

`codecs` turns those raw payload bytes into typed columns — Avro, JSON, Protobuf, text —
including Confluent Schema Registry framing, so a stream's real schema is known to the
plan rather than hidden inside a `map_batches`.

`broker_sinks` is the write side: Pulsar, Kinesis, Pub/Sub and Event Hubs sinks on the one
`BrokerStreamSink` capability contract the Kafka sink (`kafka_sink`) also implements.
"""

from __future__ import annotations

from batcher.io.formats.streaming.autoloader import IncrementalFileSource
from batcher.io.formats.streaming.broker import (
    BrokerMessage,
    BrokerSource,
    BrokerSplit,
    broker_schema,
)
from batcher.io.formats.streaming.broker_sinks import (
    BrokerStreamSink,
    EventHubsStreamSink,
    KinesisStreamSink,
    PubSubStreamSink,
    PulsarStreamSink,
    SinkCapabilities,
)
from batcher.io.formats.streaming.codecs import (
    CODECS,
    PayloadCodec,
    SchemaRegistry,
    resolve_codec,
)
from batcher.io.formats.streaming.dev import (
    RateMicroBatchSource,
    RateSource,
    SocketSource,
)
from batcher.io.formats.streaming.eventhubs import EventHubsSource
from batcher.io.formats.streaming.kafka import KafkaSource
from batcher.io.formats.streaming.kafka_sink import KafkaStreamSink
from batcher.io.formats.streaming.kinesis import KinesisSource
from batcher.io.formats.streaming.pubsub import PubSubSource
from batcher.io.formats.streaming.pulsar import PulsarSource
from batcher.io.formats.streaming.sinks import (
    STREAM_SINKS,
    DeltaStreamSink,
    FileStreamSink,
    ForeachWriter,
    NoopStreamSink,
    StreamSink,
    TransactionalStreamSink,
    memory_table,
)

__all__ = [
    "CODECS",
    "STREAM_SINKS",
    "BrokerMessage",
    "BrokerSource",
    "BrokerSplit",
    "BrokerStreamSink",
    "DeltaStreamSink",
    "EventHubsSource",
    "EventHubsStreamSink",
    "FileStreamSink",
    "ForeachWriter",
    "IncrementalFileSource",
    "KafkaSource",
    "KafkaStreamSink",
    "KinesisSource",
    "KinesisStreamSink",
    "NoopStreamSink",
    "PayloadCodec",
    "PubSubSource",
    "PubSubStreamSink",
    "PulsarSource",
    "PulsarStreamSink",
    "RateMicroBatchSource",
    "RateSource",
    "SchemaRegistry",
    "SinkCapabilities",
    "SocketSource",
    "StreamSink",
    "TransactionalStreamSink",
    "broker_schema",
    "memory_table",
    "resolve_codec",
]
