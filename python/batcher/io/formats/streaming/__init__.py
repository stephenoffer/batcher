"""`io.formats.streaming` — broker + incremental-file sources, behind the registry.

Importing this package imports every streaming source module, so each registers
itself into the ``SOURCES`` registry as a side effect (``"kafka"``,
``"kinesis"``, ``"eventhubs"``, ``"pubsub"``, ``"pulsar"``,
``"files_incremental"``). Broker sources deliver raw message ``bytes`` plus
coordinates at batch granularity; the incremental file source replicates
Databricks Auto Loader (``cloudFiles``). Each broker's client dependency is an
optional extra, deferred until construction.
"""

from __future__ import annotations

from batcher.io.formats.streaming.autoloader import IncrementalFileSource
from batcher.io.formats.streaming.broker import (
    BrokerMessage,
    BrokerSource,
    BrokerSplit,
    broker_schema,
)
from batcher.io.formats.streaming.dev import RateSource, SocketSource
from batcher.io.formats.streaming.eventhubs import EventHubsSource
from batcher.io.formats.streaming.kafka import KafkaSource
from batcher.io.formats.streaming.kinesis import KinesisSource
from batcher.io.formats.streaming.pubsub import PubSubSource
from batcher.io.formats.streaming.pulsar import PulsarSource
from batcher.io.formats.streaming.sinks import (
    STREAM_SINKS,
    DeltaStreamSink,
    FileStreamSink,
    StreamSink,
    memory_table,
)

__all__ = [
    "STREAM_SINKS",
    "BrokerMessage",
    "BrokerSource",
    "BrokerSplit",
    "DeltaStreamSink",
    "EventHubsSource",
    "FileStreamSink",
    "IncrementalFileSource",
    "KafkaSource",
    "KinesisSource",
    "PubSubSource",
    "PulsarSource",
    "RateSource",
    "SocketSource",
    "StreamSink",
    "broker_schema",
    "memory_table",
]
