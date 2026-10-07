"""Message-broker streaming sinks on one capability contract (`contract.BrokerStreamSink`).

Importing this package registers the Pulsar, Kinesis, Pub/Sub and Event Hubs sinks into
``STREAM_SINKS``; the Kafka sink, which predates the package, lives beside the Kafka source
and implements the same contract.
"""

from __future__ import annotations

from batcher.io.formats.streaming.broker_sinks.contract import (
    DEDUP_ID_HEADER,
    BrokerRecords,
    BrokerStreamSink,
    SinkCapabilities,
)
from batcher.io.formats.streaming.broker_sinks.eventhubs import EventHubsStreamSink
from batcher.io.formats.streaming.broker_sinks.kinesis import KinesisStreamSink
from batcher.io.formats.streaming.broker_sinks.pubsub import PubSubStreamSink
from batcher.io.formats.streaming.broker_sinks.pulsar import PulsarStreamSink

__all__ = [
    "DEDUP_ID_HEADER",
    "BrokerRecords",
    "BrokerStreamSink",
    "EventHubsStreamSink",
    "KinesisStreamSink",
    "PubSubStreamSink",
    "PulsarStreamSink",
    "SinkCapabilities",
]
