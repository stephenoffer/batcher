"""Kafka broker source — one Split per topic-partition, exactly-once commits.

Backed by ``confluent-kafka`` (the optional ``kafka`` extra). A
:class:`KafkaSource` is an unbounded :class:`BrokerSource`: it polls a batch of
messages with ``Consumer.consume(num_messages=N)`` and assembles them into one
Arrow batch via the shared ``_make_batch`` helper.

Exactly-once delivery is achieved through the consumer group: offsets are
committed (synchronously) only *after* a batch is assembled, so a crash before
the commit re-delivers the batch on restart, and a crash after never re-delivers
it. ``splits()`` returns one split per topic-partition (each carrying its
partition id as the offset locator), so a distributed reader assigns one consumer
per partition.

The ``confluent-kafka`` import is deferred to construction; if the extra is
missing a :class:`BackendError` instructs the user to install it.
"""

from __future__ import annotations

from typing import Any

from batcher._internal.errors import BackendError
from batcher.io.formats.base import SOURCES
from batcher.io.formats.streaming.broker import BrokerMessage, BrokerSource

__all__ = ["KafkaSource"]


def _import_consumer() -> Any:
    """Import ``confluent_kafka.Consumer`` or raise a guiding ``BackendError``."""
    try:
        from confluent_kafka import Consumer
    except ImportError as exc:
        raise BackendError(
            "reading from Kafka needs the kafka extra: pip install 'batcher[kafka]'"
        ) from exc
    return Consumer


@SOURCES.register("kafka")
class KafkaSource(BrokerSource):
    """An unbounded Kafka topic, consumed via ``confluent-kafka``.

    Options (``**options``) map to ``confluent-kafka`` consumer config, with two
    conveniences: ``bootstrap_servers`` (→ ``bootstrap.servers``) and ``group``
    (→ ``group.id``). ``partitions`` restricts the source to specific
    topic-partitions (set by :class:`BrokerSplit` on a worker); omit it to consume
    all partitions of the topic.
    """

    format_name = "kafka"

    __slots__ = ("_consumer", "_partitions")

    def __init__(
        self,
        topic: str,
        *,
        poll_size: int = 16_384,
        partitions: list[int] | None = None,
        bootstrap_servers: str = "localhost:9092",
        group: str = "batcher",
        **options: Any,
    ) -> None:
        super().__init__(
            topic,
            poll_size=poll_size,
            bootstrap_servers=bootstrap_servers,
            group=group,
            **options,
        )
        self._partitions = partitions
        self._consumer: Any = None

    def _client(self) -> Any:
        """Lazily construct and subscribe the underlying consumer."""
        if self._consumer is not None:
            return self._consumer
        consumer_cls = _import_consumer()
        opts = dict(self._options)
        config = {
            "bootstrap.servers": opts.pop("bootstrap_servers"),
            "group.id": opts.pop("group"),
            "enable.auto.commit": False,  # we commit per batch for exactly-once.
            "auto.offset.reset": opts.pop("auto_offset_reset", "earliest"),
            **{k.replace("_", "."): v for k, v in opts.items()},
        }
        self._consumer = consumer_cls(config)
        if self._partitions is None:
            self._consumer.subscribe([self.topic])
        else:
            from confluent_kafka import TopicPartition

            self._consumer.assign([TopicPartition(self.topic, p) for p in self._partitions])
        return self._consumer

    def _discover_partitions(self) -> list[int]:
        if self._partitions is not None:
            return list(self._partitions)
        consumer = self._client()
        meta = consumer.list_topics(self.topic)
        topic_meta = meta.topics[self.topic]
        return sorted(topic_meta.partitions.keys())

    def _poll(self) -> list[BrokerMessage] | None:
        consumer = self._client()
        records = consumer.consume(num_messages=self.poll_size, timeout=1.0)
        messages = [
            BrokerMessage(
                value=rec.value() or b"",
                partition=rec.partition(),
                offset=rec.offset(),
                timestamp=rec.timestamp()[1],
                topic=rec.topic(),
                key=rec.key(),
            )
            for rec in records
            if rec.error() is None
        ]
        if messages:
            # Commit only after the batch is consumed → exactly-once on restart.
            consumer.commit(asynchronous=False)
        return messages
