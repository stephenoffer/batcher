"""Kafka broker source — one Split per topic-partition, exactly-once commits.

Backed by ``confluent-kafka`` (the optional ``kafka`` extra). A
:class:`KafkaSource` is an unbounded :class:`BrokerSource`: it polls a batch of
messages with ``Consumer.consume(num_messages=N)`` and assembles them into one
Arrow batch via the shared ``_make_batch`` helper.

Offsets advance only *after* an epoch is **published** — not when it is polled. The engine
write-aheads the position it consumed, publishes, and then `_commit_delivered` moves the
consumer group forward; a crash in between re-delivers the batch, which an idempotent sink
absorbs. Committing at poll time instead (as this once did) makes the broker believe messages
were handled the moment they were *read*, so a crash before the publish skips them forever —
at-most-once, and silent. The ordering is chosen so the failure mode is a duplicate, never a
gap. On restart `_on_assign` resumes each partition from Batcher's checkpoint rather than the
group's own offset, so the engine's log — not the broker's — is the source of truth.

``splits()`` returns one split per topic-partition (each carrying its partition id as the
offset locator), so a distributed reader assigns one consumer per partition; that path
write-aheads positions through the driver and never relies on group commits at all.

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
            "reading from Kafka needs the kafka extra: pip install 'batcher-engine[kafka]'"
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
            self._consumer.subscribe([self.topic], on_assign=self._on_assign)
        else:
            from confluent_kafka import TopicPartition

            self._consumer.assign([TopicPartition(self.topic, p) for p in self._partitions])
        return self._consumer

    def _on_assign(self, consumer: Any, partitions: list[Any]) -> None:
        """On a group rebalance, resume each assigned partition from *Batcher's* checkpoint.

        Subscribe mode cannot seek eagerly — the partitions this consumer owns are not known
        until the group assigns them — so the resume has to happen in the assignment callback.
        Without it, `_apply_seek` was simply a no-op here and recovery silently fell back to
        the consumer group's committed offset, ignoring the checkpoint the engine had written.
        That is what made the premature poll-time commit lose data rather than merely duplicate
        it: the group offset had already advanced past messages the engine never published, and
        nothing repositioned the consumer back to them.

        A partition with no checkpointed position keeps the offset the broker assigned
        (``auto.offset.reset``), which is the right default for a partition this consumer has
        not read before.
        """
        for tp in partitions:
            token = self._resume_from.get(tp.partition)
            if token is not None:
                tp.offset = int(token) + 1  # resume strictly *after* the last published row
        consumer.assign(partitions)

    def _apply_seek(self, partition: int, token: Any) -> None:
        """Resume a checkpointed partition strictly after its published offset.

        With partitions explicitly assigned (the distributed split path) the consumer already
        owns them, so it is repositioned immediately. In subscribe mode ownership is decided by
        a group rebalance, so the seek cannot happen yet — `seek` has recorded the position in
        `_resume_from` and `_on_assign` applies it the moment the group hands us the partition.
        """
        if self._partitions is None:
            return  # deferred to `_on_assign` — see above
        from confluent_kafka import TopicPartition

        consumer = self._client()
        consumer.seek(TopicPartition(self.topic, partition, int(token) + 1))

    def _discover_partitions(self) -> list[int]:
        if self._partitions is not None:
            return list(self._partitions)
        consumer = self._client()
        meta = consumer.list_topics(self.topic)
        topic_meta = meta.topics[self.topic]
        return sorted(topic_meta.partitions.keys())

    def _poll(self) -> list[BrokerMessage] | None:
        # Deliberately does not commit. Polling is not processing: the engine has not staged,
        # let alone published, this batch yet. `BrokerSource.iter_batches` calls
        # `_commit_delivered` once the epoch is published — the only correct moment.
        consumer = self._client()
        records = consumer.consume(num_messages=self.poll_size, timeout=1.0)
        return [
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

    def _commit_delivered(self) -> None:
        """Commit the group offsets of the epoch that was just published.

        Synchronous on purpose: the commit must land before the next epoch's poll, or a crash
        in between reopens the very replay window this ordering exists to close. A duplicate
        (crash *after* publish, *before* commit) re-delivers the batch and the idempotent sink
        absorbs it; a *skip* would be unrecoverable, so the ordering favours the duplicate.
        """
        if self._consumer is not None:
            self._consumer.commit(asynchronous=False)

    def close(self) -> None:
        """Close the consumer, leaving the group cleanly and releasing its threads.

        `Consumer.close()` also triggers a final offset commit and a graceful group leave, so
        skipping it does more than leak a socket and a poll thread: the group waits out
        `session.timeout.ms` before rebalancing, stalling the partitions this consumer held.
        `BrokerSource.iter_batches` calls this from a `finally`, so it runs even when a
        consumer abandons the generator mid-stream.
        """
        if self._consumer is not None:
            self._consumer.close()
            self._consumer = None
