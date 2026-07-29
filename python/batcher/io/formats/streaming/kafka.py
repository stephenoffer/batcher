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


#: librdkafka's "you have reached the end of this partition" pseudo-error. It is delivered as
#: a message whose `error()` is set, and it is entirely routine: every idle partition reports
#: it. Numeric rather than `KafkaError._PARTITION_EOF` so the check needs no import on a poll.
_PARTITION_EOF = -191

#: librdkafka's "this consumer has no stored offset to commit" code (``KafkaError._NO_OFFSET``).
_NO_OFFSET = -168


def _is_no_offset(exc: BaseException) -> bool:
    """Whether a commit failure is the benign "nothing stored to commit" condition."""
    args = getattr(exc, "args", ())
    err = args[0] if args else None
    code = getattr(err, "code", None)
    try:
        return code is not None and code() == _NO_OFFSET
    except Exception:  # pragma: no cover - a client whose args[0] is not a KafkaError
        return False


def _is_benign_record_error(err: Any) -> bool:
    """Whether a per-message error is routine back-pressure rather than a failure.

    Every errored record used to be dropped on the floor. That is right for a partition-EOF
    marker and catastrophic for anything else: an unknown topic, a failed SASL handshake, or
    an out-of-range offset arrives as an errored record on *every* poll, so the source
    returned an empty list forever. The query stayed "running", read nothing, reported no
    failure, and the back-off in `BrokerSource._poll_loop` made it look idle rather than
    broken.

    Benign means partition EOF, or an error librdkafka itself marks retriable (a transient
    leader election, a broker restart) — those resolve on the next poll. Anything else is a
    real failure and must reach the user.

    Args:
        err: The ``KafkaError`` from ``Message.error()``.

    Returns:
        True when the record may be skipped silently, False when it must be raised.
    """
    try:
        if err.code() == _PARTITION_EOF:
            return True
    except Exception:  # pragma: no cover - a client without `code()`
        return False
    try:
        return bool(err.retriable())
    except Exception:  # pragma: no cover - older clients lack `retriable()`
        return False


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

    __slots__ = ("_consumer", "_metadata_timeout", "_partitions", "_poll_timeout")

    def __init__(
        self,
        topic: str,
        *,
        poll_size: int = 16_384,
        partitions: list[int] | None = None,
        bootstrap_servers: str = "localhost:9092",
        group: str = "batcher",
        poll_timeout: float = 1.0,
        metadata_timeout: float = 10.0,
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
        # How long the *first* record of a poll is waited for. It bounds the micro-batch
        # loop's stop latency (a `stop()` is observed only between polls); the records that
        # follow are drained without blocking, so this is no longer a floor on batch latency.
        # Kept out of `options` so it never leaks into the confluent-kafka config as a bogus
        # `poll.timeout` key.
        self._poll_timeout = poll_timeout
        # Bounds the cluster-metadata fetch in `_discover_partitions`, which otherwise blocks
        # the driver forever against an unreachable bootstrap server. Also kept out of
        # `options` for the same reason.
        self._metadata_timeout = metadata_timeout

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
        """The topic's partition ids, from cluster metadata.

        Bounded by ``metadata_timeout`` on purpose. ``list_topics`` with no timeout blocks
        forever when the bootstrap servers are wrong or unreachable, and this runs on the
        *driver* while planning a distributed read — so a typo in ``bootstrap_servers``
        presented as a hung `collect()` with no error and no traceback rather than as the
        configuration mistake it is.

        A missing topic is likewise reported rather than swallowed: metadata for an unknown
        topic comes back as an entry carrying an ``error``, and indexing it used to raise a
        bare ``KeyError`` on the topic name.
        """
        if self._partitions is not None:
            return list(self._partitions)
        consumer = self._client()
        meta = consumer.list_topics(self.topic, timeout=self._metadata_timeout)
        topic_meta = meta.topics.get(self.topic)
        if topic_meta is None:
            raise BackendError(
                f"Kafka topic {self.topic!r} was not found on the cluster "
                f"({self._options.get('bootstrap_servers')})"
            )
        if getattr(topic_meta, "error", None) is not None:
            raise BackendError(
                f"Kafka metadata for topic {self.topic!r} reports: {topic_meta.error}"
            )
        return sorted(topic_meta.partitions.keys())

    def _poll(self) -> list[BrokerMessage] | None:
        # Deliberately does not commit. Polling is not processing: the engine has not staged,
        # let alone published, this batch yet. `BrokerSource.iter_batches` calls
        # `_commit_delivered` once the epoch is published — the only correct moment.
        return self._decode(self._consume_eagerly())

    def _consume_eagerly(self) -> list[Any]:
        """Block only until the *first* record, then drain what is already buffered.

        This is the single largest source of end-to-end latency on a low-rate topic, and it
        was invisible because it looks like a correct call. ``consume(num_messages=N,
        timeout=T)`` does not return as soon as data is available: librdkafka keeps popping
        against the *remaining* timeout until it has N messages, so a topic producing ten
        records a second answered a 16,384-message request only when the full ``poll_timeout``
        expired. Every micro-batch therefore cost a fixed second of latency no matter how
        promptly the record arrived, and raising ``poll_size`` for throughput made it worse.

        Blocking for one record and then draining with a zero timeout inverts that: the wait
        ends the instant the first record lands, and the follow-up passes sweep up everything
        librdkafka has already fetched into its queue. Latency becomes time-to-first-record;
        throughput is unchanged, because a backlogged partition fills the queue and the drain
        returns the whole batch in one extra call.

        Returns:
            Up to ``poll_size`` raw client records, oldest first.
        """
        consumer = self._client()
        records = consumer.consume(num_messages=1, timeout=self._poll_timeout)
        if not records:
            return records
        # Drain the already-fetched queue without blocking. Loop rather than making a single
        # call so a queue that refills during the drain (a high-throughput partition) still
        # fills the batch, and stop the moment a pass comes back short — that means the queue
        # is empty and another pass would only add syscalls.
        remaining = self.poll_size - len(records)
        while remaining > 0:
            more = consumer.consume(num_messages=remaining, timeout=0)
            if not more:
                break
            records.extend(more)
            remaining -= len(more)
        return records

    def _decode(self, records: list[Any]) -> list[BrokerMessage]:
        """Turn client records into broker messages, raising on a real record error."""
        messages: list[BrokerMessage] = []
        for rec in records:
            err = rec.error()
            if err is not None:
                if _is_benign_record_error(err):
                    continue
                raise BackendError(f"Kafka read failed on topic {self.topic!r}: {err}")
            messages.append(
                BrokerMessage(
                    value=rec.value() or b"",
                    partition=rec.partition(),
                    offset=rec.offset(),
                    timestamp=rec.timestamp()[1],
                    topic=rec.topic(),
                    key=rec.key(),
                )
            )
        return messages

    def _commit_delivered(self) -> None:
        """Commit the group offsets of the epoch that was just published.

        Synchronous on purpose: the commit must land before the next epoch's poll, or a crash
        in between reopens the very replay window this ordering exists to close. A duplicate
        (crash *after* publish, *before* commit) re-delivers the batch and the idempotent sink
        absorbs it; a *skip* would be unrecoverable, so the ordering favours the duplicate.

        A "nothing to commit" answer is not a failure. librdkafka raises ``_NO_OFFSET`` when
        the consumer holds no stored position for its assignment — which happens whenever a
        rebalance revoked the partitions between the poll and the publish. Batcher's own
        offset log already carries the position (`_on_assign` resumes from it), so the group
        commit is advisory here; letting ``_NO_OFFSET`` escape killed a query for a condition
        the very next poll resolves.
        """
        if self._consumer is None:
            return
        try:
            self._consumer.commit(asynchronous=False)
        except Exception as exc:
            if not _is_no_offset(exc):
                raise

    def close(self) -> None:
        """Close the consumer, leaving the group cleanly and releasing its threads.

        `Consumer.close()` also triggers a final offset commit and a graceful group leave, so
        skipping it does more than leak a socket and a poll thread: the group waits out
        `session.timeout.ms` before rebalancing, stalling the partitions this consumer held.
        `BrokerSource.iter_batches` calls this from a `finally`, so it runs even when a
        consumer abandons the generator mid-stream.

        The handle is dropped in a `finally`: a `close()` that raises (a broker already gone,
        a group coordinator that timed out on the leave) used to leave `_consumer` set, so the
        next `close()` — and `iter_batches` guarantees one — re-closed a dead consumer and
        raised again, this time out of a `finally` where it masks the original error.
        """
        if self._consumer is not None:
            consumer, self._consumer = self._consumer, None
            consumer.close()
