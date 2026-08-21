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
from batcher._internal.optional import require
from batcher.io.formats.base import SOURCES
from batcher.io.formats.streaming.broker import BrokerMessage, BrokerSource

__all__ = ["KafkaSource"]


def _import_consumer() -> Any:
    """Import ``confluent_kafka.Consumer`` or raise a guiding ``BackendError``."""
    return require(
        "confluent_kafka",
        "Consumer",
        feature="Kafka support",
        provides="confluent-kafka",
        extra="kafka",
    )


#: librdkafka's "you have reached the end of this partition" pseudo-error. It is delivered as
#: a message whose `error()` is set, and it is entirely routine: every idle partition reports
#: it. Numeric rather than `KafkaError._PARTITION_EOF` so the check needs no import on a poll.
_PARTITION_EOF = -191

#: librdkafka's "this consumer has no stored offset to commit" code (``KafkaError._NO_OFFSET``).
_NO_OFFSET = -168

#: The two errors that mean "the offset you asked for is gone from the log" — the broker's
#: retention deleted it, or the partition was truncated. This is the *data loss* condition
#: Spark's ``failOnDataLoss`` is about, and the only one it is about: every other error
#: stays fatal whatever that option says.
#: ``_AUTO_OFFSET_RESET`` (-140) is librdkafka's local report; ``OFFSET_OUT_OF_RANGE`` (1)
#: is the broker's.
_DATA_LOSS_CODES = (-140, 1)

#: What ``starting_offsets`` accepts as a whole-topic position, and the
#: ``auto.offset.reset`` value each maps to.
_STARTING_POSITIONS = {"earliest": "earliest", "latest": "latest"}


def _is_data_loss(err: Any) -> bool:
    """Whether a record error means the requested offset is no longer in the log."""
    try:
        return err.code() in _DATA_LOSS_CODES
    except Exception:  # pragma: no cover - a client without `code()`
        return False


def _parse_starting_offsets(value: Any, topic: str) -> tuple[str, dict[int, int]]:
    """Split ``starting_offsets`` into an ``auto.offset.reset`` and per-partition seeks.

    Spark's ``startingOffsets`` is one option carrying two different things: a whole-topic
    position (``"earliest"`` / ``"latest"``) or an explicit per-partition map. Both are
    accepted here, because a job ported from Spark passes whichever it already used and a
    resume-from-a-recorded-position workflow needs the map.

    The map may be Spark's nested ``{"topic": {"0": 123}}`` or the flat ``{0: 123}`` that
    is more natural when there is only one topic. A partition offset of ``-2`` means
    earliest and ``-1`` latest, as in Spark.

    Args:
        value: The option as given.
        topic: The topic being read, used to unwrap the nested form.

    Returns:
        ``(auto_offset_reset, {partition: offset})``.

    Raises:
        PlanError: If the option is neither a recognized position nor a mapping.
    """
    from batcher._internal.errors import PlanError, suggestion

    if value is None:
        return "earliest", {}
    if isinstance(value, str):
        if value in _STARTING_POSITIONS:
            return _STARTING_POSITIONS[value], {}
        try:
            import json

            value = json.loads(value)
        except ValueError:
            hint = suggestion(value, tuple(_STARTING_POSITIONS))
            raise PlanError(
                f"unknown starting_offsets {value!r}; use 'earliest', 'latest', or a "
                "{partition: offset} mapping." + (f" {hint}" if hint else "")
            ) from None
    if not isinstance(value, dict):
        raise PlanError(
            f"starting_offsets must be 'earliest', 'latest', or a mapping, not "
            f"{type(value).__name__}"
        )
    inner = value.get(topic, value)
    if not isinstance(inner, dict):
        raise PlanError(f"starting_offsets for topic {topic!r} must be a mapping of partition")
    seeks = {int(p): int(o) for p, o in inner.items()}
    # Spark's sentinels, so a map copied out of a Spark job keeps meaning what it meant.
    reset = "latest" if any(o == -1 for o in seeks.values()) else "earliest"
    return reset, {p: o for p, o in seeks.items() if o >= 0}


def _parse_ending_offsets(value: Any, topic: str) -> tuple[bool, dict[int, int]] | None:
    """Split ``ending_offsets`` into "stop at the head" and explicit per-partition ends.

    Spark's ``endingOffsets`` is what turns a topic into a *bounded* relation: a batch read
    of an offset range, which is how a backfill or a reprocess is expressed. Without it a
    Kafka source can only be consumed by a streaming query, and `collect()` on one correctly
    refuses because it could never terminate.

    The accepted forms mirror `_parse_starting_offsets`, so the two options are written the
    same way: ``"latest"`` for the head of each partition as of query start, or a
    ``{partition: offset}`` map (Spark's nested ``{"topic": {...}}`` form included). The end
    is **exclusive**, as in Spark: an end of 100 reads offsets up to and including 99.

    Args:
        value: The option as given.
        topic: The topic being read, used to unwrap the nested form.

    Returns:
        None when unbounded, else ``(stop_at_head, {partition: end_offset})``. The flag and
        the map are not exclusive: a map may name some partitions and leave the rest to the
        head, which is what a partial backfill needs.

    Raises:
        PlanError: If the option is neither a recognized position nor a mapping.
    """
    from batcher._internal.errors import PlanError, suggestion

    if value is None:
        return None
    if isinstance(value, str):
        if value == "latest":
            return True, {}
        try:
            import json

            value = json.loads(value)
        except ValueError:
            hint = suggestion(value, ("latest",))
            raise PlanError(
                f"unknown ending_offsets {value!r}; use 'latest' or a {{partition: offset}} "
                "mapping. 'earliest' is not an ending position." + (f" {hint}" if hint else "")
            ) from None
    if not isinstance(value, dict):
        raise PlanError(f"ending_offsets must be 'latest' or a mapping, not {type(value).__name__}")
    inner = value.get(topic, value)
    if not isinstance(inner, dict):
        raise PlanError(f"ending_offsets for topic {topic!r} must be a mapping of partition")
    ends = {int(p): int(o) for p, o in inner.items()}
    # Spark's `-1` sentinel means "the latest offset" on the ending side too.
    stop_at_head = any(offset == -1 for offset in ends.values())
    return stop_at_head, {p: o for p, o in ends.items() if o >= 0}


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


def _payload_bytes(records: list[Any]) -> int:
    """Total payload size of `records`, without materializing any of it.

    ``Message.len()`` is the size librdkafka already knows; `value()` would copy every
    payload into a Python `bytes` a second time — once here to measure it and once in
    `_decode` to keep it — which on the megabyte-message topics this bound exists for is
    the whole batch, twice. A client without `len()` falls back to the copy rather than
    losing the bound.
    """
    total = 0
    for rec in records:
        size = getattr(rec, "len", None)
        if size is not None:
            total += size() or 0
        else:  # pragma: no cover - a client whose Message has no `len()`
            total += len(rec.value() or b"")
    return total


@SOURCES.register("kafka")
class KafkaSource(BrokerSource):
    """An unbounded Kafka topic, consumed via ``confluent-kafka``.

    Options (``**options``) map to ``confluent-kafka`` consumer config, with two
    conveniences: ``bootstrap_servers`` (→ ``bootstrap.servers``) and ``group``
    (→ ``group.id``). ``partitions`` restricts the source to specific
    topic-partitions (set by :class:`BrokerSplit` on a worker); omit it to consume
    all partitions of the topic. ``poll_bytes`` bounds a poll by payload size as
    ``poll_size`` bounds it by count — raise it for a topic of small messages, lower it for
    one whose ``message.max.bytes`` is large.
    """

    format_name = "kafka"

    __slots__ = (
        "_consumer",
        "_drained",
        "_end_at",
        "_end_spec",
        "_fail_on_data_loss",
        "_finished",
        "_metadata_timeout",
        "_offset_reset",
        "_partitions",
        "_poll_timeout",
        "_start_at",
    )

    def __init__(
        self,
        topic: str,
        *,
        poll_size: int = 16_384,
        poll_bytes: int | None = None,
        partitions: list[int] | None = None,
        bootstrap_servers: str = "localhost:9092",
        group: str = "batcher",
        poll_timeout: float = 1.0,
        metadata_timeout: float = 10.0,
        starting_offsets: Any = None,
        ending_offsets: Any = None,
        fail_on_data_loss: bool = True,
        **options: Any,
    ) -> None:
        super().__init__(
            topic,
            poll_size=poll_size,
            poll_bytes=poll_bytes,
            bootstrap_servers=bootstrap_servers,
            group=group,
            **options,
        )
        self._partitions = partitions
        self._consumer: Any = None
        # Spark's `startingOffsets` and `failOnDataLoss`, kept out of `options` so they
        # never reach the confluent-kafka config as bogus keys. See `_parse_starting_offsets`
        # and `_is_data_loss`.
        self._offset_reset, self._start_at = _parse_starting_offsets(starting_offsets, topic)
        # `ending_offsets` makes the topic a *bounded* relation — an offset range rather than
        # a live stream — which is what a backfill or a reprocess is. Parsed here so a typo
        # is reported at construction; the per-partition end offsets are resolved against
        # the cluster's watermarks on the first poll, which is the earliest moment "latest"
        # has a value.
        self._end_spec = _parse_ending_offsets(ending_offsets, topic)
        self._end_at: dict[int, int] | None = None
        self._finished: set[int] = set()
        self._drained = False
        self._fail_on_data_loss = fail_on_data_loss
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

    @property
    def bounded(self) -> bool:
        """Whether this read terminates, so `collect()` is allowed to materialize it.

        False for a live topic and True for an offset range (``ending_offsets=``). It has to
        be answerable *before* the consumer is built, because it is what decides at plan
        time whether a terminal is a `collect` or a streaming query — so it is derived from
        the parsed option rather than from any cluster state.

        Examples:
            .. doctest::

                >>> from batcher.io.formats.streaming.kafka import KafkaSource
                >>> KafkaSource("t", partitions=[0]).bounded
                False

                >>> KafkaSource("t", partitions=[0], ending_offsets={0: 100}).bounded
                True

        Returns:
            True when the source stops at a declared end offset.
        """
        return self._end_spec is not None

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
            # `starting_offsets` is the Spark spelling and wins; `auto_offset_reset` stays
            # for a caller reaching for the librdkafka name directly.
            "auto.offset.reset": opts.pop("auto_offset_reset", self._offset_reset),
            **{k.replace("_", "."): v for k, v in opts.items()},
        }
        self._consumer = consumer_cls(config)
        if self._partitions is not None:
            from confluent_kafka import TopicPartition

            self._consumer.assign([TopicPartition(self.topic, p) for p in self._partitions])
        elif self._end_spec is not None:
            # A bounded range read assigns every partition instead of joining the group.
            # End-detection needs the full assigned set up front, and a group subscription
            # cannot supply it: the set is decided by a rebalance that may hand this
            # consumer a subset, so the read would stop at the end of *its* partitions and
            # silently omit the rest of the range. Spark's batch Kafka source assigns for
            # the same reason. `self._consumer` is already set, so the discovery below
            # reuses this client rather than recursing into `_client`.
            self._assign_all()
        else:
            self._consumer.subscribe([self.topic], on_assign=self._on_assign)
        return self._consumer

    def _assign_all(self) -> None:
        """Assign every partition of the topic, seeking each to its configured start."""
        from confluent_kafka import TopicPartition

        assigned = []
        for partition in self._discover_partitions():
            token = self._resume_from.get(partition)
            if token is not None:
                offset = int(token) + 1
            elif partition in self._start_at:
                offset = self._start_at[partition]
            else:
                from confluent_kafka import OFFSET_BEGINNING, OFFSET_END

                offset = OFFSET_END if self._offset_reset == "latest" else OFFSET_BEGINNING
            assigned.append(TopicPartition(self.topic, partition, offset))
        self._consumer.assign(assigned)

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
            elif tp.partition in self._start_at:
                # A first run with an explicit `starting_offsets` map. Only when there is no
                # checkpointed position: the checkpoint is the source of truth once a query
                # has run, or a restart would rewind to the configured start every time.
                tp.offset = self._start_at[tp.partition]
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

    def _split_options(self) -> dict[str, Any]:
        """The four settings this source consumes by name, so a worker rebuilds them.

        Without these a distributed read silently ran a *different* query than the
        single-node one: ``starting_offsets="latest"`` reverted to ``earliest`` and
        replayed the whole topic on every worker, and ``fail_on_data_loss=False`` reverted
        to True and stopped the query on exactly the condition the user chose to tolerate.

        ``starting_offsets`` is re-derived rather than stored raw, so the Spark sentinels
        and the nested ``{"topic": {...}}`` form are normalized once, here, instead of
        being re-parsed identically on every worker.

        Returns:
            The constructor keyword arguments this class consumed.
        """
        starting: Any = self._offset_reset if not self._start_at else dict(self._start_at)
        options: dict[str, Any] = {
            "starting_offsets": starting,
            "fail_on_data_loss": self._fail_on_data_loss,
            "poll_timeout": self._poll_timeout,
            "metadata_timeout": self._metadata_timeout,
        }
        if self._end_spec is not None:
            # The *resolved* ends where they are known, so every worker stops at the same
            # offsets the driver saw. Re-resolving `"latest"` per worker would end each
            # partition at whatever the head was when that worker started, making the
            # bounded read's answer depend on scheduling.
            stop_at_head, explicit = self._end_spec
            resolved = self._end_at if self._end_at is not None else explicit
            options["ending_offsets"] = dict(resolved) if resolved or not stop_at_head else "latest"
        return options

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
        if self._end_spec is None:
            return self._decode(self._consume_eagerly())
        if self._drained:
            return None  # every partition reached its end offset on an earlier poll
        return self._trim_to_range(self._decode(self._consume_eagerly()))

    def _resolve_ends(self) -> dict[int, int]:
        """The exclusive end offset per assigned partition, resolved once against the cluster.

        ``"latest"`` has no value until a client can ask, so it is resolved here rather than
        at construction: the high watermark as of the *first poll* is the range's end, which
        is what makes a backfill reproducible — a partition that keeps growing during the
        read does not extend it.

        A partition whose end is at or below its low watermark has nothing in range and is
        marked finished immediately. Without that, a range entirely behind the retention
        cutoff would wait forever for a message that can never arrive, which presents as a
        hung `collect()`.

        Returns:
            ``{partition: exclusive_end_offset}`` for every partition being read.
        """
        from confluent_kafka import TopicPartition

        stop_at_head, explicit = self._end_spec  # type: ignore[misc]
        consumer = self._client()
        ends: dict[int, int] = {}
        for partition in self._discover_partitions():
            if partition not in explicit and not stop_at_head:
                # Named neither explicitly nor by `"latest"`: this partition is outside the
                # requested range, so it contributes nothing and must not hold the read open.
                self._finished.add(partition)
                continue
            # One watermark call per partition, not two: it is a broker round trip, and a
            # thousand-partition topic pays it once per partition at query start.
            low, high = consumer.get_watermark_offsets(
                TopicPartition(self.topic, partition), timeout=self._metadata_timeout
            )
            end = int(explicit.get(partition, high))
            ends[partition] = end
            if end <= low:
                self._finished.add(partition)
        return ends

    def _trim_to_range(self, messages: list[BrokerMessage]) -> list[BrokerMessage]:
        """Drop messages at or past their partition's end, and retire finished partitions.

        Trimming rather than trusting the poll to stop: nothing tells librdkafka about the
        range, so a poll routinely returns messages past the end, and keeping them would
        make the bounded read return more rows than the range it was asked for.

        A retired partition is `pause`d so the client stops fetching it, which matters on a
        wide topic where one long partition otherwise keeps every other one's fetch traffic
        flowing for the rest of the read.
        """
        if self._end_at is None:
            self._end_at = self._resolve_ends()
        kept: list[BrokerMessage] = []
        retired: list[int] = []
        for message in messages:
            end = self._end_at.get(message.partition)
            if end is None or message.partition in self._finished:
                continue  # outside the range, or a partition already retired
            if message.offset < end:
                kept.append(message)
            if message.offset >= end - 1:
                self._finished.add(message.partition)
                retired.append(message.partition)
        if retired:
            self._pause(retired)
        if self._finished.issuperset(self._end_at):
            # Report the last rows now and end on the next poll: returning None here would
            # discard the very messages that completed the range.
            self._drained = True
        return kept

    def _pause(self, partitions: list[int]) -> None:
        """Stop fetching partitions that have reached their end. Best effort.

        A client that cannot pause (an older librdkafka, a test double) simply keeps
        fetching; the trim above still bounds what is returned, so this can cost throughput
        and never correctness.
        """
        from contextlib import suppress

        with suppress(Exception):
            from confluent_kafka import TopicPartition

            self._client().pause([TopicPartition(self.topic, p) for p in partitions])

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

        The drain is bounded by `poll_bytes` as well as by ``poll_size``, because a count
        says nothing about memory when a message can be a megabyte — see the attribute's
        own note for what a 16,384-message poll costs on such a topic.

        Returns:
            Up to ``poll_size`` raw client records, oldest first, and no more than roughly
            ``poll_bytes`` of payload.
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
        budget = self.poll_bytes - _payload_bytes(records)
        while remaining > 0 and budget > 0:
            more = consumer.consume(num_messages=remaining, timeout=0)
            if not more:
                break
            records.extend(more)
            remaining -= len(more)
            budget -= _payload_bytes(more)
        return records

    def _decode(self, records: list[Any]) -> list[BrokerMessage]:
        """Turn client records into broker messages, raising on a real record error."""
        messages: list[BrokerMessage] = []
        for rec in records:
            err = rec.error()
            if err is not None:
                if _is_benign_record_error(err):
                    continue
                if _is_data_loss(err) and not self._fail_on_data_loss:
                    # `fail_on_data_loss=False` is an explicit "I would rather keep running
                    # than stop on a gap". It must still be *loud*: rows were skipped, and a
                    # stream that logs nothing here is indistinguishable from one that lost
                    # nothing at all.
                    from batcher._internal.logging import get_logger

                    get_logger("io").warning(
                        "Kafka topic %r: requested offsets are no longer available (%s). "
                        "fail_on_data_loss=False, so the consumer resets to %r and rows "
                        "between the two positions are skipped.",
                        self.topic,
                        err,
                        self._offset_reset,
                    )
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
                    headers=rec.headers() if self._include_headers else None,
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
