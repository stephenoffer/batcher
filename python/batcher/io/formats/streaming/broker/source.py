"""`BrokerSource` — the abstract unbounded message source and its poll loop.

Owns the schema, the batch assembly, the poll/publish/commit cycle, the checkpoint
position bookkeeping, and split generation. Concrete brokers subclass it and supply two
primitives: discover the partitions and poll a batch of messages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from batcher.io.formats.streaming.broker.schema import (
    BrokerMessage,
    broker_schema,
    redact_broker_options,
)
from batcher.io.formats.streaming.codecs.base import build_payload_codecs
from batcher.io.splits import Split

__all__ = ["BrokerSource"]


def _options_fingerprint(options: Any) -> str:
    """The connection's contribution to an identity — redacted, then fingerprinted."""
    from batcher.io.formats.sql._common import connection_fingerprint

    return connection_fingerprint(redact_broker_options(options))


@dataclass(slots=True)
class _PollBudget:
    """One poll's remaining payload-byte allowance, and where to start spending it.

    Mutable on purpose: it is threaded through a partition sweep and each partition spends
    from the one object, which is what makes "stop when the batch is full" a single check
    rather than a running total every loop has to maintain identically.
    """

    remaining: int
    rotation: int

    def spend(self, nbytes: int) -> None:
        """Charge `nbytes` of payload against this poll."""
        self.remaining -= nbytes

    @property
    def spent(self) -> bool:
        """Whether this poll has taken all the payload bytes it may.

        Returns:
            True once the allowance is exhausted, so the sweep should stop early and leave
            the rest for the next epoch.
        """
        return self.remaining <= 0

    def order(self, items: list[Any]) -> list[Any]:
        """`items` rotated so a different one leads each poll.

        Args:
            items: The partitions (or shards) to sweep, in their natural order.

        Returns:
            The same items, rotated by this poll's turn.
        """
        if len(items) < 2:
            return items
        start = self.rotation % len(items)
        return items[start:] + items[:start]


class BrokerSource(ABC):
    """Base for an unbounded, message-based streaming source.

    Subclasses set ``format_name`` and implement ``_discover_partitions`` and
    ``_poll``. The base owns the fixed schema, batch assembly, the (infinite)
    poll loop, and per-partition split generation.
    """

    format_name: str = "broker"
    bounded = False  # an infinite poll loop — collect() must not materialize it
    #: Partitions are independent, offset-addressable work units, so a micro-batch can be
    #: read partition-per-worker across the cluster (see `BrokerSplit.read_epoch`).
    partitionable = True
    #: Per-partition offsets carry forward, so a fresh poll loop resumes where the last one
    #: stopped rather than replaying the topic. See `io.source.continues_across_passes`.
    continues_across_passes = True
    #: Every polled batch carries which partition each message came from, so an event-time
    #: watermark over this source can be the *minimum* over per-partition maxima rather than
    #: one global maximum. Without the attribution the fastest partition sets the watermark
    #: for the whole topic and the slower ones' rows are dropped as late — the exact reason
    #: these two columns are named here instead of being an internal detail of the schema.
    #: `topic` rides along because a subscription may span topics whose partition ids
    #: collide, and two different partitions sharing one key would take a minimum over the
    #: wrong set.
    watermark_partition_columns = ("topic", "partition")

    #: Payload bytes one poll may accumulate before it stops early, whatever `poll_size` says.
    #:
    #: `poll_size` bounds a batch by *count*, and a count says nothing about memory when a
    #: message can be a megabyte. Kafka's own `message.max.bytes` defaults to about 1 MiB and
    #: is routinely raised, so the default 16,384-message poll is a 16 GiB micro-batch on such
    #: a topic — held in the poll, again as an Arrow batch, and again by every operator that
    #: touches it. It also walks into a hard Arrow limit well before that: a `binary` column
    #: has 32-bit offsets, so the batch fails to build at all past 2 GiB, as an opaque
    #: overflow from inside the array builder rather than as anything naming the poll.
    #:
    #: This is the same bound the media sources already put on blob batching, for the same
    #: reason. A drain that reaches it stops early and the rest of the queue is the next
    #: epoch's, which costs nothing: the messages are already buffered client-side.
    DEFAULT_POLL_BYTES = 128 << 20

    __slots__ = (
        "_admission_limit",
        "_codec_config",
        "_configured_poll_size",
        "_include_headers",
        "_key_codec",
        "_options",
        "_poll_rotation",
        "_positions",
        "_resume_from",
        "_schema",
        "_should_stop",
        "_value_codec",
        "poll_bytes",
        "topic",
    )

    def __init__(
        self,
        topic: str,
        *,
        poll_size: int = 16_384,
        poll_bytes: int | None = None,
        max_offsets_per_trigger: int | None = None,
        max_bytes_per_trigger: int | None = None,
        include_headers: bool = False,
        value_format: Any = None,
        value_schema: Any = None,
        value_subject: str | None = None,
        value_decode_mode: str = "fail",
        key_format: Any = None,
        key_schema: Any = None,
        key_subject: str | None = None,
        key_decode_mode: str = "fail",
        schema_registry: Any = None,
        schema_registry_auth: str | None = None,
        **options: Any,
    ) -> None:
        """Create a broker source for ``topic`` polling ``poll_size`` per batch.

        ``poll_bytes`` additionally bounds a poll by payload size (see `DEFAULT_POLL_BYTES`);
        ``options`` are passed through to the concrete client (broker addresses,
        credentials, consumer group, …); subclasses document what they accept.

        ``value_format`` / ``key_format`` name a wire format from
        `io.formats.streaming.codecs` (``"avro"``, ``"json"``, ``"protobuf"``, ``"string"``,
        ``"bytes"``), so the payload arrives as a typed column instead of raw bytes and the
        source's declared schema says so. ``value_schema`` supplies the reader schema, or
        ``schema_registry`` (a URL or a `SchemaRegistry`) resolves it from the subject —
        ``"{topic}-value"`` / ``"{topic}-key"`` by the standard naming, overridable with
        ``value_subject`` / ``key_subject``. ``*_decode_mode`` is ``"fail"`` (the default) or
        ``"permissive"``, matching Spark's ``FAILFAST`` / ``PERMISSIVE``.

        ``max_offsets_per_trigger`` and ``max_bytes_per_trigger`` are the Spark spellings of
        the same two bounds, accepted so a ported job's options carry over verbatim. One
        poll *is* one micro-batch here, so they are exact synonyms rather than an
        approximation — but a reader who knows the Spark names should not have to discover
        that, and a job that passed `maxOffsetsPerTrigger` as an unknown option used to have
        it forwarded silently into the client config, where librdkafka rejected it or, worse,
        ignored it.
        """
        if max_offsets_per_trigger is not None:
            poll_size = max_offsets_per_trigger
        if max_bytes_per_trigger is not None:
            poll_bytes = max_bytes_per_trigger
        if poll_size < 1:
            from batcher._internal.errors import PlanError

            raise PlanError(
                f"broker poll_size / max_offsets_per_trigger must be >= 1, got {poll_size}"
            )
        if poll_bytes is not None and poll_bytes < 1:
            from batcher._internal.errors import PlanError

            raise PlanError(
                f"broker poll_bytes / max_bytes_per_trigger must be >= 1, got {poll_bytes}"
            )
        # Spark's `includeHeaders`, off by default and opt-in for the same reason: headers
        # are per-message metadata most pipelines never read, and a nested Arrow column
        # costs on every message of every poll. A broker that cannot supply them simply
        # leaves the column null rather than refusing the option.
        self._include_headers = include_headers
        self.topic = topic
        # Kept verbatim (not as built codec objects) because a `BrokerSplit` is pickled to a
        # worker and a live `SchemaRegistry` holds a lock, while a parsed Avro schema holds
        # nothing a worker could reuse anyway. The worker rebuilds from this.
        self._codec_config: dict[str, Any] = {
            "value_format": value_format,
            "value_schema": value_schema,
            "value_subject": value_subject,
            "value_decode_mode": value_decode_mode,
            "key_format": key_format,
            "key_schema": key_schema,
            "key_subject": key_subject,
            "key_decode_mode": key_decode_mode,
            "schema_registry": schema_registry,
            "schema_registry_auth": schema_registry_auth,
        }
        self._value_codec, self._key_codec = build_payload_codecs(topic, self._codec_config)
        self._schema: pa.Schema | None = None
        # What the operator asked for. The live `poll_size` below holds it under whatever the
        # rate controller has narrowed this trigger to — see `set_admission_limit`.
        self._configured_poll_size = poll_size
        self._admission_limit: int | None = None
        self.poll_bytes = self.DEFAULT_POLL_BYTES if poll_bytes is None else poll_bytes
        self._options = options
        # The latest position delivered per partition this run (offset or native
        # `resume_token`). A streaming checkpoint write-aheads this via
        # `snapshot_position`, so recovery resumes strictly after it.
        self._positions: dict[int, Any] = {}
        # Per-partition position to resume strictly after, set by `seek` on recovery
        # and applied to the live client by `_apply_seek`.
        self._resume_from: dict[int, Any] = {}
        # Set by a driver that can be stopped; consulted between polls. See `set_stop_signal`.
        self._should_stop: Any = None
        # Advances once per poll so a multi-partition broker starts its sweep somewhere new.
        # See `_poll_budget`.
        self._poll_rotation = -1

    @property
    def poll_size(self) -> int:
        """Messages one poll may take — the configured size, held under any live rate limit.

        A property rather than a plain attribute so every broker subclass is throttled by the
        one they already read. Kafka, Kinesis, Pulsar, Pub/Sub and Event Hubs each bound their
        own poll by `poll_size`, so narrowing it here reaches all five without a line in any
        of them.

        Returns:
            The effective per-poll message count, never below 1.
        """
        if self._admission_limit is None:
            return self._configured_poll_size
        return max(1, min(self._configured_poll_size, self._admission_limit))

    def set_admission_limit(self, max_rows: int | None) -> None:
        """Narrow this trigger's poll to `max_rows` messages, or `None` to lift the narrowing.

        The `io.source.RateLimited` seam a streaming rate controller acts through. One poll is
        one micro-batch for a broker source, so a row cap and a poll size are the same bound —
        which is why `max_offsets_per_trigger` is accepted as an exact synonym for
        `poll_size` rather than as an approximation of it.

        **Only ever narrows.** The configured `poll_size` remains the ceiling, so a controller
        cannot hand a source more than its operator allowed. An admission cap changes how much
        of a stream a trigger reads, never what the query computes from the rows it read, so
        this can never change a result.

        Args:
            max_rows: The cap, or `None` to read up to the configured `poll_size` again.
        """
        self._admission_limit = None if max_rows is None or max_rows < 1 else int(max_rows)

    def set_stop_signal(self, should_stop: Any) -> None:
        """Register a predicate the poll loop checks between polls, ending the stream.

        Without it, stopping a query that is parked on an idle topic was not merely slow,
        it did not happen: `iter_batches` polls until data arrives, so the driver thread sat
        inside `next()` and the `stop()` that had already been signalled blocked on joining
        it — indefinitely on a topic that never published again. The loop now ends itself
        the way a bounded source does, which the driver already reads as "no more epochs".

        Checked *between* polls, so the worst-case stop latency is one poll timeout rather
        than unbounded. Sources that a driver never attaches to are unaffected.

        Args:
            should_stop: A zero-argument predicate that becomes true when the stream should
                end. ``None`` clears it.
        """
        self._should_stop = should_stop

    # ---- shared, do-not-override ------------------------------------------
    def schema(self) -> pa.Schema:
        """This source's declared schema, built once.

        Memoized because `_make_batch` asks for it on every poll, and a decoded source
        cannot answer from the module-level shared instance the undecoded one returns — so
        without this the codec branch reallocates a field list per micro-batch on the
        latency-critical path, which is the cost the shared schema exists to avoid. A
        codec's `arrow_type()` is fixed at construction, so there is nothing to invalidate.
        """
        if self._schema is None:
            self._schema = broker_schema(
                self._include_headers,
                value_type=None if self._value_codec is None else self._value_codec.arrow_type(),
                key_type=None if self._key_codec is None else self._key_codec.arrow_type(),
            )
        return self._schema

    def row_count(self) -> int | None:
        return None  # unbounded stream

    def identity(self) -> str:
        """A stats key that distinguishes the same topic name on different clusters.

        The connection has to be part of the key. ``kafka:events`` names a topic, not a
        relation: the same topic exists on the production cluster and on staging, and keyed
        on the name alone the two share one learned-statistics entry. Kyber then plans the
        thousand-message staging topic with the billion-message production cardinalities and
        nothing errors — the query is merely planned for the wrong data.

        The connection is folded in as a `connection_fingerprint`, which is `sha256` and not
        `hash()` on purpose: Python salts `hash()` per process, so a `hash()`-based identity
        would differ on every run and no statistic would ever be reused — a feedback loop
        that looks alive while never improving a plan. Options are redacted first, so the
        persisted key never carries a credential and survives a rotation unchanged.
        """
        return f"{self.format_name}:{self.topic}:{_options_fingerprint(self._options)}"

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        """Materialize the stream — only valid for a bounded broker.

        This used to be ``list(self.iter_batches(...))`` unconditionally, which on an
        unbounded broker is not slow but non-terminating: `iter_batches` polls forever by
        contract, so `read()` accumulates every message ever published into a list until the
        process dies of memory exhaustion. Nothing raises and nothing logs — a `collect()` on
        a Kafka topic simply never returns, which reads as a hang rather than as the misuse
        it is.

        A bounded broker (a test broker whose ``_poll`` returns ``None`` at end-of-stream)
        still materializes, so the `Source` protocol is satisfied where it can be. An
        unbounded one refuses with an actionable error naming `iter_batches`, following
        `RateSource.read`, which already declined for exactly this reason.

        Args:
            projection: Columns the scan must produce. All columns when omitted.

        Returns:
            Every batch of a bounded broker, materialized.

        Raises:
            PlanError: If the broker is unbounded, where materializing cannot terminate.
        """
        if not self.bounded:
            from batcher._internal.errors import PlanError

            raise PlanError(
                f"{self.format_name!r} is an unbounded stream: read()/collect() would never "
                "terminate. Use iter_batches(), or a streaming query with a trigger."
            )
        return list(self.iter_batches(projection))

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Poll the broker forever, yielding one batch per non-empty poll.

        Empty polls (no messages available) are skipped — the generator simply
        keeps polling. A subclass whose ``_poll`` returns ``None`` signals
        end-of-stream (a bounded test broker), which stops the loop.
        """
        try:
            yield from self._poll_loop(projection)
        finally:
            # A consumer abandons this generator constantly — `break`ing out of a
            # micro-batch loop, a trigger firing, an exception upstream. Without a `finally`
            # the client socket and its background threads live until the generator is
            # garbage-collected, which for a reference cycle is *never*. A long-running
            # driver then leaks one broker connection per query restart.
            self.close()

    def _poll_loop(self, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        """The poll/assemble/publish/commit cycle, wrapped by `iter_batches`' cleanup."""
        import time

        # Adaptive back-off between *empty* polls. A broker whose `_poll` returns immediately
        # when there is nothing to read (Kinesis `GetRecords` always returns; a Kinesis shard
        # skipped for throttling returns empty at once) would otherwise spin this loop as fast
        # as the CPU allows — burning a core and hammering a rate-limited API into the very
        # throttling it should avoid. The back-off grows to a low cap so a stop is still
        # observed promptly and the first record after idle is barely delayed; a broker whose
        # `_poll` already blocks (Kafka's consume timeout) rarely reaches this and pays nothing.
        #
        # The nap is *discounted by how long the poll itself blocked*. Without that, a broker
        # whose `_poll` already waits (Kafka blocks up to `poll_timeout` for its first record)
        # paid the back-off on top of a wait it had already served — adding up to 250ms of
        # pure latency to the first record after an idle stretch, on the exact path this
        # engine is meant to be fast on. Discounting keeps the "do not spin a core" property
        # for a fast-returning broker while charging a blocking one nothing at all.
        idle = 0.0
        should_stop = self._should_stop
        while True:
            if should_stop is not None and should_stop():
                return
            started = time.monotonic()
            messages = self._poll()
            if messages is None:
                return
            if not messages:
                idle = min(idle * 2, 0.25) if idle else 0.01
                nap = idle - (time.monotonic() - started)
                if nap > 0:
                    time.sleep(nap)
                continue
            idle = 0.0
            self._track_positions(messages)
            yield self._make_batch(messages, projection)
            # Control reaches here only when the consumer asks for the *next* batch — and the
            # consumer is the micro-batch loop, which asks only after it has staged,
            # write-ahead-logged, and **published** the epoch this batch became
            # (`core/streaming_query.py::_process_next`). So this is the first moment the
            # broker's own offsets may safely advance. Committing them any earlier — inside
            # `_poll`, at read time — means a crash between the poll and the publish leaves
            # the broker believing those messages were handled: on restart it resumes past
            # them and they are never processed. That is at-most-once, i.e. silent data loss,
            # and it is the exact opposite of what this module promises.
            self._commit_delivered()

    # ---- exactly-once checkpoint/resume (Checkpointable protocol) ----------
    def close(self) -> None:  # noqa: B027
        """Release the client this source opened. Idempotent; safe on a never-opened source.

        The base is a no-op — a broker with no client (a test broker) has nothing to release.
        A broker holding a socket, a consumer, or background threads overrides this.
        """

    def _commit_delivered(self) -> None:  # noqa: B027
        """Advance the broker's own offsets to the last *published* batch.

        Called after an epoch is published, never before. The base is a no-op: a broker whose
        only offset store is Batcher's checkpoint log has nothing to advance. A broker that
        also keeps server-side offsets (a Kafka consumer group) overrides this to commit them.
        """

    def _track_positions(self, messages: list[BrokerMessage]) -> None:
        """Record the latest resume position per partition from a poll.

        Messages arrive in delivery order, so the last message of each partition
        carries the position to resume strictly after — a native ``resume_token``
        when the client supplies one, otherwise the int64 ``offset``.
        """
        for m in messages:
            self._positions[m.partition] = m.offset if m.resume_token is None else m.resume_token

    def snapshot_position(self) -> dict:
        """The latest position delivered per partition (for checkpoint/resume).

        Returns a JSON-serializable ``{"offsets": {partition: position}}`` the
        offset log write-aheads before a micro-batch is processed, so recovery
        resumes strictly after the last *committed* batch (exactly-once for a
        replayable broker + idempotent sink).
        """
        return {"offsets": {str(p): tok for p, tok in self._positions.items()}}

    def seek(self, position: dict) -> None:
        """Resume each partition strictly after its checkpointed position.

        Restores the in-memory positions and repositions the live client per
        partition via ``_apply_seek`` (a native seek in a concrete broker).
        """
        for p_str, tok in position.get("offsets", {}).items():
            p = int(p_str)
            self._positions[p] = tok
            self._resume_from[p] = tok
            self._apply_seek(p, tok)

    def _apply_seek(self, partition: int, token: Any) -> None:  # noqa: B027
        """Reposition the live client to resume strictly after ``token``.

        The base default records the target in ``_resume_from`` for a ``_poll``
        that consults it; a broker with a native seek (Kafka ``seek``, Kinesis
        ``AFTER_SEQUENCE_NUMBER``) overrides this to drive the client directly.
        Intentionally a no-op beyond the ``_resume_from`` already set by ``seek``.
        """

    def watermark_partitions(self) -> list[tuple[str, int]]:
        """The ``(topic, partition)`` pairs this source expects to read from.

        Answers the startup question a per-partition watermark otherwise gets wrong: until
        partition 1 delivers its first message, a minimum over "partitions seen so far" is a
        minimum over partition 0, which over-claims event time and rules partition 1's first
        rows late. Declaring the assigned set holds the watermark back until every partition
        has spoken or gone idle.

        Same discovery `splits` uses, so a broker that can enumerate partitions for parallel
        reads can enumerate them for this. `io.source.watermark_partitions` treats a failure
        as "cannot enumerate", so a broker that will not answer still starts the query.

        Returns:
            One ``(topic, partition)`` pair per partition backing the topic.
        """
        return [(self.topic, p) for p in self._discover_partitions()]

    def _poll_budget(self) -> _PollBudget:
        """A fresh per-poll payload-byte budget, and the partition order to spend it in.

        `poll_bytes` was honoured by Kafka and Pulsar and **silently ignored** by Kinesis,
        Pub/Sub and Event Hubs, so the memory bound the option exists for held on two
        brokers out of five. What it guards is not a nicety: a `binary` Arrow column has
        32-bit offsets, so a poll past 2 GiB fails inside the array builder with an overflow
        that names nothing, and well before that the batch is held three times over.

        The budget also carries a **rotating start index** for a broker that polls several
        partitions in one pass. Stopping at the budget always walks the same order, so the
        last partitions are read only when the earlier ones are quiet — and under a
        per-partition watermark a partition that is never read never advances, so the
        stream's frontier is the minimum over a partition that is being starved rather than
        one that is idle. The whole query stalls until the idleness timeout fires, and
        nothing says why. Rotating the start makes every partition the first one in turn.

        Returns:
            A budget to spend across this poll.
        """
        self._poll_rotation += 1
        return _PollBudget(remaining=self.poll_bytes, rotation=self._poll_rotation)

    def _split_options(self) -> dict[str, Any]:
        """Constructor options a split must be handed back that ``_options`` does not carry.

        A concrete broker pulls its own settings out of ``**options`` into named keyword
        parameters — Kafka's ``starting_offsets`` and ``fail_on_data_loss``, Pulsar's
        ``num_partitions``, Pub/Sub's ``pull_timeout`` — precisely so they never reach the
        client config as bogus keys. The consequence is that ``self._options``, which is
        what a `BrokerSplit` carries, no longer holds them, so a worker rebuilding the
        source got the **constructor defaults** for every one.

        That is not a degraded distributed read, it is a different query. A Kafka source
        configured ``starting_offsets="latest"`` replayed the whole topic from the
        beginning on every worker; one configured ``fail_on_data_loss=False`` died on the
        aged-out offsets the user had explicitly chosen to tolerate. Nothing raised on the
        single-node path, where the same object keeps its own attributes, so the divergence
        appeared only under distribution.

        A broker that consumes a named option therefore re-declares it here.
        `tests/io/test_streaming_split_fidelity.py` holds every broker to that mechanically,
        so a new option cannot reintroduce the bug by being forgotten.

        Returns:
            Constructor keyword arguments to merge into the split's options. Empty for a
            broker that consumes nothing beyond what it forwards to this base.
        """
        return {}

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        """One :class:`BrokerSplit` per partition/shard (offset-locator only)."""
        from batcher.io.formats.streaming.broker.split import BrokerSplit

        return [
            BrokerSplit(
                format_name=self.format_name,
                topic=self.topic,
                partition=p,
                poll_size=self.poll_size,
                poll_bytes=self.poll_bytes,
                include_headers=self._include_headers,
                codecs=dict(self._codec_config),
                options={**self._options, **self._split_options()},
            )
            for p in self._discover_partitions()
        ]

    def _make_batch(
        self, messages: list[BrokerMessage], projection: list[str] | None = None
    ) -> pa.RecordBatch:
        """Assemble polled messages into one Arrow batch (column-at-a-time).

        Builds each column from the whole message list in one pass — no per-row Python
        beyond the unavoidable attribute reads — and returns a batch in the broker schema
        this source was configured for.

        `projection` is applied **while building**, not after. The difference is the whole
        cost of the poll on the two expensive columns: a `headers` column is assembled with
        a per-row Python call each, and a decoded `value` runs the wire-format codec over
        every payload. Building them and then calling `.select` — which is what the callers
        used to do — paid both in full for a query that reads only `key` and `timestamp`,
        and the codecs made that the dominant cost of the read rather than a rounding error.

        Args:
            messages: The poll's messages, in delivery order.
            projection: Columns the batch must carry, in order. All of them when omitted.

        Returns:
            One `RecordBatch` in the projected schema.
        """
        from batcher.io.formats.streaming.broker.schema import HEADERS_TYPE

        schema = self.schema()
        if projection is not None:
            schema = pa.schema([schema.field(name) for name in projection])
        wanted = set(schema.names)

        builders: dict[str, Any] = {
            "key": lambda: pa.array([m.key for m in messages], type=pa.binary()),
            "value": lambda: pa.array([m.value for m in messages], type=pa.binary()),
            "partition": lambda: pa.array([m.partition for m in messages], type=pa.int64()),
            "offset": lambda: pa.array([m.offset for m in messages], type=pa.int64()),
            "timestamp": lambda: pa.array([m.timestamp for m in messages], type=pa.int64()),
            "topic": lambda: pa.array([m.topic for m in messages], type=pa.string()),
            "headers": lambda: pa.array(
                [_header_rows(m.headers) for m in messages], type=HEADERS_TYPE
            ),
        }
        columns: dict[str, Any] = {name: builders[name]() for name in schema.names}
        # Decode here rather than downstream: one call per column per poll, on the batch the
        # source already holds, so the wire format never becomes per-row work in the plan.
        if self._value_codec is not None and "value" in wanted:
            columns["value"] = self._value_codec.decode(columns["value"])
        if self._key_codec is not None and "key" in wanted:
            columns["key"] = self._key_codec.decode(columns["key"])
        return pa.record_batch(columns, schema=schema)

    # ---- override points --------------------------------------------------
    @abstractmethod
    def _discover_partitions(self) -> list[int]:
        """Return the partition/shard ids backing the topic (for ``splits``)."""

    @abstractmethod
    def _poll(self) -> list[BrokerMessage] | None:
        """Poll up to ``poll_size`` messages.

        Returns a (possibly empty) list of messages, or ``None`` to signal
        end-of-stream for a bounded source.
        """


def _header_rows(headers: list[tuple[str, Any]] | None) -> list[dict[str, Any]] | None:
    """One message's headers as the ``[{"key": ..., "value": ...}]`` the column holds.

    None (rather than an empty list) for a message with no headers, so "this broker does
    not carry headers" and "this message had none" stay distinguishable — the same
    distinction Spark's null-vs-empty-array draws.
    """
    if not headers:
        return None
    return [{"key": str(name), "value": value} for name, value in headers]


#: Live per-partition consumers, keyed by split identity, reused across micro-batches within
#: one worker process. See `BrokerSplit._epoch_reader` for why this exists and why reuse is
#: conditional on contiguous resumption.
