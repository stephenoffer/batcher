"""`BrokerSource` — the abstract unbounded message source and its poll loop.

Owns the schema, the batch assembly, the poll/publish/commit cycle, the checkpoint
position bookkeeping, and split generation. Concrete brokers subclass it and supply two
primitives: discover the partitions and poll a batch of messages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from batcher.io.formats.streaming.broker.schema import (
    BrokerMessage,
    broker_schema,
    redact_broker_options,
)
from batcher.io.splits import Split

__all__ = ["BrokerSource"]


def _options_fingerprint(options: Any) -> str:
    """The connection's contribution to an identity — redacted, then fingerprinted."""
    from batcher.io.formats.sql._common import connection_fingerprint

    return connection_fingerprint(redact_broker_options(options))


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

    __slots__ = ("_options", "_positions", "_resume_from", "_should_stop", "poll_size", "topic")

    def __init__(self, topic: str, *, poll_size: int = 16_384, **options: Any) -> None:
        """Create a broker source for ``topic`` polling ``poll_size`` per batch.

        ``options`` are passed through to the concrete client (broker addresses,
        credentials, consumer group, …); subclasses document what they accept.
        """
        self.topic = topic
        self.poll_size = poll_size
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
        return broker_schema()

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
            batch = self._make_batch(messages)
            yield batch.select(projection) if projection is not None else batch
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

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        """One :class:`BrokerSplit` per partition/shard (offset-locator only)."""
        from batcher.io.formats.streaming.broker.split import BrokerSplit

        return [
            BrokerSplit(
                format_name=self.format_name,
                topic=self.topic,
                partition=p,
                poll_size=self.poll_size,
                options=dict(self._options),
            )
            for p in self._discover_partitions()
        ]

    @staticmethod
    def _make_batch(messages: list[BrokerMessage]) -> pa.RecordBatch:
        """Assemble polled messages into one Arrow batch (column-at-a-time).

        Builds each column from the whole message list in one pass — no per-row
        Python beyond the unavoidable attribute reads — and returns a batch in
        the fixed broker schema.
        """
        return pa.record_batch(
            {
                "key": pa.array([m.key for m in messages], type=pa.binary()),
                "value": pa.array([m.value for m in messages], type=pa.binary()),
                "partition": pa.array([m.partition for m in messages], type=pa.int64()),
                "offset": pa.array([m.offset for m in messages], type=pa.int64()),
                "timestamp": pa.array([m.timestamp for m in messages], type=pa.int64()),
                "topic": pa.array([m.topic for m in messages], type=pa.string()),
            },
            schema=broker_schema(),
        )

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


#: Live per-partition consumers, keyed by split identity, reused across micro-batches within
#: one worker process. See `BrokerSplit._epoch_reader` for why this exists and why reuse is
#: conditional on contiguous resumption.
