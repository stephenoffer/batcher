"""Shared base for row/message-based streaming brokers (Kafka, Kinesis, …).

A *broker* source models an unbounded stream of raw messages. Unlike file
formats, the payload is opaque: each message is delivered as raw ``bytes`` plus
its coordinates (partition/shard, offset, timestamp, topic) — decoding and
schema-registry handling are downstream concerns expressed as ordinary
expressions over the ``value`` column.

The contract:

* the schema is **fixed** —
  ``{key: binary, value: binary, partition: int64, offset: int64,
  timestamp: int64, topic: string}``;
* ``row_count()`` is ``None`` (the stream is unbounded);
* ``iter_batches()`` is an (infinite) generator that polls ``poll_size``
  messages at a time and assembles each poll into **one** ``RecordBatch`` —
  batch-granularity assembly, never per-row Python in the hot path;
* ``splits()`` returns one picklable :class:`Split` per partition/shard so a
  distributed reader consumes partitions in parallel.

Concrete brokers subclass this and implement two primitives: discover the
partitions/shards (``_discover_partitions``) and poll a batch of messages from
one partition (``_poll``). All Arrow assembly lives here in ``_make_batch``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from batcher.io.splits import Split

__all__ = ["BrokerMessage", "BrokerSource", "BrokerSplit", "broker_schema"]


def broker_schema() -> pa.Schema:
    """The fixed broker message schema shared by every broker source."""
    return pa.schema(
        [
            pa.field("key", pa.binary()),
            pa.field("value", pa.binary()),
            pa.field("partition", pa.int64()),
            pa.field("offset", pa.int64()),
            pa.field("timestamp", pa.int64()),
            pa.field("topic", pa.string()),
        ]
    )


@dataclass(frozen=True, slots=True)
class BrokerMessage:
    """One polled message: raw bytes plus its broker coordinates.

    ``key`` may be ``None`` (an unkeyed message); all other fields are required.
    ``timestamp`` is milliseconds since the Unix epoch.

    ``resume_token`` is the *native* position a client seeks strictly after to
    replay from this message on recovery (a Kinesis sequence number, a Pulsar
    message id, …). It is checkpoint bookkeeping only — never a schema column —
    and defaults to ``None``, in which case the int64 ``offset`` is the token.
    """

    value: bytes
    partition: int
    offset: int
    timestamp: int
    topic: str
    key: bytes | None = None
    resume_token: Any = None


class BrokerSource(ABC):
    """Base for an unbounded, message-based streaming source.

    Subclasses set ``format_name`` and implement ``_discover_partitions`` and
    ``_poll``. The base owns the fixed schema, batch assembly, the (infinite)
    poll loop, and per-partition split generation.
    """

    format_name: str = "broker"
    bounded = False  # an infinite poll loop — collect() must not materialize it

    __slots__ = ("_options", "_positions", "_resume_from", "poll_size", "topic")

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

    # ---- shared, do-not-override ------------------------------------------
    def schema(self) -> pa.Schema:
        return broker_schema()

    def row_count(self) -> int | None:
        return None  # unbounded stream

    def identity(self) -> str:
        return f"{self.format_name}:{self.topic}"

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        """Materialize the stream — only safe for a bounded test broker.

        An unbounded broker never terminates here; production reads go through
        ``iter_batches``. Provided so a broker satisfies the ``Source`` protocol.
        """
        return list(self.iter_batches(projection))

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Poll the broker forever, yielding one batch per non-empty poll.

        Empty polls (no messages available) are skipped — the generator simply
        keeps polling. A subclass whose ``_poll`` returns ``None`` signals
        end-of-stream (a bounded test broker), which stops the loop.
        """
        while True:
            messages = self._poll()
            if messages is None:
                return
            if not messages:
                continue
            self._track_positions(messages)
            batch = self._make_batch(messages)
            yield batch.select(projection) if projection is not None else batch

    # ---- exactly-once checkpoint/resume (Checkpointable protocol) ----------
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


@dataclass(frozen=True, slots=True)
class BrokerSplit:
    """One partition/shard of a broker, reconstructed on the worker.

    Carries only picklable offset *locators* (format name, topic, partition id,
    poll size, client options) — never live client handles or data. ``read``
    rebuilds the concrete broker source from the format registry, scoped to this
    single partition.
    """

    format_name: str
    topic: str
    partition: int
    poll_size: int
    options: dict[str, Any] = field(default_factory=dict)

    def _reader(self) -> BrokerSource:
        from batcher.io.formats.base import SOURCES

        cls = SOURCES.get(self.format_name)
        return cls(  # type: ignore[no-any-return]
            self.topic,
            poll_size=self.poll_size,
            partitions=[self.partition],
            **self.options,
        )

    def schema(self) -> pa.Schema:
        return broker_schema()

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return self._reader().read(projection)

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        yield from self._reader().iter_batches(projection)

    def read_epoch(
        self, start_offset: Any | None = None, projection: list[str] | None = None
    ) -> tuple[list[pa.RecordBatch], Any | None]:
        """Read **one** micro-batch of this partition, resuming after `start_offset`.

        The distributed streaming path needs a partition read that *ends*: `read` and
        `iter_batches` poll an unbounded broker forever, which is right for a consumer and
        useless for an epoch. This polls once and reports the offset it stopped at, so the
        driver can write-ahead the position and hand it back for the next epoch — the same
        resume-strictly-after contract the single-node checkpoint uses, evaluated on the
        worker that owns the partition.

        Returns the batch (empty when the poll had nothing) and the new resume position.
        """
        src = self._reader()
        if start_offset is not None:
            src.seek({"offsets": {str(self.partition): start_offset}})
        messages = src._poll()
        if not messages:  # None (end of stream) or an empty poll
            return [], start_offset
        src._track_positions(messages)
        batch = src._make_batch(messages)
        if projection is not None:
            batch = batch.select(projection)
        return [batch], src.snapshot_position()["offsets"].get(str(self.partition))

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"{self.format_name}:{self.topic}:p{self.partition}"
