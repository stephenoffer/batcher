"""Apache Pulsar broker source — one Split per partition, via ``pulsar-client``.

Backed by ``pulsar-client`` (the optional ``pulsar`` extra). A
:class:`PulsarSource` consumes a topic with a shared subscription, draining up to
``poll_size`` messages per poll (``Consumer.receive(timeout_millis=…)``) and
assembling them into one Arrow batch via the shared ``_make_batch`` helper.

Messages are acknowledged only after an epoch is **published**, never when it is polled.
`_poll` holds the raw messages and `_commit_delivered` acks them once the engine has staged,
logged, and published the batch they became. Acking at poll time instead (as this once did,
under a comment claiming "ack after the batch is assembled") makes Pulsar believe messages
were handled the moment they were *read* — a crash before the publish drops them forever,
at-most-once and silent. The ordering is chosen so the failure mode is a duplicate, which an
idempotent sink absorbs, never a gap, which nothing can recover.

``splits()`` returns one split per partition (the partition index is the offset
locator); for a non-partitioned topic this is a single split. The Pulsar
``MessageId`` is opaque, so the message's ledger/entry pair is folded into the
int64 ``offset`` column to fit the fixed broker schema.

The ``pulsar`` import is deferred to construction; if the extra is missing a
:class:`BackendError` instructs the user to install it.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any

from batcher._internal.errors import BackendError
from batcher.io.formats.base import SOURCES
from batcher.io.formats.streaming.broker import BrokerMessage, BrokerSource

__all__ = ["PulsarSource"]

#: How long a *drain* `receive` waits. Not zero: the Pulsar client reads ``0`` as "block
#: until a message arrives", which would turn the drain into a hang on an idle topic. One
#: millisecond is long enough to pick up anything already in the client's local queue and
#: short enough that finding it empty costs nothing measurable.
_DRAIN_TIMEOUT_MILLIS = 1

#: Bits of a Pulsar ``MessageId``'s entry id preserved in the int64 ``offset`` column.
#: Twenty was too few: a BookKeeper ledger routinely holds far more than a million entries,
#: and past that the entry id wrapped, so two different messages in one ledger folded to the
#: same offset — silently breaking the ordering and de-duplication the column exists for.
#: Thirty-two leaves 31 bits for the ledger id, which stays inside int64.
_ENTRY_ID_BITS = 32


def _import_pulsar() -> Any:
    """Import the ``pulsar`` client module or raise a guiding ``BackendError``."""
    try:
        import pulsar
    except ImportError as exc:
        raise BackendError(
            "reading from Pulsar needs the pulsar extra: pip install 'batcher-engine[pulsar]'"
        ) from exc
    return pulsar


@SOURCES.register("pulsar")
class PulsarSource(BrokerSource):
    """An unbounded Pulsar topic, consumed via ``pulsar-client``.

    Options: ``service_url`` (default ``"pulsar://localhost:6650"``),
    ``subscription`` (default ``"batcher"``), ``num_partitions`` (how many
    partitions the topic has — used by ``splits``; default ``1``), and
    ``partitions`` (the specific partition indices to read — set by
    :class:`BrokerSplit` on a worker).
    """

    format_name = "pulsar"

    __slots__ = (
        "_client_obj",
        "_consumer",
        "_num_partitions",
        "_partitions",
        "_receive_timeout_millis",
        "_unacked",
    )

    def __init__(
        self,
        topic: str,
        *,
        poll_size: int = 16_384,
        partitions: list[int] | None = None,
        service_url: str = "pulsar://localhost:6650",
        subscription: str = "batcher",
        num_partitions: int = 1,
        receive_timeout_millis: int = 1000,
        **options: Any,
    ) -> None:
        super().__init__(
            topic,
            poll_size=poll_size,
            service_url=service_url,
            subscription=subscription,
            **options,
        )
        self._partitions = partitions
        self._num_partitions = num_partitions
        self._client_obj: Any = None
        self._consumer: Any = None
        # How long a single `receive` blocks before a poll settles for what it has. It
        # bounds the micro-batch loop's stop latency and how long an idle topic waits before
        # yielding an empty poll; kept off `_options` so it never leaks to the Pulsar client.
        self._receive_timeout_millis = receive_timeout_millis
        # Messages polled but not yet published, held so `_commit_delivered` can ack them
        # at the only correct moment — after the epoch they became is published.
        self._unacked: list[Any] = []

    def _topic_names(self) -> list[str]:
        """The concrete topic name(s) this source consumes.

        For a partition subset, address each partition's physical topic
        (``<topic>-partition-<n>``); otherwise the base topic (the client fans
        out across partitions for a shared subscription).
        """
        if self._partitions is None:
            return [self.topic]
        return [f"{self.topic}-partition-{p}" for p in self._partitions]

    def _client(self) -> Any:
        """The live consumer, built on first use.

        The import is inside the construction branch, not above it. `_client()` runs on every
        poll and every commit, and `_import_pulsar` was re-entering the import machinery each
        time — a module-cache lookup plus a try/except on the latency-critical path, for a
        module the already-built consumer proves is present.
        """
        if self._consumer is not None:
            return self._consumer
        pulsar = _import_pulsar()
        if self._client_obj is None:
            self._client_obj = pulsar.Client(self._options["service_url"])
        self._consumer = self._client_obj.subscribe(
            self._topic_names(),
            subscription_name=self._options["subscription"],
            consumer_type=pulsar.ConsumerType.Shared,
        )
        return self._consumer

    def _discover_partitions(self) -> list[int]:
        if self._partitions is not None:
            return list(self._partitions)
        return list(range(max(1, self._num_partitions)))

    def _poll(self) -> list[BrokerMessage] | None:
        pulsar = _import_pulsar()
        consumer = self._client()
        messages: list[BrokerMessage] = []
        raw: list[Any] = []
        # Wait out `receive_timeout_millis` for the *first* message only, then drain what the
        # client has already buffered with a near-zero timeout. Charging every `receive` the
        # full timeout meant the last one — the one that finds the queue empty and ends the
        # drain — cost a full second on *every* poll, so a topic delivering ten messages a
        # trigger paid a fixed second of latency to notice it had run out. That is the same
        # shape as the Kafka `consume()` trap, and it hid in the same place: a loop that looks
        # like it is reading is in fact mostly waiting to be told there is nothing left.
        for i in range(self.poll_size):
            try:
                timeout = self._receive_timeout_millis if i == 0 else _DRAIN_TIMEOUT_MILLIS
                msg = consumer.receive(timeout_millis=timeout)
            except pulsar.Timeout:
                break
            raw.append(msg)
            mid = msg.message_id()
            messages.append(
                BrokerMessage(
                    value=msg.data(),
                    partition=msg.partition() if hasattr(msg, "partition") else 0,
                    offset=_message_id_to_offset(mid),
                    # The int64 `offset` folds (ledger, entry) lossily, so it cannot be
                    # seeked to. The serialized `MessageId` is the exact position, and it is
                    # what `_apply_seek` hands back to `Consumer.seek`.
                    resume_token=_serialize_message_id(mid),
                    timestamp=msg.publish_timestamp(),
                    topic=msg.topic_name() if hasattr(msg, "topic_name") else self.topic,
                    key=msg.partition_key().encode("utf-8") if msg.partition_key() else None,
                )
            )
        # Deliberately does not acknowledge. Assembling a batch is not publishing it: the
        # engine has not staged, logged, or published this epoch yet. Acking here made Pulsar
        # believe the messages were handled the moment they were *read*, so a crash between
        # the poll and the publish dropped them permanently — at-most-once, and silent. They
        # are held for `_commit_delivered`, which runs after the publish.
        self._unacked.extend(raw)
        return messages

    def _commit_delivered(self) -> None:
        """Acknowledge the messages of the epoch that was just published.

        Called by `BrokerSource.iter_batches` only after the consumer has asked for the next
        batch, which it does only once the current epoch is published. A crash before this
        re-delivers the batch and an idempotent sink absorbs it; the ordering favours a
        duplicate over a gap, because a gap is unrecoverable.
        """
        if not self._unacked:
            return
        consumer = self._client()
        for msg in self._unacked:
            consumer.acknowledge(msg)
        self._unacked = []

    def seek(self, position: dict) -> None:
        """Resume from a checkpoint, refusing an ambiguous multi-partition one.

        A Pulsar seek is per *consumer*, and one consumer here can span several partitions.
        The base `seek` walks the checkpoint partition by partition, so a multi-partition
        checkpoint issued one seek per entry and the last one silently won: every other
        partition resumed at a position belonging to a different partition, replaying or
        skipping records with nothing to show for it. Recovery that quietly loses data is
        worse than recovery that refuses, so this names the condition and the fix.

        Args:
            position: The checkpointed ``{"offsets": {partition: token}}`` mapping.

        Raises:
            PlanError: If the checkpoint carries distinct positions for more than one
                partition while this consumer covers them all at once.
        """
        offsets = position.get("offsets", {})
        if len(offsets) > 1 and (self._partitions is None or len(self._partitions) > 1):
            from batcher._internal.errors import PlanError

            raise PlanError(
                f"cannot resume Pulsar topic {self.topic!r}: a Pulsar seek repositions the "
                f"whole consumer, but the checkpoint carries {len(offsets)} distinct "
                "partition positions. Read the topic with one split per partition (a "
                "distributed or partitioned read) so each partition seeks its own consumer."
            )
        super().seek(position)

    def _apply_seek(self, partition: int, token: Any) -> None:  # noqa: ARG002
        """Reposition the consumer to a checkpointed `MessageId`.

        Without this the base `_apply_seek` was a no-op, so `seek` recorded a position that
        nothing ever applied: a restart resumed from wherever the *subscription* happened to
        sit, silently ignoring the checkpoint and replaying or skipping accordingly. Pulsar
        seeks are per-consumer rather than per-partition, and `seek` above guarantees this
        consumer covers exactly one checkpointed partition, so the partition id is implicit.
        """
        message_id = _deserialize_message_id(token)
        if message_id is not None:
            self._client().seek(message_id)
        # A seek invalidates everything polled but not yet acknowledged: those messages are
        # about to be delivered again from the new position, and acking the stale handles
        # would acknowledge records the engine never published.
        self._unacked = []

    def close(self) -> None:
        """Close the consumer and client, releasing their sockets and background threads.

        Each handle is dropped *before* the call that releases it. A `close()` that raises
        (a broker already gone, a client whose IO threads have died) used to leave the
        attribute set, so the next `close()` — and `iter_batches` guarantees one from its
        `finally` — re-closed a dead handle and raised again, masking the original error.
        The client is closed even when the consumer's close fails, so a half-failed shutdown
        cannot strand the client's background threads.
        """
        consumer, self._consumer = self._consumer, None
        client, self._client_obj = self._client_obj, None
        self._unacked = []
        try:
            if consumer is not None:
                consumer.close()
        finally:
            if client is not None:
                client.close()


def _message_id_to_offset(message_id: Any) -> int:
    """Fold a Pulsar ``MessageId`` (ledger, entry) into one int64 offset.

    Pulsar offsets are a ``(ledger_id, entry_id)`` pair; combine them so messages
    within a ledger remain monotonically ordered in the fixed int64 column.
    """
    try:
        ledger = int(message_id.ledger_id())
        entry = int(message_id.entry_id())
        mask = (1 << _ENTRY_ID_BITS) - 1
        return ((ledger << _ENTRY_ID_BITS) | (entry & mask)) % (1 << 63)
    except (AttributeError, ValueError):
        return _stable_offset(str(message_id))


def _stable_offset(text: str) -> int:
    """A process-stable int64 for an id that exposes no numeric position.

    `hash()` was the obvious spelling and is the wrong one: Python salts `str` hashing per
    process (PYTHONHASHSEED), so the same message got a different `offset` on every run and
    on every worker. That column is what downstream de-duplication and ordering key off, so
    the value silently stopped being comparable across exactly the boundaries — restart,
    distributed read — where comparing it is the entire point. `sha256` is the same one-line
    call and is stable everywhere, which is why the SQL connectors' `connection_fingerprint`
    made the same switch.
    """
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") % (1 << 63)


def _serialize_message_id(message_id: Any) -> str | None:
    """A Pulsar ``MessageId`` as a JSON-safe string for the checkpoint log.

    The checkpoint is written as JSON, so the client's native ``serialize()`` bytes cannot go
    in directly — they are base64-encoded here and decoded back in `_deserialize_message_id`.
    """
    try:
        return base64.b64encode(message_id.serialize()).decode("ascii")
    except (AttributeError, TypeError, ValueError):
        return None


def _deserialize_message_id(token: Any) -> Any:
    """A checkpointed token back into a Pulsar ``MessageId``, or None if unusable."""
    if not isinstance(token, str):
        return None
    pulsar = _import_pulsar()
    try:
        return pulsar.MessageId.deserialize(base64.b64decode(token))
    except (AttributeError, TypeError, ValueError):
        return None
