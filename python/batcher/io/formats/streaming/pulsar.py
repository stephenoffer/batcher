"""Apache Pulsar broker source — one Split per partition, via ``pulsar-client``.

Backed by ``pulsar-client`` (the optional ``pulsar`` extra). A
:class:`PulsarSource` consumes a topic with a shared subscription, draining up to
``poll_size`` messages per poll (one blocking ``Consumer.receive(timeout_millis=…)`` for
the first message, then a single ``batch_receive()`` for whatever the client already holds)
and assembling them into one Arrow batch via the shared ``_make_batch`` helper.

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

from batcher._internal.optional import require
from batcher.io.formats.base import SOURCES
from batcher.io.formats.streaming.broker import BrokerMessage, BrokerSource, as_header_pairs

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
    return require("pulsar", feature="Pulsar support", provides="pulsar-client", extra="pulsar")


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
        "_starting_position",
        "_unacked",
    )

    def __init__(
        self,
        topic: str,
        *,
        poll_size: int = 16_384,
        poll_bytes: int | None = None,
        partitions: list[int] | None = None,
        service_url: str = "pulsar://localhost:6650",
        subscription: str = "batcher",
        num_partitions: int = 1,
        receive_timeout_millis: int = 1000,
        starting_position: str = "earliest",
        **options: Any,
    ) -> None:
        from batcher.io.formats.streaming.broker.schema import normalize_starting_position

        # One option name across every broker, mapped here onto Pulsar's `InitialPosition`.
        self._starting_position = normalize_starting_position(
            starting_position, aliases={"earliest": "Earliest", "latest": "Latest"}
        )
        super().__init__(
            topic,
            poll_size=poll_size,
            poll_bytes=poll_bytes,
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
        kwargs: dict[str, Any] = {}
        policy = getattr(pulsar, "ConsumerBatchReceivePolicy", None)
        if policy is not None:
            # Sized for the *drain*, which is the only thing `batch_receive` is used for:
            # take everything the client has already buffered, up to a poll's budget, and
            # come back within a millisecond if it has nothing. The blocking wait for the
            # first message stays a plain `receive` with the full timeout, so an idle topic
            # still parks instead of spinning.
            #
            # The policy's *byte* bound carries `poll_bytes` — the point being that the
            # client stops filling the batch, rather than this code receiving a huge one and
            # then having to put messages back. There is no putting them back: an unacked
            # Pulsar message is redelivered only after `ackTimeout`, so trimming an
            # over-large batch here would reorder the partition rather than defer it.
            kwargs["batch_receive_policy"] = policy(
                self.poll_size, self.poll_bytes, _DRAIN_TIMEOUT_MILLIS
            )
        # Where a *new* subscription starts. Pulsar remembers a subscription's cursor
        # server-side, so this only bites the first time — which is exactly when a reader
        # who wanted `latest` discovers they replayed the whole topic instead.
        # Doubly defensive: an older client may not expose `InitialPosition` at all, and a
        # client that does may not know a position name. Either way the subscription simply
        # takes the broker's default rather than the read failing on an option.
        initial = getattr(getattr(pulsar, "InitialPosition", None), self._starting_position, None)
        if initial is not None:
            kwargs["initial_position"] = initial
        self._consumer = self._client_obj.subscribe(
            self._topic_names(),
            subscription_name=self._options["subscription"],
            consumer_type=pulsar.ConsumerType.Shared,
            **kwargs,
        )
        return self._consumer

    def _split_options(self) -> dict[str, Any]:
        """The three settings this source consumes by name, so a worker rebuilds them.

        ``num_partitions`` is the load-bearing one: Pulsar does not tell a client how many
        partitions a topic has, so it is declared — and a split that lost it rebuilt a
        one-partition reader, which discovers the wrong partition set on the very worker
        that was given a partition outside it. ``starting_position`` is already normalized
        to Pulsar's own spelling, which `normalize_starting_position` accepts back
        unchanged.

        Returns:
            The constructor keyword arguments this class consumed.
        """
        return {
            "num_partitions": self._num_partitions,
            "receive_timeout_millis": self._receive_timeout_millis,
            "starting_position": self._starting_position,
        }

    def _discover_partitions(self) -> list[int]:
        if self._partitions is not None:
            return list(self._partitions)
        return list(range(max(1, self._num_partitions)))

    def _drain(self, consumer: Any, pulsar: Any) -> list[Any]:
        """Everything the client has already buffered, in as few client calls as possible.

        `batch_receive()` hands back a whole buffered batch in **one** call, bounded by the
        `batch_receive_policy` set at subscribe time. The drain used to be one `receive()`
        per message, so a poll that collected a full 16,384-message budget crossed the
        Python/C++ boundary 16,384 times, plus one more to be told the queue was empty —
        pure per-message overhead on the latency-critical path, for messages the client
        already held in memory.

        Older `pulsar-client` builds have no `batch_receive`, so the per-message loop stays
        as the fallback rather than becoming a hard version requirement on an optional extra.
        """
        budget = self.poll_size - 1
        if budget <= 0:
            return []
        batch_receive = getattr(consumer, "batch_receive", None)
        if batch_receive is not None:
            try:
                return list(batch_receive())[:budget]
            except pulsar.Timeout:
                return []
        drained: list[Any] = []
        while budget > 0:
            try:
                drained.append(consumer.receive(timeout_millis=_DRAIN_TIMEOUT_MILLIS))
            except pulsar.Timeout:
                break
            budget -= 1
        return drained

    def _to_message(self, msg: Any) -> BrokerMessage:
        """One Pulsar message as the broker schema's fixed shape."""
        mid = msg.message_id()
        return BrokerMessage(
            value=msg.data(),
            partition=msg.partition() if hasattr(msg, "partition") else 0,
            offset=_message_id_to_offset(mid),
            # The int64 `offset` folds (ledger, entry) lossily, so it cannot be seeked to.
            # The serialized `MessageId` is the exact position, and it is what `_apply_seek`
            # hands back to `Consumer.seek`.
            resume_token=_serialize_message_id(mid),
            timestamp=msg.publish_timestamp(),
            topic=msg.topic_name() if hasattr(msg, "topic_name") else self.topic,
            key=msg.partition_key().encode("utf-8") if msg.partition_key() else None,
            # Pulsar's *properties* are the same idea as Kafka's headers, and they reached
            # the `headers` column as nulls until now: the option was accepted and the data
            # silently dropped. Read only when asked for, because it is a per-message dict.
            headers=(as_header_pairs(_properties(msg)) if self._include_headers else None),
        )

    def _poll(self) -> list[BrokerMessage] | None:
        pulsar = _import_pulsar()
        consumer = self._client()
        # Wait out `receive_timeout_millis` for the *first* message only, then drain what the
        # client has already buffered with a near-zero timeout. Charging every `receive` the
        # full timeout meant the last one — the one that finds the queue empty and ends the
        # drain — cost a full second on *every* poll, so a topic delivering ten messages a
        # trigger paid a fixed second of latency to notice it had run out. That is the same
        # shape as the Kafka `consume()` trap, and it hid in the same place: a loop that looks
        # like it is reading is in fact mostly waiting to be told there is nothing left.
        try:
            first = consumer.receive(timeout_millis=self._receive_timeout_millis)
        except pulsar.Timeout:
            return []
        raw: list[Any] = [first, *self._drain(consumer, pulsar)]
        messages = [self._to_message(m) for m in raw]
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


def _properties(msg: Any) -> Any:
    """One Pulsar message's properties, or None on a client that does not expose them."""
    getter = getattr(msg, "properties", None)
    if getter is None:
        return None
    try:
        return getter()
    except Exception:  # pragma: no cover - a client whose `properties()` refuses
        return None
