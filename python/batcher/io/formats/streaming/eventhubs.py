"""Azure Event Hubs broker source — one Split per partition, via ``azure-eventhub``.

Backed by ``azure-eventhub`` (the optional ``eventhubs`` extra). A
:class:`EventHubsSource` polls a partition with a partition-scoped consumer
(``EventHubConsumerClient`` / ``receive_batch``) and assembles each poll into one
Arrow batch via the shared ``_make_batch`` helper.

``splits()`` returns one split per partition (the partition id is the offset
locator). Note: Event Hubs also exposes a Kafka-protocol endpoint, so a
``KafkaSource`` pointed at ``<namespace>.servicebus.windows.net:9093`` with SASL
is a valid alternative; this native client avoids the Kafka dependency for users
already on the Azure SDK.

The ``azure-eventhub`` import is deferred to construction; if the extra is missing
a :class:`BackendError` instructs the user to install it.
"""

from __future__ import annotations

from typing import Any

from batcher._internal.optional import require
from batcher.io.formats.base import SOURCES
from batcher.io.formats.streaming.broker import (
    BrokerMessage,
    BrokerSource,
    as_header_pairs,
    opaque_offset,
)

__all__ = ["EventHubsSource"]

#: How long the *first* partition of a poll waits for events. The partitions after it drain
#: what has already arrived without waiting again — otherwise an idle eight-partition hub cost
#: eight full waits per poll, serially, and the trigger cadence became `partitions x wait`.
_FIRST_WAIT_SECONDS = 1.0
_DRAIN_WAIT_SECONDS = 0.01


def _as_bytes(value: Any) -> bytes | None:
    """Coerce an Event Hub field to the ``bytes`` the broker schema declares.

    Event Hub payloads and partition keys can arrive as ``str`` or ``bytes`` depending on
    how the producer sent them. The broker schema is fixed at ``binary`` for both ``value``
    and ``key``, so a ``str`` reaches `_make_batch` and raises `ArrowTypeError` there. A
    ``None`` (an unkeyed message) passes through unchanged.
    """
    if value is None or isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return bytes(value)


def _event_body(ev: Any) -> bytes:
    """One ``EventData``'s payload as the raw ``bytes`` the ``value`` column declares.

    `EventData` has no ``body_as_bytes``; it exposes ``body``, whose shape follows the AMQP
    body type. A ``DATA`` body — what an ordinary producer sends — is ``bytes`` **or an
    iterable of ``bytes``**, one element per AMQP data section, so a multi-section event has
    to be joined rather than assumed scalar. A ``SEQUENCE`` or ``VALUE`` body is structured
    data with no byte encoding of its own, so it is rendered as JSON, which the ``.json``
    accessor can then parse downstream; that is a statable encoding rather than a `repr`.

    Deliberately not `body_as_str`: a binary payload — a protobuf, an Avro frame, a
    compressed blob — is not valid UTF-8, so decoding raises and a decodable-but-not-UTF-8
    payload is mangled by the decode/re-encode round trip.
    """
    import json

    try:
        body = ev.body
    except ValueError:
        return b""  # `EventData.body` raises this for an empty event
    if isinstance(body, bytes):
        return body
    if isinstance(body, bytearray | memoryview):
        return bytes(body)
    if isinstance(body, str):
        return body.encode("utf-8")
    try:
        sections = list(body)
    except TypeError:
        return json.dumps(body, default=str).encode("utf-8")
    if sections and all(isinstance(s, bytes | bytearray | memoryview) for s in sections):
        return b"".join(bytes(s) for s in sections)
    return json.dumps(sections, default=str).encode("utf-8")


def _enqueued_ms(ev: Any) -> int:
    """The event's enqueued time in epoch milliseconds, or 0 when the hub recorded none.

    `EventData` exposes ``enqueued_time`` as a timezone-aware `datetime`, not the
    ``enqueued_time_utc_ms`` this connector used to read — an attribute that does not exist
    on any released `azure-eventhub`, so every poll raised `AttributeError` before a single
    event was assembled.
    """
    enqueued = getattr(ev, "enqueued_time", None)
    return 0 if enqueued is None else int(enqueued.timestamp() * 1000)


def _event_to_message(
    ev: Any, partition_id: int, topic: str, *, include_headers: bool = False
) -> BrokerMessage:
    """Turn one ``EventData`` into a `BrokerMessage`, preserving raw bytes.

    ``resume_token`` carries the Event Hub *offset string* — the exact `event_position` a
    recovering consumer resumes from. The int64 ``offset`` column is a lossy projection of
    it, fine for ordering but not for seeking, so the raw string is kept for the checkpoint.
    The projection goes through `opaque_offset` rather than a bare `int(...)`, because an
    Event Hubs offset is typed `str` and is only *conventionally* numeric — a namespace that
    hands back a non-numeric token used to raise `ValueError` and kill the query.
    """
    offset = ev.offset
    return BrokerMessage(
        value=_event_body(ev),
        partition=partition_id,
        offset=opaque_offset(str(offset)) if offset is not None else 0,
        resume_token=str(offset) if offset is not None else None,
        timestamp=_enqueued_ms(ev),
        topic=topic,
        key=_as_bytes(ev.partition_key),
        # Event Hubs' application *properties* are this broker's headers, and they reached
        # the column as nulls until now. Read only when asked for: it is a per-message dict.
        headers=as_header_pairs(getattr(ev, "properties", None)) if include_headers else None,
    )


def _sink(buffer: list[Any]) -> Any:
    """An ``on_event_received`` callback that appends what it is handed to `buffer`.

    `EventHubConsumer.receive(batch=True)` calls the callback with a **list** of events;
    with ``batch=False`` it calls it with a single event, or `None` when the wait expired
    with nothing to deliver. Accepting all three shapes means the drain is correct whichever
    way the consumer is driven, and a `None` never lands in the buffer as a phantom event.
    """

    def _on_event(events: Any) -> None:
        if events is None:
            return
        if isinstance(events, list):
            buffer.extend(e for e in events if e is not None)
        else:
            buffer.append(events)

    return _on_event


def _import_consumer() -> Any:
    """Import ``EventHubConsumerClient`` or raise a guiding ``BackendError``."""
    return require(
        "azure.eventhub",
        "EventHubConsumerClient",
        feature="Event Hubs support",
        provides="azure-eventhub",
        extra="eventhubs",
    )


def _event_hubs_position(value: str) -> str:
    """Map the shared `starting_position` onto Event Hubs' own offset sentinels.

    ``-1`` is the start of a partition and ``@latest`` its tail, which is a vocabulary
    nobody guesses. An explicit offset string is passed through, because resuming from a
    recorded position is the other reason to set this at all.
    """
    from batcher.io.formats.streaming.broker.schema import normalize_starting_position

    if value not in ("earliest", "latest", "-1", "@latest"):
        return value  # an explicit offset/sequence number the caller recorded earlier
    return normalize_starting_position(value, aliases={"earliest": "-1", "latest": "@latest"})


@SOURCES.register("eventhubs")
class EventHubsSource(BrokerSource):
    """An unbounded Event Hub, consumed via ``azure-eventhub``.

    The ``topic`` is the Event Hub name. Required option: ``connection_str`` (the
    namespace connection string). Options: ``consumer_group`` (default
    ``"$Default"``), ``starting_position`` (``"earliest"`` / ``"latest"``, the name every
    broker here shares, or an explicit Event Hubs offset string),
    and ``partitions`` (the specific partition ids to read — set by
    :class:`BrokerSplit` on a worker).

    Alternatively, point :class:`~batcher.io.formats.streaming.kafka.KafkaSource`
    at the namespace's Kafka endpoint (port 9093, SASL/PLAIN) to avoid this extra.
    """

    format_name = "eventhubs"

    __slots__ = ("_buffers", "_client_obj", "_consumers", "_partition_ids", "_partitions")

    def __init__(
        self,
        topic: str,
        *,
        poll_size: int = 16_384,
        partitions: list[int] | None = None,
        connection_str: str = "",
        consumer_group: str = "$Default",
        starting_position: str = "earliest",
        **options: Any,
    ) -> None:
        super().__init__(
            topic,
            poll_size=poll_size,
            connection_str=connection_str,
            consumer_group=consumer_group,
            starting_position=_event_hubs_position(starting_position),
            **options,
        )
        self._partitions = partitions
        self._client_obj: Any = None
        # Per-partition AMQP consumers, kept across polls. See `_consumer`.
        self._consumers: dict[int, Any] = {}
        # Where each partition's consumer *delivers*. An `EventHubConsumer` hands events to
        # the `on_event_received` callback it was built with and returns nothing, so the
        # callback has to own a destination that outlives the call. See `_consumer`.
        self._buffers: dict[int, list[Any]] = {}
        # The hub's partition ids, fetched once. `_poll` asked the service for them on
        # *every* poll, a management round-trip per trigger for a value that changes only
        # when the hub is rescaled.
        self._partition_ids: list[int] | None = None

    def _client(self) -> Any:
        if self._client_obj is None:
            consumer_cls = _import_consumer()
            self._client_obj = consumer_cls.from_connection_string(
                conn_str=self._options["connection_str"],
                consumer_group=self._options["consumer_group"],
                eventhub_name=self.topic,
            )
        return self._client_obj

    def _discover_partitions(self) -> list[int]:
        if self._partitions is not None:
            return list(self._partitions)
        if self._partition_ids is None:
            client = self._client()
            self._partition_ids = sorted(int(pid) for pid in client.get_partition_ids())
        return list(self._partition_ids)

    def _consumer(self, partition_id: int) -> tuple[Any, list[Any]]:
        """A partition-scoped consumer, opened once and reused across polls.

        Opening one per poll and closing it in a `finally` fixed a leak (S7) at the cost of
        a full AMQP link setup and teardown *per partition, per poll*. On a 100ms trigger
        over an eight-partition hub that is eighty link negotiations a second before a
        single event is read, and it also throws away the consumer's own prefetch — every
        poll started with an empty local buffer.

        Reuse is safe because the consumer's position advances with what it has delivered,
        which is exactly the "resume strictly after" contract the checkpoint wants.
        `_apply_seek` drops the cached consumer for a partition being repositioned, so a
        recovery still rebuilds at the checkpointed offset.

        Returns the consumer together with the list it delivers into. An `EventHubConsumer`
        is *callback*-driven: `receive` pushes events to the `on_event_received` it was
        constructed with and returns `None`. The callback here was ``lambda *_: None``, so
        every event the client fetched was dropped on the floor — one of the reasons this
        source could not make progress at all. The buffer is per partition and lives as long
        as its consumer, and `_poll` drains it after each `receive`.
        """
        consumer = self._consumers.get(partition_id)
        if consumer is not None:
            return consumer, self._buffers[partition_id]
        client = self._client()
        # Resume from the checkpointed offset when recovering this partition; otherwise the
        # configured start. Without this the source always restarted from
        # `starting_position`, silently replaying or skipping on every recovery — every
        # other broker here (`kafka`, `kinesis`, `pulsar`) honors its checkpoint, and this
        # one quietly did not. `seek` populates `_resume_from`; a live offset is exclusive,
        # so the consumer resumes strictly after the last delivered event.
        event_position = self._resume_from.get(partition_id, self._options["starting_position"])
        buffer: list[Any] = []
        consumer = client._create_consumer(
            consumer_group=self._options["consumer_group"],
            partition_id=str(partition_id),
            event_position=event_position,
            on_event_received=_sink(buffer),
        )
        self._consumers[partition_id] = consumer
        self._buffers[partition_id] = buffer
        return consumer, buffer

    def _apply_seek(self, partition: int, token: Any) -> None:  # noqa: ARG002
        """Drop this partition's cached consumer so the next poll reopens it at `token`.

        The base records the position in `_resume_from`; a consumer already open is still
        sitting at its old position, so without this a recovery would silently keep reading
        from wherever the pre-crash consumer had got to.
        """
        consumer = self._consumers.pop(partition, None)
        self._buffers.pop(partition, None)
        if consumer is not None:
            import contextlib

            with contextlib.suppress(Exception):
                consumer.close()

    def close(self) -> None:
        """Close the consumer client, releasing its AMQP connection and background threads.

        `BrokerSource.iter_batches` calls this from a `finally`, so it runs even when a
        consumer abandons the generator mid-stream — previously the AMQP link stayed open
        until garbage collection, which for a reference cycle never happens.
        """
        import contextlib

        consumers, self._consumers = self._consumers, {}
        self._buffers = {}
        for consumer in consumers.values():
            with contextlib.suppress(Exception):
                consumer.close()
        # Dropped before the call that releases it: a `close()` that raises used to leave the
        # attribute set, so the second `close()` `iter_batches` guarantees from its `finally`
        # re-closed a dead client and raised again over the original error.
        if self._client_obj is not None:
            client, self._client_obj = self._client_obj, None
            client.close()

    def _poll(self) -> list[BrokerMessage] | None:
        """One micro-batch across every partition this reader owns.

        Reads through `EventHubConsumer.receive`, which is the method the client actually
        has. This used to call ``receive_message_batch``, which exists on no released
        `azure-eventhub` — so the very first poll raised `AttributeError` and the source
        never delivered a single row. `receive` returns `None` and delivers into the
        consumer's callback, so each partition's buffer is drained here rather than read
        from a return value.
        """
        messages: list[BrokerMessage] = []
        budget = self._poll_budget()
        # Only the first partition waits. The rest take whatever has already arrived, so an
        # idle hub costs one wait per poll rather than one per partition — the serial loop
        # made the effective trigger cadence `partitions x max_wait_time`, which on eight
        # partitions was eight seconds of latency for a stream that had nothing to say.
        #
        # Rotated, so the partition that gets the long wait — and the one guaranteed to be
        # read before the byte budget runs out — is a different one each poll. Always
        # leading with partition 0 gives it the only real wait, and starves the tail of a
        # wide hub; under a per-partition watermark a starved partition stalls the stream's
        # frontier just as a silent one does.
        for index, partition_id in enumerate(budget.order(self._discover_partitions())):
            if budget.spent:
                break  # the rest of the sweep is the next epoch's; nothing is lost
            consumer, buffer = self._consumer(partition_id)
            wait = _FIRST_WAIT_SECONDS if index == 0 else _DRAIN_WAIT_SECONDS
            consumer.receive(batch=True, max_batch_size=self.poll_size, max_wait_time=wait)
            if not buffer:
                continue
            # Swap the list out rather than clearing in place: `receive` can deliver more
            # than one callback's worth, and a buffer emptied under a still-registered
            # callback would drop whatever arrived between the read and the clear.
            events, buffer[:] = list(buffer), []
            decoded = [
                _event_to_message(
                    ev, partition_id, self.topic, include_headers=self._include_headers
                )
                for ev in events
            ]
            budget.spend(sum(len(m.value or b"") for m in decoded))
            messages.extend(decoded)
        return messages
