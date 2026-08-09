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

from batcher._internal.errors import BackendError
from batcher.io.formats.base import SOURCES
from batcher.io.formats.streaming.broker import BrokerMessage, BrokerSource

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


def _event_to_message(ev: Any, partition_id: int, topic: str) -> BrokerMessage:
    """Turn one ``EventData`` into a `BrokerMessage`, preserving raw bytes.

    Reads the body as raw bytes (`body_as_bytes`), not `body_as_str`: a binary Event Hub
    payload — a protobuf, an Avro frame, a compressed blob — is not valid UTF-8, so
    decoding it raised `UnicodeDecodeError` and a non-UTF-8-but-decodable payload was
    silently mangled by the decode/re-encode round trip. The partition key is coerced to
    bytes for the same fixed-`binary` schema reason.

    ``resume_token`` carries the Event Hub *offset string* — the exact `event_position` a
    recovering consumer resumes from. The int64 ``offset`` column is a lossy `int(...)` of
    it, fine for ordering but not for seeking, so the raw string is kept for the checkpoint.
    """
    return BrokerMessage(
        value=_as_bytes(ev.body_as_bytes()) or b"",
        partition=partition_id,
        offset=int(ev.offset) if ev.offset is not None else 0,
        resume_token=str(ev.offset) if ev.offset is not None else None,
        timestamp=ev.enqueued_time_utc_ms or 0,
        topic=topic,
        key=_as_bytes(ev.partition_key),
    )


def _import_consumer() -> Any:
    """Import ``EventHubConsumerClient`` or raise a guiding ``BackendError``."""
    try:
        from azure.eventhub import EventHubConsumerClient
    except ImportError as exc:
        raise BackendError(
            "reading Event Hubs needs the eventhubs extra: pip install 'batcher-engine[eventhubs]'"
        ) from exc
    return EventHubConsumerClient


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

    __slots__ = ("_client_obj", "_consumers", "_partition_ids", "_partitions")

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

    def _consumer(self, partition_id: int) -> Any:
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
        """
        consumer = self._consumers.get(partition_id)
        if consumer is not None:
            return consumer
        client = self._client()
        # Resume from the checkpointed offset when recovering this partition; otherwise the
        # configured start. Without this the source always restarted from
        # `starting_position`, silently replaying or skipping on every recovery — every
        # other broker here (`kafka`, `kinesis`, `pulsar`) honors its checkpoint, and this
        # one quietly did not. `seek` populates `_resume_from`; a live offset is exclusive,
        # so the consumer resumes strictly after the last delivered event.
        event_position = self._resume_from.get(partition_id, self._options["starting_position"])
        consumer = client._create_consumer(
            consumer_group=self._options["consumer_group"],
            partition_id=str(partition_id),
            event_position=event_position,
            on_event_received=lambda *_: None,
        )
        self._consumers[partition_id] = consumer
        return consumer

    def _apply_seek(self, partition: int, token: Any) -> None:  # noqa: ARG002
        """Drop this partition's cached consumer so the next poll reopens it at `token`.

        The base records the position in `_resume_from`; a consumer already open is still
        sitting at its old position, so without this a recovery would silently keep reading
        from wherever the pre-crash consumer had got to.
        """
        consumer = self._consumers.pop(partition, None)
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
        messages: list[BrokerMessage] = []
        # Only the first partition waits. The rest take whatever has already arrived, so an
        # idle hub costs one wait per poll rather than one per partition — the serial loop
        # made the effective trigger cadence `partitions x max_wait_time`, which on eight
        # partitions was eight seconds of latency for a stream that had nothing to say.
        for index, partition_id in enumerate(self._discover_partitions()):
            consumer = self._consumer(partition_id)
            wait = _FIRST_WAIT_SECONDS if index == 0 else _DRAIN_WAIT_SECONDS
            events = consumer.receive_message_batch(
                max_batch_size=self.poll_size, max_wait_time=wait
            )
            messages.extend(_event_to_message(ev, partition_id, self.topic) for ev in events)
        return messages
