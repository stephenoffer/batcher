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


@SOURCES.register("eventhubs")
class EventHubsSource(BrokerSource):
    """An unbounded Event Hub, consumed via ``azure-eventhub``.

    The ``topic`` is the Event Hub name. Required option: ``connection_str`` (the
    namespace connection string). Options: ``consumer_group`` (default
    ``"$Default"``), ``starting_position`` (default ``"-1"`` = start of stream),
    and ``partitions`` (the specific partition ids to read — set by
    :class:`BrokerSplit` on a worker).

    Alternatively, point :class:`~batcher.io.formats.streaming.kafka.KafkaSource`
    at the namespace's Kafka endpoint (port 9093, SASL/PLAIN) to avoid this extra.
    """

    format_name = "eventhubs"

    __slots__ = ("_client_obj", "_partitions")

    def __init__(
        self,
        topic: str,
        *,
        poll_size: int = 16_384,
        partitions: list[int] | None = None,
        connection_str: str = "",
        consumer_group: str = "$Default",
        starting_position: str = "-1",
        **options: Any,
    ) -> None:
        super().__init__(
            topic,
            poll_size=poll_size,
            connection_str=connection_str,
            consumer_group=consumer_group,
            starting_position=starting_position,
            **options,
        )
        self._partitions = partitions
        self._client_obj: Any = None

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
        client = self._client()
        return sorted(int(pid) for pid in client.get_partition_ids())

    def close(self) -> None:
        """Close the consumer client, releasing its AMQP connection and background threads.

        `BrokerSource.iter_batches` calls this from a `finally`, so it runs even when a
        consumer abandons the generator mid-stream — previously the AMQP link stayed open
        until garbage collection, which for a reference cycle never happens.
        """
        if self._client_obj is not None:
            self._client_obj.close()
            self._client_obj = None

    def _poll(self) -> list[BrokerMessage] | None:
        client = self._client()
        messages: list[BrokerMessage] = []
        partitions = (
            self._partitions
            if self._partitions is not None
            else [int(p) for p in client.get_partition_ids()]
        )
        for partition_id in partitions:
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
            # Each `_create_consumer` opens its own AMQP link. Without this `finally` a
            # continuous stream leaked one consumer (a socket plus background threads) per
            # partition on *every* poll — seconds apart, for the life of the query.
            try:
                events = consumer.receive_message_batch(
                    max_batch_size=self.poll_size, max_wait_time=1.0
                )
                messages.extend(_event_to_message(ev, partition_id, self.topic) for ev in events)
            finally:
                consumer.close()
        return messages
