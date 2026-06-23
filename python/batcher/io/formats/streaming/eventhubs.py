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


def _import_consumer() -> Any:
    """Import ``EventHubConsumerClient`` or raise a guiding ``BackendError``."""
    try:
        from azure.eventhub import EventHubConsumerClient
    except ImportError as exc:
        raise BackendError(
            "reading from Event Hubs needs the eventhubs extra: pip install 'batcher[eventhubs]'"
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

    def _poll(self) -> list[BrokerMessage] | None:
        client = self._client()
        messages: list[BrokerMessage] = []
        partitions = (
            self._partitions
            if self._partitions is not None
            else [int(p) for p in client.get_partition_ids()]
        )
        for partition_id in partitions:
            consumer = client._create_consumer(
                consumer_group=self._options["consumer_group"],
                partition_id=str(partition_id),
                event_position=self._options["starting_position"],
                on_event_received=lambda *_: None,
            )
            events = consumer.receive_message_batch(
                max_batch_size=self.poll_size, max_wait_time=1.0
            )
            for ev in events:
                messages.append(
                    BrokerMessage(
                        value=bytes(ev.body_as_str(), "utf-8"),
                        partition=partition_id,
                        offset=int(ev.offset) if ev.offset is not None else 0,
                        timestamp=ev.enqueued_time_utc_ms or 0,
                        topic=self.topic,
                        key=ev.partition_key,
                    )
                )
        return messages
