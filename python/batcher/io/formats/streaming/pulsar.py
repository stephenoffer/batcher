"""Apache Pulsar broker source — one Split per partition, via ``pulsar-client``.

Backed by ``pulsar-client`` (the optional ``pulsar`` extra). A
:class:`PulsarSource` consumes a topic with a shared subscription, draining up to
``poll_size`` messages per poll (``Consumer.receive(timeout_millis=…)``),
assembling them into one Arrow batch via the shared ``_make_batch`` helper, and
acknowledging them — ack only after a batch is assembled so a crash before ack
re-delivers.

``splits()`` returns one split per partition (the partition index is the offset
locator); for a non-partitioned topic this is a single split. The Pulsar
``MessageId`` is opaque, so the message's ledger/entry pair is folded into the
int64 ``offset`` column to fit the fixed broker schema.

The ``pulsar`` import is deferred to construction; if the extra is missing a
:class:`BackendError` instructs the user to install it.
"""

from __future__ import annotations

from typing import Any

from batcher._internal.errors import BackendError
from batcher.io.formats.base import SOURCES
from batcher.io.formats.streaming.broker import BrokerMessage, BrokerSource

__all__ = ["PulsarSource"]


def _import_pulsar() -> Any:
    """Import the ``pulsar`` client module or raise a guiding ``BackendError``."""
    try:
        import pulsar
    except ImportError as exc:
        raise BackendError(
            "reading from Pulsar needs the pulsar extra: pip install 'batcher[pulsar]'"
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

    __slots__ = ("_client_obj", "_consumer", "_num_partitions", "_partitions")

    def __init__(
        self,
        topic: str,
        *,
        poll_size: int = 16_384,
        partitions: list[int] | None = None,
        service_url: str = "pulsar://localhost:6650",
        subscription: str = "batcher",
        num_partitions: int = 1,
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
        pulsar = _import_pulsar()
        if self._client_obj is None:
            self._client_obj = pulsar.Client(self._options["service_url"])
        if self._consumer is None:
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
        for _ in range(self.poll_size):
            try:
                msg = consumer.receive(timeout_millis=1000)
            except pulsar.Timeout:
                break
            raw.append(msg)
            mid = msg.message_id()
            messages.append(
                BrokerMessage(
                    value=msg.data(),
                    partition=msg.partition() if hasattr(msg, "partition") else 0,
                    offset=_message_id_to_offset(mid),
                    timestamp=msg.publish_timestamp(),
                    topic=msg.topic_name() if hasattr(msg, "topic_name") else self.topic,
                    key=msg.partition_key().encode("utf-8") if msg.partition_key() else None,
                )
            )
        for msg in raw:
            consumer.acknowledge(msg)  # ack after the batch is assembled.
        return messages


def _message_id_to_offset(message_id: Any) -> int:
    """Fold a Pulsar ``MessageId`` (ledger, entry) into one int64 offset.

    Pulsar offsets are a ``(ledger_id, entry_id)`` pair; combine them so messages
    within a ledger remain monotonically ordered in the fixed int64 column.
    """
    try:
        ledger = int(message_id.ledger_id())
        entry = int(message_id.entry_id())
        return ((ledger << 20) | (entry & 0xFFFFF)) % (1 << 63)
    except (AttributeError, ValueError):
        return abs(hash(str(message_id))) % (1 << 63)
