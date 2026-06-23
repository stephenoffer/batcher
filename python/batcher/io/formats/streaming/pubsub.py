"""Google Cloud Pub/Sub broker source — subscription pull batches.

Backed by ``google-cloud-pubsub`` (the optional ``pubsub`` extra). A
:class:`PubSubSource` pulls a batch of messages from a subscription
(``SubscriberClient.pull(max_messages=N)``), assembles them into one Arrow batch
via the shared ``_make_batch`` helper, and acknowledges them — at-least-once
delivery; ack only after a batch is assembled so a crash before ack re-delivers.

Pub/Sub has no user-visible partitions, so the stream is modeled as a single
logical partition (``0``): ``splits()`` returns one split. The opaque message id
is hashed into the int64 ``offset`` column to fit the fixed broker schema.

The ``google-cloud-pubsub`` import is deferred to construction; if the extra is
missing a :class:`BackendError` instructs the user to install it.
"""

from __future__ import annotations

from typing import Any

from batcher._internal.errors import BackendError
from batcher.io.formats.base import SOURCES
from batcher.io.formats.streaming.broker import BrokerMessage, BrokerSource

__all__ = ["PubSubSource"]


def _import_subscriber() -> Any:
    """Import ``pubsub_v1.SubscriberClient`` or raise a guiding ``BackendError``."""
    try:
        from google.cloud import pubsub_v1
    except ImportError as exc:
        raise BackendError(
            "reading from Pub/Sub needs the pubsub extra: pip install 'batcher[pubsub]'"
        ) from exc
    return pubsub_v1.SubscriberClient


@SOURCES.register("pubsub")
class PubSubSource(BrokerSource):
    """An unbounded Pub/Sub subscription, consumed via ``google-cloud-pubsub``.

    The ``topic`` is the fully-qualified subscription path
    (``projects/<project>/subscriptions/<sub>``). Pub/Sub does not expose
    partitions, so the stream is a single logical partition; ``partitions`` is
    accepted (for the :class:`BrokerSplit` round-trip) but ignored beyond
    confirming the single partition.
    """

    format_name = "pubsub"

    __slots__ = ("_client_obj",)

    def __init__(
        self,
        topic: str,
        *,
        poll_size: int = 16_384,
        partitions: list[int] | None = None,  # noqa: ARG002 (single logical partition)
        **options: Any,
    ) -> None:
        super().__init__(topic, poll_size=poll_size, **options)
        self._client_obj: Any = None

    def _client(self) -> Any:
        if self._client_obj is None:
            subscriber_cls = _import_subscriber()
            self._client_obj = subscriber_cls()
        return self._client_obj

    def _discover_partitions(self) -> list[int]:
        return [0]  # Pub/Sub has no user-visible partitions.

    def _poll(self) -> list[BrokerMessage] | None:
        client = self._client()
        response = client.pull(request={"subscription": self.topic, "max_messages": self.poll_size})
        received = response.received_messages
        messages = [
            BrokerMessage(
                value=rm.message.data,
                partition=0,
                offset=abs(hash(rm.message.message_id)) % (1 << 63),
                timestamp=int(rm.message.publish_time.timestamp() * 1000),
                topic=self.topic,
                key=(rm.message.ordering_key or "").encode("utf-8") or None,
            )
            for rm in received
        ]
        if received:
            # Ack only after assembling the batch → no message lost on crash.
            client.acknowledge(
                request={
                    "subscription": self.topic,
                    "ack_ids": [rm.ack_id for rm in received],
                }
            )
        return messages
