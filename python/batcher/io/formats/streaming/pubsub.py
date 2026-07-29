"""Google Cloud Pub/Sub broker source — subscription pull batches.

Backed by ``google-cloud-pubsub`` (the optional ``pubsub`` extra). A
:class:`PubSubSource` pulls a batch of messages from a subscription
(``SubscriberClient.pull(max_messages=N)``) and assembles them into one Arrow batch
via the shared ``_make_batch`` helper.

Acks happen only after an epoch is **published**, never when it is pulled: `_poll` holds the
ack ids and `_commit_delivered` sends them once the engine has published the batch. Acking at
pull time (as this once did) told Pub/Sub the messages were handled the moment they were
*read*, so a crash before the publish meant no redelivery and silent data loss. The ordering
favours a duplicate, absorbed by an idempotent sink, over an unrecoverable gap.

Pub/Sub has no user-visible partitions, so the stream is modeled as a single
logical partition (``0``): ``splits()`` returns one split. The opaque message id
is hashed into the int64 ``offset`` column to fit the fixed broker schema.

The ``google-cloud-pubsub`` import is deferred to construction; if the extra is
missing a :class:`BackendError` instructs the user to install it.
"""

from __future__ import annotations

import hashlib
from typing import Any

from batcher._internal.errors import BackendError
from batcher.io.formats.base import SOURCES
from batcher.io.formats.streaming.broker import BrokerMessage, BrokerSource

__all__ = ["PubSubSource"]

#: Pub/Sub caps a synchronous `pull` at 1,000 messages and rejects a larger request with
#: `InvalidArgument`. The engine's 16,384-row default poll size is therefore not a legal
#: request, so every poll is clamped to the API's ceiling rather than sent.
_PULL_MAX_MESSAGES = 1_000

#: Ack ids per `acknowledge` request. The API bounds a request at 512 KiB, and an ack id runs
#: to a couple of hundred bytes, so a full poll's worth in one call is over the limit and
#: fails the *commit* — the step that turns an at-least-once read into a completed epoch.
#: Chunking keeps each request comfortably inside the bound.
_ACK_CHUNK = 1_000


def _stable_offset(message_id: str) -> int:
    """A process-stable int64 offset for an opaque Pub/Sub message id.

    This was `abs(hash(message_id))`, and Python salts `str` hashing per process, so the same
    message got a different `offset` on every run and on every worker. The `offset` column is
    what downstream de-duplication and ordering key off, so it silently stopped being
    comparable across restarts and across a distributed read — the two places comparing it
    matters. `sha256` costs the same call and is stable everywhere.
    """
    return int.from_bytes(hashlib.sha256(message_id.encode("utf-8")).digest()[:8], "big") % (
        1 << 63
    )


def _is_pull_timeout(exc: BaseException) -> bool:
    """Whether `exc` is a Pub/Sub pull that timed out with nothing to deliver.

    An idle subscription has no messages, so a bounded pull hits its deadline. Google raises
    ``DeadlineExceeded`` (or wraps it in ``RetryError``); both mean "no data within the
    window", not a failure. Matched by class name so the optional ``google-api-core`` need
    not be importable to recognize the idle case.
    """
    names = {type(exc).__name__ for exc in (exc, exc.__cause__) if exc is not None}
    return bool(names & {"DeadlineExceeded", "RetryError", "_MultiThreadedRendezvous"})


def _import_subscriber() -> Any:
    """Import ``pubsub_v1.SubscriberClient`` or raise a guiding ``BackendError``."""
    try:
        from google.cloud import pubsub_v1
    except ImportError as exc:
        raise BackendError(
            "reading from Pub/Sub needs the pubsub extra: pip install 'batcher-engine[pubsub]'"
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

    __slots__ = ("_client_obj", "_pending_acks", "_pull_timeout")

    def __init__(
        self,
        topic: str,
        *,
        poll_size: int = 16_384,
        partitions: list[int] | None = None,  # noqa: ARG002 (single logical partition)
        pull_timeout: float = 10.0,
        **options: Any,
    ) -> None:
        super().__init__(topic, poll_size=poll_size, **options)
        self._client_obj: Any = None
        # Ack ids pulled but not yet published, acked by `_commit_delivered`.
        self._pending_acks: list[str] = []
        # Bound a single pull so an idle subscription does not block the poll — and with it
        # the trigger cadence and `stop()` — indefinitely. A deadline hit with no messages is
        # treated as an empty poll (the loop simply polls again on the next trigger).
        self._pull_timeout = pull_timeout

    def _client(self) -> Any:
        if self._client_obj is None:
            subscriber_cls = _import_subscriber()
            self._client_obj = subscriber_cls()
        return self._client_obj

    def _discover_partitions(self) -> list[int]:
        return [0]  # Pub/Sub has no user-visible partitions.

    def _poll(self) -> list[BrokerMessage] | None:
        client = self._client()
        try:
            response = client.pull(
                request={
                    "subscription": self.topic,
                    "max_messages": min(self.poll_size, _PULL_MAX_MESSAGES),
                },
                timeout=self._pull_timeout,
            )
        except Exception as exc:
            if _is_pull_timeout(exc):
                return []  # idle subscription: no data within the deadline, poll again later
            raise
        received = response.received_messages
        messages = [
            BrokerMessage(
                value=rm.message.data,
                partition=0,
                offset=_stable_offset(rm.message.message_id),
                # Pub/Sub is not replayable by position — the ack id is what the *next*
                # commit needs, and it is held in `_pending_acks` rather than checkpointed.
                timestamp=int(rm.message.publish_time.timestamp() * 1000),
                topic=self.topic,
                key=(rm.message.ordering_key or "").encode("utf-8") or None,
            )
            for rm in received
        ]
        # Deliberately does not acknowledge. Assembling a batch is not publishing it: acking
        # here told Pub/Sub the messages were handled the moment they were *read*, so a crash
        # between the pull and the publish meant they were never redelivered and the data was
        # gone — at-most-once, and silent. The ack ids are held for `_commit_delivered`,
        # which runs only after the epoch is published.
        self._pending_acks.extend(rm.ack_id for rm in received)
        return messages

    def _commit_delivered(self) -> None:
        """Acknowledge the epoch that was just published.

        A crash before this re-delivers the batch, which an idempotent sink absorbs; the
        ordering favours a duplicate over a gap, because a gap cannot be recovered.
        """
        if not self._pending_acks:
            return
        client = self._client()
        pending = self._pending_acks
        # Chunked: a single `acknowledge` carrying a whole poll's ack ids exceeds the API's
        # 512 KiB request bound and fails, which fails the *commit* rather than the read —
        # so the epoch was published and then the acks were lost, and every message came
        # back on the next poll.
        for start in range(0, len(pending), _ACK_CHUNK):
            client.acknowledge(
                request={
                    "subscription": self.topic,
                    "ack_ids": pending[start : start + _ACK_CHUNK],
                }
            )
        self._pending_acks = []
