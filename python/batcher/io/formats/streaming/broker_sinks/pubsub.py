"""Google Cloud Pub/Sub streaming sink — publish each micro-batch through ``google-cloud-pubsub``.

The write side of `pubsub.PubSubSource`, on the shared `BrokerStreamSink` contract. Each
row becomes one ``PublisherClient.publish`` call, whose returned future is awaited before
the micro-batch is reported written. The client batches the publishes itself (its
``BatchSettings``), so one micro-batch goes out as a handful of requests.

**Ordering** is per ordering key, and only with ``ordered=True``. That builds the publisher
with ``PublisherOptions(enable_message_ordering=True)`` and sends each row's ``key`` as its
``ordering_key``; the subscription must also have message ordering enabled for a consumer
to see the order. When a publish with an ordering key fails, the client pauses that key,
so this sink calls ``resume_publish`` for it before failing the epoch, letting the replay
publish again. Pub/Sub has no message key otherwise, so a ``key`` column without
``ordered=True`` is refused rather than dropped.

**Retry identity.** Pub/Sub's exactly-once delivery deduplicates redelivery to a
subscriber, not a publisher's republish, so a replayed epoch publishes again. ``dedup_ids=``
adds a ``batcher-dedup-id`` attribute to deduplicate on downstream. Headers become message
attributes, which Pub/Sub carries as text.

Not yet verified against a live Pub/Sub; see tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

import time
from typing import Any

import pyarrow as pa

from batcher._internal.errors import IOError, PlanError
from batcher._internal.optional import require
from batcher.io.formats.streaming.broker_sinks.contract import (
    BrokerRecords,
    BrokerStreamSink,
    SinkCapabilities,
)
from batcher.io.formats.streaming.sinks import STREAM_SINKS

__all__ = ["PubSubStreamSink"]


def _import_pubsub() -> Any:
    """Import ``google.cloud.pubsub_v1`` or raise a guiding ``BackendError``."""
    return require(
        "google.cloud",
        "pubsub_v1",
        feature="Pub/Sub support",
        provides="google-cloud-pubsub",
        extra="pubsub",
    )


@STREAM_SINKS.register("pubsub")
class PubSubStreamSink(BrokerStreamSink):
    """Publish each micro-batch's rows to a Pub/Sub topic, one message per row.

    Args:
        topic: The full topic path (``projects/<project>/topics/<topic>``) for rows whose
            ``topic`` column is null or absent.
        ordered: Publish each row's ``key`` as its ordering key.
        flush_timeout: Seconds a micro-batch may wait for its publish futures.
        dedup_ids: A stable writer name; each message then carries a ``batcher-dedup-id``
            attribute.
        options: Codec options (``value_format=`` ...).
    """

    capabilities = SinkCapabilities(
        broker="pubsub",
        display="Pub/Sub",
        columns=("value", "key", "topic", "headers"),
        ordering="per ordering key, only with ordered=True (the key column is the ordering "
        "key; the subscription must enable ordering)",
        batching="the publisher client's BatchSettings; every future awaited per micro-batch",
        retry_identity="none on publish; dedup_ids= attribute for downstream dedup",
    )

    def __init__(
        self,
        *,
        topic: str | None = None,
        ordered: bool = False,
        flush_timeout: float = 60.0,
        dedup_ids: str | None = None,
        **options: Any,
    ) -> None:
        codecs = self.split_codec_options(options)
        if options:
            raise PlanError(f"pubsub sink: unknown option(s) {sorted(options)}")
        super().__init__(
            topic=topic, flush_timeout=flush_timeout, dedup_ids=dedup_ids, codec_options=codecs
        )
        self._ordered = ordered
        self._client: Any = None

    def _connect(self) -> None:
        pubsub_v1 = _import_pubsub()
        if self._ordered:
            options = pubsub_v1.types.PublisherOptions(enable_message_ordering=True)
            self._client = pubsub_v1.PublisherClient(publisher_options=options)
        else:
            self._client = pubsub_v1.PublisherClient()

    def _validate_extra(self, table: pa.Table) -> None:
        if not self._ordered and table.schema.get_field_index("key") >= 0:
            raise PlanError(
                "the Pub/Sub sink has no message key: pass ordered=True to publish the 'key' "
                "column as the ordering key, or drop the column"
            )

    def _publish(self, records: BrokerRecords, batch_id: int) -> None:
        futures: list[tuple[Any, str, str]] = []
        for i, value in enumerate(records.values):
            attributes = records.text_properties(i, broker="Pub/Sub")
            key = (records.text_key(i, broker="Pub/Sub") or "") if self._ordered else ""
            destination = records.destinations[i]
            kwargs: dict[str, Any] = dict(attributes)
            if key:
                kwargs["ordering_key"] = key
            futures.append((self._client.publish(destination, value, **kwargs), destination, key))
        deadline = time.monotonic() + self._flush_timeout
        failures: list[str] = []
        for future, destination, key in futures:
            try:
                future.result(timeout=max(deadline - time.monotonic(), 0.0))
            except Exception as exc:
                failures.append(f"{destination}: {exc}")
                if key:
                    # The client pauses an ordering key after a failed publish; unpause it so
                    # the replayed epoch can publish that key again.
                    self._client.resume_publish(destination, key)
        if failures:
            raise IOError(
                f"pubsub sink: {len(failures)} message(s) of micro-batch {batch_id} were not "
                f"published; first: {failures[0]}. The epoch is not durable."
            )

    def _disconnect(self) -> None:
        if self._client is None:
            return
        client, self._client = self._client, None
        stop = getattr(client, "stop", None)
        if callable(stop):
            stop()
