"""Apache Pulsar streaming sink — publish each micro-batch through ``pulsar-client``.

The write side of `pulsar.PulsarSource`, on the shared `BrokerStreamSink` contract. One
producer per destination topic is created on first use and kept for the query's life.
Records are sent with ``Producer.send_async`` and the micro-batch is reported written only
once ``Producer.flush`` has returned and every send callback has reported ``Result.Ok``.

**Ordering** is per partition. A row's ``key`` becomes the message's ``partition_key``, so
rows sharing a key route to the same partition of a partitioned topic and keep their order
there.

**Retry identity.** Pulsar offers broker-side message deduplication, keyed on a producer's
name and a monotonically increasing sequence id, when it is enabled for the namespace or
topic. Pass ``producer_name=`` and this sink sets each record's ``sequence_id`` to
``batch_id << 32 | row``, which increases across micro-batches and repeats exactly for a
replayed epoch, so the broker drops records it already persisted. Without
``producer_name`` the client picks the sequence ids and a replay republishes.

Not yet verified against a live Pulsar; see tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

import threading
from typing import Any

from batcher._internal.errors import IOError, PlanError
from batcher._internal.optional import require
from batcher.io.formats.streaming.broker_sinks.contract import (
    BrokerRecords,
    BrokerStreamSink,
    SinkCapabilities,
)
from batcher.io.formats.streaming.sinks import STREAM_SINKS

__all__ = ["PulsarStreamSink"]

#: Rows a micro-batch may hold when ``producer_name`` derives sequence ids from it: the row
#: index occupies the low 32 bits of the id.
_MAX_ROWS_PER_BATCH = 1 << 32


def _import_pulsar() -> Any:
    """Import ``pulsar`` or raise a guiding ``BackendError``."""
    return require("pulsar", feature="Pulsar support", provides="pulsar-client", extra="pulsar")


class _Acks:
    """Counts outstanding ``send_async`` callbacks and collects their failures.

    The callbacks run on the client's own thread, so the count is guarded and `wait`
    blocks on a condition rather than polling.
    """

    def __init__(self, ok: Any) -> None:
        self._ok = ok
        self._pending = 0
        self._failures: list[str] = []
        self._done = threading.Condition()

    def expect(self) -> None:
        with self._done:
            self._pending += 1

    def callback(self, result: Any, _message_id: Any) -> None:
        with self._done:
            if result != self._ok:
                self._failures.append(str(result))
            self._pending -= 1
            self._done.notify_all()

    def wait(self, timeout: float) -> tuple[int, list[str]]:
        """Block until every send is acknowledged or `timeout` passes."""
        with self._done:
            self._done.wait_for(lambda: self._pending <= 0, timeout=timeout)
            return self._pending, list(self._failures)


@STREAM_SINKS.register("pulsar")
class PulsarStreamSink(BrokerStreamSink):
    """Publish each micro-batch's rows to Pulsar as one message per row.

    Args:
        topic: Destination topic for rows whose ``topic`` column is null or absent.
        service_url: The Pulsar service URL, as for the source.
        producer_name: A stable producer name; enables sequence ids for broker-side
            deduplication (see the module docstring).
        auth_token: A JWT for token authentication, or a secret reference to one
            (``env:NAME``, ``file:PATH``, ...), resolved on the worker.
        flush_timeout: Seconds a micro-batch may wait for acknowledgements.
        dedup_ids: A stable writer name; each record then carries a ``batcher-dedup-id``
            property.
        options: Codec options (``value_format=`` ...) plus any further
            ``Client.create_producer`` keyword arguments, such as ``compression_type``.
    """

    capabilities = SinkCapabilities(
        broker="pulsar",
        display="Pulsar",
        columns=("value", "key", "topic", "headers"),
        ordering="per partition; the key is the partition_key, so a key keeps its order",
        batching="the client's producer batching (batching_enabled), flushed per micro-batch",
        retry_identity="broker-side deduplication on producer_name + sequence_id, when "
        "enabled on the namespace; dedup_ids= property otherwise",
    )

    def __init__(
        self,
        *,
        topic: str | None = None,
        service_url: str = "pulsar://localhost:6650",
        producer_name: str | None = None,
        auth_token: str | None = None,
        flush_timeout: float = 30.0,
        dedup_ids: str | None = None,
        **options: Any,
    ) -> None:
        codecs = self.split_codec_options(options)
        super().__init__(
            topic=topic, flush_timeout=flush_timeout, dedup_ids=dedup_ids, codec_options=codecs
        )
        self._service_url = service_url
        self._producer_name = producer_name
        self._auth_token = auth_token
        self._producer_options = options
        self._client: Any = None
        self._producers: dict[str, Any] = {}
        self._ok: Any = None

    def _connect(self) -> None:
        """Build the client; producers are created per destination on first use."""
        from batcher.io.credentials import resolve_secret

        pulsar = _import_pulsar()
        kwargs: dict[str, Any] = {}
        if self._auth_token is not None:
            token = resolve_secret(self._auth_token, what="Pulsar auth_token")
            kwargs["authentication"] = pulsar.AuthenticationToken(token)
        self._client = pulsar.Client(self._service_url, **kwargs)
        self._ok = pulsar.Result.Ok
        self._producers = {}

    def _producer(self, topic: str) -> Any:
        producer = self._producers.get(topic)
        if producer is None:
            kwargs = dict(self._producer_options)
            if self._producer_name is not None:
                kwargs["producer_name"] = self._producer_name
            # Block on a full client queue instead of failing the send: a full queue is
            # backpressure on a fast producer, not an error.
            kwargs.setdefault("block_if_queue_full", True)
            producer = self._client.create_producer(topic, **kwargs)
            self._producers[topic] = producer
        return producer

    def _publish(self, records: BrokerRecords, batch_id: int) -> None:
        if self._producer_name is not None and len(records) >= _MAX_ROWS_PER_BATCH:
            raise PlanError(
                f"pulsar sink: micro-batch {batch_id} has {len(records)} rows, more than "
                "producer_name= sequence ids can number; lower the trigger's batch size"
            )
        acks = _Acks(self._ok)
        used: set[str] = set()
        for i, value in enumerate(records.values):
            destination = records.destinations[i]
            producer = self._producer(destination)
            used.add(destination)
            kwargs: dict[str, Any] = {}
            key = records.text_key(i, broker="Pulsar")
            if key is not None:
                kwargs["partition_key"] = key
            properties = records.text_properties(i, broker="Pulsar")
            if properties:
                kwargs["properties"] = properties
            if self._producer_name is not None:
                kwargs["sequence_id"] = (batch_id << 32) | i
            acks.expect()
            producer.send_async(value, acks.callback, **kwargs)
        for destination in used:
            self._producers[destination].flush()
        pending, failures = acks.wait(self._flush_timeout)
        if pending:
            raise IOError(
                f"pulsar sink: {pending} message(s) of micro-batch {batch_id} were still "
                f"unacknowledged after {self._flush_timeout}s. The epoch is not durable; "
                "raise flush_timeout, or check broker reachability."
            )
        if failures:
            raise IOError(
                f"pulsar sink: {len(failures)} message(s) of micro-batch {batch_id} were "
                f"rejected by the broker; first: {failures[0]}"
            )

    def _disconnect(self) -> None:
        """Flush and close every producer, then the client. Idempotent."""
        producers, self._producers = self._producers, {}
        for producer in producers.values():
            producer.flush()
            producer.close()
        if self._client is not None:
            client, self._client = self._client, None
            client.close()
