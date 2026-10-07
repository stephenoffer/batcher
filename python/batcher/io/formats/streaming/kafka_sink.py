"""Kafka streaming sink — publish each micro-batch to a topic (Spark ``format("kafka")``).

The write side of `kafka.KafkaSource`, and the sink a stream-processing pipeline ends in
when its output is another stream rather than a table. Backed by ``confluent-kafka`` (the
optional ``kafka`` extra), the same client the source uses.

**The column contract is Spark's**, so a ported job needs no reshaping: ``value`` is
required, ``key`` / ``topic`` / ``partition`` / ``headers`` are optional, and a
``topic=`` option supplies the destination for rows that do not carry one. String and
binary are both accepted for the payload columns; anything else is refused at open time
with the column and its type named, rather than at the first `produce` call from inside a
delivery callback where the message is lost.

**Delivery is at-least-once, and says so.** Kafka's exactly-once story is transactional
produce, which requires the consumer of this topic to read committed-only *and* the
producer to own the transaction across the engine's own commit — a coupling Batcher does
not have and Spark's Kafka sink does not attempt either. What this sink does guarantee is
that a micro-batch is fully acknowledged before it is reported as written: `write_batch`
flushes and fails the query if any record was rejected, so a replayed epoch republishes
its rows rather than losing them. Downstream consumers must be idempotent, or must dedup
on a key, or on the ``batcher-dedup-id`` header ``dedup_ids=`` attaches.

The column contract, the codecs, the dedup header and the acknowledgement rule are the
shared `broker_sinks.contract.BrokerStreamSink`; this module is only the librdkafka client.
"""

from __future__ import annotations

from typing import Any

from batcher._internal.errors import IOError
from batcher._internal.optional import require
from batcher.io.formats.streaming.broker_sinks.contract import (
    BrokerRecords,
    BrokerStreamSink,
    SinkCapabilities,
)
from batcher.io.formats.streaming.sinks import STREAM_SINKS

__all__ = ["KafkaStreamSink"]


def _import_producer() -> Any:
    """Import ``confluent_kafka.Producer`` or raise a guiding ``BackendError``."""
    return require(
        "confluent_kafka",
        "Producer",
        feature="Kafka support",
        provides="confluent-kafka",
        extra="kafka",
    )


@STREAM_SINKS.register("kafka")
class KafkaStreamSink(BrokerStreamSink):
    """Publish each micro-batch's rows to Kafka as one message per row.

    Args:
        topic: Destination topic for rows whose `topic` column is null or absent. A
            stream that always carries a `topic` column may omit it.
        bootstrap_servers: The Kafka bootstrap servers, as for the source.
        flush_timeout: Seconds `write_batch` waits for outstanding deliveries before
            declaring the micro-batch failed.
        dedup_ids: A stable writer name; each record then carries a ``batcher-dedup-id``
            header ``<name>:<batch id>:<row>``.
        options: Any further ``confluent-kafka`` producer configuration. Underscores
            become dots, so ``compression_type="zstd"`` sets ``compression.type``.
    """

    capabilities = SinkCapabilities(
        broker="kafka",
        display="Kafka",
        columns=("value", "key", "topic", "partition", "headers"),
        ordering="per partition; rows with the same key hash to the same partition",
        batching="librdkafka's client-side batches (linger.ms, batch.size), flushed per "
        "micro-batch",
        retry_identity="idempotent producer via enable_idempotence=True (dedups the "
        "client's own retries within one producer session); dedup_ids= header for replays",
        null_values=True,
        max_attempts_option="retries",
    )

    def __init__(
        self,
        *,
        topic: str | None = None,
        bootstrap_servers: str = "localhost:9092",
        flush_timeout: float = 30.0,
        dedup_ids: str | None = None,
        **options: Any,
    ) -> None:
        # The same codec vocabulary the source reads with, so a pipeline that decodes Avro
        # off one topic and writes Avro to another names the format once per side rather
        # than hand-rolling a serializer in a `map_batches`.
        codecs = self.split_codec_options(options)
        super().__init__(
            topic=topic, flush_timeout=flush_timeout, dedup_ids=dedup_ids, codec_options=codecs
        )
        self._config = {
            "bootstrap.servers": bootstrap_servers,
            **{k.replace("_", "."): v for k, v in options.items()},
        }
        self._producer: Any = None
        # Delivery failures reported by the client's callback thread since the last flush.
        # A `produce()` that succeeds has not delivered anything yet, so this is the only
        # place a broker-side rejection can be observed.
        self._reported: list[str] = []

    def _connect(self) -> None:
        """Construct the producer, resolving secret references on this worker."""
        from batcher.io.credentials import resolve_client_secrets
        from batcher.io.formats.streaming.broker.schema import _BROKER_SECRET_HINTS

        self._producer = _import_producer()(
            resolve_client_secrets(self._config, what="kafka", hints=_BROKER_SECRET_HINTS)
        )
        self._reported = []

    def _publish(self, records: BrokerRecords, batch_id: int) -> None:
        """Enqueue every record, flush, and fail the epoch on anything unacknowledged.

        The flush is what makes the epoch's report honest: `produce` only enqueues, so a
        sink that returned as soon as the loop finished would tell the engine a
        micro-batch was durable while its records were still in a client-side queue that a
        crash discards. Waiting here costs one round trip per micro-batch and turns a
        silent loss into a failed query the checkpoint can replay.
        """
        producer = self._producer
        for i, value in enumerate(records.values):
            record: dict[str, Any] = {"value": value}
            if records.keys is not None and records.keys[i] is not None:
                record["key"] = records.keys[i]
            if records.partitions is not None and records.partitions[i] is not None:
                record["partition"] = int(records.partitions[i])
            headers = records.header_pairs(i)
            if headers:
                record["headers"] = headers
            self._produce_one(producer, records.destinations[i], record)
        remaining = producer.flush(self._flush_timeout)
        if remaining:
            raise IOError(
                f"kafka sink: {remaining} message(s) of micro-batch {batch_id} were still "
                f"unacknowledged after {self._flush_timeout}s. The epoch is not durable; "
                "raise flush_timeout, or check broker reachability and acks/retries config."
            )
        if self._reported:
            failures, self._reported = self._reported, []
            raise IOError(
                f"kafka sink: {len(failures)} message(s) of micro-batch {batch_id} were "
                f"rejected by the broker; first: {failures[0]}"
            )

    def _disconnect(self) -> None:
        """Flush anything still queued and drop the producer. Idempotent."""
        if self._producer is None:
            return
        producer, self._producer = self._producer, None
        producer.flush(self._flush_timeout)

    def _produce_one(self, producer: Any, topic: str, record: dict[str, Any]) -> None:
        """Enqueue one record, draining the client queue when it is full rather than failing.

        ``BufferError`` is librdkafka saying its local queue is at ``queue.buffering.max.
        messages`` — routine backpressure on a fast producer, not an error. Polling gives
        the delivery callbacks a chance to run and free slots; without this, a micro-batch
        larger than the client's queue failed the whole epoch on a condition that resolves
        in milliseconds.
        """
        while True:
            try:
                producer.produce(topic, on_delivery=self._on_delivery, **record)
                return
            except BufferError:
                producer.poll(0.5)

    def _on_delivery(self, err: Any, msg: Any) -> None:
        """Record a broker-side rejection; the flush turns it into a failed micro-batch."""
        if err is not None:
            self._reported.append(f"{msg.topic() if msg is not None else '?'}: {err}")
