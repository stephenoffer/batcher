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
on a key.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError, IOError, PlanError
from batcher.io.formats.streaming.sinks import STREAM_SINKS

__all__ = ["KafkaStreamSink"]

#: The only column a Kafka write cannot proceed without. `key`, `topic`, `partition` and
#: `headers` are optional, exactly as in Spark's Kafka sink schema.
_REQUIRED = ("value",)

#: Arrow types a payload column (`key` / `value`) may have. Kafka carries bytes; a string
#: column is UTF-8 encoded on the way out, which is what every other client does too.
_PAYLOAD_TYPES = ("binary", "large_binary", "string", "large_string")


def _import_producer() -> Any:
    """Import ``confluent_kafka.Producer`` or raise a guiding ``BackendError``."""
    try:
        from confluent_kafka import Producer
    except ImportError as exc:
        raise BackendError(
            "writing to Kafka needs the kafka extra: pip install 'batcher-engine[kafka]'"
        ) from exc
    return Producer


def _payload_column(table: pa.Table, name: str) -> list[bytes | None]:
    """A payload column as Python bytes, UTF-8 encoding a string column on the way.

    One vectorized `to_pylist` per column rather than a per-row read: the encode is the
    user's chosen wire format, not engine work, and `confluent-kafka` takes one record at
    a time regardless.
    """
    column = table.column(name)
    if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
        return [None if v is None else v.encode("utf-8") for v in column.to_pylist()]
    return column.to_pylist()


def _record_headers(value: Any) -> list[tuple[str, bytes | None]]:
    """One row's `headers` column as the ``[(key, bytes)]`` confluent-kafka wants.

    Two shapes are accepted because both are natural to produce with expressions: a map
    column arrives as a list of ``(key, value)`` pairs, and a
    ``list<struct<key, value>>`` column arrives as a list of dicts. Anything else is left
    to the client to reject, which it does by name.
    """
    headers: list[tuple[str, bytes | None]] = []
    for item in value:
        if isinstance(item, dict):
            key, payload = item.get("key"), item.get("value")
        else:
            key, payload = item
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        headers.append((str(key), payload))
    return headers


def _check_payload_type(schema: pa.Schema, name: str) -> None:
    """Reject a `key`/`value` column Kafka cannot carry, naming the column and its type."""
    index = schema.get_field_index(name)
    if index < 0:
        return
    field_type = schema.field(index).type
    if str(field_type) not in _PAYLOAD_TYPES:
        raise PlanError(
            f"the Kafka sink's {name!r} column must be binary or string, not {field_type}; "
            f"serialize it first, e.g. .with_columns({name}=col({name}).cast('string'))"
        )


@STREAM_SINKS.register("kafka")
class KafkaStreamSink:
    """Publish each micro-batch's rows to Kafka as one message per row.

    Args:
        topic: Destination topic for rows whose `topic` column is null or absent. A
            stream that always carries a `topic` column may omit it.
        bootstrap_servers: The Kafka bootstrap servers, as for the source.
        flush_timeout: Seconds `write_batch` waits for outstanding deliveries before
            declaring the micro-batch failed.
        options: Any further ``confluent-kafka`` producer configuration. Underscores
            become dots, so ``compression_type="zstd"`` sets ``compression.type``.
    """

    __slots__ = (
        "_config",
        "_flush_timeout",
        "_producer",
        "_reported",
        "_topic",
    )

    def __init__(
        self,
        *,
        topic: str | None = None,
        bootstrap_servers: str = "localhost:9092",
        flush_timeout: float = 30.0,
        **options: Any,
    ) -> None:
        if flush_timeout <= 0:
            raise PlanError(f"kafka sink flush_timeout must be > 0, got {flush_timeout}")
        self._topic = topic
        self._flush_timeout = flush_timeout
        self._config = {
            "bootstrap.servers": bootstrap_servers,
            **{k.replace("_", "."): v for k, v in options.items()},
        }
        self._producer: Any = None
        # Delivery failures reported by the client's callback thread since the last flush.
        # A `produce()` that succeeds has not delivered anything yet, so this is the only
        # place a broker-side rejection can be observed.
        self._reported: list[str] = []

    def open(self) -> None:
        """Construct the producer. Deferred to here so a plan can be built without the extra."""
        self._producer = _import_producer()(self._config)
        self._reported = []

    def write_batch(self, batch_id: int, table: pa.Table) -> str | None:
        """Publish every row, wait for the acknowledgements, and report what was written.

        The flush is what makes the epoch's report honest: `produce` only enqueues, so a
        sink that returned as soon as the loop finished would tell the engine a
        micro-batch was durable while its records were still in a client-side queue that a
        crash discards. Waiting here costs one round trip per micro-batch and turns a
        silent loss into a failed query the checkpoint can replay.

        Args:
            batch_id: The micro-batch's id, used only in the returned token.
            table: The micro-batch's output.

        Returns:
            A ``kafka:<topic>:<batch_id>:<rows>`` receipt for the commit log.

        Raises:
            PlanError: If the table has no `value` column, or a payload column has a type
                Kafka cannot carry.
            IOError: If any record was rejected, or deliveries did not complete within
                `flush_timeout`.
        """
        self._validate(table.schema)
        if table.num_rows == 0:
            return f"kafka:{self._topic}:{batch_id}:0"
        self._produce_all(table)
        remaining = self._producer.flush(self._flush_timeout)
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
        return f"kafka:{self._topic}:{batch_id}:{table.num_rows}"

    def close(self) -> None:
        """Flush anything still queued and drop the producer. Idempotent."""
        if self._producer is None:
            return
        producer, self._producer = self._producer, None
        producer.flush(self._flush_timeout)

    # --- internals --------------------------------------------------------
    def _validate(self, schema: pa.Schema) -> None:
        """Refuse a table Kafka cannot carry, before a single record is enqueued."""
        missing = [c for c in _REQUIRED if schema.get_field_index(c) < 0]
        if missing:
            raise PlanError(
                f"the Kafka sink needs a {missing[0]!r} column; the write schema is "
                f"{schema.names}. Project one, e.g. "
                ".select(value=col('payload').cast('string'))"
            )
        if self._topic is None and schema.get_field_index("topic") < 0:
            raise PlanError(
                "the Kafka sink needs a destination: pass topic=... to write.kafka(), or "
                "project a 'topic' column"
            )
        for name in ("key", "value"):
            _check_payload_type(schema, name)

    def _produce_all(self, table: pa.Table) -> None:
        """Enqueue one record per row, servicing the delivery queue as it fills."""
        values = _payload_column(table, "value")
        keys = _payload_column(table, "key") if table.schema.get_field_index("key") >= 0 else None
        topics = (
            table.column("topic").to_pylist()
            if table.schema.get_field_index("topic") >= 0
            else None
        )
        partitions = (
            table.column("partition").to_pylist()
            if table.schema.get_field_index("partition") >= 0
            else None
        )
        headers = (
            table.column("headers").to_pylist()
            if table.schema.get_field_index("headers") >= 0
            else None
        )
        producer = self._producer
        for i, value in enumerate(values):
            record: dict[str, Any] = {"value": value}
            if keys is not None and keys[i] is not None:
                record["key"] = keys[i]
            if partitions is not None and partitions[i] is not None:
                record["partition"] = int(partitions[i])
            if headers is not None and headers[i]:
                record["headers"] = _record_headers(headers[i])
            destination = self._topic
            if topics is not None and topics[i] is not None:
                destination = topics[i]
            self._produce_one(producer, destination, record)

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
