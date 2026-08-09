"""Kafka sink contracts: the column schema, the flush, and what a rejection does.

None of these need a broker. They drive `KafkaStreamSink` against a fake producer that
models the `confluent_kafka.Producer` surface the sink actually uses — `produce`,
`poll`, `flush`, and the delivery callback — which is where every behavior worth pinning
lives. A live-broker test would observe the same things only as "it worked".
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher._internal.errors import IOError, PlanError
from batcher.io.formats.streaming.kafka_sink import KafkaStreamSink
from batcher.io.formats.streaming.sinks import STREAM_SINKS


class _Msg:
    def __init__(self, topic: str) -> None:
        self._topic = topic

    def topic(self) -> str:
        return self._topic


class _FakeProducer:
    """Records what was produced; optionally fails delivery or leaves records queued."""

    def __init__(
        self,
        *,
        reject: int | None = None,
        unflushed: int = 0,
        buffer_full_for: int = 0,
    ) -> None:
        self.produced: list[dict] = []
        self.polls = 0
        self.flushes = 0
        self._reject = reject
        self._unflushed = unflushed
        self._buffer_full_for = buffer_full_for

    def produce(self, topic, **record):
        if self._buffer_full_for > 0:
            self._buffer_full_for -= 1
            raise BufferError("queue full")
        self.produced.append({"topic": topic, **record})

    def poll(self, _timeout):
        self.polls += 1
        return 0

    def flush(self, _timeout):
        self.flushes += 1
        for i, rec in enumerate(self.produced):
            callback = rec.get("on_delivery")
            if callback is None:
                continue
            failed = self._reject is not None and i == self._reject
            callback("broker said no" if failed else None, _Msg(rec["topic"]))
        return self._unflushed


def _sink(producer: _FakeProducer, **kwargs) -> KafkaStreamSink:
    # Skip `open()`: it constructs the real client. Injecting the fake is the same seam.
    sink = KafkaStreamSink(**kwargs)
    sink._producer = producer
    sink._reported = []
    return sink


def test_the_sink_is_registered_under_its_industry_name():
    assert STREAM_SINKS.get("kafka") is KafkaStreamSink


def test_a_table_with_no_value_column_is_refused_by_name():
    sink = _sink(_FakeProducer(), topic="out")
    with pytest.raises(PlanError, match="needs a 'value' column"):
        sink.write_batch(0, pa.table({"payload": ["a"]}))


def test_a_value_column_kafka_cannot_carry_is_refused_with_its_type():
    sink = _sink(_FakeProducer(), topic="out")
    with pytest.raises(PlanError, match="must be binary or string, not int64"):
        sink.write_batch(0, pa.table({"value": [1, 2]}))


def test_no_destination_at_all_is_refused():
    sink = _sink(_FakeProducer())
    with pytest.raises(PlanError, match="needs a destination"):
        sink.write_batch(0, pa.table({"value": ["a"]}))


def test_a_topic_column_overrides_the_default_destination_per_row():
    producer = _FakeProducer()
    sink = _sink(producer, topic="fallback")
    sink.write_batch(0, pa.table({"value": ["a", "b"], "topic": ["other", None]}))
    assert [r["topic"] for r in producer.produced] == ["other", "fallback"]


def test_string_payloads_are_utf8_encoded_and_binary_passes_through():
    producer = _FakeProducer()
    sink = _sink(producer, topic="out")
    sink.write_batch(0, pa.table({"value": ["hé"], "key": [b"k"]}))
    assert producer.produced[0]["value"] == "hé".encode()
    assert producer.produced[0]["key"] == b"k"


def test_a_null_key_is_omitted_rather_than_sent_as_none():
    producer = _FakeProducer()
    sink = _sink(producer, topic="out")
    sink.write_batch(0, pa.table({"value": ["a"], "key": pa.array([None], type=pa.string())}))
    assert "key" not in producer.produced[0]


def test_an_explicit_partition_column_is_passed_through():
    producer = _FakeProducer()
    sink = _sink(producer, topic="out")
    sink.write_batch(0, pa.table({"value": ["a"], "partition": pa.array([3], type=pa.int32())}))
    assert producer.produced[0]["partition"] == 3


def test_headers_reach_the_client_as_key_value_pairs():
    producer = _FakeProducer()
    sink = _sink(producer, topic="out")
    headers = pa.array(
        [[{"key": "trace", "value": b"abc"}]],
        type=pa.list_(pa.struct([("key", pa.string()), ("value", pa.binary())])),
    )
    sink.write_batch(0, pa.table({"value": ["a"], "headers": headers}))
    assert producer.produced[0]["headers"] == [("trace", b"abc")]


def test_every_micro_batch_is_flushed_before_it_is_reported_written():
    """`produce` only enqueues. A sink that returned without flushing would report a
    micro-batch durable while its records sat in a client queue a crash discards."""
    producer = _FakeProducer()
    sink = _sink(producer, topic="out")
    token = sink.write_batch(7, pa.table({"value": ["a", "b"]}))
    assert producer.flushes == 1
    assert token == "kafka:out:7:2"


def test_a_broker_rejection_fails_the_micro_batch():
    producer = _FakeProducer(reject=1)
    sink = _sink(producer, topic="out")
    with pytest.raises(IOError, match="rejected by the broker"):
        sink.write_batch(0, pa.table({"value": ["a", "b"]}))


def test_a_rejection_does_not_poison_the_next_micro_batch():
    producer = _FakeProducer(reject=0)
    sink = _sink(producer, topic="out")
    with pytest.raises(IOError):
        sink.write_batch(0, pa.table({"value": ["a"]}))
    producer.produced.clear()
    producer._reject = None
    assert sink.write_batch(1, pa.table({"value": ["b"]})) == "kafka:out:1:1"


def test_records_still_unacknowledged_after_the_timeout_fail_the_epoch():
    producer = _FakeProducer(unflushed=2)
    sink = _sink(producer, topic="out", flush_timeout=0.01)
    with pytest.raises(IOError, match="unacknowledged"):
        sink.write_batch(0, pa.table({"value": ["a", "b"]}))


def test_a_full_client_queue_is_backpressure_not_a_failure():
    """librdkafka raises BufferError when its local queue is full — routine on a fast
    producer. Failing the epoch on it would kill a query for a millisecond condition."""
    producer = _FakeProducer(buffer_full_for=2)
    sink = _sink(producer, topic="out")
    sink.write_batch(0, pa.table({"value": ["a", "b"]}))
    assert producer.polls == 2
    assert len(producer.produced) == 2


def test_an_empty_micro_batch_produces_nothing_but_still_reports():
    producer = _FakeProducer()
    sink = _sink(producer, topic="out")
    assert (
        sink.write_batch(4, pa.table({"value": pa.array([], type=pa.string())})) == "kafka:out:4:0"
    )
    assert producer.produced == []


def test_a_nonpositive_flush_timeout_is_refused_at_construction():
    with pytest.raises(PlanError, match="flush_timeout"):
        KafkaStreamSink(topic="out", flush_timeout=0)


def test_close_flushes_and_is_idempotent():
    producer = _FakeProducer()
    sink = _sink(producer, topic="out")
    sink.close()
    sink.close()
    assert producer.flushes == 1


def test_producer_options_become_dotted_client_config():
    sink = KafkaStreamSink(topic="out", bootstrap_servers="b:9092", compression_type="zstd")
    assert sink._config == {"bootstrap.servers": "b:9092", "compression.type": "zstd"}
