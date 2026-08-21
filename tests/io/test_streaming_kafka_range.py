"""Bounded Kafka reads: `ending_offsets` turns a topic into an offset range.

Without it a Kafka source can only be consumed by a streaming query, and `collect()` on one
correctly refuses because it could never terminate — so a backfill or a reprocess over a
known range had no spelling at all. These hold the range's edges, which is where a bounded
read of an unbounded thing goes wrong: one row too many, one row too few, or a partition
that never reports itself done and hangs the read.
"""

from __future__ import annotations

import sys
import types

import pytest

from batcher._internal.errors import PlanError
from batcher.io.formats.streaming.kafka import KafkaSource

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def confluent_kafka_stub():
    """Stand in for `confluent_kafka`'s value types, which the range path constructs.

    Only `TopicPartition` and the two offset sentinels are needed: they are plain locators
    the source hands to the consumer, so a stub is a faithful stand-in and lets the range
    logic be tested without a broker or the optional extra. A real installation is left
    alone — the stub is only inserted when the module is genuinely absent.
    """
    if "confluent_kafka" in sys.modules:
        yield
        return
    module = types.ModuleType("confluent_kafka")

    class TopicPartition:
        def __init__(self, topic, partition, offset=None):
            self.topic = topic
            self.partition = partition
            self.offset = offset

    module.TopicPartition = TopicPartition
    module.OFFSET_BEGINNING = -2
    module.OFFSET_END = -1
    sys.modules["confluent_kafka"] = module
    try:
        yield
    finally:
        del sys.modules["confluent_kafka"]


class _Rec:
    """One client record, in the shape `confluent_kafka.Message` presents."""

    def __init__(self, offset: int, partition: int = 0) -> None:
        self._offset = offset
        self._partition = partition

    def error(self):
        return None

    def value(self):
        return b"v"

    def key(self):
        return None

    def len(self):
        return 1

    def partition(self):
        return self._partition

    def offset(self):
        return self._offset

    def timestamp(self):
        return (1, 1000 + self._offset)

    def topic(self):
        return "t"


class _RangeConsumer:
    """A consumer with declared watermarks, so end resolution is testable with no broker."""

    def __init__(self, batches, watermarks) -> None:
        self.batches = list(batches)
        self.watermarks = dict(watermarks)
        self.paused: list[int] = []

    def consume(self, num_messages, timeout):
        return self.batches.pop(0) if self.batches else []

    def get_watermark_offsets(self, tp, timeout=None):
        return self.watermarks[tp.partition]

    def pause(self, partitions):
        self.paused.extend(tp.partition for tp in partitions)

    def commit(self, asynchronous):
        pass

    def close(self):
        pass


def _source(consumer, **kwargs) -> KafkaSource:
    """A Kafka source wired to `consumer`, so `_client()` never dials a broker."""
    source = KafkaSource("t", partitions=[0], **kwargs)
    source._consumer = consumer
    return source


# --- what the option means -------------------------------------------------


def test_a_topic_is_unbounded_until_an_end_is_declared():
    assert KafkaSource("t", partitions=[0]).bounded is False


def test_declaring_an_end_makes_the_read_bounded_before_any_client_exists():
    """`bounded` decides at plan time whether a terminal is a collect or a streaming query,
    so it must not depend on reaching the cluster."""
    assert KafkaSource("t", partitions=[0], ending_offsets={0: 100}).bounded is True


def test_an_unbounded_kafka_source_still_refuses_to_materialize():
    with pytest.raises(PlanError, match="unbounded stream"):
        KafkaSource("t", partitions=[0]).read()


def test_earliest_is_refused_as_an_ending_position():
    with pytest.raises(PlanError, match="ending_offsets"):
        KafkaSource("t", partitions=[0], ending_offsets="earliest")


def test_a_non_mapping_end_is_refused_by_type():
    with pytest.raises(PlanError, match="ending_offsets must be"):
        KafkaSource("t", partitions=[0], ending_offsets=42)


def test_the_spark_nested_topic_form_is_accepted():
    source = KafkaSource("t", partitions=[0], ending_offsets={"t": {"0": 7}})
    assert source._end_spec == (False, {0: 7})


# --- the range's edges -----------------------------------------------------


def test_the_end_offset_is_exclusive():
    """Spark's semantics: an end of 3 reads offsets 0, 1, 2 and not 3."""
    consumer = _RangeConsumer([[_Rec(i) for i in range(5)]], {0: (0, 5)})
    source = _source(consumer, ending_offsets={0: 3}, poll_size=100)
    assert [m.offset for m in source._poll()] == [0, 1, 2]


def test_the_read_ends_once_every_partition_reaches_its_end():
    consumer = _RangeConsumer([[_Rec(i) for i in range(3)]], {0: (0, 5)})
    source = _source(consumer, ending_offsets={0: 3}, poll_size=100)
    assert len(source._poll()) == 3
    assert source._poll() is None, "a drained range must end the poll loop"


def test_the_last_rows_are_not_discarded_by_the_end_of_the_range():
    """Returning None on the poll that completed the range would drop its final rows."""
    consumer = _RangeConsumer([[_Rec(i) for i in range(3)]], {0: (0, 3)})
    source = _source(consumer, ending_offsets="latest", poll_size=100)
    assert [m.offset for m in source._poll()] == [0, 1, 2]
    assert source._poll() is None


def test_latest_resolves_to_the_head_as_of_the_first_poll():
    """A partition that keeps growing during the read must not extend the range, or a
    backfill would not be reproducible."""
    consumer = _RangeConsumer([[_Rec(0), _Rec(1)], [_Rec(2), _Rec(3)]], {0: (0, 2)})
    source = _source(consumer, ending_offsets="latest", poll_size=100)
    assert [m.offset for m in source._poll()] == [0, 1]
    assert source._poll() is None


def test_a_range_entirely_behind_retention_finishes_instead_of_hanging():
    """A `collect()` waiting for a message that retention already deleted reads as a hang."""
    consumer = _RangeConsumer([[]], {0: (100, 200)})
    source = _source(consumer, ending_offsets={0: 50}, poll_size=100)
    assert source._poll() == []
    assert source._poll() is None


def test_a_partition_outside_the_range_does_not_hold_the_read_open():
    consumer = _RangeConsumer(
        [[_Rec(0, partition=0), _Rec(0, partition=1)]], {0: (0, 10), 1: (0, 10)}
    )
    source = KafkaSource("t", partitions=[0, 1], ending_offsets={0: 1}, poll_size=100)
    source._consumer = consumer
    kept = source._poll()
    assert [(m.partition, m.offset) for m in kept] == [(0, 0)], (
        "partition 1 was not named in the range, so its rows are not read"
    )
    assert source._poll() is None, "and it does not hold the read open either"


def test_a_retired_partition_is_paused_so_the_client_stops_fetching_it():
    consumer = _RangeConsumer([[_Rec(0, 0), _Rec(1, 0), _Rec(0, 1)]], {0: (0, 10), 1: (0, 10)})
    source = KafkaSource("t", partitions=[0, 1], ending_offsets={0: 2, 1: 5}, poll_size=100)
    source._consumer = consumer
    source._poll()
    assert consumer.paused == [0], "only the finished partition is paused"


def test_a_client_that_cannot_pause_still_returns_the_right_rows():
    """Pausing is a throughput optimization, so it must never be load-bearing."""

    class _NoPause(_RangeConsumer):
        def pause(self, partitions):
            raise NotImplementedError

    consumer = _NoPause([[_Rec(i) for i in range(5)]], {0: (0, 5)})
    source = _source(consumer, ending_offsets={0: 2}, poll_size=100)
    assert [m.offset for m in source._poll()] == [0, 1]


# --- the distributed path --------------------------------------------------


def test_a_split_carries_the_resolved_ends_so_every_worker_stops_at_the_same_offsets():
    """Re-resolving "latest" per worker makes the answer depend on scheduling."""
    consumer = _RangeConsumer([[_Rec(0)]], {0: (0, 4)})
    source = _source(consumer, ending_offsets="latest", poll_size=100)
    source._poll()  # resolves the head
    assert source._split_options()["ending_offsets"] == {0: 4}


def test_a_split_of_an_unresolved_range_still_carries_the_declared_ends():
    source = KafkaSource("t", partitions=[0], ending_offsets={0: 9})
    assert source._split_options()["ending_offsets"] == {0: 9}


def test_a_split_of_an_unbounded_source_declares_no_end():
    source = KafkaSource("t", partitions=[0])
    assert "ending_offsets" not in source._split_options()
