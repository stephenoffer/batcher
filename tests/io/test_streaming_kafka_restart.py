"""Kafka restart positions: every assigned partition is checkpointed, not just busy ones.

A checkpoint used to record only the partitions that had delivered a message, so a partition
that sat idle (or was added to the topic later) fell back to ``auto.offset.reset`` on restart.
Under ``starting_offsets="latest"`` that is the head *at restart*, and every message written to
it while the query was down was skipped without an error. These drive `KafkaSource` through a
group assignment against a fake consumer, so the offsets it assigns are directly visible.
"""

from __future__ import annotations

import sys
import types

import pytest

from batcher.io.formats.streaming.kafka import KafkaSource

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def confluent_kafka_stub():
    """Stand in for `confluent_kafka.TopicPartition` when the optional extra is absent."""
    if "confluent_kafka" in sys.modules:
        yield
        return
    module = types.ModuleType("confluent_kafka")

    class TopicPartition:
        def __init__(self, topic, partition, offset=-1001):
            self.topic = topic
            self.partition = partition
            self.offset = offset

    module.TopicPartition = TopicPartition
    sys.modules["confluent_kafka"] = module
    try:
        yield
    finally:
        del sys.modules["confluent_kafka"]


class _Rec:
    def __init__(self, partition: int, offset: int) -> None:
        self._partition = partition
        self._offset = offset

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


class _GroupConsumer:
    """A subscribe-mode consumer: the group hands it `partitions` on its first poll."""

    def __init__(self, partitions, watermarks, *, records=(), committed=None) -> None:
        self.partitions = list(partitions)
        self.watermarks = dict(watermarks)
        self.records = list(records)
        self.committed_offsets = dict(committed or {})
        self.assigned: dict[int, int] = {}
        self._on_assign = None

    def subscribe(self, topics, on_assign=None):
        self._on_assign = on_assign

    def rebalance(self) -> dict[int, int]:
        from confluent_kafka import TopicPartition

        self._on_assign(self, [TopicPartition("t", p) for p in self.partitions])
        return self.assigned

    def assign(self, partitions):
        self.assigned = {tp.partition: tp.offset for tp in partitions}

    def consume(self, num_messages, timeout):
        if not self.assigned:
            self.rebalance()
        out, self.records = self.records, []
        return out

    def get_watermark_offsets(self, tp, timeout=None):
        return self.watermarks[tp.partition]

    def committed(self, partitions, timeout=None):
        from confluent_kafka import TopicPartition

        return [
            TopicPartition("t", tp.partition, self.committed_offsets.get(tp.partition, -1001))
            for tp in partitions
        ]

    def commit(self, asynchronous):
        pass

    def close(self):
        pass


def _source(consumer: _GroupConsumer, **kwargs) -> KafkaSource:
    src = KafkaSource("t", **kwargs)
    src._consumer = consumer  # `_client()` hands this back instead of dialling a broker
    consumer.subscribe(["t"], on_assign=src._on_assign)
    return src


def test_an_idle_partition_is_checkpointed_so_a_restart_reads_what_landed_on_it():
    """Partition 1 delivered nothing in the first run. It still has to be in the checkpoint,
    at the head it started from, or the restart starts it at the *new* head and the ten
    messages written while the query was down are never read."""
    first = _GroupConsumer([0, 1], {0: (0, 5), 1: (0, 3)}, records=[_Rec(0, 5)])
    src = _source(first, starting_offsets="latest")
    next(src.iter_batches())
    assert first.assigned == {0: 5, 1: 3}  # "latest", resolved to the heads at start
    checkpoint = src.snapshot_position()
    assert checkpoint == {"offsets": {"0": 5, "1": 2}}

    # Down for a while: one more message on partition 0, ten on partition 1.
    restarted = _GroupConsumer([0, 1], {0: (0, 7), 1: (0, 13)})
    resumed = _source(restarted, starting_offsets="latest")
    resumed.seek(checkpoint)
    assert restarted.rebalance() == {0: 6, 1: 3}


def test_a_partition_added_after_the_checkpoint_starts_at_the_earliest_offset():
    """The query never read partition 2, so none of it was processed: it starts at the
    earliest offset still in the log, as Spark starts a new partition, not at the head."""
    consumer = _GroupConsumer([0, 1, 2], {0: (0, 9), 1: (0, 9), 2: (4, 20)})
    src = _source(consumer, starting_offsets="latest")
    src.seek({"offsets": {"0": 5, "1": 2}})
    assert consumer.rebalance() == {0: 6, 1: 3, 2: 4}
    # And it is carried in the next checkpoint even before it delivers anything.
    assert src.snapshot_position()["offsets"]["2"] == 3


def test_a_first_run_still_honours_the_groups_committed_offset_and_the_explicit_map():
    """Resolving every start concretely must not change *which* offset a fresh query starts
    at: the group's committed offset outranks the reset policy, as it does in the client, and
    an explicit `starting_offsets` entry outranks both."""
    consumer = _GroupConsumer([0, 1, 2], {0: (0, 9), 1: (0, 9), 2: (0, 9)}, committed={1: 4, 2: 6})
    src = _source(consumer, starting_offsets={"0": 2, "2": 1})
    assert consumer.rebalance() == {0: 2, 1: 4, 2: 1}
    assert src.snapshot_position() == {"offsets": {"0": 1, "1": 3, "2": 0}}


def test_a_restore_rolls_back_positions_read_after_the_checkpoint():
    """An in-process recovery seeks back to the last committed epoch. A partition first read
    in the abandoned epoch is not in that checkpoint, and must be re-read from where it began,
    not resumed after rows that were never published."""
    consumer = _GroupConsumer([0, 1], {0: (0, 9), 1: (0, 9)})
    src = _source(consumer, starting_offsets="earliest")
    src._positions[1] = 7  # delivered in the epoch that failed, never committed
    src.seek({"offsets": {"0": 3}})
    assert consumer.rebalance() == {0: 4, 1: 0}
