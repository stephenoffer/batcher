"""A Kinesis checkpoint must survive a reshard — the identity is the shard, not its slot.

A checkpoint is ``{partition: sequence_number}``, and a Kinesis sequence number is valid
only against the shard that produced it. When `partition` was the shard's *position* in a
`list_shards` response, a reshard — a split or merge, the ordinary event partitioning
exists to absorb — shifted every position. A sequence checkpointed for position 2 was then
replayed with ``AFTER_SEQUENCE_NUMBER`` against whatever shard now occupied position 2:
records silently skipped or re-read across the restart, with no error.

The fix keys everything on the shard's stable number (parsed from its ShardId). These tests
drive a fake boto3 client through a snapshot → reshard → seek cycle and assert the resumed
read targets the *same* shards it checkpointed, and that a brand-new child shard starts from
the configured iterator type rather than inheriting a foreign sequence.
"""

from __future__ import annotations

import pytest

from batcher.io.formats.streaming.kinesis import KinesisSource, _shard_number

pytestmark = pytest.mark.unit


class _FakeKinesis:
    """A boto3-kinesis stand-in: a fixed shard set and a record of every iterator request."""

    def __init__(self, shard_ids: list[str]) -> None:
        self._shard_ids = shard_ids
        #: (shard_id, iterator_type, starting_sequence_or_None) for each get_shard_iterator.
        self.iterator_requests: list[tuple[str, str, str | None]] = []

    def list_shards(self, **kwargs):
        return {"Shards": [{"ShardId": s} for s in self._shard_ids]}

    def get_shard_iterator(self, **kwargs):
        self.iterator_requests.append(
            (kwargs["ShardId"], kwargs["ShardIteratorType"], kwargs.get("StartingSequenceNumber"))
        )
        return {"ShardIterator": f"iter-{kwargs['ShardId']}"}

    def get_records(self, **kwargs):
        return {"Records": [], "NextShardIterator": None}


def _source(fake: _FakeKinesis, **kw) -> KinesisSource:
    source = KinesisSource(topic="events", region="us-east-1", **kw)
    source._client_obj = fake  # inject the fake, bypassing boto3
    return source


def test_shard_number_parses_the_stable_suffix() -> None:
    assert _shard_number("shardId-000000000002") == 2
    assert _shard_number("shardId-000000000042") == 42


def test_shard_number_is_deterministic_for_a_nonstandard_id() -> None:
    """A stub or future scheme still resolves to one stable number, not a per-run hash."""
    assert _shard_number("weird-id") == _shard_number("weird-id")


def test_partition_is_the_shard_number_not_the_list_position() -> None:
    """`_discover_partitions` reports shard numbers; a two-shard stream at 0 and 2 says so."""
    fake = _FakeKinesis(["shardId-000000000000", "shardId-000000000002"])
    assert _source(fake)._discover_partitions() == [0, 2]


def test_a_checkpoint_resumes_the_same_shard_after_a_reshard() -> None:
    # Before: two shards, 0 and 1. We checkpoint a sequence for shard 1.
    before = _source(_FakeKinesis(["shardId-000000000000", "shardId-000000000001"]))
    before.seek({"offsets": {"1": "49590000000000000000000000001"}})

    # After a reshard, shard 1 has split into 2 and 3; the *positions* have shifted, but
    # shard 1's number has not. A fresh source (a restart) sees the new shard set.
    after_fake = _FakeKinesis(
        ["shardId-000000000000", "shardId-000000000002", "shardId-000000000003"]
    )
    after = _source(after_fake)
    after.seek({"offsets": {"1": "49590000000000000000000000001"}})
    after._poll()

    by_shard = {shard: (kind, seq) for shard, kind, seq in after_fake.iterator_requests}
    # Shard 1 is gone, so nothing resumes its foreign sequence against a live shard...
    assert "shardId-000000000001" not in by_shard
    # ...and the surviving/new shards were NOT handed shard 1's sequence.
    for _shard, kind, seq in after_fake.iterator_requests:
        assert (kind, seq) != ("AFTER_SEQUENCE_NUMBER", "49590000000000000000000000001")
    # The brand-new child shards start from the configured iterator type, not a checkpoint.
    assert by_shard["shardId-000000000002"][0] == "TRIM_HORIZON"
    assert by_shard["shardId-000000000003"][0] == "TRIM_HORIZON"


def test_the_correct_shard_resumes_after_its_sequence_when_it_survives() -> None:
    """When the checkpointed shard is still present, it resumes exactly — no replay."""
    fake = _FakeKinesis(["shardId-000000000000", "shardId-000000000001"])
    source = _source(fake)
    source.seek({"offsets": {"1": "49590000000000000000000000009"}})
    source._poll()

    by_shard = {shard: (kind, seq) for shard, kind, seq in fake.iterator_requests}
    assert by_shard["shardId-000000000001"] == (
        "AFTER_SEQUENCE_NUMBER",
        "49590000000000000000000000009",
    )
    # An unrelated shard with no checkpoint starts fresh.
    assert by_shard["shardId-000000000000"][0] == "TRIM_HORIZON"
