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
    """A boto3-kinesis stand-in: a mutable shard set and a record of every iterator request.

    ``shards`` entries are either a bare id or ``(id, *parent_ids)``, so a test can describe
    the lineage a reshard produces — the ``ParentShardId`` / ``AdjacentParentShardId`` the
    real ``list_shards`` reports and that the reader follows to adopt a child.
    """

    def __init__(self, shards: list[str | tuple[str, ...]]) -> None:
        self.shards = list(shards)
        #: (shard_id, iterator_type, starting_sequence_or_None) for each get_shard_iterator.
        self.iterator_requests: list[tuple[str, str, str | None]] = []
        #: Shard ids whose `get_records` reports the shard drained (no NextShardIterator).
        self.drained: set[str] = set()
        #: shard_id -> the sequence numbers it hands back on the next poll.
        self.records: dict[str, list[str]] = {}
        self.list_shards_calls = 0

    def _descriptor(self, entry):
        if isinstance(entry, str):
            return {"ShardId": entry}
        shard_id, *parents = entry
        out = {"ShardId": shard_id}
        for key, parent in zip(("ParentShardId", "AdjacentParentShardId"), parents, strict=False):
            out[key] = parent
        return out

    def list_shards(self, **kwargs):
        self.list_shards_calls += 1
        return {"Shards": [self._descriptor(e) for e in self.shards]}

    def get_shard_iterator(self, **kwargs):
        self.iterator_requests.append(
            (kwargs["ShardId"], kwargs["ShardIteratorType"], kwargs.get("StartingSequenceNumber"))
        )
        return {"ShardIterator": f"iter-{kwargs['ShardId']}"}

    def get_records(self, **kwargs):
        shard_id = kwargs["ShardIterator"].removeprefix("iter-")
        sequences = self.records.pop(shard_id, [])
        records = [{"Data": b"x", "SequenceNumber": s, "PartitionKey": "k"} for s in sequences]
        drained = shard_id in self.drained
        return {
            "Records": records,
            "NextShardIterator": None if drained else f"iter-{shard_id}",
        }


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


# --------------------------------------------------------------------------
# A reshard mid-run: the children must be taken over, not left unread.
# --------------------------------------------------------------------------
_P0 = "shardId-000000000000"
_P1 = "shardId-000000000001"
_C2 = "shardId-000000000002"
_C3 = "shardId-000000000003"


def _reshard(fake: _FakeKinesis, parent: str, children: list[str]) -> None:
    """Replace `parent` with `children`, as `list_shards` reports it after a split."""
    fake.shards = [s for s in fake.shards if s != parent] + [(c, parent) for c in children]


def test_a_pinned_reader_adopts_the_children_of_its_own_drained_shard() -> None:
    """The silent-loss case, and it is the *only* shape the distributed path uses.

    `BrokerSplit` rebuilds the source as ``partitions=[n]``, so a worker polled strictly the
    shard numbers in its pinned set. A reshard gives the replacement shards **new** numbers,
    which were in nobody's set — so once the parent drained, the worker went quiet on that
    key range forever. Every poll returned nothing, the empty-poll back-off made it look
    like an idle stream, and every record written to the children was lost with no error.
    """
    fake = _FakeKinesis([_P0, _P1])
    fake.drained = {_P1}
    source = _source(fake, partitions=[1])

    source._poll()  # drains shard 1
    _reshard(fake, _P1, [_C2, _C3])

    assert [sid for _n, sid in source._active_shards()] == [_C2, _C3]


def test_an_adopted_child_starts_at_the_beginning_not_at_the_configured_position() -> None:
    """A child holds the continuation of a range this reader was already following, and its
    records begin at the reshard. Under ``LATEST`` everything written between the reshard
    and the child's first poll is skipped — the same silent loss, one step later."""
    fake = _FakeKinesis([_P1])
    fake.drained = {_P1}
    source = _source(fake, partitions=[1], starting_position="latest")

    source._poll()
    _reshard(fake, _P1, [_C2])
    source._poll()

    by_shard = {shard: kind for shard, kind, _seq in fake.iterator_requests}
    assert by_shard[_P1] == "LATEST"  # what the user asked for, for the shard they named
    assert by_shard[_C2] == "TRIM_HORIZON"  # but never for an adopted child


def test_records_from_an_adopted_child_reach_the_caller() -> None:
    """Adoption is only worth anything if the records actually come through."""
    fake = _FakeKinesis([_P1])
    fake.drained = {_P1}
    source = _source(fake, partitions=[1])
    source._poll()

    _reshard(fake, _P1, [_C2])
    fake.records = {_C2: ["49590000000000000000000000007"]}
    messages = source._poll()

    assert [m.resume_token for m in messages] == ["49590000000000000000000000007"]
    assert [m.partition for m in messages] == [2]


def test_a_reader_does_not_adopt_a_child_of_a_shard_it_does_not_own() -> None:
    """Adoption must not become duplication: the reader holding shard 0 never takes over
    the children of shard 1, which belong to whoever holds shard 1."""
    fake = _FakeKinesis([_P0, _P1])
    fake.drained = {_P0, _P1}
    source = _source(fake, partitions=[0])

    source._poll()
    _reshard(fake, _P1, [_C2, _C3])

    assert [sid for _n, sid in source._active_shards()] == []


def test_a_merge_child_is_adopted_by_exactly_one_of_its_parents_owners() -> None:
    """A merge child has two parents, which may sit on two readers that cannot see each
    other. Adopting it on both delivers its records twice; on neither, loses them. The owner
    is defined as the holder of the lowest-numbered parent — evaluated from the child's own
    lineage, so both readers reach the same answer with no coordination."""
    merged = "shardId-000000000005"

    def reader_for(owned: int) -> KinesisSource:
        fake = _FakeKinesis([_P0, _P1])
        fake.drained = {_P0, _P1}
        source = _source(fake, partitions=[owned])
        source._poll()
        fake.shards = [(merged, _P0, _P1)]
        return source

    owner_of_0 = reader_for(0)
    owner_of_1 = reader_for(1)

    assert [sid for _n, sid in owner_of_0._active_shards()] == [merged]
    assert [sid for _n, sid in owner_of_1._active_shards()] == []


def test_adoption_is_transitive_across_a_second_reshard() -> None:
    """A child can itself be resharded before the query restarts."""
    fake = _FakeKinesis([_P1])
    fake.drained = {_P1, _C2}
    source = _source(fake, partitions=[1])

    source._poll()
    _reshard(fake, _P1, [_C2])
    source._poll()  # adopts and drains the child
    _reshard(fake, _C2, [_C3])

    assert [sid for _n, sid in source._active_shards()] == [_C3]
