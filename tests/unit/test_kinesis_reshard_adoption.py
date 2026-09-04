"""A resharded Kinesis stream must lose no record and deliver none twice.

Resharding is routine on a stream large enough to need Batcher: AWS splits a hot shard and
merges cold ones, and each event replaces a shard with **new** shards carrying **new** ids.
Every reader here is pinned to a shard set, because `BrokerSplit` rebuilds the source as
``partitions=[n]`` on each worker — so the children of a reshard are in nobody's pinned set,
and a reader that only polls its own set goes permanently quiet on that key range the moment
its parent drains.

That failure is silent in every direction. Nothing raises, the empty polls look exactly like
an idle stream, the back-off makes them cheap, and no count is wrong anywhere it can be
compared. `_adopt_children` is the fix and it had **no test**: the surrounding suite covers
the `GetRecords` limit, `ListShards` pagination and closed-shard retirement, and mentions
adoption only in a comment.

The properties here are arithmetic over a shard lineage, so they need no AWS — which is
exactly why the bug could survive a green suite, and exactly why they are worth pinning.
Two of them fail in opposite directions and both matter:

* **Under-adoption loses data.** Nobody polls the children.
* **Over-adoption duplicates it.** A merge child has *two* parents, which may sit on two
  different readers that cannot see each other's state. If both adopt, every record in the
  child is delivered twice.
"""

from __future__ import annotations

from typing import Any

import pytest

from batcher.io.formats.streaming.kinesis import KinesisSource, _shard_number

pytestmark = pytest.mark.unit


def _sid(n: int) -> str:
    """A realistic ShardId, so `_shard_number` parses it rather than hashing it."""
    return f"shardId-{n:012d}"


def _shard(n: int, *parents: int) -> dict[str, Any]:
    """One `list_shards` descriptor: a shard and the shards it replaced."""
    out: dict[str, Any] = {"ShardId": _sid(n)}
    for key, parent in zip(("ParentShardId", "AdjacentParentShardId"), parents, strict=False):
        out[key] = _sid(parent)
    return out


class _FakeKinesis:
    """A boto3 `kinesis` client whose shard listing can be resharded mid-test."""

    def __init__(self, shards: list[dict[str, Any]]) -> None:
        self.shards = shards
        self.closed: set[str] = set()
        self.polled: list[str] = []

    def reshard(self, shards: list[dict[str, Any]]) -> None:
        """Replace the listing, as AWS does once a split or merge completes."""
        self.shards = shards

    def list_shards(self, **kwargs: Any) -> dict[str, Any]:
        return {"Shards": list(self.shards)}

    def get_shard_iterator(self, **kwargs: Any) -> dict[str, Any]:
        return {"ShardIterator": f"iter::{kwargs['ShardId']}"}

    def get_records(self, **kwargs: Any) -> dict[str, Any]:
        shard_id = kwargs["ShardIterator"].split("::", 1)[1]
        self.polled.append(shard_id)
        if shard_id in self.closed:
            # A drained, closed shard: no next iterator. This is the only signal Kinesis
            # gives that a reshard replaced it.
            return {"Records": []}
        return {"Records": [], "NextShardIterator": f"iter::{shard_id}"}


def _source(fake: _FakeKinesis, partitions: list[int] | None = None) -> KinesisSource:
    kwargs: dict[str, Any] = {} if partitions is None else {"partitions": partitions}
    src = KinesisSource("my-stream", **kwargs)
    src._client_obj = fake
    return src


def _active(src: KinesisSource) -> list[str]:
    return [sid for _, sid in src._active_shards()]


def _reshard(src: KinesisSource, fake: _FakeKinesis, shards: list[dict[str, Any]]) -> None:
    """Put the source in the state a completed reshard leaves it in.

    Two things happen in the engine and both are needed: AWS starts reporting the new
    listing, and the source drops its cached one. The drop is `_advance`'s job, on the poll
    that finds no `NextShardIterator` -- so it is *bypassed* here and proved end to end by
    `TestTheCacheIsInvalidatedOnDrain` instead. Setting it directly keeps the tests below
    about the adoption arithmetic, which is the part with the ownership rule in it.
    """
    fake.reshard(shards)
    src._shards_cache = None


class TestSplit:
    """A split replaces one shard with two children of the same parent."""

    def test_a_pinned_reader_polls_only_its_own_shard_before_the_reshard(self):
        fake = _FakeKinesis([_shard(1), _shard(2)])
        src = _source(fake, partitions=[_shard_number(_sid(1))])
        assert _active(src) == [_sid(1)]

    def test_the_children_are_adopted_once_the_parent_drains(self):
        """The whole point: without this the reader goes quiet on shard 1's key range."""
        fake = _FakeKinesis([_shard(1), _shard(2)])
        src = _source(fake, partitions=[_shard_number(_sid(1))])
        assert _active(src) == [_sid(1)]
        src._closed.add(_sid(1))
        _reshard(src, fake, [_shard(2), _shard(3, 1), _shard(4, 1)])
        assert _active(src) == [_sid(3), _sid(4)]

    def test_another_readers_shard_is_never_adopted(self):
        """Over-adoption is the other failure: two readers on one shard deliver twice."""
        fake = _FakeKinesis([_shard(1), _shard(2)])
        src = _source(fake, partitions=[_shard_number(_sid(1))])
        src._closed.add(_sid(1))
        # Shard 2 also reshards, but it belongs to the other reader.
        _reshard(src, fake, [_shard(3, 1), _shard(5, 2), _shard(6, 2)])
        assert _active(src) == [_sid(3)]

    def test_a_child_is_not_read_before_its_parent_is_drained(self):
        """Kinesis orders a child strictly after its parents, so adopting early would
        deliver the key range out of order."""
        fake = _FakeKinesis([_shard(1), _shard(3, 1), _shard(4, 1)])
        src = _source(fake, partitions=[_shard_number(_sid(1))])
        assert _active(src) == [_sid(1)]

    def test_adoption_is_transitive(self):
        """A child can itself be resharded, and often is — a hot key stays hot."""
        fake = _FakeKinesis([_shard(1)])
        src = _source(fake, partitions=[_shard_number(_sid(1))])
        src._closed.update({_sid(1), _sid(3)})
        _reshard(src, fake, [_shard(3, 1), _shard(7, 3), _shard(8, 3)])
        assert _active(src) == [_sid(7), _sid(8)]

    def test_a_drained_parent_is_not_polled_again(self):
        fake = _FakeKinesis([_shard(1)])
        src = _source(fake, partitions=[_shard_number(_sid(1))])
        src._closed.add(_sid(1))
        _reshard(src, fake, [_shard(1), _shard(3, 1)])
        assert _sid(1) not in _active(src)


class TestMerge:
    """A merge child has two parents, and exactly one reader may take it."""

    def test_the_lowest_numbered_parents_owner_adopts_it(self):
        fake = _FakeKinesis([_shard(1), _shard(2)])
        low = _source(fake, partitions=[_shard_number(_sid(1))])
        low._closed.add(_sid(1))
        _reshard(low, fake, [_shard(9, 1, 2)])
        assert _active(low) == [_sid(9)]

    def test_the_other_parents_owner_does_not(self):
        """Both readers evaluate the same rule from the child's own lineage, with no
        coordination, and must name the same owner. If they disagree, every record in the
        merged shard is delivered twice."""
        fake = _FakeKinesis([_shard(1), _shard(2)])
        high = _source(fake, partitions=[_shard_number(_sid(2))])
        high._closed.add(_sid(2))
        _reshard(high, fake, [_shard(9, 1, 2)])
        assert _active(high) == []

    def test_the_rule_does_not_depend_on_the_order_the_parents_are_listed(self):
        """`ParentShardId` and `AdjacentParentShardId` are not ordered by AWS, so a rule
        that took the first would name a different owner on each reader."""
        fake = _FakeKinesis([_shard(1), _shard(2)])
        low = _source(fake, partitions=[_shard_number(_sid(1))])
        low._closed.add(_sid(1))
        _reshard(
            low,
            fake,
            [{"ShardId": _sid(9), "ParentShardId": _sid(2), "AdjacentParentShardId": _sid(1)}],
        )
        assert _active(low) == [_sid(9)]

    def test_exactly_one_of_the_two_readers_adopts(self):
        """Stated as the property itself rather than as two separate expectations: across
        the readers that could take it, the count is one -- not zero (loss), not two
        (duplication)."""
        fake = _FakeKinesis([_shard(1), _shard(2)])
        readers = []
        for parent in (1, 2):
            src = _source(fake, partitions=[_shard_number(_sid(parent))])
            src._closed.add(_sid(parent))
            readers.append(src)
        _reshard(src, fake, [_shard(9, 1, 2)])
        assert sum(_sid(9) in _active(src) for src in readers) == 1


class TestWholeStreamReader:
    """A source with no pinned partitions owns everything and needs no adoption."""

    def test_it_polls_every_shard(self):
        fake = _FakeKinesis([_shard(1), _shard(2)])
        assert _active(_source(fake)) == [_sid(1), _sid(2)]

    def test_it_picks_up_children_without_adopting(self):
        fake = _FakeKinesis([_shard(1)])
        src = _source(fake)
        src._closed.add(_sid(1))
        _reshard(src, fake, [_shard(3, 1), _shard(4, 1)])
        assert _active(src) == [_sid(3), _sid(4)]


class TestTheCacheIsInvalidatedOnDrain:
    """Adoption cannot happen at all if the children are never listed.

    The shard list is cached for the life of the source, so without invalidation
    `_adopt_children` runs against a listing that predates the reshard and finds no
    children to adopt. This is the link between the two halves, and it is the one that
    fails end to end rather than in the arithmetic.
    """

    def test_draining_a_shard_makes_the_children_discoverable(self):
        fake = _FakeKinesis([_shard(1)])
        src = _source(fake, partitions=[_shard_number(_sid(1))])
        assert _active(src) == [_sid(1)]  # populates the cache
        fake.closed.add(_sid(1))
        fake.reshard([_shard(3, 1), _shard(4, 1)])
        src._poll()  # the poll that sees no NextShardIterator and retires shard 1
        assert _active(src) == [_sid(3), _sid(4)]

    def test_the_reader_then_actually_polls_the_children(self):
        """The positive control for the assertion above: `_active_shards` naming them is
        not the same as records flowing, and it is records that were being lost."""
        fake = _FakeKinesis([_shard(1)])
        src = _source(fake, partitions=[_shard_number(_sid(1))])
        fake.closed.add(_sid(1))
        src._poll()
        fake.reshard([_shard(3, 1), _shard(4, 1)])
        fake.polled.clear()
        src._poll()
        assert sorted(set(fake.polled)) == [_sid(3), _sid(4)]
