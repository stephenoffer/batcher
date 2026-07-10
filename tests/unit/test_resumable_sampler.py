"""`ResumableSampler` and the padded `drop_last=False` shard: the DDP contract.

Two failures are silent and catastrophic in distributed training, so they are pinned
exhaustively rather than by example:

* **unequal rank counts hang the job.** Every rank must yield the same number of
  samples, or the short rank reaches the all-reduce barrier while the others still
  wait. `drop_last=True` trims the tail; `drop_last=False` now *pads* it (repeating a
  few samples, as `torch.utils.data.DistributedSampler` does) rather than handing the
  ranks unequal counts.
* **a bad resume silently retrains on seen samples.** A `state_dict` taken at a step
  boundary must partition the epoch into exactly "already seen" and "not yet seen",
  for every rank and every boundary.

Both are checked over a grid of sizes rather than one case, because the interesting
inputs are precisely those where `num_samples % world_size != 0`.
"""

from __future__ import annotations

from itertools import islice

import pytest

from batcher.ml import ResumableSampler
from batcher.ml.streaming_sampler import (
    elastic_shard,
    epoch_order,
    epoch_positions,
    rank_shard,
    usable_length,
)

pytestmark = pytest.mark.unit

_SIZES = [0, 1, 2, 3, 5, 7, 8, 12, 17]
_WORLDS = [1, 2, 3, 4, 8]


@pytest.mark.parametrize("n", _SIZES)
@pytest.mark.parametrize("world_size", _WORLDS)
@pytest.mark.parametrize("drop_last", [True, False])
def test_every_rank_gets_an_equal_count(n, world_size, drop_last):
    """The DDP-hang guard: equal shard sizes in both `drop_last` modes."""
    order = epoch_order(n, seed=1)
    sizes = {
        len(rank_shard(order, world_size=world_size, rank=r, drop_last=drop_last))
        for r in range(world_size)
    }
    assert len(sizes) == 1, f"ranks got different counts: {sizes}"


@pytest.mark.parametrize("n", _SIZES)
@pytest.mark.parametrize("world_size", _WORLDS)
def test_drop_last_covers_a_prefix_without_duplicates(n, world_size):
    order = epoch_order(n, seed=1)
    union = [x for r in range(world_size) for x in rank_shard(order, world_size=world_size, rank=r)]
    assert len(union) == len(set(union)) == usable_length(n, world_size)


@pytest.mark.parametrize("n", [x for x in _SIZES if x])
@pytest.mark.parametrize("world_size", _WORLDS)
def test_padding_covers_every_sample(n, world_size):
    """`drop_last=False` must lose nothing — the remainder is padded, not dropped."""
    order = epoch_order(n, seed=1)
    union = [
        x
        for r in range(world_size)
        for x in rank_shard(order, world_size=world_size, rank=r, drop_last=False)
    ]
    assert set(union) == set(range(n))
    assert len(union) == n + (-n % world_size)


def test_padding_repeats_cyclically_when_world_exceeds_samples():
    """With more ranks than samples, padding wraps rather than running out."""
    positions = epoch_positions([7, 8], world_size=5, drop_last=False)
    assert len(positions) == 5
    assert set(positions) == {7, 8}


def test_padding_of_empty_order_stays_empty():
    assert epoch_positions([], world_size=4, drop_last=False) == []


def test_usable_length_rounds_down_then_up():
    assert usable_length(10, 4) == 8  # drop_last: round down
    assert usable_length(10, 4, drop_last=False) == 12  # pad: round up
    assert usable_length(8, 4, drop_last=False) == 8  # already aligned


@pytest.mark.parametrize("n", [10, 12, 17])
@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
def test_resume_at_every_step_boundary_partitions_the_epoch(n, world_size):
    """seen ++ remaining == the full epoch, for every rank and every checkpoint point."""
    full = {
        r: list(ResumableSampler(n, world_size=world_size, rank=r, seed=3))
        for r in range(world_size)
    }
    per_rank = len(full[0])
    for step in range(per_rank + 1):
        for rank in range(world_size):
            sampler = ResumableSampler(n, world_size=world_size, rank=rank, seed=3)
            seen = list(islice(sampler, step))
            state = sampler.state_dict()
            assert state["global_consumed"] == step * world_size

            resumed = ResumableSampler(n, world_size=world_size, rank=rank, seed=3)
            resumed.load_state_dict(state)
            assert len(resumed) == per_rank - step
            assert seen + list(resumed) == full[rank]


def test_len_matches_what_iteration_yields():
    sampler = ResumableSampler(17, world_size=4, rank=2, seed=9)
    assert len(sampler) == len(list(sampler))


def test_global_consumed_counts_all_ranks():
    sampler = ResumableSampler(12, world_size=3, rank=0, seed=1)
    list(islice(sampler, 2))
    assert sampler.global_consumed == 6  # 2 steps * 3 ranks


def test_set_epoch_reshuffles_and_rewinds():
    sampler = ResumableSampler(10, seed=1)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert sampler.epoch == 1
    assert first != second, "a new epoch must reshuffle"
    assert sorted(first) == sorted(second), "a new epoch must see the same samples"


def test_set_epoch_rewinds_a_partially_consumed_epoch():
    sampler = ResumableSampler(10, seed=1)
    list(islice(sampler, 3))
    sampler.set_epoch(1)
    assert sampler.global_consumed == 0
    assert len(list(sampler)) == 10


def test_shuffle_false_is_the_identity_order():
    sampler = ResumableSampler(6, world_size=2, rank=0, shuffle=False)
    assert list(sampler) == [0, 2, 4]


def test_resume_onto_a_different_world_size_is_allowed():
    """The global order is world-size independent, so an elastic job may resize."""
    sampler = ResumableSampler(24, world_size=4, rank=0, seed=5)
    list(islice(sampler, 2))
    state = sampler.state_dict()

    resized = ResumableSampler(24, world_size=2, rank=0, seed=5)
    resized.load_state_dict(state)
    assert len(resized) == (24 - state["global_consumed"]) // 2


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [("num_samples", {"num_samples": 99}), ("seed", {"seed": 77}), ("shuffle", {"shuffle": False})],
)
def test_resume_refuses_a_state_from_a_different_order(field, kwargs):
    """Restoring across a different corpus or seed would re-shuffle already-seen
    samples back into the remainder — refuse rather than silently retrain on them."""
    state = ResumableSampler(10, seed=1).state_dict()
    state.update(kwargs)
    sampler = ResumableSampler(10, seed=1)
    with pytest.raises(ValueError, match=field):
        sampler.load_state_dict(state)


def test_constructor_validates_rank_and_world_size():
    with pytest.raises(ValueError, match="world_size must be positive"):
        ResumableSampler(10, world_size=0)
    with pytest.raises(ValueError, match="out of range"):
        ResumableSampler(10, world_size=2, rank=2)
    with pytest.raises(ValueError, match="non-negative"):
        ResumableSampler(-1)


def test_elastic_shard_respects_padding():
    """`elastic_shard` must stride the same padded positions `rank_shard` does."""
    order = epoch_order(10, seed=2)
    for rank in range(4):
        assert elastic_shard(order, world_size=4, rank=rank, drop_last=False) == rank_shard(
            order, world_size=4, rank=rank, drop_last=False
        )


# --- streaming_split's single-rank shard --------------------------------------
#
# The streaming counterpart of the same balance contract: a rank that keeps every
# `i % world_size == rank` batch takes `ceil(n / world_size)` of them while the last
# rank takes `floor(...)`, so the low ranks hit the all-reduce barrier one extra time
# and the job hangs. Only complete rounds may be emitted.


@pytest.mark.parametrize("n", list(range(12)))
@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
def test_single_rank_stream_shards_are_equal_and_disjoint(n, world_size):
    from batcher.ml.loader import _rank_shard_stream

    shards = [list(_rank_shard_stream(range(n), world_size, r)) for r in range(world_size)]
    assert {len(s) for s in shards} == {n // world_size}, [len(s) for s in shards]

    seen = [b for s in shards for b in s]
    assert len(seen) == len(set(seen)), "a batch reached two ranks"
    assert set(seen) == set(range((n // world_size) * world_size)), "not a prefix of the stream"


def test_single_rank_stream_drops_the_trailing_partial_round():
    from batcher.ml.loader import _rank_shard_stream

    # 7 batches over 3 ranks: batch 6 would give rank 0 a third batch and hang ranks 1-2.
    assert [list(_rank_shard_stream(range(7), 3, r)) for r in range(3)] == [
        [0, 3],
        [1, 4],
        [2, 5],
    ]


def test_single_rank_stream_is_lazy():
    """It must not materialize the stream — an unbounded source is the point."""
    from itertools import count, islice

    from batcher.ml.loader import _rank_shard_stream

    assert list(islice(_rank_shard_stream(count(), 2, 1), 3)) == [1, 3, 5]
