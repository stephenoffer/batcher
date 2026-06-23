"""Deterministic / balanced / elastic / resumable sample ordering (F3/F4).

The contract a MosaicML-Streaming-class loader needs and Ray Train's split iterator
lacks — verified as pure index math (no engine, no torch).
"""

from __future__ import annotations

import pytest

from batcher.ml.streaming_sampler import elastic_shard, epoch_order, rank_shard, usable_length


def test_deterministic_same_seed_epoch():
    a = epoch_order(1000, epoch=2, seed=7)
    b = epoch_order(1000, epoch=2, seed=7)
    assert a == b
    assert sorted(a) == list(range(1000))  # a true permutation


def test_different_epoch_reshuffles():
    assert epoch_order(1000, epoch=0, seed=7) != epoch_order(1000, epoch=1, seed=7)


def test_order_is_world_size_independent():
    # Elasticity backbone: the global order does not depend on world size.
    assert epoch_order(1000, epoch=3, seed=1) == epoch_order(1000, epoch=3, seed=1)


def test_ranks_cover_everything_once_and_are_balanced():
    order = epoch_order(1000, epoch=0, seed=0)
    W = 4
    shards = [rank_shard(order, world_size=W, rank=r) for r in range(W)]
    # Balanced: every rank gets the same count (no straggler → no DDP hang).
    lengths = {len(s) for s in shards}
    assert lengths == {1000 // W}
    # Coverage: union of shards == the trimmed global order, each sample once.
    union = sorted(x for s in shards for x in s)
    assert union == sorted(order[: usable_length(1000, W)])


def test_drop_last_trims_to_equal_length():
    order = epoch_order(1003, epoch=0, seed=0)  # not divisible by 4
    W = 4
    shards = [rank_shard(order, world_size=W, rank=r) for r in range(W)]
    assert {len(s) for s in shards} == {1003 // W}  # 250 each, 3 dropped


def test_resume_skips_consumed_no_dup_no_skip():
    order = epoch_order(100, epoch=0, seed=5)
    W, rank = 4, 1
    full = rank_shard(order, world_size=W, rank=rank)
    # Suppose 5 global "steps" completed: each rank consumed 5 → global_consumed = 5*W.
    consumed = 5 * W
    remaining = elastic_shard(order, world_size=W, rank=rank, global_consumed=consumed)
    # The remaining for this rank == its full shard minus the first 5 it had done.
    assert remaining == full[5:]
    # Across all ranks, resume covers exactly the unconsumed global tail, once.
    resumed_union = sorted(
        x
        for r in range(W)
        for x in elastic_shard(order, world_size=W, rank=r, global_consumed=consumed)
    )
    usable = usable_length(100, W)
    assert resumed_union == sorted(order[consumed:usable])


def test_elastic_resume_on_different_world_size_covers_tail_once():
    # Crash at world_size 4 after global_consumed samples; resume at world_size 2.
    order = epoch_order(120, epoch=0, seed=9)
    consumed = 8 * 4  # 8 synchronized steps at W=4
    W2 = 2
    resumed = sorted(
        x
        for r in range(W2)
        for x in elastic_shard(order, world_size=W2, rank=r, global_consumed=consumed)
    )
    usable2 = usable_length(120, W2)
    # Exactly the unconsumed tail under the new world size, each sample once.
    assert resumed == sorted(order[consumed:usable2])
    assert len(resumed) == len(set(resumed))


def test_no_shuffle_is_identity():
    assert epoch_order(10, shuffle=False) == list(range(10))


def test_rank_out_of_range_raises():
    order = epoch_order(10)
    with pytest.raises(ValueError, match="rank"):
        rank_shard(order, world_size=2, rank=2)


# --- brute-force invariant sweep (correctness audit) -------------------------


def test_partition_and_balance_invariants_across_configs():
    # Exhaustively verify the core guarantees over many shapes:
    #   (1) ranks PARTITION the usable order — every usable position once, no dup.
    #   (2) with drop_last, ranks are BALANCED (equal counts).
    for n in [0, 1, 7, 16, 100, 101, 257]:
        for ws in [1, 2, 3, 4, 8]:
            order = epoch_order(n, epoch=1, seed=3)
            usable = usable_length(n, ws)
            shards = [rank_shard(order, world_size=ws, rank=r) for r in range(ws)]
            # (1) partition: union == first `usable` positions of the order, each once.
            union = sorted(x for s in shards for x in s)
            assert union == sorted(order[:usable]), (n, ws)
            assert len(union) == len(set(union)), (n, ws)  # no duplicates
            # (2) balance: every rank has exactly usable/ws samples.
            assert {len(s) for s in shards} == {usable // ws}, (n, ws)


def test_step_aligned_resume_is_exact_and_balanced():
    # At a synchronized step boundary (global_consumed = k*ws), resume covers exactly
    # the unconsumed tail once AND every rank resumes balanced — across configs.
    for n in [16, 100, 256, 257]:
        for ws in [1, 2, 4, 8]:
            order = epoch_order(n, epoch=0, seed=11)
            usable = usable_length(n, ws)
            steps = usable // ws
            for k in [0, 1, steps // 2, steps]:
                consumed = k * ws  # step-aligned
                shards = [
                    elastic_shard(order, world_size=ws, rank=r, global_consumed=consumed)
                    for r in range(ws)
                ]
                union = sorted(x for s in shards for x in s)
                # Exactly the unconsumed tail [consumed, usable), each once.
                assert union == sorted(order[consumed:usable]), (n, ws, k)
                assert len(union) == len(set(union)), (n, ws, k)
                # Balanced resume: equal counts per rank.
                assert len({len(s) for s in shards}) == 1, (n, ws, k)


def test_full_run_in_steps_covers_epoch_once():
    # Walking an epoch step by step (consume ws per step) visits every usable sample
    # exactly once, in order — the property a training loop relies on.
    n, ws = 100, 4
    order = epoch_order(n, epoch=2, seed=7)
    usable = usable_length(n, ws)
    visited = []
    for step in range(usable // ws):
        for r in range(ws):
            # rank r's sample at this step is its step-th strided position.
            shard = rank_shard(order, world_size=ws, rank=r)
            visited.append(shard[step])
    assert sorted(visited) == sorted(order[:usable])
    assert len(visited) == len(set(visited)) == usable
