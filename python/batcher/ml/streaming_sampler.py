"""Deterministic, resumable, elastic sample ordering for distributed training.

The hard part of a streaming training loader (MosaicML-Streaming's signature
feature, and where Ray Train's ``StreamSplitDataIterator`` struggles — rank hangs,
no mid-epoch resume): give every rank a sample sequence that is

* **deterministic** — same ``(seed, epoch)`` → same global order (reproducible runs);
* **balanced** — every rank gets the *same* number of samples (``drop_last``), so no
  rank finishes early and stalls the others at the DDP all-reduce barrier;
* **elastic** — the global order is independent of ``world_size``, so a job can
  resume on a differently-sized cluster and still see each sample exactly once;
* **resumable** — checkpoint a global sample position and resume mid-epoch with no
  repeated and no skipped samples.

This module is pure index arithmetic — no engine, no framework, no I/O — so it is
exhaustively unit-testable. A loader layers shard reads, prefetch, and tensor
collation on top (those need the engine / torch); the *ordering contract* lives
here and is verified independently.

Note: the order is materialized as a list — fine for the per-rank index sequence at
realistic sizes. A bounded block-shuffle for billion-sample corpora is a scaling
follow-up that preserves this same contract.
"""

from __future__ import annotations

import random

__all__ = ["elastic_shard", "epoch_order", "rank_shard", "usable_length"]


def epoch_order(
    num_samples: int, *, epoch: int = 0, seed: int = 0, shuffle: bool = True
) -> list[int]:
    """The global sample order for one epoch — a permutation of ``range(num_samples)``.

    Seeded by ``(seed, epoch)`` and **independent of world size**, so it is the
    stable backbone every rank strides over (the basis for elasticity). With
    ``shuffle=False`` it is the identity order.
    """
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    order = list(range(num_samples))
    if shuffle:
        # Combine (seed, epoch) into a single int seed — deterministic and
        # collision-free across realistic epoch counts.
        random.Random(seed * 1_000_003 + epoch).shuffle(order)
    return order


def usable_length(total: int, world_size: int, *, drop_last: bool = True) -> int:
    """How many of ``total`` samples are used this epoch. With ``drop_last`` (the
    default) it is trimmed to a multiple of ``world_size`` so every rank gets an
    equal count — the condition that prevents a straggler-rank DDP hang."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    return (total // world_size) * world_size if drop_last else total


def rank_shard(
    order: list[int], *, world_size: int, rank: int, drop_last: bool = True
) -> list[int]:
    """Rank ``rank`` of ``world_size``'s samples: a strided slice of ``order``.

    Striding (``order[rank::world_size]``) means the union over all ranks is exactly
    the (``drop_last``-trimmed) global order — every sample covered once, none
    duplicated — and with ``drop_last`` every rank gets the same count.
    """
    if not (0 <= rank < world_size):
        raise ValueError(f"rank {rank} out of range for world_size {world_size}")
    usable = usable_length(len(order), world_size, drop_last=drop_last)
    return [order[p] for p in range(rank, usable, world_size)]


def elastic_shard(
    order: list[int],
    *,
    world_size: int,
    rank: int,
    global_consumed: int = 0,
    drop_last: bool = True,
) -> list[int]:
    """Rank ``rank``'s *remaining* samples after ``global_consumed`` were already
    processed globally this epoch — the resume path.

    Because ``order`` is world-size-independent, ``global_consumed`` is a position in
    the global order, so a job can resume under a **different** ``world_size`` and
    still cover the unconsumed tail (each remaining position is taken by exactly one
    rank — the strided classes partition ``[global_consumed, usable)``).

    **Precondition for balance:** ``global_consumed`` must be a multiple of
    ``world_size`` — i.e. the count at a *synchronized step boundary*, which is
    exactly what a DDP/checkpoint records (every rank completes step *k* together, so
    ``global_consumed == k * world_size``). At such a boundary the consumed positions
    are precisely ``[0, global_consumed)`` and every rank resumes with an equal count
    (no straggler). A non-aligned value still skips nothing and duplicates nothing,
    but would hand ranks unequal counts — re-introducing the DDP hang this avoids.
    Cross-world-size note: with ``drop_last`` and a total not divisible by both world
    sizes, the trimmed tail differs between sizes, so resume at a new size may include
    a few tail samples the original size had dropped (never a dup of a processed one).
    """
    if not (0 <= rank < world_size):
        raise ValueError(f"rank {rank} out of range for world_size {world_size}")
    usable = usable_length(len(order), world_size, drop_last=drop_last)
    start = max(0, global_consumed)
    return [order[p] for p in range(rank, usable, world_size) if p >= start]
