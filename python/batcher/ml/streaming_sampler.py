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

and, the constraint that decides the design,

* **O(1) memory** — the global order is *computed*, never materialized. A shuffled list
  of every index costs ~28 bytes per sample in CPython: 280 GB of driver RAM for a
  10-billion-sample corpus, 28 TB for a trillion. Instead `epoch_permutation` is a
  keyed pseudorandom bijection on ``[0, num_samples)`` — index in, shuffled index out,
  no state — so a rank streams its slice of an exabyte corpus in constant memory and
  can seek to any position instantly (which is what makes mid-epoch resume O(1) too).

This module is pure index arithmetic — no engine, no framework, no I/O — so it is
exhaustively unit-testable. A loader layers shard reads, prefetch, and tensor
collation on top (those need the engine / torch); the *ordering contract* lives
here and is verified independently.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from batcher.ml.permutation import _FeistelPermutation, epoch_permutation

__all__ = [
    "ResumableSampler",
    "elastic_shard",
    "epoch_order",
    "epoch_permutation",
    "epoch_positions",
    "num_rank_batches",
    "rank_index_batches",
    "rank_shard",
    "usable_length",
]


def epoch_order(
    num_samples: int, *, epoch: int = 0, seed: int = 0, shuffle: bool = True
) -> list[int]:
    """`epoch_permutation` materialized as a list — for callers that need random access.

    O(`num_samples`) memory. Prefer `epoch_permutation` (or `rank_index_batches`, which
    is built on it) for a corpus that does not fit in driver RAM.
    """
    return list(epoch_permutation(num_samples, epoch=epoch, seed=seed, shuffle=shuffle))


def usable_length(total: int, world_size: int, *, drop_last: bool = True) -> int:
    """How many sample *positions* this epoch spans — always a multiple of ``world_size``.

    Equal per-rank counts are what keep every rank arriving at the DDP all-reduce
    barrier together; an unequal split hangs the job. So both modes round to a
    multiple of ``world_size``, and the flag only picks the direction:
    ``drop_last=True`` (the default) rounds **down**, dropping the remainder;
    ``drop_last=False`` rounds **up**, padding it (see `epoch_positions`), exactly as
    `torch.utils.data.DistributedSampler` does.
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if drop_last:
        return (total // world_size) * world_size
    return total + (-total % world_size)


def epoch_positions(order: list[int], *, world_size: int, drop_last: bool = True) -> list[int]:
    """`order` trimmed or padded to `usable_length` — the positions the ranks stride over.

    With ``drop_last`` the tail remainder is dropped. Without it the remainder is kept
    and the sequence is padded back up to a multiple of ``world_size`` by repeating
    samples from the front (cyclically, so it holds even when ``world_size`` exceeds
    the sample count). Padding duplicates a handful of samples rather than dropping
    them — the trade `DistributedSampler(drop_last=False)` makes, and the only way to
    see every sample *and* keep the ranks balanced.
    """
    usable = usable_length(len(order), world_size, drop_last=drop_last)
    if usable <= len(order):
        return order[:usable]
    if not order:
        return []  # nothing to repeat from
    return order + [order[i % len(order)] for i in range(usable - len(order))]


def rank_shard(
    order: list[int], *, world_size: int, rank: int, drop_last: bool = True
) -> list[int]:
    """Rank ``rank`` of ``world_size``'s samples: a strided slice of `epoch_positions`.

    Striding (``positions[rank::world_size]``) means the union over all ranks is
    exactly the epoch's positions — every sample covered, none dropped — and every
    rank gets the same count in both `drop_last` modes.
    """
    if not (0 <= rank < world_size):
        raise ValueError(f"rank {rank} out of range for world_size {world_size}")
    positions = epoch_positions(order, world_size=world_size, drop_last=drop_last)
    return positions[rank::world_size]


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
    positions = epoch_positions(order, world_size=world_size, drop_last=drop_last)
    start = max(0, global_consumed)
    return [positions[p] for p in range(rank, len(positions), world_size) if p >= start]


def _rank_positions(
    num_samples: int, world_size: int, rank: int, global_consumed: int, drop_last: bool
) -> range:
    """This rank's remaining epoch *positions*, as a `range` — no list, O(1) to build.

    Positions are strided (``rank``, ``rank + world_size``, …) up to `usable_length`,
    skipping the ones a checkpoint says were already consumed. Returning a `range`
    rather than a list is what keeps a trillion-sample epoch free: it slices, counts
    and iterates lazily.
    """
    if not (0 <= rank < world_size):
        raise ValueError(f"rank {rank} out of range for world_size {world_size}")
    usable = usable_length(num_samples, world_size, drop_last=drop_last)
    start = max(0, global_consumed)
    first = rank
    if start > rank:
        # The first position >= start that is congruent to `rank` modulo `world_size`.
        first = rank + -(-(start - rank) // world_size) * world_size
    return range(min(first, usable), usable, world_size)


def num_rank_batches(
    num_samples: int,
    *,
    batch_size: int,
    world_size: int = 1,
    rank: int = 0,
    drop_last: bool = True,
    global_consumed: int = 0,
) -> int:
    """How many batches `rank_index_batches` will yield — a loader's ``__len__``.

    Equal across ranks, so `DataLoader`'s length (and any epoch-length barrier built on
    it) agrees on every rank.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    total = len(_rank_positions(num_samples, world_size, rank, global_consumed, drop_last))
    return total // batch_size if drop_last else -(-total // batch_size)


def rank_index_batches(
    num_samples: int,
    *,
    batch_size: int,
    world_size: int = 1,
    rank: int = 0,
    epoch: int = 0,
    seed: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
    global_consumed: int = 0,
) -> Iterator[list[int]]:
    """Stream this rank's sample indices for one epoch, `batch_size` at a time.

    The scalable entry point: it holds one batch of indices at a time, never the epoch,
    so shuffling a corpus of any size costs O(`batch_size`) driver memory. Same
    guarantees as `rank_shard` — deterministic in ``(seed, epoch)``, world-size
    independent, balanced across ranks — computed instead of materialized.

    With `drop_last` a trailing partial batch is dropped (so every rank yields the same
    number of batches); without it the last batch may be short, and the epoch's tail is
    padded rather than trimmed.

    Args:
        num_samples: Size of the corpus.
        batch_size: Indices per yielded batch.
        world_size / rank: This process's slot in the data-parallel group.
        epoch / seed / shuffle: Select the deterministic global order.
        drop_last: Drop the epoch's tail and any partial final batch.
        global_consumed: Samples already consumed this epoch (the resume point).

    Yields:
        Lists of `batch_size` sample indices into the corpus.

    Examples:
        .. doctest::

            >>> from batcher.ml import rank_index_batches
            >>> list(rank_index_batches(8, batch_size=2, world_size=2, rank=0, shuffle=False))
            [[0, 2], [4, 6]]
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    import numpy as np

    permutation = epoch_permutation(num_samples, epoch=epoch, seed=seed, shuffle=shuffle)
    positions = _rank_positions(num_samples, world_size, rank, global_consumed, drop_last)
    total = len(positions)
    limit = (total // batch_size) * batch_size if drop_last else total

    for start in range(0, limit, batch_size):
        count = min(batch_size, limit - start)
        first = positions.start + start * world_size
        chunk = np.arange(first, first + count * world_size, world_size, dtype=np.uint64)
        # Positions beyond the corpus exist only in the padded (`drop_last=False`) tail,
        # where the epoch repeats samples from the front — hence the wrap.
        chunk %= np.uint64(num_samples)
        if isinstance(permutation, _FeistelPermutation):
            chunk = permutation.take(chunk)
        yield chunk.tolist()


class ResumableSampler:
    """A per-rank index stream that survives a checkpoint — ``state_dict`` / ``load_state_dict``.

    The three functions above are the arithmetic; this is the object a training loop
    actually holds. It owns the ``(epoch, global_consumed)`` pair the loop would
    otherwise have to thread by hand — the gap that made mid-epoch resume a
    caller-assembled protocol rather than a feature (MosaicML `StreamingDataset` has
    this; `DistributedSampler` does not, and Ray Train's split iterator cannot).

    Iterating yields this rank's sample indices and advances the global position by
    ``world_size`` per yielded sample, so a `state_dict` taken **between steps** —
    where every rank has consumed the same count — resumes with no sample repeated and
    none skipped. `set_epoch` reshuffles and rewinds, the `DistributedSampler` protocol.

    Examples:
        .. doctest::

            >>> from itertools import islice
            >>> from batcher.ml import ResumableSampler
            >>> sampler = ResumableSampler(10, world_size=2, rank=0, seed=1)
            >>> len(sampler)
            5
            >>> seen = list(islice(sampler, 2))  # train two steps, then checkpoint
            >>> state = sampler.state_dict()
            >>> resumed = ResumableSampler(10, world_size=2, rank=0, seed=1)
            >>> resumed.load_state_dict(state)
            >>> len(resumed)  # the three samples this rank had not reached
            3
            >>> set(seen) & set(resumed)  # and none of them repeat what was seen
            set()
    """

    __slots__ = (
        "_consumed",
        "_drop_last",
        "_epoch",
        "_num_samples",
        "_rank",
        "_seed",
        "_shuffle",
        "_world_size",
    )

    def __init__(
        self,
        num_samples: int,
        *,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        """Bind the sampler to a corpus size and this process's place in the world."""
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if not (0 <= rank < world_size):
            raise ValueError(f"rank {rank} out of range for world_size {world_size}")
        self._num_samples = num_samples
        self._world_size = world_size
        self._rank = rank
        self._seed = seed
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._epoch = 0
        self._consumed = 0

    @property
    def epoch(self) -> int:
        """The epoch whose order is currently being yielded."""
        return self._epoch

    @property
    def global_consumed(self) -> int:
        """Samples consumed across **all** ranks so far this epoch."""
        return self._consumed

    def set_epoch(self, epoch: int) -> None:
        """Start `epoch`: reshuffle the global order and rewind to its first sample.

        Call this once per epoch, on every rank, before iterating — the same contract
        as ``DistributedSampler.set_epoch``. Every rank must pass the same value or
        the ranks stride over different permutations.
        """
        self._epoch = epoch
        self._consumed = 0

    def __len__(self) -> int:
        """How many samples this rank still yields this epoch (equal across ranks)."""
        usable = usable_length(self._num_samples, self._world_size, drop_last=self._drop_last)
        return max(0, usable - self._consumed) // self._world_size

    def __iter__(self) -> Iterator[int]:
        """Yield this rank's remaining sample indices, advancing the global position.

        Nothing is materialized: the order is a keyed bijection evaluated per index and
        the positions are a `range`, so an epoch over a trillion samples costs the same
        memory as one over ten.
        """
        permutation = epoch_permutation(
            self._num_samples, epoch=self._epoch, seed=self._seed, shuffle=self._shuffle
        )
        positions = _rank_positions(
            self._num_samples, self._world_size, self._rank, self._consumed, self._drop_last
        )
        for position in positions:
            # Count the sample *before* handing it over: a `state_dict` taken after
            # receiving k samples must say k were consumed, and post-yield code only
            # runs once the consumer asks for the next one (or never, if it stops).
            self._consumed += self._world_size
            yield permutation[position % self._num_samples]

    def state_dict(self) -> dict[str, Any]:
        """The resume point: everything needed to rebuild this stream mid-epoch.

        Take it **between steps**, when every rank has consumed the same number of
        samples; `global_consumed` is then a multiple of `world_size` and the resumed
        ranks stay balanced. `world_size` is recorded but not required to match on
        restore — the global order is world-size independent, so a job may resume on a
        differently sized cluster.
        """
        return {
            "num_samples": self._num_samples,
            "epoch": self._epoch,
            "global_consumed": self._consumed,
            "seed": self._seed,
            "shuffle": self._shuffle,
            "drop_last": self._drop_last,
            "world_size": self._world_size,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a `state_dict`, resuming the epoch where it left off.

        Raises:
            ValueError: If `state` describes a different corpus (`num_samples`) or a
                different ordering (`seed`/`shuffle`), which would silently re-shuffle
                the samples this rank has already trained on.
        """
        for key in ("num_samples", "seed", "shuffle"):
            if key in state and state[key] != getattr(self, f"_{key}"):
                raise ValueError(
                    f"cannot resume: state has {key}={state[key]!r} but this sampler "
                    f"has {getattr(self, f'_{key}')!r}; the sample order would differ"
                )
        self._epoch = int(state["epoch"])
        self._consumed = int(state["global_consumed"])
