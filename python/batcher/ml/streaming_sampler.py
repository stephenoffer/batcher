"""Deterministic, resumable, elastic sample ordering for distributed training.

The hard part of a streaming training loader (MosaicML-Streaming's signature feature, and
where Ray Train's ``StreamSplitDataIterator`` struggles — rank hangs, no mid-epoch resume):
give every rank a sample sequence that is

* **deterministic** — same ``(seed, epoch)`` → same global order (reproducible runs);
* **balanced** — every rank gets the *same* number of samples (``drop_last``), so no rank
  finishes early and stalls the others at the DDP all-reduce barrier;
* **elastic** — the global order is independent of ``world_size``, so a job can resume on
  a differently-sized cluster and still see each sample exactly once;
* **resumable** — checkpoint a global sample position and resume mid-epoch with no
  repeated and no skipped samples.

and, the constraint that decides the design,

* **O(1) memory** — the global order is *computed*, never materialized: a shuffled index
  list costs ~28 bytes per sample in CPython (280 GB of driver RAM for a 10-billion-sample
  corpus, 28 TB for a trillion), so `epoch_permutation` is a keyed pseudorandom bijection
  on ``[0, num_samples)`` instead — index in, shuffled index out, no state. A rank streams
  its slice of an exabyte corpus in constant memory and seeks to any position instantly,
  which is what makes mid-epoch resume O(1) too.

This module is pure index arithmetic — no engine, no framework, no I/O — so it is
exhaustively unit-testable. A loader layers shard reads, prefetch, and tensor collation on
top (those need the engine / torch); the *ordering contract* lives here, verified alone.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from batcher.ml.converters import _worker_stride
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

    O(`num_samples`) memory — prefer `epoch_permutation` past driver RAM.

    Examples:
        .. doctest::

            >>> from batcher.ml import epoch_order
            >>> epoch_order(4, seed=7)
            [1, 3, 2, 0]

    Args:
        num_samples: Size of the corpus.
        epoch: Selects this epoch's order (together with `seed`).
        seed: Keys the permutation.
        shuffle: Return the identity order instead when false.

    Returns:
        Every index in ``[0, num_samples)``, in this epoch's order.
    """
    return list(epoch_permutation(num_samples, epoch=epoch, seed=seed, shuffle=shuffle))


def usable_length(total: int, world_size: int, *, drop_last: bool = True) -> int:
    """How many sample *positions* this epoch spans — always a multiple of ``world_size``.

    Equal per-rank counts are what keep every rank arriving at the DDP all-reduce barrier
    together; an unequal split hangs the job. So both modes round to a multiple of
    ``world_size`` and the flag only picks the direction: `drop_last` (the default) rounds
    **down**, dropping the remainder; otherwise it rounds **up**, padding it (see
    `epoch_positions`), exactly as `torch.utils.data.DistributedSampler` does.

    Examples:
        .. doctest::

            >>> from batcher.ml import usable_length
            >>> usable_length(10, 4)  # 2 samples dropped
            8
            >>> usable_length(10, 4, drop_last=False)  # 2 samples repeated
            12

    Args:
        total: Samples in the corpus.
        world_size: Ranks in the data-parallel group.
        drop_last: Round down (dropping the remainder) rather than up (padding it).

    Returns:
        The epoch's length in sample positions, a multiple of `world_size`.
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if drop_last:
        return (total // world_size) * world_size
    return total + (-total % world_size)


def epoch_positions(order: Sequence[int], *, world_size: int, drop_last: bool = True) -> list[int]:
    """`order` trimmed or padded to `usable_length` — the positions the ranks stride over.

    With ``drop_last`` the tail remainder is dropped. Without it the remainder is kept and the
    sequence padded back up to a multiple of ``world_size`` by repeating samples from the front
    (cyclically, so it holds even when ``world_size`` exceeds the sample count): a handful of
    duplicates rather than dropped samples — the trade `DistributedSampler(drop_last=False)`
    makes, and the only way to see every sample *and* keep the ranks balanced.
    """
    total = len(order)
    usable = usable_length(total, world_size, drop_last=drop_last)
    return [order[p if p < total else (p - total) % total] for p in range(usable)]


def rank_shard(
    order: list[int], *, world_size: int, rank: int, drop_last: bool = True
) -> list[int]:
    """Rank ``rank`` of ``world_size``'s samples: a strided slice of `epoch_positions`.

    Striding (``positions[rank::world_size]``) means the union over all ranks is exactly
    the epoch's positions — every sample covered, none dropped — and every rank gets the
    same count in both modes. This is `elastic_shard` from the top of the epoch, and shares
    its implementation so the two can never drift.
    """
    return elastic_shard(order, world_size=world_size, rank=rank, drop_last=drop_last)


def elastic_shard(
    order: Sequence[int],
    *,
    world_size: int,
    rank: int,
    global_consumed: int = 0,
    drop_last: bool = True,
) -> list[int]:
    """Rank ``rank``'s samples left after ``global_consumed`` globally — the resume path.

    Because ``order`` is world-size-independent, ``global_consumed`` is a position in the
    global order, so a job can resume under a **different** ``world_size`` and still cover
    the unconsumed tail (each remaining position is taken by exactly one rank — the strided
    classes partition ``[global_consumed, usable)``).

    **Precondition for balance:** ``global_consumed`` must be a multiple of ``world_size``
    — the count at a *synchronized step boundary*, which is what a DDP checkpoint records
    (every rank completes step *k* together, so ``global_consumed == k * world_size``).
    There the consumed positions are precisely ``[0, global_consumed)`` and every rank
    resumes with an equal count. A non-aligned value still skips nothing and duplicates
    nothing, but hands ranks unequal counts — re-introducing the DDP hang this avoids.
    Cross-world-size note: with ``drop_last`` and a total not divisible by both world sizes
    the trimmed tail differs, so a resume at a new size may include a few tail samples the
    original dropped (never a dup of a processed one).

    Only this rank's ``1 / world_size`` share is built: materializing `epoch_positions` first
    would cost O(``len(order)``) — the regression the O(1) design exists to avoid.
    """
    total = len(order)
    positions = _rank_positions(total, world_size, rank, global_consumed, drop_last)
    # Positions past the corpus are the padded tail, which repeats it from the front.
    return [order[p if p < total else (p - total) % total] for p in positions]


def _rank_positions(
    num_samples: int, world_size: int, rank: int, global_consumed: int, drop_last: bool
) -> range:
    """This rank's remaining epoch *positions*, as a `range` — no list, O(1) to build.

    Positions are strided (``rank``, ``rank + world_size``, …) up to `usable_length`, skipping
    the ones a checkpoint says were already consumed. Returning a `range` rather than a list
    is what keeps a trillion-sample epoch free: it slices, counts and iterates lazily.
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

    Equal across ranks, so `DataLoader`'s length (and any epoch-length barrier built on it)
    agrees everywhere.
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

    The scalable entry point: it holds one batch of indices at a time, never the epoch, so
    shuffling a corpus of any size costs O(`batch_size`) driver memory. Same guarantees as
    `rank_shard` — deterministic in ``(seed, epoch)``, world-size independent, balanced
    across ranks — computed instead of materialized. With `drop_last` a trailing partial
    batch is dropped (so every rank yields the same number of batches); without it the last
    batch may be short, and the epoch's tail is padded rather than trimmed.

    Args:
        num_samples: Size of the corpus.
        batch_size: Indices per yielded batch.
        world_size: Ranks in the data-parallel group.
        rank: This process's slot in that group.
        epoch: Selects this epoch's order (together with `seed`).
        seed: Keys the permutation.
        shuffle: Stream the identity order instead when false.
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

    The functions above are the arithmetic; this is the object a training loop holds, owning
    the ``(epoch, global_consumed)`` pair the loop would otherwise thread by hand — the gap
    that made mid-epoch resume a caller-assembled protocol rather than a feature (MosaicML
    `StreamingDataset` has it; `DistributedSampler` does not, and Ray Train's split iterator
    cannot). Iterating yields this rank's sample indices and advances the global position by
    ``world_size`` per sample, so a `state_dict` taken **between steps** — where every rank
    has consumed the same count — resumes with no sample repeated and none skipped.
    `set_epoch` reshuffles and rewinds, the `DistributedSampler` protocol.

    Examples:
        .. doctest::

            >>> from itertools import islice
            >>> from batcher.ml import ResumableSampler
            >>> sampler = ResumableSampler(10, world_size=2, rank=0, seed=1)
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
        """The epoch whose order is currently being yielded.

        Examples:
            .. doctest::

                >>> from batcher.ml import ResumableSampler
                >>> ResumableSampler(10).epoch
                0

        Returns:
            The current epoch, as last set by `set_epoch`.
        """
        return self._epoch

    @property
    def global_consumed(self) -> int:
        """Samples consumed across **all** ranks so far this epoch.

        Examples:
            .. doctest::

                >>> from batcher.ml import ResumableSampler
                >>> sampler = ResumableSampler(10, world_size=2, rank=0)
                >>> _ = next(iter(sampler))  # one sample here, one on the other rank
                >>> sampler.global_consumed
                2

        Returns:
            The global sample position this epoch — the resume point.
        """
        return self._consumed

    def set_epoch(self, epoch: int) -> None:
        """Start `epoch`: reshuffle the global order and rewind to its first sample.

        Call this once per epoch, on every rank, before iterating — the ``DistributedSampler``
        contract. Every rank must pass the same value or they stride different permutations.

        Examples:
            .. doctest::

                >>> from batcher.ml import ResumableSampler
                >>> sampler = ResumableSampler(10)
                >>> sampler.set_epoch(1)  # every rank passes the same value

        Args:
            epoch: The epoch to start. The same value on every rank.
        """
        self._epoch = epoch
        self._consumed = 0

    def __len__(self) -> int:
        """How many samples this rank still yields this epoch (equal across ranks).

        Examples:
            .. doctest::

                >>> from batcher.ml import ResumableSampler
                >>> len(ResumableSampler(10, world_size=4))  # 8 usable / 4 ranks
                2

        Returns:
            This rank's remaining sample count for the epoch.
        """
        usable = usable_length(self._num_samples, self._world_size, drop_last=self._drop_last)
        return max(0, usable - self._consumed) // self._world_size

    def __iter__(self) -> Iterator[int]:
        """Yield this rank's remaining sample indices, advancing the global position.

        Nothing is materialized: the order is a keyed bijection evaluated per index and the
        positions are a `range`, so an epoch over a trillion samples costs the same memory
        as one over ten. The sequence is strided across DataLoader workers, each of which gets
        its own copy of this sampler and would otherwise replay the whole stream — training on
        every sample ``num_workers`` times, silently. The global position advances over skipped
        samples too, so a `state_dict` means the same thing in every worker.

        Examples:
            .. doctest::

                >>> from batcher.ml import ResumableSampler
                >>> sampler = ResumableSampler(6, world_size=2, rank=1, shuffle=False)
                >>> list(sampler)
                [1, 3, 5]

        Yields:
            This rank's sample indices into the corpus, in epoch order.
        """
        permutation = epoch_permutation(
            self._num_samples, epoch=self._epoch, seed=self._seed, shuffle=self._shuffle
        )
        positions = _rank_positions(
            self._num_samples, self._world_size, self._rank, self._consumed, self._drop_last
        )
        offset, stride = _worker_stride()
        for i, position in enumerate(positions):
            # Count the sample *before* handing it over: a `state_dict` taken after
            # receiving k samples must say k were consumed, and post-yield code only
            # runs once the consumer asks for the next one (or never, if it stops).
            self._consumed += self._world_size
            if i % stride == offset:
                yield permutation[position % self._num_samples]

    def state_dict(self) -> dict[str, Any]:
        """The resume point: everything needed to rebuild this stream mid-epoch.

        Take it **between steps**, when every rank has consumed the same number of samples;
        `global_consumed` is then a multiple of `world_size` and the resumed ranks stay
        balanced. `world_size` is recorded but not required to match on restore — the
        global order is world-size independent, so a job may resume on a differently sized
        cluster.

        Examples:
            .. doctest::

                >>> from batcher.ml import ResumableSampler
                >>> sampler = ResumableSampler(10, world_size=2, seed=1)
                >>> sampler.state_dict()["global_consumed"]
                0

        Returns:
            The sampler's resume state, to hand back to `load_state_dict`.
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

        Examples:
            .. doctest::

                >>> from batcher.ml import ResumableSampler
                >>> sampler = ResumableSampler(10, world_size=2)
                >>> sampler.load_state_dict({"epoch": 1, "global_consumed": 4})
                >>> len(sampler)  # 10 usable positions, 4 already consumed
                3

        Args:
            state: A `state_dict` from a sampler over the same corpus and order.

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
