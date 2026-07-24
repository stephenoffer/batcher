"""`ResumableSampler` — the per-rank index stream a training loop holds across a checkpoint.

`ordering` is the arithmetic: pure functions from ``(seed, epoch, rank, world_size)`` to the
indices a rank should see. This module is the one piece of state on top of it, owning the
``(epoch, global_consumed)`` pair a loop would otherwise thread by hand, and the
``state_dict`` / ``load_state_dict`` pair that makes mid-epoch resume a feature rather than a
caller-assembled protocol.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from batcher.ml.converters import _worker_stride
from batcher.ml.permutation import epoch_permutation
from batcher.ml.streaming_sampler.ordering import _rank_positions, usable_length

__all__ = ["ResumableSampler"]


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
