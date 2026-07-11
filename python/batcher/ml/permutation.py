"""The keyed epoch permutation — a shuffled sample order that is computed, not stored.

Shuffling a training corpus normally means permuting a list of every sample index. At
~28 bytes per CPython int that list is 280 GB for ten billion samples, and it must be
built (and held) before the first batch is read — the wall a PB-scale pretraining
pipeline hits first.

`epoch_permutation` removes the list. It returns a **keyed pseudorandom bijection** on
``[0, num_samples)``: hand it an index, get that index's shuffled position, in constant
time and constant memory. An epoch over an exabyte of samples then costs the same
driver memory as one over ten, and seeking to any offset (a mid-epoch resume) is
arithmetic rather than a walk.

The construction is a 4-round balanced Feistel network over the smallest power-of-two
domain containing `num_samples`, narrowed to exactly that many values by cycle walking.
Feistel is a bijection for *any* round function, so nothing has to be tracked to keep it
one-to-one; four rounds and a SplitMix64 round function make it look shuffled. This is
about statistical quality, not secrecy — nothing here resists an adversary.

Everything is integer arithmetic on `uint64`, written out rather than delegated to
`random` or `hashlib`, because every rank must compute the *identical* permutation: a
rank that disagreed would stride a different order and the epoch would both repeat and
skip samples, silently.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["epoch_permutation"]


_U64 = (1 << 64) - 1


def _mix64(x: int) -> int:
    """SplitMix64's finalizer — an avalanching 64-bit integer hash.

    Written out rather than pulled from `random`/`hashlib` because it must be
    *identical* on every rank and every Python build: a rank whose round function
    disagrees would stride a different permutation and the epoch would both repeat and
    skip samples.
    """
    x = (x + 0x9E3779B97F4A7C15) & _U64
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & _U64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & _U64
    return x ^ (x >> 31)


def _mix64_vec(np: Any, x: Any) -> Any:
    """`_mix64` over a NumPy `uint64` array. Unsigned arithmetic wraps, so the masks
    the scalar form needs are implicit — the two agree bit for bit (asserted in tests)."""
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = x ^ (x >> np.uint64(30))
    x = x * np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27))
    x = x * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


class _FeistelPermutation(Sequence[int]):
    """A keyed bijection on ``[0, n)``, computed per index in O(1) time and memory.

    A 4-round balanced Feistel network is a permutation of ``[0, 2^(2h))`` for *any*
    round function, so no bookkeeping is needed to keep it one-to-one. The power-of-two
    domain is narrowed to exactly ``n`` by **cycle walking**: re-encrypt any output
    ``>= n`` until it lands in range. That is still a bijection (it walks the cycles of
    a permutation, and every cycle through the out-of-range region returns), and since
    the domain is under 4x ``n``, it terminates in under 4 rounds on average.

    Four rounds is about statistical quality, not secrecy — nothing here defends against
    an adversary, it just has to look shuffled and be reproducible.
    """

    __slots__ = ("_half", "_key", "_mask", "_n")

    def __init__(self, n: int, key: int) -> None:
        self._n = n
        self._key = key & _U64
        # Half-width of the Feistel block: the smallest h with 2^(2h) >= n.
        bits = max(2, (n - 1).bit_length()) if n > 1 else 2
        self._half = (bits + 1) // 2
        self._mask = (1 << self._half) - 1

    def __len__(self) -> int:
        return self._n

    def _round(self, x: int) -> int:
        left, right = x >> self._half, x & self._mask
        for r in range(4):
            f = _mix64(self._key ^ (r * 0x2545F4914F6CDD1D) ^ right) & self._mask
            left, right = right, left ^ f
        return (left << self._half) | right

    def __getitem__(self, index: int) -> int:  # type: ignore[override]
        """The shuffled position of `index` — the whole permutation, one entry at a time."""
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self._n))]  # type: ignore[return-value]
        if index < 0:
            index += self._n
        if not 0 <= index < self._n:
            raise IndexError(f"index {index} out of range for permutation of {self._n}")
        x = index
        while True:
            x = self._round(x)
            if x < self._n:
                return x

    def take(self, indices: Any) -> Any:
        """`__getitem__` over a whole NumPy array of indices at once.

        Per-index Python arithmetic runs at ~240k indices/s — enough for a large model,
        but a bottleneck for a small one on many GPUs. The same Feistel rounds over a
        `uint64` array run ~100x faster, and cycle walking vectorizes too: re-encrypt
        only the lanes still out of range, of which there are geometrically few. Bound
        the walk anyway and fall back per-index, so an unlucky lane can never spin.
        """
        import numpy as np

        half = np.uint64(self._half)
        mask = np.uint64(self._mask)
        key = np.uint64(self._key)
        rounds = [np.uint64(r * 0x2545F4914F6CDD1D) for r in range(4)]

        def feistel(x: Any) -> Any:
            left, right = x >> half, x & mask
            for rk in rounds:
                f = _mix64_vec(np, key ^ rk ^ right) & mask
                left, right = right, left ^ f
            return (left << half) | right

        n = np.uint64(self._n)
        out = feistel(np.asarray(indices, dtype=np.uint64))
        for _ in range(64):  # geometric; 64 is a safety bound, not an expected count
            over = out >= n
            if not over.any():
                return out
            out[over] = feistel(out[over])
        # Astronomically unlikely; finish the stragglers exactly as `__getitem__` would.
        out = out.tolist()
        return np.array(
            [v if v < self._n else self[int(i)] for i, v in zip(indices, out, strict=True)],
            dtype=np.uint64,
        )


def epoch_permutation(
    num_samples: int, *, epoch: int = 0, seed: int = 0, shuffle: bool = True
) -> Sequence[int]:
    """The global sample order for one epoch, as a **lazy** sequence.

    Supports ``len`` and ``[i]`` without ever materializing the order, so a corpus far
    larger than driver RAM shuffles for free. Keyed by ``(seed, epoch)`` and independent
    of world size — the stable backbone every rank strides over, and the basis for both
    elasticity and O(1) resume. With ``shuffle=False`` it is ``range(num_samples)``.

    Args:
        num_samples: Size of the corpus.
        epoch: Reshuffles the order; pass the same value on every rank.
        seed: Reproduces the order across runs.
        shuffle: When false, the identity order.

    Returns:
        A lazy sequence that is a permutation of ``range(num_samples)``.

    Raises:
        ValueError: If `num_samples` is negative.

    Examples:
        .. doctest::

            >>> from batcher.ml import epoch_permutation
            >>> order = epoch_permutation(10**12, seed=1)  # a trillion samples, instantly
            >>> len(order)
            1000000000000
            >>> sorted(epoch_permutation(5, seed=1))  # still a permutation
            [0, 1, 2, 3, 4]
    """
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if not shuffle or num_samples <= 1:
        return range(num_samples)
    # One key per (seed, epoch): a new epoch reshuffles, the same pair reproduces.
    return _FeistelPermutation(num_samples, _mix64(seed * 0x9E3779B1 + epoch))
