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


class _KeyedPermutation(Sequence[int]):
    """A computed bijection on ``[0, n)`` that can also map a whole NumPy array at once.

    The two implementations below differ only in *locality*: `_FeistelPermutation` scatters
    neighbouring positions across the whole corpus, and `_BlockPermutation` keeps them inside
    one block. Callers switch on this base rather than on either concrete class, so a
    permutation added later is vectorized by every caller without touching them.
    """

    __slots__ = ()

    def take(self, indices: Any) -> Any:
        """Map a whole NumPy `uint64` array of positions at once."""
        raise NotImplementedError


class _FeistelPermutation(_KeyedPermutation):
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


class _BlockPermutation(_KeyedPermutation):
    """A keyed bijection on ``[0, n)`` that shuffles **within blocks and between blocks**.

    `_FeistelPermutation` shuffles perfectly and reads terribly. Every sample it hands out
    is uniform over the whole corpus, so a batch of 1,024 samples lands in up to 1,024
    different shards — and a shard is the unit a sharded corpus is *stored* and *cached* in.
    With a corpus of 10,000 shards, a loader with any bounded shard cache therefore misses on
    nearly every sample and reads a whole shard to use one row of it: the epoch reads the
    corpus thousands of times over. The cache is not too small; a global shuffle simply has
    no working set for it to hold.

    So this permutes at two scales instead. The corpus is cut into blocks of `block` samples;
    the *blocks* are permuted, and each block's rows are permuted *inside* it. Consecutive
    positions therefore stay inside one block, and a block is sized to the shard cache — so
    the loader reads each shard once per epoch while still seeing a different, seed-keyed
    order every epoch. This is the ``py1b``/``py1e`` trade MosaicML-Streaming makes and the
    reason WebDataset pairs a shard shuffle with a sample buffer: locality bought with a
    shuffle that is random within a window rather than across the corpus.

    Both scales are Feistel bijections, so the whole map is a bijection with no bookkeeping
    and no materialized order: memory stays O(1) and a mid-epoch seek stays arithmetic. The
    trailing ``n % block`` samples are a short final block, permuted among themselves.
    """

    __slots__ = ("_block", "_blocks", "_boundary", "_full", "_key", "_n", "_tail")

    def __init__(self, n: int, block: int, key: int) -> None:
        self._n = n
        self._block = block
        self._key = key & _U64
        self._full = n // block
        self._boundary = self._full * block
        # The block order, and the order within the short trailing block. `None` where the
        # domain has fewer than two entries and a permutation would be the identity anyway.
        self._blocks = (
            _FeistelPermutation(self._full, _mix64(self._key ^ 0x9E3779B97F4A7C15))
            if self._full > 1
            else None
        )
        tail = n - self._boundary
        self._tail = (
            _FeistelPermutation(tail, _mix64(self._key ^ 0xC2B2AE3D27D4EB4F)) if tail > 1 else None
        )

    def __len__(self) -> int:
        return self._n

    def _inner(self, index_block: int) -> _FeistelPermutation:
        """The permutation of the rows *inside* index-block `index_block`.

        Keyed on the block, so two blocks never share an order, and rebuilt per call rather
        than cached: the object is four integers, and caching one per block would reintroduce
        the O(number of blocks) state this design exists to avoid.
        """
        return _FeistelPermutation(
            self._block, _mix64(self._key ^ ((index_block * 0x2545F4914F6CDD1D) & _U64))
        )

    def __getitem__(self, index: int) -> int:  # type: ignore[override]
        """The shuffled sample at position `index`."""
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self._n))]  # type: ignore[return-value]
        if index < 0:
            index += self._n
        if not 0 <= index < self._n:
            raise IndexError(f"index {index} out of range for permutation of {self._n}")
        if index >= self._boundary:  # the short trailing block, permuted among itself
            offset = index - self._boundary
            return self._boundary + (self._tail[offset] if self._tail is not None else offset)
        position_block, offset = divmod(index, self._block)
        j = self._blocks[position_block] if self._blocks is not None else position_block
        return j * self._block + self._inner(j)[offset]

    def take(self, indices: Any) -> Any:
        """`__getitem__` over a whole NumPy array of positions at once.

        Positions are grouped by their block, so the per-block Feistel runs once per block
        rather than once per sample. A loader reads consecutive positions, so a batch spans
        one or two blocks and this is one or two vectorized passes.
        """
        import numpy as np

        pos = np.asarray(indices, dtype=np.uint64)
        out = np.empty_like(pos)
        boundary = np.uint64(self._boundary)
        block = np.uint64(self._block)

        tail_mask = pos >= boundary
        if tail_mask.any():
            offsets = pos[tail_mask] - boundary
            out[tail_mask] = boundary + (
                self._tail.take(offsets) if self._tail is not None else offsets
            )
        head_mask = ~tail_mask
        if not head_mask.any():
            return out
        head = pos[head_mask]
        position_blocks = head // block
        offsets = head - position_blocks * block
        target = self._blocks.take(position_blocks) if self._blocks is not None else position_blocks
        mapped = np.empty_like(head)
        # One vectorized Feistel per distinct target block, not per sample.
        for j in np.unique(target):
            in_block = target == j
            mapped[in_block] = j * block + self._inner(int(j)).take(offsets[in_block])
        out[head_mask] = mapped
        return out


def epoch_permutation(
    num_samples: int,
    *,
    epoch: int = 0,
    seed: int = 0,
    shuffle: bool = True,
    block_size: int | None = None,
) -> Sequence[int]:
    """The global sample order for one epoch, as a **lazy** sequence.

    Supports ``len`` and ``[i]`` without ever materializing the order, so a corpus far
    larger than driver RAM shuffles for free. Keyed by ``(seed, epoch)`` and independent
    of world size — the stable backbone every rank strides over, and the basis for both
    elasticity and O(1) resume. With ``shuffle=False`` it is ``range(num_samples)``.

    `block_size` trades shuffle radius for **read locality**. Without it the order is
    uniform over the whole corpus, which is ideal for convergence and pathological for a
    sharded corpus on disk: consecutive samples land in unrelated shards, so a bounded shard
    cache misses on nearly every one and the epoch reads the corpus many times over. With it,
    the corpus is cut into blocks of `block_size` samples, the blocks are shuffled, and each
    block is shuffled internally — so a batch stays inside one block, the working set is one
    block wide, and each shard is read once per epoch. Size the block to the reader's cache
    (a whole number of shards) and the shuffle is still random within a window far larger
    than any batch.

    Args:
        num_samples: Size of the corpus.
        epoch: Reshuffles the order; pass the same value on every rank.
        seed: Reproduces the order across runs.
        shuffle: When false, the identity order.
        block_size: Shuffle within blocks of this many samples (and shuffle the blocks)
            instead of across the whole corpus. ``None`` shuffles globally.

    Returns:
        A lazy sequence that is a permutation of ``range(num_samples)``.

    Raises:
        ValueError: If `num_samples` is negative, or `block_size` is below 1.

    Examples:
        .. doctest::

            >>> from batcher.ml import epoch_permutation
            >>> order = epoch_permutation(10**12, seed=1)  # a trillion samples, instantly
            >>> len(order)
            1000000000000
            >>> sorted(epoch_permutation(5, seed=1))  # still a permutation
            [0, 1, 2, 3, 4]
            >>> blocked = epoch_permutation(8, seed=1, block_size=4)
            >>> sorted(blocked[:4]) == [0, 1, 2, 3] or sorted(blocked[:4]) == [4, 5, 6, 7]
            True
    """
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if block_size is not None and block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if not shuffle or num_samples <= 1:
        return range(num_samples)
    # One key per (seed, epoch): a new epoch reshuffles, the same pair reproduces.
    key = _mix64(seed * 0x9E3779B1 + epoch)
    if block_size is not None and block_size < num_samples:
        return _BlockPermutation(num_samples, block_size, key)
    return _FeistelPermutation(num_samples, key)
