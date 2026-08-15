"""The block-local epoch permutation: still a bijection, but with a working set.

A globally shuffled epoch and a bounded shard cache cannot both work: every batch of a
global shuffle touches as many shards as it has samples, so the cache misses on nearly all
of them and the epoch re-reads the corpus once per batch. `epoch_permutation(block_size=...)`
buys a working set back by shuffling within blocks and shuffling the blocks. These tests hold
it to the properties the loader depends on — it must still be a permutation, it must still be
reproducible from ``(seed, epoch)``, and its locality must be real.
"""

from __future__ import annotations

import numpy as np
import pytest

from batcher.ml.permutation import _BlockPermutation, _KeyedPermutation, epoch_permutation


@pytest.mark.parametrize("n", [1, 2, 3, 7, 16, 17, 100, 1000])
@pytest.mark.parametrize("block", [1, 2, 3, 16, 64, 997])
def test_a_blocked_order_is_still_a_permutation(n, block):
    order = epoch_permutation(n, seed=5, epoch=2, block_size=block)
    assert sorted(order[i] for i in range(n)) == list(range(n))


@pytest.mark.parametrize("block", [1, 5, 64, 512])
def test_vectorized_take_agrees_with_per_index_lookup(block):
    # The loader reads through `take`; a divergence between the two would give the vectorized
    # path a different epoch from the scalar one, silently.
    n = 1000
    order = epoch_permutation(n, seed=3, block_size=block)
    assert isinstance(order, _KeyedPermutation)
    positions = np.arange(n, dtype=np.uint64)
    assert order.take(positions).tolist() == [order[i] for i in range(n)]


def test_a_block_of_positions_lands_in_one_block_of_samples():
    n, block = 1_000_000, 65_536
    window = np.arange(0, 8192, dtype=np.uint64)
    blocked = epoch_permutation(n, seed=1, block_size=block).take(window)
    globally = epoch_permutation(n, seed=1).take(window)
    assert len({int(v) // block for v in blocked}) == 1
    # The same window under a global shuffle is spread over essentially every block, which
    # is the read pattern a bounded shard cache cannot serve.
    assert len({int(v) // block for v in globally}) > 10


def test_the_order_is_reproducible_and_epoch_dependent():
    take = lambda **kw: epoch_permutation(500, block_size=32, **kw).take(  # noqa: E731
        np.arange(500, dtype=np.uint64)
    )
    assert np.array_equal(take(seed=7, epoch=0), take(seed=7, epoch=0))
    assert not np.array_equal(take(seed=7, epoch=0), take(seed=7, epoch=1))
    assert not np.array_equal(take(seed=7, epoch=0), take(seed=8, epoch=0))


def test_a_block_at_least_as_wide_as_the_corpus_is_the_global_shuffle():
    # No point paying for two-scale arithmetic when one block covers everything.
    assert not isinstance(epoch_permutation(100, seed=1, block_size=100), _BlockPermutation)
    assert not isinstance(epoch_permutation(100, seed=1, block_size=1000), _BlockPermutation)
    assert isinstance(epoch_permutation(100, seed=1, block_size=99), _BlockPermutation)


def test_a_block_size_below_one_is_rejected():
    with pytest.raises(ValueError, match="block_size must be >= 1"):
        epoch_permutation(10, block_size=0)


def test_the_trailing_partial_block_is_covered_and_shuffled():
    # 250 = 3 full blocks of 64 plus a 58-sample remainder. The remainder is the easiest
    # thing to drop or to double-count, and either would corrupt an epoch.
    order = epoch_permutation(250, seed=2, block_size=64)
    values = [order[i] for i in range(250)]
    assert sorted(values) == list(range(250))
    tail = values[192:]
    assert sorted(tail) == list(range(192, 250))
    assert tail != list(range(192, 250)), "the trailing block was left unshuffled"
