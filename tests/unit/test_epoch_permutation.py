"""The keyed epoch permutation: a bijection, computed in O(1) memory, identical everywhere.

A shuffled *list* of every sample index costs ~28 bytes per sample in CPython — 280 GB
of driver RAM for a 10-billion-sample corpus. `epoch_permutation` replaces it with a
keyed pseudorandom bijection on ``[0, n)``: index in, shuffled index out, no state. That
makes an exabyte-scale epoch (and an O(1) mid-epoch seek) possible, but only if three
things hold, so each is pinned here:

* it really is a **bijection** — a permutation that repeats one index and drops another
  silently trains twice on one sample and never on the other;
* the vectorized batch path is **bit-identical** to the scalar one — a rank that
  disagrees with its peers strides a different order, and the epoch both repeats and
  skips samples across the job;
* it is **deterministic** across processes — so no `random` module, no hash
  randomization, no float arithmetic.

Cheap statistical checks guard against a permutation that is a bijection but obviously
structured (e.g. an affine map), which would correlate a rank's samples with its index.
"""

from __future__ import annotations

import numpy as np
import pytest

from batcher.ml import epoch_permutation, rank_index_batches
from batcher.ml.permutation import _FeistelPermutation, _mix64, _mix64_vec
from batcher.ml.streaming_sampler import elastic_shard, epoch_order, num_rank_batches

pytestmark = pytest.mark.unit

_SIZES = [0, 1, 2, 3, 5, 8, 17, 64, 65, 100, 1000, 4096, 4097]


# --- bijection ---------------------------------------------------------------


@pytest.mark.parametrize("n", _SIZES)
def test_permutation_is_a_bijection(n):
    perm = epoch_permutation(n, seed=3, epoch=1)
    assert len(perm) == n
    assert sorted(perm[i] for i in range(n)) == list(range(n))


@pytest.mark.parametrize("n", [100003, 65536])
def test_permutation_is_a_bijection_at_size(n):
    perm = epoch_permutation(n, seed=9)
    assert sorted(perm[i] for i in range(n)) == list(range(n))


def test_permutation_without_shuffle_is_the_identity():
    assert list(epoch_permutation(10, shuffle=False)) == list(range(10))


def test_epoch_order_materializes_the_same_permutation():
    perm = epoch_permutation(50, seed=2, epoch=4)
    assert epoch_order(50, seed=2, epoch=4) == [perm[i] for i in range(50)]


# --- determinism -------------------------------------------------------------


def test_same_seed_and_epoch_reproduce_the_order():
    a = list(epoch_permutation(64, seed=7, epoch=2))
    b = list(epoch_permutation(64, seed=7, epoch=2))
    assert a == b


def test_a_new_epoch_reshuffles():
    a = list(epoch_permutation(64, seed=7, epoch=2))
    b = list(epoch_permutation(64, seed=7, epoch=3))
    assert a != b
    assert sorted(a) == sorted(b)


def test_a_new_seed_reshuffles():
    assert list(epoch_permutation(64, seed=7)) != list(epoch_permutation(64, seed=8))


def test_out_of_range_index_raises():
    perm = epoch_permutation(10, seed=1)
    with pytest.raises(IndexError):
        perm[10]


# --- scalar / vectorized agreement -------------------------------------------


def test_mix64_scalar_and_vectorized_agree():
    xs = [0, 1, 2, 3, 2**63, 2**64 - 1, 12345678901234567]
    got = _mix64_vec(np, np.array(xs, dtype=np.uint64))
    assert [int(v) for v in got] == [_mix64(x) for x in xs]


@pytest.mark.parametrize("n", [2, 3, 17, 64, 65, 1000, 4097, 100003])
def test_vectorized_take_matches_index_by_index(n):
    """A rank using the batch path must see exactly what a rank using `[i]` sees."""
    perm = epoch_permutation(n, seed=11, epoch=2)
    assert isinstance(perm, _FeistelPermutation)
    vectorized = [int(v) for v in perm.take(np.arange(n, dtype=np.uint64))]
    assert vectorized == [perm[i] for i in range(n)]


# --- it actually shuffles ----------------------------------------------------


def test_permutation_has_few_fixed_points():
    """A random permutation of n leaves ~1 index in place, whatever n is."""
    perm = epoch_permutation(10_000, seed=1)
    fixed = sum(1 for i in range(10_000) if perm[i] == i)
    assert fixed < 10, f"{fixed} fixed points — the permutation is barely shuffling"


def test_permutation_is_not_an_affine_map():
    """Consecutive indices must not map to consecutive (or evenly-spaced) outputs."""
    perm = epoch_permutation(65_536, seed=42)
    gaps = {perm[i + 1] - perm[i] for i in range(200)}
    assert len(gaps) > 150, "outputs of neighbouring indices are structured"


def test_permutation_spreads_indices_across_the_range():
    """The first 1/4 of indices should land roughly uniformly, not in one region."""
    n = 65_536
    perm = epoch_permutation(n, seed=42)
    bins = [0] * 16
    for i in range(n // 4):
        bins[perm[i] * 16 // n] += 1
    expected = (n // 4) / 16
    assert all(0.8 * expected < b < 1.2 * expected for b in bins), bins


# --- lazy per-rank batching --------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 5, 8, 13, 20, 37])
@pytest.mark.parametrize("world_size", [1, 2, 3])
@pytest.mark.parametrize("batch_size", [1, 2, 5])
@pytest.mark.parametrize("drop_last", [True, False])
def test_lazy_batches_equal_the_materialized_path(n, world_size, batch_size, drop_last):
    """`rank_index_batches` must be the streaming spelling of `elastic_shard`, exactly."""
    for rank in range(world_size):
        order = epoch_order(n, seed=4, epoch=1)
        reference = elastic_shard(order, world_size=world_size, rank=rank, drop_last=drop_last)
        limit = (len(reference) // batch_size) * batch_size if drop_last else len(reference)
        expected = [reference[i : i + batch_size] for i in range(0, limit, batch_size)]

        got = list(
            rank_index_batches(
                n,
                batch_size=batch_size,
                world_size=world_size,
                rank=rank,
                epoch=1,
                seed=4,
                drop_last=drop_last,
            )
        )
        assert got == expected


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_lazy_batches_resume_from_a_checkpoint(world_size):
    n, batch_size = 40, 2
    consumed = 4 * world_size  # four synchronized steps
    for rank in range(world_size):
        order = epoch_order(n, seed=5)
        reference = elastic_shard(order, world_size=world_size, rank=rank, global_consumed=consumed)
        limit = (len(reference) // batch_size) * batch_size
        expected = [reference[i : i + batch_size] for i in range(0, limit, batch_size)]
        got = list(
            rank_index_batches(
                n,
                batch_size=batch_size,
                world_size=world_size,
                rank=rank,
                seed=5,
                global_consumed=consumed,
            )
        )
        assert got == expected


@pytest.mark.parametrize("n", [0, 7, 16, 33])
@pytest.mark.parametrize("world_size", [1, 2, 3])
@pytest.mark.parametrize("drop_last", [True, False])
def test_num_rank_batches_predicts_what_is_yielded(n, world_size, drop_last):
    """A loader's `__len__` must equal what its `__iter__` produces — on every rank."""
    counts = set()
    for rank in range(world_size):
        predicted = num_rank_batches(
            n, batch_size=3, world_size=world_size, rank=rank, drop_last=drop_last
        )
        actual = sum(
            1
            for _ in rank_index_batches(
                n, batch_size=3, world_size=world_size, rank=rank, drop_last=drop_last
            )
        )
        assert predicted == actual
        counts.add(actual)
    assert len(counts) == 1, f"ranks disagree on epoch length: {counts}"


def test_batches_cover_the_rank_shard_without_duplicates():
    n, world_size, batch_size = 30, 3, 4
    seen = []
    for rank in range(world_size):
        for batch in rank_index_batches(
            n, batch_size=batch_size, world_size=world_size, rank=rank, seed=2, drop_last=False
        ):
            seen.extend(batch)
    assert sorted(seen) == list(range(n))


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        next(rank_index_batches(10, batch_size=0))
    with pytest.raises(ValueError, match="batch_size must be positive"):
        num_rank_batches(10, batch_size=0)


def test_a_huge_corpus_costs_constant_memory():
    """The whole point: an epoch over 10^15 samples must not allocate per-sample."""
    import tracemalloc
    from itertools import islice

    tracemalloc.start()
    batches = list(
        islice(rank_index_batches(10**15, batch_size=4096, world_size=1024, rank=7, seed=1), 3)
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert all(len(b) == 4096 for b in batches)
    assert all(0 <= i < 10**15 for b in batches for i in b)
    assert peak < 50_000_000, f"peak {peak:,} bytes — the epoch is being materialized"


def test_a_huge_corpus_seeks_in_constant_time():
    """Resuming near the end of a trillion-sample epoch must not walk to it."""
    from itertools import islice

    batches = list(
        islice(
            rank_index_batches(
                10**12,
                batch_size=8,
                world_size=1024,
                rank=7,
                seed=1,
                global_consumed=900_000_000_000,
            ),
            1,
        )
    )
    assert len(batches[0]) == 8
