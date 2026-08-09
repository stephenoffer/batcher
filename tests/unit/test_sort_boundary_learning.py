"""Metadata-driven range-sort boundaries: the quantile grids the SAMPLE barrier measures
are persisted by sort shape and reused on later runs, so a repeat sort range-partitions
without re-executing its mapped prefix over the whole input.

The loop is result-preserving — boundaries decide only which reducer a row lands on, and
the ordered concat is correct for any monotone boundary list — so these tests pin the
persistence semantics and the balance property. `tests/integration/` proves the sorted
relation itself is unchanged end to end.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.partition_io import merge_boundaries
from batcher.dist.sort_boundaries import (
    load_learned_grids,
    persist_grids,
    sort_shape_key,
)

pytestmark = pytest.mark.unit


def test_sort_shape_key_is_stable_and_shape_specific():
    key = sort_shape_key("MAP_IR", "k")
    assert key == sort_shape_key("MAP_IR", "k")  # deterministic
    assert len(key) == 16  # short hash

    # The mapped prefix is part of the shape: the same column behind a different
    # predicate has a genuinely different key distribution and must not share a grid.
    assert sort_shape_key("MAP_IR_FILTERED", "k") != key
    # And so is the key column.
    assert sort_shape_key("MAP_IR", "other") != key


def test_learned_grids_round_trip_and_none_when_never_measured():
    key = sort_shape_key("MAP_IR_ROUNDTRIP", "k")

    # Never sampled → None, so the caller knows it must run the barrier.
    assert load_learned_grids(key) is None

    persist_grids(key, [([0.0, 5.0, 10.0], 100), ([2.0, 6.0, 12.0], 50)])
    assert load_learned_grids(key) == [([0.0, 5.0, 10.0], 100), ([2.0, 6.0, 12.0], 50)]


def test_grids_describing_no_rows_are_not_stored():
    """An empty split contributes nothing to the mixture, so storing it only inflates
    the payload a later run has to read back."""
    key = sort_shape_key("MAP_IR_EMPTY_SPLITS", "k")
    persist_grids(key, [([], 0), ([1.0, 2.0], 10), ([3.0, 4.0], 0)])
    assert load_learned_grids(key) == [([1.0, 2.0], 10)]

    # All-empty stores nothing at all, so the next run samples rather than reusing a
    # grid that describes no rows.
    all_empty = sort_shape_key("MAP_IR_ALL_EMPTY", "k")
    persist_grids(all_empty, [([], 0)])
    assert load_learned_grids(all_empty) is None


def test_learned_grids_recut_to_whatever_bucket_count_this_run_resolved():
    """Grids persist, not boundaries.

    The reducer count moves between runs (the learned shuffle fan-out and the
    `max_shuffle_partitions` cap both trim it), and `merge_boundaries` must emit exactly
    `n_buckets - 1` boundaries — more would route rows past the last bucket. Storing the
    grids keeps that re-cut available; storing a boundary list would not.
    """
    key = sort_shape_key("MAP_IR_RECUT", "k")
    persist_grids(key, [([float(i) for i in range(33)], 1_000)])
    grids = load_learned_grids(key)
    assert grids is not None

    for n_buckets in (2, 4, 8):
        boundaries = merge_boundaries(grids, n_buckets)
        assert len(boundaries) == n_buckets - 1
        # Monotone and deduplicated, which is what makes the ordered concat correct.
        assert boundaries == sorted(boundaries)
        assert len(set(boundaries)) == len(boundaries)


def test_a_stale_grid_still_yields_usable_monotone_boundaries():
    """Staleness costs balance, never correctness.

    A grid learned on one distribution and re-cut against a bucket count it never saw
    still produces monotone, deduplicated boundaries, which is the entire precondition
    the range partitioner needs. This is why the optimization is safe to apply without
    validating that the data still matches.
    """
    key = sort_shape_key("MAP_IR_STALE", "k")
    persist_grids(key, [([0.0, 1.0, 2.0, 3.0], 10)])
    boundaries = merge_boundaries(load_learned_grids(key), 4)
    assert boundaries == sorted(boundaries)
    assert len(boundaries) == 3
