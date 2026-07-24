"""Sizing the salt fan-out for a skewed join key.

Salting splits a hot key across sub-partitions so one reducer does not receive the whole
key. How many sub-partitions that takes is not a constant: with `P` reducers the average
reducer gets `N/P` rows while the hot key's reducer gets `f·N`, so the overload is `f·P` —
a quantity that grows with the cluster. These tests pin that the fan-out tracks it.
"""

from __future__ import annotations

import pytest

from batcher.dist.skew import salt_factor

pytestmark = pytest.mark.unit


def test_salt_levels_the_hot_reducer_with_the_average():
    """After salting, the hot key's per-reducer load must be at or below average."""
    for fraction in (0.05, 0.1, 0.25, 0.5):
        for partitions in (4, 16, 64, 200):
            salt = salt_factor(fraction, partitions)
            # Hot reducer holds `f/salt` of the data; the average reducer holds `1/P`.
            assert fraction / salt <= 1.0 / partitions + 1e-12 or salt == 64, (
                f"f={fraction} P={partitions} salt={salt} leaves the hot reducer overloaded"
            )


def test_a_wide_shuffle_needs_more_fan_out_than_a_narrow_one():
    """The failure a constant fan-out has: it is sized for one cluster width.

    A 10% key across 200 reducers overloads by 20x; a fan-out of 4 leaves it 5x over
    average, which is the straggler salting exists to remove.
    """
    assert salt_factor(0.1, 200) > salt_factor(0.1, 16)
    assert salt_factor(0.1, 200) >= 20


def test_salt_is_bounded_so_replication_cannot_run_away():
    """Each sub-partition replicates the matching build rows, so the fan-out is capped."""
    assert salt_factor(1.0, 100_000) <= 64


def test_salting_at_all_means_at_least_two_ways():
    assert salt_factor(0.001, 2) == 2
    assert salt_factor(0.0, 100) == 2
    assert salt_factor(0.5, 1) == 2


def test_hotness_scales_with_the_shuffle_width():
    """A fixed fraction cannot be right for every cluster size.

    Every reducer's fair share is `1/P`, so a value at 5% is harmless across 4 reducers and a
    10x straggler across 200.
    """
    from batcher.kyber.stats.skew import _overloading

    freq = {"7": 0.05}
    assert _overloading(freq, {}, 100.0, 100.0, 4) == set()
    assert _overloading(freq, {}, 100.0, 100.0, 200) == {"7"}


def test_a_join_is_a_product_so_unremarkable_frequencies_can_still_overload():
    """The straggler an input-side fraction test cannot see.

    The join's whole output is `S·|L||R|` for a total match probability `S` far below 1, so a
    value's share of the *output* is `f_L·f_R/S` — amplified by the key's distinct count.
    Two frequencies that are individually well under one reducer's fair share can still hand
    one reducer most of the join.
    """
    from batcher.kyber.stats.skew import _overloading

    # 1% on each side of a key with 100,000 distinct values: neither input is hot at 32
    # reducers (fair share 3.1%), but the pair carries a large share of the output.
    left, right = {"7": 0.01}, {"7": 0.01}
    assert _overloading(left, right, 100_000.0, 100_000.0, 32) == {"7"}
    # A key with few distinct values spreads the output evenly, so the same frequencies are
    # genuinely harmless.
    assert _overloading(left, right, 20.0, 20.0, 32) == set()


def test_frequencies_on_different_values_do_not_multiply():
    """Only a value present on *both* sides concentrates output rows."""
    from batcher.kyber.stats.skew import _overloading

    assert _overloading({"7": 0.01}, {"3": 0.01}, 100_000.0, 100_000.0, 32) == set()


def test_a_single_partition_shuffle_has_no_skew_to_fix():
    from batcher.kyber.stats.skew import _overloading

    assert _overloading({"7": 0.9}, {"7": 0.9}, 10.0, 10.0, 1) == set()
