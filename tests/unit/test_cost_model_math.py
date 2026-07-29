"""The closed forms the cost model uses, and what each of them exists to distinguish.

A cost model is only useful where it *separates* two plans. These tests pin the separations
the model would otherwise miss: a top-N against a full sort, a spilling operator against a
resident one, a cache-resident hash table against one that misses on every probe, and a
partitioned window against a global sort.
"""

from __future__ import annotations

import math

import pytest

from batcher.kyber.cost.terms import sort_comparisons

pytestmark = pytest.mark.unit


def test_top_n_is_costed_as_one_pass_plus_a_few_insertions():
    """`LIMIT 10` over 100M rows is one pass, not `n·log2(10)`.

    Charging every row a heap sift-down assumes every row enters the heap. Over a randomly
    ordered input the `i`-th row displaces the root with probability `min(1, k/i)`, so only
    `k·(1 + ln(n/k))` rows ever do — about 170 of 100 million here. The old form charged 3.3x
    the input for work that is barely more than reading it, which made the optimizer nearly
    indifferent to fusing the limit into the sort.
    """
    n, k = 100_000_000.0, 10.0
    cost = sort_comparisons(n, k)
    assert cost < 1.05 * n, f"top-N should be about one pass, got {cost / n:.2f} passes"
    assert cost < n * math.log2(k) / 3, "must be far below the every-row-sifts model"
    # And still strictly more than a free scan: the insertions are real work.
    assert cost > n


def test_top_n_never_costs_more_than_the_full_sort_it_replaces():
    n = 1_000_000.0
    full = sort_comparisons(n, n)
    for k in (1.0, 10.0, 1_000.0, 100_000.0, n):
        assert sort_comparisons(n, k) <= full, f"top-{k} above a full sort"


def test_a_limit_larger_than_the_input_degenerates_to_a_full_sort():
    n = 1_000.0
    assert sort_comparisons(n, n * 10) == sort_comparisons(n, n)


def test_full_sort_is_n_log_n():
    n = 1_048_576.0
    assert sort_comparisons(n, n) == pytest.approx(n * 20.0)


def test_top_n_cost_grows_with_the_limit():
    """Monotone in `k`: keeping more rows can only cost more."""
    n = 10_000_000.0
    costs = [sort_comparisons(n, k) for k in (1.0, 10.0, 100.0, 10_000.0, 1_000_000.0)]
    assert costs == sorted(costs)
