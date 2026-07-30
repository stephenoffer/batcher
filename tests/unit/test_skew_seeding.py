"""Skew is the one distribution the estimator cannot approximate without measuring it.

Under uniformity an equality is `1/ndv`, which is exactly wrong on a skewed key — and a
skewed key is the one a join is usually built on. `learn_column_stats` measures the
most-common values, but only *after* a query of the shape has run, so the first one plans
blind. For a **resident** source the data is already in memory, which is the same argument
that justifies seeding the distinct count there.

Measured on 100,000 rows with one value at 50%: `k = 7` was estimated at 20 rows against a
true 49,868 — a 2,487x under-estimate that then sizes the filter, the join built on its
output, and that join's memory envelope.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import core
from batcher.api.source_stats import collect_source_stats, column_bounds_needed
from batcher.api.terminal._metadata import seed_column_ndv
from batcher.kyber import load_learned_stats
from batcher.kyber.cardinality import CardinalityEstimator

pytestmark = pytest.mark.unit

_N = 20_000


def _skewed(hot_fraction: float, hot=7):
    rng = np.random.default_rng(0)
    arr = np.where(rng.random(_N) < hot_fraction, hot, rng.integers(100, 5000, _N))
    return bt.from_arrow(pa.table({"k": pa.array(arr.astype("int64"))})), arr


def _cold_estimate(ds, plan):
    """The estimate a *first* run gets — statistics seeded, nothing measured by execution."""
    hub = core.default_hub()
    stats = collect_source_stats(ds._sources, hub, need_columns=column_bounds_needed(plan))
    seed_column_ndv(hub, ds._sources, plan)
    estimator = CardinalityEstimator(ds._sources, load_learned_stats(hub), source_stats=stats)
    return estimator.estimate(plan).rows


def test_a_hot_key_is_estimated_from_its_measured_frequency():
    ds, arr = _skewed(0.5)
    plan = ds.filter(bt.col("k") == 7)._plan
    true = int((arr == 7).sum())
    estimate = _cold_estimate(ds, plan)
    # Within 25% of the truth, where `1/ndv` uniformity was three orders of magnitude under.
    assert 0.75 * true <= estimate <= 1.25 * true, (estimate, true)


def test_a_cold_hot_key_beats_the_uniform_guess_by_orders_of_magnitude():
    ds, arr = _skewed(0.5)
    plan = ds.filter(bt.col("k") == 7)._plan
    uniform = _N / len(np.unique(arr))  # what `1/ndv` would have said
    assert _cold_estimate(ds, plan) > 100 * uniform


def test_an_unskewed_column_is_not_disturbed():
    """The safety property: a uniform column has no heavy hitter, so nothing is recorded
    and the estimate is the uniform one it always was."""
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 1000, _N)
    ds = bt.from_arrow(pa.table({"k": pa.array(arr.astype("int64"))}))
    plan = ds.filter(bt.col("k") == 7)._plan
    true = int((arr == 7).sum())
    estimate = _cold_estimate(ds, plan)
    assert 0.5 * true <= estimate <= 2.0 * true, (estimate, true)


def test_the_distinct_count_is_still_seeded():
    """Adding most-common-values must not displace the statistic that was already seeded."""
    ds, arr = _skewed(0.5)
    plan = ds.group_by("k").agg(n=bt.col("k").count())._plan
    groups = len(np.unique(arr))
    estimate = _cold_estimate(ds, plan)
    assert 0.9 * groups <= estimate <= 1.1 * groups, (estimate, groups)


def test_execution_still_converges_to_the_truth():
    """Seeding is a cold-start floor, not a replacement for measurement.

    The post-run pass records the *measured* selectivity for the shape, which is exact —
    so a second run must be at least as good as the seeded first one, never worse.
    """
    ds, arr = _skewed(0.5)
    query = ds.filter(bt.col("k") == 7)
    true = int((arr == 7).sum())
    assert query.collect().num_rows == true
    warm = _cold_estimate(ds, query._plan)
    assert 0.9 * true <= warm <= 1.1 * true, (warm, true)
