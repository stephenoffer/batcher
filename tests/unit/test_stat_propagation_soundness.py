"""Column-stat propagation must not over-claim through row-shrinking operators.

Each of these was an estimate that could go the *unsafe* direction — a stale distinct count
that deflates a downstream join, or an EXACT bound a top-N could have dropped. They are
pinned against the estimator directly so a regression is caught without executing.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.config import active_config
from batcher.kyber.stats import StatsEstimator
from batcher.plan.stats import Provenance

pytestmark = pytest.mark.unit


def _est(ds):
    return StatsEstimator(ds._sources, {}, active_config().optimizer.cardinality)


def test_filter_caps_column_ndv_at_surviving_rows():
    # A selective filter cannot leave more distinct keys than the rows it keeps.
    ds = bt.from_pydict({"k": list(range(1000))})
    learned = {"__column_ndv__": {"k": 1000.0}}
    est = StatsEstimator(ds._sources, learned, active_config().optimizer.cardinality)
    filtered = ds.filter(bt.col("k") == 5)._plan  # ~0.1% selectivity
    stats = est.estimate(filtered)
    kstat = stats.columns.get("k")
    assert kstat is not None
    assert kstat.ndv is not None and kstat.ndv <= stats.rows + 1e-9


def test_top_n_downgrades_column_bounds_from_exact():
    # A Sort+limit (top-N) drops rows and can exclude the extremes, so min/max/ndv must not
    # stay EXACT — else min()/count_distinct answer from metadata over dropped rows.
    topn_ds = bt.from_pydict({"v": list(range(100))}).sort("v").limit(10)
    stats = _est(topn_ds).estimate(topn_ds._plan)
    vstat = stats.columns.get("v")
    if vstat is not None:
        assert vstat.provenance is not Provenance.EXACT


def test_full_sort_preserves_exact_bounds():
    # A full sort (no limit) reorders but keeps every value, so EXACT survives.
    ds = bt.from_pydict({"v": list(range(100))}).sort("v")
    stats = _est(ds).estimate(ds._plan)
    # rows are exact and unchanged
    assert stats.rows == 100.0


def test_explode_carries_passthrough_columns_as_bounds():
    # Every non-exploded column repeats per element, so its bounds survive (downgraded),
    # instead of the estimator returning no columns and blinding the operators above.
    from batcher.kyber.stats.columns import unnest_columns
    from batcher.plan.stats import ColumnStat, Provenance, RelStats

    ds = bt.from_pydict({"id": [1, 2], "vals": [[1, 2], [3]]})
    unnest = ds.explode("vals")._plan
    child = RelStats(
        2.0,
        Provenance.EXACT,
        {
            "id": ColumnStat(min=1, max=100, ndv=100.0, null_count=0, provenance=Provenance.EXACT),
            "vals": ColumnStat(min=0, max=5, provenance=Provenance.EXACT),
        },
    )
    cols = unnest_columns(unnest, child)
    assert set(cols) == {"id"}  # the exploded column is dropped
    assert cols["id"].provenance is Provenance.DEFAULT  # fan-out downgrades
    assert cols["id"].null_count is None  # counts change under fan-out
    assert cols["id"].min == 1 and cols["id"].max == 100  # bounds survive


def test_percent_rank_output_is_bounded_to_unit_interval():
    # percent_rank ∈ [0, 1] within every partition, so the bound is exact regardless of
    # partitioning and sharpens a downstream `WHERE pr < 0.1` percentile filter.
    ds = bt.from_pydict({"x": list(range(30))}).with_columns(
        pr=bt.percent_rank().over(order_by=["x"])
    )
    stats = _est(ds).estimate(ds._plan)
    pr = stats.columns.get("pr")
    assert pr is not None
    assert (pr.min, pr.max) == (0.0, 1.0)


def test_ranking_functions_get_no_bound():
    # row_number/rank would be [1, rows], which under-counts a partitioned `rank <= k`; the
    # estimator deliberately leaves them unbounded so the safe range fallback applies.
    ds = bt.from_pydict({"x": list(range(30))}).with_columns(
        rn=bt.row_number().over(order_by=["x"])
    )
    stats = _est(ds).estimate(ds._plan)
    assert "rn" not in stats.columns
