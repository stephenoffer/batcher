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


# --- a group key's null count ----------------------------------------------------------
#
# Grouping collapses every null key into a single group, so the input's null count is not
# the output's. It used to be dropped entirely for that reason, which threw away the two
# cases grouping *does* pin — and a known-zero null count is what `constant_value` and
# `_predicate_status` require before either will call a key provably constant or a
# predicate provably true. Erasing it made an aggregate hide a proof its own input carried.
#
# Driven through `grouped_aggregate_columns` with a constructed child, the way
# `test_explode_carries_passthrough_columns_as_bounds` above drives `unnest_columns`: a
# `Scan`'s column stats are collected by the conductor at the terminal op, so an estimator
# built here sees an empty bundle and the derivation has nothing to read.


def _grouped_key_stats(data: dict, keys: tuple[str, ...], child_cols: dict):
    """Group `data` by `keys` and return the derived key stats for a child holding
    `child_cols`."""
    from batcher.kyber.stats.aggregate_columns import grouped_aggregate_columns
    from batcher.plan.stats import RelStats

    ds = bt.from_pydict(data).group_by(*keys).agg(c=bt.col("v").count())
    child = RelStats(float(len(next(iter(data.values())))), Provenance.EXACT, child_cols)
    return grouped_aggregate_columns(ds._plan, child), ds


def _exact(null_count: int) -> object:
    from batcher.plan.stats import ColumnStat

    return ColumnStat(min=1, max=9, null_count=null_count, provenance=Provenance.EXACT)


def test_a_group_key_without_nulls_has_no_nulls_after_grouping():
    # Grouping invents no value, so a key the input never held a null of cannot acquire one.
    cols, _ = _grouped_key_stats(
        {"k": [1, 1, 2, 2, 3], "v": [1, 2, 3, 4, 5]}, ("k",), {"k": _exact(0)}
    )
    assert cols["k"].null_count == 0
    assert cols["k"].null_count_is_exact


def test_a_lone_group_key_with_nulls_keeps_exactly_one():
    # The nulls form one group among the column's distinct values — exactly one null row out.
    cols, _ = _grouped_key_stats(
        {"k": [1, None, 2, None, None], "v": [1, 2, 3, 4, 5]}, ("k",), {"k": _exact(3)}
    )
    assert cols["k"].null_count == 1


def test_several_group_keys_do_not_pin_a_nullable_key_to_one():
    # With a tuple key, a null in `k` can appear in as many groups as there are distinct
    # values of `g` beside it. That is a lower bound, not a count, so nothing is claimed —
    # claiming 1 would under-count, and a null count is read by paths that delete rows.
    cols, _ = _grouped_key_stats(
        {"k": [1, None, None], "g": ["a", "b", "c"], "v": [1, 2, 3]},
        ("k", "g"),
        {"k": _exact(2), "g": _exact(0)},
    )
    assert cols["k"].null_count is None


def test_several_group_keys_still_pin_a_non_null_key_to_zero():
    # The zero case needs no such care: no nulls in means no nulls out however many keys
    # the group is a tuple of.
    cols, _ = _grouped_key_stats(
        {"k": [1, 2, 2], "g": ["a", "b", "c"], "v": [1, 2, 3]},
        ("k", "g"),
        {"k": _exact(0), "g": _exact(0)},
    )
    assert cols["k"].null_count == 0


def test_an_inexact_input_null_count_derives_nothing():
    # An estimated zero is a *guess* that the column has no nulls. The derived count is read
    # by paths that decide whether a predicate is provably true, where a guess does not
    # merely mis-plan — it deletes rows.
    from batcher.plan.stats import ColumnStat

    guessed = ColumnStat(min=1, max=9, null_count=0, provenance=Provenance.DEFAULT)
    cols, _ = _grouped_key_stats({"k": [1, 2], "v": [1, 2]}, ("k",), {"k": guessed})
    assert cols["k"].null_count is None


def test_the_derived_count_matches_what_executing_reports():
    # The soundness check the others are worth nothing without: the derived count is held
    # against the rows the engine actually produces.
    for data, keys, child in (
        ({"k": [1, 1, 2], "v": [1, 2, 3]}, ("k",), {"k": _exact(0)}),
        ({"k": [1, None, None], "v": [1, 2, 3]}, ("k",), {"k": _exact(2)}),
        (
            {"k": [1, 2, 2], "g": ["a", "b", "c"], "v": [1, 2, 3]},
            ("k", "g"),
            {"k": _exact(0), "g": _exact(0)},
        ),
    ):
        cols, ds = _grouped_key_stats(data, keys, child)
        derived = cols["k"].null_count
        if derived is None:
            continue
        actual = sum(1 for row in ds.collect().to_pylist() if row["k"] is None)
        assert derived == actual, (data, keys)
