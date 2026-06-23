"""Sketch-driven cardinality: recorded column stats sharpen the estimator (W2)."""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.learning import load_learned_stats, record_column_stats
from batcher.metadata import MetadataHub
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.plan.logical import Filter


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


def test_record_and_load_column_stats_round_trip():
    hub = _hub()
    record_column_stats(
        hub,
        {"a": 100.0},
        {"a": {"probs": [0.0, 0.5, 1.0], "values": [0.0, 50.0, 100.0]}},
    )
    learned = load_learned_stats(hub)
    assert learned["__column_ndv__"]["a"] == 100.0
    assert learned["__column_quantiles__"]["a"]["values"][-1] == 100.0


def test_recording_is_merge_not_overwrite():
    hub = _hub()
    record_column_stats(hub, {"a": 10.0}, {})
    record_column_stats(hub, {"b": 20.0}, {})
    learned = load_learned_stats(hub)
    assert learned["__column_ndv__"] == {"a": 10.0, "b": 20.0}


def test_ndv_sharpens_equality_selectivity():
    ds = bt.from_pydict({"a": list(range(100))})
    plan = ds.filter(col("a") == 5)._plan
    assert isinstance(plan, Filter)

    # Default (no learned ndv): equality uses the small constant selectivity.
    base = CardinalityEstimator(ds._sources, {}).estimate(plan).rows
    # With a learned ndv of 100, `a = 5` keeps ~1/100 of 100 rows = 1.
    learned = {"__column_ndv__": {"a": 100.0}}
    sharp = CardinalityEstimator(ds._sources, learned).estimate(plan).rows
    assert sharp < base
    assert abs(sharp - 1.0) < 1e-6


def test_record_and_load_avg_byte_width():
    hub = _hub()
    record_column_stats(hub, {"a": 10.0}, {}, {"a": 4096.0})
    learned = load_learned_stats(hub)
    assert learned["__column_avg_bytes__"]["a"] == 4096.0

    # row_width sums learned widths over the node's output columns; with no
    # measured width it returns the supplied flat default (cold-start parity).
    ds = bt.from_pydict({"a": list(range(10))})
    est = CardinalityEstimator(ds._sources, learned)
    assert est.row_width(ds._plan, default=64.0) == 4096.0
    cold = CardinalityEstimator(ds._sources, {})
    assert cold.row_width(ds._plan, default=64.0) == 64.0


def test_record_column_stats_best_effort_on_none_hub():
    # Must never raise when there is no hub or nothing to record.
    record_column_stats(None, {"a": 1.0}, {})
    record_column_stats(_hub(), {}, {})
    record_column_stats(_hub(), {}, {}, {})


def test_grouped_aggregate_cardinality_uses_ndv():
    # 100 rows grouped by `k` (4 distinct) should estimate ~4 groups with learned
    # ndv, not the flat 10%-of-input (=10) default.
    ds = bt.from_pydict({"k": [i % 4 for i in range(100)], "v": list(range(100))})
    plan = ds.group_by("k").agg(total=col("v").sum())._plan

    default_est = CardinalityEstimator(ds._sources, {}).estimate(plan).rows
    learned = {"__column_ndv__": {"k": 4.0}}
    ndv_est = CardinalityEstimator(ds._sources, learned).estimate(plan).rows

    assert ndv_est == 4.0
    assert ndv_est < default_est  # 4 groups vs the 10% (=10) default


def test_two_key_aggregate_cardinality_is_capped_at_input():
    # product of ndvs (50 * 50 = 2500) is capped at the 100-row input.
    ds = bt.from_pydict({"a": list(range(100)), "b": list(range(100)), "v": list(range(100))})
    plan = ds.group_by("a", "b").agg(total=col("v").sum())._plan
    learned = {"__column_ndv__": {"a": 50.0, "b": 50.0}}
    est = CardinalityEstimator(ds._sources, learned).estimate(plan).rows
    assert est == 100.0  # capped at input size
