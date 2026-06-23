"""Measured filter selectivity flows Core → Kyber and drives estimates.

This closes a metadata loop that was previously dead: the estimator's learned
`selectivity` key was never populated. Now a filter's measured kept-fraction is
recorded by signature and — being a ratio — generalizes across input sizes, unlike
a learned absolute row count.

Uses an isolated in-dict hub so the test never depends on the process-wide
MetadataHub's accumulated state.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col, count, kyber
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.learning import _NAMESPACE, _filter_over_scan
from batcher.kyber.signature import plan_signature
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend
from batcher.plan.stats import Provenance


def _DictHub() -> MetadataHub:
    """An isolated in-process MetadataHub for the learning loop (per-key param store)."""
    return MetadataHub(InProcessBackend())


def _filter_plan(rows: int):
    ds = bt.from_arrow(pa.table({"lsx": list(range(rows))})).filter(col("lsx") < 400)
    return ds._plan, ds._sources


def test_measured_selectivity_recorded_and_generalizes():
    hub = _DictHub()
    plan, sources = _filter_plan(1000)  # lsx<400 over 1000 → kept 400 → selectivity 0.4
    kyber.record_selectivity(hub, plan, sources, 400)

    learned = hub.load_keyed_params(_NAMESPACE)
    sig = plan_signature(plan)
    assert abs(learned[sig]["selectivity"] - 0.4) < 1e-6

    # Kyber applies the measured ratio to a DIFFERENT (2000-row) input: 2000 * 0.4.
    plan2, sources2 = _filter_plan(2000)
    est = CardinalityEstimator(sources2, learned)
    e = est.estimate(plan2)
    assert e.provenance is Provenance.LEARNED
    assert abs(e.rows - 800) < 1  # not the stale absolute 400


def test_selectivity_smoothing_across_runs():
    hub = _DictHub()
    plan, sources = _filter_plan(1000)
    kyber.record_selectivity(hub, plan, sources, 400)  # 0.4
    kyber.record_selectivity(hub, plan, sources, 200)  # 0.2 → smoothed 0.5*0.2+0.5*0.4
    sig = plan_signature(plan)
    assert abs(hub.load_keyed_params(_NAMESPACE)[sig]["selectivity"] - 0.3) < 1e-6


def test_record_execution_preserves_selectivity():
    hub = _DictHub()
    plan, sources = _filter_plan(100)
    kyber.record_selectivity(hub, plan, sources, 50)  # selectivity 0.5
    kyber.record_execution(hub, plan, 50)  # must not clobber selectivity
    entry = hub.load_keyed_params(_NAMESPACE)[plan_signature(plan)]
    assert "selectivity" in entry and "rows" in entry


def test_filter_over_scan_only():
    # A non-filter-topped plan records no selectivity.
    agg = bt.from_arrow(pa.table({"k": [1, 2]})).group_by("k").agg(n=count())._plan
    assert _filter_over_scan(agg) is None
    hub = _DictHub()
    kyber.record_selectivity(hub, agg, [], 1)
    assert hub.load_keyed_params(_NAMESPACE) == {}  # nothing recorded
