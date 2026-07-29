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
import pytest

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


# --- a filter *inside* a larger plan also learns, from the per-operator history ---------
#
# `record_selectivity` above is handed the whole query's output row count, so it can only
# speak for a filter that *is* the plan (`_filter_over_scan`). Core meanwhile measures every
# filter's own rows_in/rows_out and files it under the signature `annotate_ops` stamped — on
# every run, profiled or not. That history went unread, so a filter under a join, aggregate
# or sort re-derived a structural guess forever. These cover the derivation that closes it.


def _feedback(sig, sel, *, kind="filter", op_id=3):
    from batcher.plan.feedback import OperatorFeedback
    from batcher.plan.ids import OpId

    return OperatorFeedback(
        op_id=OpId(op_id),
        kind=kind,
        n_actual=int(1000 * sel),
        t_op_ms=1.0,
        m_peak_bytes=0,
        selectivity=sel,
        batch_size=16384,
        n_input=1000,
        signature=sig,
    )


def _min_samples():
    from batcher.config import active_config

    return active_config().optimizer.cardinality_correction_min_samples


def test_measured_history_yields_a_selectivity_for_a_nested_filter():
    from batcher.kyber.learning import load_learned_stats

    hub = _DictHub()
    for _ in range(_min_samples()):
        hub.record(_feedback("sig-nested", 0.25))
    assert load_learned_stats(hub)["sig-nested"]["selectivity"] == pytest.approx(0.25)


def test_one_observation_is_not_enough():
    """Below `min_samples` the structural estimate stands — one run cannot anchor a plan."""
    from batcher.kyber.learning import load_learned_stats

    hub = _DictHub()
    hub.record(_feedback("sig-thin", 0.25))
    if _min_samples() > 1:
        assert "sig-thin" not in load_learned_stats(hub)


def test_the_mean_of_recent_observations_is_used():
    from batcher.kyber.learning import load_learned_stats

    hub = _DictHub()
    hub.record(_feedback("sig-mean", 0.2))
    hub.record(_feedback("sig-mean", 0.4))
    assert load_learned_stats(hub)["sig-mean"]["selectivity"] == pytest.approx(0.3)


def test_only_filters_contribute():
    """A join's or aggregate's ratio is not a filter selectivity and must not be read as one."""
    from batcher.kyber.learning import load_learned_stats

    hub = _DictHub()
    for kind in ("hash_join", "aggregate", "sort"):
        for _ in range(_min_samples() + 1):
            hub.record(_feedback(f"sig-{kind}", 0.25, kind=kind))
    stats = load_learned_stats(hub)
    assert all(
        "selectivity" not in (stats.get(f"sig-{k}") or {})
        for k in ("hash_join", "aggregate", "sort")
    )


def test_an_unsigned_row_contributes_nothing():
    """Distributed workers report op ids for their own sub-plan and send no signature."""
    from batcher.kyber.learning import load_learned_stats

    hub = _DictHub()
    for _ in range(_min_samples() + 1):
        hub.record(_feedback("", 0.25))
    assert load_learned_stats(hub) == {}


def test_an_explicitly_recorded_selectivity_wins():
    """`record_selectivity`'s keyed param is authoritative; the derivation only fills gaps."""
    from batcher.kyber.learning import load_learned_stats

    hub = _DictHub()
    plan, sources = _filter_plan(1000)
    kyber.record_selectivity(hub, plan, sources, 400)  # 0.4, stored under the plan signature
    sig = plan_signature(plan)
    for _ in range(_min_samples() + 1):
        hub.record(_feedback(sig, 0.9))  # a contradicting measured history
    assert load_learned_stats(hub)[sig]["selectivity"] == pytest.approx(0.4)
