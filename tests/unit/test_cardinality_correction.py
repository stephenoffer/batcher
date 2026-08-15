"""The learned cardinality-correction loop: Core measures q-error, Kyber corrects.

Every relational engine estimates cardinality structurally and gets it wrong. Batcher's
distinguishing claim is that it *measures* how wrong, per operator shape, and folds that
back into the next plan. This file pins the contract of that loop:

* the factor is a **geometric** average of `actual / estimated` (multiplicative, so a 4x
  over- and a 4x under-estimate cancel), recency-weighted and shrunk toward "no correction"
  by how much the samples agree — a consistent bias passes through, a noisy one does not,
* only the most recent `window` samples count (the structural estimator sharpens over
  time, so a stale correction must decay),
* an operator whose estimate did not come from the structural estimator contributes no
  sample — otherwise re-running a query would wash out its own correction,
* and the correction never upgrades provenance or touches an `EXACT` count.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

import batcher as bt
from batcher.config import Config, active_config
from batcher.kyber.correction import correction_factor
from batcher.kyber.learning import CARDINALITY_CORRECTION_KEY, load_learned_stats
from batcher.kyber.stats import StatsEstimator
from batcher.metadata import MetadataHub
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.plan.feedback import OperatorFeedback
from batcher.plan.ids import OpId
from batcher.plan.stats import Provenance


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


def _feed(hub: MetadataHub, signature: str, *, est: float, actual: int, kind: str = "hash_join"):
    hub.record(
        OperatorFeedback(
            op_id=OpId(0),
            kind=kind,
            n_actual=actual,
            t_op_ms=1.0,
            m_peak_bytes=0,
            selectivity=1.0,
            batch_size=1024,
            signature=signature,
            n_estimated=est,
        )
    )


def _config_with(**overrides) -> Config:
    cfg = active_config()
    return dataclasses.replace(cfg, optimizer=dataclasses.replace(cfg.optimizer, **overrides))


def _corrections(hub: MetadataHub) -> dict[str, float]:
    return load_learned_stats(hub).get(CARDINALITY_CORRECTION_KEY, {})


# --- deriving the factor ----------------------------------------------------------


@pytest.mark.unit
def test_no_correction_below_min_samples():
    hub = _hub()
    _feed(hub, "sig", est=100, actual=800)
    assert _corrections(hub) == {}, "a single observation must not steer a plan"


@pytest.mark.unit
def test_a_consistent_bias_passes_through_as_the_geometric_mean():
    hub = _hub()
    for _ in range(6):
        _feed(hub, "sig", est=100, actual=800)  # q = 8, every time
    # Samples that agree carry no uncertainty to shrink against, so the factor is the
    # geometric mean itself. (The arithmetic mean of q-errors would be the wrong average:
    # it does not make a 4x over- and a 4x under-estimate cancel.)
    assert _corrections(hub)["sig"] == pytest.approx(8.0, rel=1e-6)


@pytest.mark.unit
def test_scattered_samples_are_shrunk_toward_no_correction():
    # The same *average* error as above, but the runs disagree wildly about it. The mean of
    # a scattered sample is mostly noise, so it must not be stamped onto the next plan at
    # full strength: the correction lands between 1.0 and the raw geometric mean.
    hub = _hub()
    for q in (0.5, 128.0, 1.0, 64.0, 0.25, 256.0):
        _feed(hub, "sig", est=100, actual=int(100 * q))
    raw = math.exp(sum(math.log(q) for q in (0.5, 128.0, 1.0, 64.0, 0.25, 256.0)) / 6)
    factor = _corrections(hub)["sig"]
    assert 1.0 < factor < raw, f"expected shrinkage below the raw geometric mean {raw}"


@pytest.mark.unit
def test_symmetric_errors_leave_essentially_no_correction():
    hub = _hub()
    for _ in range(3):
        _feed(hub, "sig", est=100, actual=400)  # q = 4
        _feed(hub, "sig", est=100, actual=25)  # q = 1/4
    # The log q-errors average to zero, and what recency weighting leaves of the imbalance
    # is shrunk away by how much the samples disagree. An unbiased estimator must not
    # acquire a correction from its own noise.
    assert _corrections(hub).get("sig", 1.0) == pytest.approx(1.0, abs=0.05)


@pytest.mark.unit
def test_factor_is_clamped_both_ways():
    hub = _hub()
    for _ in range(3):
        _feed(hub, "under", est=1, actual=1_000_000)  # q = 1e6
        _feed(hub, "over", est=1_000_000, actual=1)  # q = 1e-6
    max_factor = active_config().optimizer.cardinality_correction_max_factor
    corr = _corrections(hub)
    assert corr["under"] == pytest.approx(max_factor)
    assert corr["over"] == pytest.approx(1.0 / max_factor)


@pytest.fixture
def narrow_window(monkeypatch):
    """Shrink the correction window to 2 observations for the duration of a test."""
    from batcher import config as config_mod

    cfg = _config_with(cardinality_correction_window=2)
    monkeypatch.setattr(config_mod, "active_config", lambda: cfg)
    monkeypatch.setattr("batcher.kyber.learning.active_config", lambda: cfg)
    return cfg


@pytest.mark.unit
def test_only_the_recent_window_counts(narrow_window):
    hub = _hub()
    # Two stale 100x under-estimates, then two runs where the estimator was right
    # (as happens once the column-stat loop learns the join key's NDV).
    _feed(hub, "sig", est=1, actual=100)
    _feed(hub, "sig", est=1, actual=100)
    _feed(hub, "sig", est=100, actual=100)
    _feed(hub, "sig", est=100, actual=100)
    # The window holds only the two accurate samples: the correction has decayed away.
    assert "sig" not in _corrections(hub)


@pytest.mark.unit
def test_stale_samples_outside_the_window_are_dropped(narrow_window):
    hub = _hub()
    _feed(hub, "sig", est=100, actual=100)  # accurate, then evicted
    _feed(hub, "sig", est=100, actual=400)  # q = 4
    _feed(hub, "sig", est=100, actual=1600)  # q = 16
    # Only the last two samples are in the window, so the factor is what those two imply —
    # strictly above what all three would (the evicted accurate run would pull it down).
    windowed = _corrections(hub)["sig"]
    assert windowed == pytest.approx(correction_factor([math.log(4.0), math.log(16.0)], 2, 100.0))
    assert windowed > correction_factor([0.0, math.log(4.0), math.log(16.0)], 2, 100.0), (
        "an evicted accurate sample must not still be damping the correction"
    )


@pytest.mark.unit
def test_rows_that_teach_nothing_are_ignored():
    hub = _hub()
    _feed(hub, "", est=100, actual=800)  # no signature (a distributed worker)
    _feed(hub, "sig", est=0.0, actual=800)  # no structural estimate
    _feed(hub, "sig", est=100, actual=0)  # empty output
    assert _corrections(hub) == {}


@pytest.mark.unit
def test_incremental_fold_matches_a_from_scratch_recompute(narrow_window):
    # The q-error fold is maintained incrementally across reads (only rows recorded since
    # the last read are absorbed). Reading the correction after every record must yield
    # exactly what a single cold hub reading the same history at the end computes — the
    # incremental cursor cannot drift from a full re-fold.
    backend = InProcessBackend()
    hub = MetadataHub(backend)
    qs = [(100, 400), (100, 1600), (100, 100), (100, 6400)]  # window keeps the last 2
    for est, actual in qs:
        _feed(hub, "sig", est=est, actual=actual)
        _corrections(hub)  # force an incremental fold after each record
    cold = _corrections(MetadataHub(backend))  # a fresh hub folds the whole history once
    assert _corrections(hub)["sig"] == pytest.approx(cold["sig"])


@pytest.mark.unit
def test_op_stats_with_signature_is_chronological_and_filtered():
    hub = _hub()
    _feed(hub, "a", est=1, actual=1)
    _feed(hub, "", est=1, actual=1)  # dropped: no signature
    _feed(hub, "b", est=1, actual=2)
    rows = hub.op_stats_with_signature()
    assert [r["signature"] for r in rows] == ["a", "b"]


# --- applying the factor ----------------------------------------------------------


def _estimator(dataset, learned: dict) -> StatsEstimator:
    """An estimator over a dataset's *bound* sources.

    Sources must be bound: an unbound scan estimates to the `unknown_rows` placeholder,
    which is not an estimate and which the correction deliberately refuses to scale.
    """
    return StatsEstimator(dataset._sources, learned, active_config().optimizer.cardinality)


def _join_dataset():
    left = bt.from_pydict({"k": [1, 2, 3], "a": [1, 2, 3]})
    right = bt.from_pydict({"k": [1, 2, 3], "b": [4, 5, 6]})
    return left.join(right, on="k")


@pytest.mark.unit
def test_correction_scales_the_estimate_and_downgrades_provenance():
    ds = _join_dataset()
    plan = ds._plan
    cold = _estimator(ds, {})
    base = cold.estimate(plan)
    sig = cold.signature_of(plan)

    corrected = _estimator(ds, {CARDINALITY_CORRECTION_KEY: {sig: 4.0}}).estimate(plan)

    assert corrected.rows == pytest.approx(base.rows * 4.0)
    assert corrected.provenance >= Provenance.LEARNED  # never upgraded by a correction


@pytest.mark.unit
def test_correction_never_touches_an_exact_estimate():
    # A bare scan's row count is EXACT and needs no correction.
    ds = bt.from_pydict({"x": [1, 2, 3]})
    plan = ds._plan
    sig = _estimator(ds, {}).signature_of(plan)
    out = _estimator(ds, {CARDINALITY_CORRECTION_KEY: {sig: 8.0}}).estimate(plan)
    assert out.rows == 3
    assert out.provenance is Provenance.EXACT


@pytest.mark.unit
def test_correction_only_applies_to_correctable_operators():
    # A Filter has its own learned-selectivity loop; correcting it too would
    # double-count the same error.
    ds = bt.from_pydict({"x": [1, 2, 3]}).filter(bt.col("x") > 1)
    est = _estimator(ds, {})
    assert est.correction_for(ds._plan) == 1.0
    assert est.reportable_estimate(ds._plan) == 0.0


@pytest.mark.unit
def test_reportable_estimate_divides_the_applied_correction_back_out():
    ds = _join_dataset()
    plan = ds._plan
    cold = _estimator(ds, {})
    raw = cold.estimate(plan).rows
    sig = cold.signature_of(plan)

    warm = _estimator(ds, {CARDINALITY_CORRECTION_KEY: {sig: 4.0}})
    assert warm.estimate(plan).rows == pytest.approx(raw * 4.0)
    # What Core reports back is the *structural* number, so the next geometric mean
    # measures the estimator's error, not the residual of an applied correction.
    assert warm.reportable_estimate(plan) == pytest.approx(raw)


@pytest.mark.unit
def test_measured_absolute_rows_suppress_the_correction_sample():
    # `record_execution` stores an absolute row count for the whole plan's signature.
    # That estimate is a past measurement, so its q-error is ~1 by construction and must
    # not be fed back — otherwise re-running a query decays its own correction to 1.0.
    ds = _join_dataset()
    plan = ds._plan
    sig = _estimator(ds, {}).signature_of(plan)
    warm = _estimator(ds, {sig: {"rows": 999.0}, CARDINALITY_CORRECTION_KEY: {sig: 4.0}})

    assert warm.estimate(plan).rows == 999.0  # the measurement wins
    assert warm.correction_for(plan) == 1.0
    assert warm.reportable_estimate(plan) == 0.0  # contributes no sample


@pytest.mark.unit
def test_factor_matches_the_estimator_applied_to_the_same_logs():
    hub = _hub()
    qs = [0.5, 2.0, 8.0]
    for q in qs:
        _feed(hub, "sig", est=1000, actual=int(1000 * q))
    expected = correction_factor(
        [math.log(q) for q in qs],
        active_config().optimizer.cardinality_correction_min_samples,
        active_config().optimizer.cardinality_correction_max_factor,
    )
    assert _corrections(hub)["sig"] == pytest.approx(expected, rel=1e-6)
    # And it sits between no correction and the raw geometric mean, in that direction.
    raw = math.exp(sum(math.log(q) for q in qs) / len(qs))
    assert 1.0 < _corrections(hub)["sig"] < raw


@pytest.mark.unit
def test_one_absurd_run_cannot_dominate_the_window():
    # A UDF that exploded once produces a single astronomically large q-error. Clipping the
    # *sample* (not just the final factor) bounds its influence to one window slot, so five
    # accurate runs still hold the correction near 1.0. Without input clipping the single
    # outlier drags the mean all the way to the clamp and pins it there.
    logs = [0.0] * 5 + [math.log(1e9)]
    assert correction_factor(logs, 2, 10.0) < 3.0


@pytest.mark.unit
def test_only_the_signatures_that_moved_are_re_derived(narrow_window):
    """Many shapes, one new observation: the answer is still the full recompute's.

    The q-error *windows* were folded incrementally, but the factors derived from them were
    not — every read re-ran the geometric mean for every shape the session had ever seen. On
    a suite that issues a hundred distinct plans, that puts a cost proportional to cumulative
    history on the critical path of each `optimize`: measured on TPC-DS at scale 1, a probe
    query took 9.4 ms with nothing else run and 21.5 ms after 100 others in one session.

    Deriving only the shapes whose window moved must not change any factor, including the
    ones left alone, which is what this holds against a hub that folds the same history cold.
    """
    backend = InProcessBackend()
    hub = MetadataHub(backend)
    for shape in range(15):
        for actual in (400, 1600):
            _feed(hub, f"s{shape}", est=100, actual=actual)
    _corrections(hub)  # summarize every shape, then move exactly one
    _feed(hub, "s3", est=100, actual=25)

    warm = _corrections(hub)
    cold = _corrections(MetadataHub(backend))
    assert warm.keys() == cold.keys()
    for sig, factor in cold.items():
        assert warm[sig] == pytest.approx(factor), sig


@pytest.mark.unit
def test_a_shape_that_stops_qualifying_loses_its_factor(narrow_window):
    """A factor is deleted when the evidence no longer supports one, not left behind.

    The full rebuild dropped a disqualified shape implicitly by not re-emitting it. An
    incremental update has to delete it on purpose, and a stale correction is exactly the
    kind of value that would otherwise steer plans forever.
    """
    hub = _hub()
    for actual in (400, 1600):  # a consistent over-estimate → a real factor
        _feed(hub, "sig", est=100, actual=actual)
    assert "sig" in _corrections(hub)
    for _ in range(6):  # the window fills with samples that agree the estimate was right
        _feed(hub, "sig", est=100, actual=100)
    assert "sig" not in _corrections(hub)
