"""Kyber cost-model calibration from measured op_stats.

Proves the feedback loop's calibration half: synthetic operator feedback moves the
fitted coefficients toward the measured per-row cost, while the sample floor and
the clamp keep a cold or noisy store from degrading the model.
"""

from __future__ import annotations

import pytest

from batcher.config import active_config
from batcher.kyber.calibration import _RECALIBRATE_AFTER, calibrate
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend
from batcher.plan.feedback import OperatorFeedback
from batcher.plan.ids import OpId

pytestmark = pytest.mark.unit


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


def _record(hub: MetadataHub, kind: str, n: int, rows: int, t_ms: float) -> None:
    for i in range(n):
        hub.record(
            OperatorFeedback(
                op_id=OpId(i % 4),
                kind=kind,
                n_actual=rows,
                t_op_ms=t_ms,
                m_peak_bytes=rows * 8,
                selectivity=1.0,
                batch_size=16384,
                backend="interp",
                n_input=rows,
            )
        )


def _record_selective(
    hub: MetadataHub, kind: str, n: int, *, rows_in: int, rows_out: int, t_ms: float
) -> None:
    """Record a family whose output rows differ from its input rows (a selective op)."""
    for i in range(n):
        hub.record(
            OperatorFeedback(
                op_id=OpId(i % 4),
                kind=kind,
                n_actual=rows_out,
                t_op_ms=t_ms,
                m_peak_bytes=rows_out * 8,
                selectivity=rows_out / rows_in,
                batch_size=16384,
                backend="interp",
                n_input=rows_in,
            )
        )


def test_no_data_returns_defaults():
    defaults = active_config().optimizer.cost_coeffs
    assert calibrate(None) == defaults
    assert calibrate(_hub()) == defaults


def test_below_sample_floor_keeps_default():
    cfg = active_config()
    hub = _hub()
    # One filter sample (< min_samples) must not move the coefficient.
    _record(hub, "filter", 1, rows=1000, t_ms=1.0)
    assert calibrate(hub, cfg).filter_row == cfg.optimizer.cost_coeffs.filter_row


def test_calibration_tracks_measured_ratio():
    cfg = active_config()
    defaults = cfg.optimizer.cost_coeffs
    n = cfg.optimizer.cost_calibration_min_samples
    hub = _hub()
    # Scan is the natural anchor; make filter cost far more per row than the default
    # model expects relative to scan, so its coefficient must rise.
    _record(hub, "scan", n, rows=1000, t_ms=1.0)  # 1 ms / 1000 rows
    _record(hub, "filter", n, rows=1000, t_ms=10.0)  # 10× the scan per-row time
    coeffs = calibrate(hub, cfg)
    # Default ratio filter_row/scan_row is 0.5; measured is ~10× → filter_row climbs.
    assert coeffs.filter_row > defaults.filter_row
    assert coeffs.scan_row > 0.0


def test_calibration_is_cached_until_new_feedback():
    # The whole-history op_stats scan must not run on every optimize: a repeated
    # calibrate reuses the prior fit, and — since a cost fit barely moves with one more
    # sample — the cache is *throttled*, holding until `_RECALIBRATE_AFTER` new feedback
    # rows accrue (a single new row does not force a re-scan). This keeps per-query
    # planning flat instead of growing O(history) with the session's query count.
    cfg = active_config()
    n = cfg.optimizer.cost_calibration_min_samples
    hub = _hub()
    _record(hub, "scan", n, rows=1000, t_ms=1.0)
    _record(hub, "filter", n, rows=1000, t_ms=10.0)

    scans = [0]
    raw = hub.op_stats_by_kind

    def counting(hw_fingerprint=None):
        scans[0] += 1
        return raw(hw_fingerprint)

    hub.op_stats_by_kind = counting  # type: ignore[method-assign]

    first = calibrate(hub, cfg)
    assert scans[0] == 1  # computed once
    again = calibrate(hub, cfg)
    assert scans[0] == 1  # second call is a pure cache hit — no re-scan
    assert again == first

    _record(hub, "scan", 1, rows=1000, t_ms=1.0)  # one new row → below the refresh interval
    calibrate(hub, cfg)
    assert scans[0] == 1  # still cached (throttled — one row doesn't force a re-scan)

    _record(hub, "scan", _RECALIBRATE_AFTER, rows=1000, t_ms=1.0)  # cross the refresh interval
    calibrate(hub, cfg)
    assert scans[0] == 2  # enough new feedback → exactly one recompute


def test_calibration_fits_against_input_rows_not_output():
    """A selective filter's per-row cost is fit against what it READ, not what it kept.

    Filter and scan both take 1 ms per 1000 *input* rows, so the filter coefficient
    should track the scan's per-row cost (a modest rise from the 0.5 default ratio) —
    it must NOT be driven to the clamp ceiling, which fitting against the filter's tiny
    output (10 rows → an apparent 100x per-row cost) would have done. This is the
    regression guard for calibrating input-bound families on `n_input`.
    """
    cfg = active_config()
    defaults = cfg.optimizer.cost_coeffs
    clamp = cfg.optimizer.cost_calibration_clamp
    n = cfg.optimizer.cost_calibration_min_samples
    hub = _hub()
    _record(hub, "scan", n, rows=1000, t_ms=1.0)  # 1 ms / 1000 rows
    # Same per-INPUT-row cost as scan, but 99% selective: output-row basis would look 100x.
    _record_selective(hub, "filter", n, rows_in=1000, rows_out=10, t_ms=1.0)
    coeffs = calibrate(hub, cfg)
    # Output-row basis would have slammed filter_row to the clamp ceiling; input-row
    # basis leaves it comfortably below it.
    assert coeffs.filter_row < defaults.filter_row * clamp - 1e-9


def test_samples_reconstruct_input_rows_from_selectivity():
    """A row persisted before `n_input` existed still calibrates on input rows.

    Reconstructs `rows_in = n_actual / selectivity` so an older SQLite/object-store
    metadata store migrates without recalibrating on output rows. Such a row also
    predates `expr_factor`, which must default to 1.0 (no expression pricing).
    """
    from batcher.kyber.calibration import _samples

    legacy_row = {"n_actual": 10, "selectivity": 0.01, "t_op_ms": 2.0}  # no n_input key
    (rin, rout, t, factor) = _samples([legacy_row])[0]
    assert rin == pytest.approx(1000.0)  # 10 / 0.01
    assert rout == pytest.approx(10.0)
    assert t == pytest.approx(2.0)
    assert factor == pytest.approx(1.0)


def test_samples_carry_the_expression_cost_factor():
    """The fit divides `expr_factor` out, so a regex-heavy workload cannot inflate the
    per-row coefficient that the cost model then multiplies by the regex's factor again."""
    from batcher.kyber.calibration import _samples

    row = {"n_input": 100, "n_actual": 10, "t_op_ms": 5.0, "expr_factor": 40.0}
    (_rin, _rout, _t, factor) = _samples([row])[0]
    assert factor == pytest.approx(40.0)


def test_clamp_bounds_runaway():
    cfg = active_config()
    defaults = cfg.optimizer.cost_coeffs
    clamp = cfg.optimizer.cost_calibration_clamp
    n = cfg.optimizer.cost_calibration_min_samples
    hub = _hub()
    # Pathologically slow filter (tiny rows, huge time) would blow the coefficient
    # up without the clamp.
    _record(hub, "scan", n, rows=1_000_000, t_ms=0.001)
    _record(hub, "filter", n, rows=1, t_ms=10_000.0)
    coeffs = calibrate(hub, cfg)
    assert coeffs.filter_row <= defaults.filter_row * clamp + 1e-9


def test_measured_jit_speedup_scales_the_prior_from_the_backend_tag():
    """`jit_speedup` is fitted from `op_stats.backend` — metadata nothing consumed before.

    Once the expression's own cost is divided out, the per-row residual of the two tiers
    should agree if the model prices them correctly. A systematic gap means the prior is
    wrong, and the fit scales it by the observed ratio. Here the interpreted bucket costs
    2x per unit of expression work, so the prior (4.0) should roughly double.
    """
    from batcher.config import active_config
    from batcher.kyber.calibration import _measured_jit_speedup

    cfg = active_config()
    n = cfg.optimizer.cost_calibration_min_samples
    by_kind = {
        "filter": (
            [
                {"n_input": 1000, "n_actual": 500, "t_op_ms": 1.0, "expr_factor": 1.0,
                 "backend": "jit"}
                for _ in range(n)
            ]
            + [
                {"n_input": 1000, "n_actual": 500, "t_op_ms": 2.0, "expr_factor": 1.0,
                 "backend": "interp"}
                for _ in range(n)
            ]
        )
    }  # fmt: skip
    measured = _measured_jit_speedup(by_kind, cfg.optimizer.cost_coeffs, cfg)
    assert measured == pytest.approx(cfg.optimizer.cost_coeffs.jit_speedup * 2.0)


def test_measured_jit_speedup_needs_both_tiers():
    """With only one tier observed there is no ratio to fit; the prior stands."""
    from batcher.config import active_config
    from batcher.kyber.calibration import _measured_jit_speedup

    cfg = active_config()
    n = cfg.optimizer.cost_calibration_min_samples
    only_jit = {
        "filter": [
            {"n_input": 1000, "n_actual": 500, "t_op_ms": 1.0, "expr_factor": 1.0, "backend": "jit"}
            for _ in range(n)
        ]
    }
    assert _measured_jit_speedup(only_jit, cfg.optimizer.cost_coeffs, cfg) is None


def test_measured_jit_speedup_ignores_mixed_tier_rows():
    """`interp+jit` blends both tiers, so it cannot calibrate either bucket."""
    from batcher.config import active_config
    from batcher.kyber.calibration import _measured_jit_speedup

    cfg = active_config()
    n = cfg.optimizer.cost_calibration_min_samples
    mixed = {
        "filter": [
            {"n_input": 1000, "n_actual": 500, "t_op_ms": 1.0, "expr_factor": 1.0,
             "backend": "interp+jit"}
            for _ in range(n * 2)
        ]
    }  # fmt: skip
    assert _measured_jit_speedup(mixed, cfg.optimizer.cost_coeffs, cfg) is None


def test_a_noisy_refit_settles_instead_of_alternating():
    """Successive fits of the same quantity must converge, not oscillate.

    The coefficients are the plan cache's key (`plan_cache._bucketed` buckets them), so a fit
    that swings run to run is not merely an imprecise estimate — it is a cache miss on every
    execution. Measured on TPC-DS q77 at scale 1, `hash_build_row` alternated between adjacent
    half-octave buckets on consecutive refits (over 40%), the memo never hit once, and the
    query paid 135 ms of optimizer time on every run against 13 ms for DuckDB's whole query.

    This feeds alternating-cost evidence for one family and asserts the published fit stops
    moving: the estimator is damped toward the live one, so noise averages out while a real
    shift still lands (the next test).
    """
    cfg = active_config()
    hub = _hub()
    _record(hub, "filter", 40, rows=1_000_000, t_ms=10.0)
    seen = [calibrate(hub, cfg).filter_row]
    for i in range(8):
        # Alternate the measured cost by ±40% — noise, not a trend.
        _record(hub, "filter", _RECALIBRATE_AFTER + 1, rows=1_000_000, t_ms=14.0 if i % 2 else 6.0)
        seen.append(calibrate(hub, cfg).filter_row)
    tail = seen[-4:]
    spread = (max(tail) - min(tail)) / max(tail)
    assert spread < 0.10, f"the fit is still swinging: {tail}"


def test_a_real_shift_still_reaches_the_published_fit():
    """Damping must slow a genuine change, not refuse it.

    The boundary of the test above: a coefficient whose measured cost moves and *stays* moved
    has to arrive, or the loop has stopped learning. It takes a few refits instead of one,
    which is the whole trade.
    """
    cfg = active_config()
    hub = _hub()
    # A second family holds the global ms-to-work-unit anchor still, so what moves below is
    # `filter`'s cost *relative* to `scan` — which is the only thing a cost model ranks on.
    _record(hub, "scan", 40, rows=1_000_000, t_ms=10.0)
    _record(hub, "filter", 40, rows=1_000_000, t_ms=5.0)
    cold = calibrate(hub, cfg).filter_row
    for _ in range(10):
        _record(hub, "scan", _RECALIBRATE_AFTER, rows=1_000_000, t_ms=10.0)
        _record(hub, "filter", _RECALIBRATE_AFTER, rows=1_000_000, t_ms=50.0)
        warm = calibrate(hub, cfg).filter_row
    assert warm > cold * 2.0, f"a 10x relative shift never reached the fit: {cold} -> {warm}"


def test_the_anchor_settles_as_history_accumulates():
    """A stationary workload must fit a stationary anchor, however long it runs.

    Every coefficient is `k x measured_ms / basis`, so the global work-per-ms anchor `k`
    multiplies all of them: an anchor that walks makes every fit walk, which moves the plan
    cache's key, which re-plans an identical query on every execution. That is what TPC-DS
    q77 did — the anchor climbed 1.78e4 -> 5.32e4 over eight identical runs, taking
    `hash_build_row` from 2.0 to 9.8 with it, and the memo never hit once.

    The shape that does it is a **cold start inside a work-weighted average**. The first
    execution of anything is slow (cold caches, first-touch allocation, no compiled
    expression yet); every execution after it is not. A ratio of summed work to summed time
    is dominated by the heaviest operators, so those first slow samples keep dragging the
    anchor and it creeps toward the warm value for as long as the session runs. A per-sample
    median crosses over as soon as most samples are warm and then stops.
    """
    cfg = active_config()
    hub = _hub()
    # Round 0 is cold: the heavy scan-bound family measures 5x what it will measure once warm.
    _record(hub, "scan", _RECALIBRATE_AFTER, rows=8_000_000, t_ms=200.0)
    _record(hub, "filter", _RECALIBRATE_AFTER, rows=10_000, t_ms=0.05)
    fits = [calibrate(hub, cfg).filter_row]
    for _ in range(7):
        _record(hub, "scan", _RECALIBRATE_AFTER, rows=8_000_000, t_ms=40.0)
        _record(hub, "filter", _RECALIBRATE_AFTER, rows=10_000, t_ms=0.05)
        fits.append(calibrate(hub, cfg).filter_row)
    tail = fits[-4:]
    spread = (max(tail) - min(tail)) / max(tail)
    assert spread < 0.05, f"the fit is still walking as history grows: {fits}"
