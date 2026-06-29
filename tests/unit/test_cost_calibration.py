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

    def counting():
        scans[0] += 1
        return raw()

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
