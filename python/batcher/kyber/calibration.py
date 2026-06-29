"""Cost-model calibration — turn measured `op_stats` into cost coefficients.

The `CostModel` coefficients ship as plain constants (`config.CostCoefficients`).
Once a workload has run, Core has recorded per-operator `OperatorFeedback`
(rows in/out, wall time) into the MetadataHub's `op_stats`. This module fits the
per-row coefficients from those measurements so the model reflects *this* engine on
*this* hardware, closing the "calibrated from measured op_stats later" gap the cost
model documents.

Method. Each operator family's dominant cost term is `coeff x basis(rows)` (e.g.
filter ~ `filter_row x rows_in`, sort ~ `sort_row x n·log₂n`). Measurements are in
milliseconds; coefficients are in abstract work units, so we anchor the two with a
single global factor `k` (work units per ms) chosen to preserve the default model's
overall scale — when reality matches the defaults, calibration is a no-op. Each
coefficient is then `median(k x t_ms / basis)` over its samples, smoothed toward the
current value and clamped to within a configured factor of its default, so timing
noise can never produce a degenerate model. Families without enough samples keep
their default. Pure function: reads the hub, returns coefficients; decides nothing.
"""

from __future__ import annotations

import dataclasses
import math
import weakref

from batcher.config import Config, CostCoefficients, active_config
from batcher.metadata import MetadataHub

__all__ = ["calibrate"]

# Per-hub memo of the calibrated coefficients, keyed weakly by the hub so a dropped
# hub (e.g. a test's process-wide reset) evicts its entry automatically. The value is
# `(hub.version, fingerprint, coeffs)`: the fit is reused while the hub has absorbed
# no new feedback (its `version` is unchanged) and the relevant config is unchanged.
# Without this, `_calibrate` re-scans + JSON-parses the *entire* op_stats history on
# every optimize — and on every adaptive sub-stage — so planning cost grows with the
# session's cumulative query count.
_CALIB_CACHE: weakref.WeakKeyDictionary[MetadataHub, tuple[int, tuple, CostCoefficients]] = (
    weakref.WeakKeyDictionary()
)

# Re-fit the cost coefficients only after this many *new* feedback rows accrue (the hub
# version bumps once per recorded operator). A small query records a handful of rows, so
# this refits roughly every few-to-ten queries — fresh enough for a cost heuristic while
# keeping per-query planning overhead flat instead of growing with session history.
_RECALIBRATE_AFTER = 64

# Each calibratable operator `kind` (the native `ExecMetrics` tag) maps to the cost
# coefficient its dominant per-row term scales, plus the `basis(rows_in, rows_out)`
# that term multiplies. `hash_build_row` is fit from `aggregate` (the purest hash-build
# signal); `hash_probe_row` from `hash_join` (its per-row work over both sides). The
# remaining coefficients (`output_row`, `map_row`, `bytes_per_row`) have no clean
# single-family signal and keep their defaults.
_KIND_COEFF: dict[str, str] = {
    "scan": "scan_row",
    "filter": "filter_row",
    "project": "project_row",
    "sort": "sort_row",
    "distinct": "distinct_row",
    "union": "union_row",
    "aggregate": "hash_build_row",
    "hash_join": "hash_probe_row",
}


def _basis(kind: str, rows_in: float, rows_out: float) -> float:
    """The row multiplier of `kind`'s dominant cost term (matches `CostModel`)."""
    if kind == "sort":
        return rows_in * math.log2(max(2.0, rows_in))
    if kind in ("scan", "project", "union"):
        return rows_out
    return rows_in  # filter, distinct, aggregate, hash_join


def _samples(rows: list[dict]) -> list[tuple[float, float, float]]:
    """Usable `(rows_in, rows_out, t_op_ms)` triples — positive rows and time only.

    `rows_in` falls back to `rows_out` (scans report them equal); a sample with no
    positive basis or no positive time carries no signal and is dropped.
    """
    out: list[tuple[float, float, float]] = []
    for r in rows:
        rin = float(r.get("rows_in", 0) or r.get("n_actual", 0))
        rout = float(r.get("n_actual", r.get("rows_out", 0)))
        t = float(r.get("t_op_ms", 0.0))
        if t > 0.0 and (rin > 0.0 or rout > 0.0):
            out.append((rin or rout, rout or rin, t))
    return out


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def calibrate(hub: MetadataHub | None, config: Config | None = None) -> CostCoefficients:
    """Fit `CostCoefficients` from the hub's measured `op_stats`.

    Returns the default coefficients unchanged when there is no hub, no measured
    data, or no family with enough samples — so a cold metadata store never degrades
    planning. Best-effort: any failure falls back to the defaults.
    """
    cfg = config or active_config()
    defaults = cfg.optimizer.cost_coeffs
    if hub is None:
        return defaults
    # Reuse the prior fit unless the hub absorbed new feedback or the relevant config
    # changed — avoids the whole-history op_stats scan on every optimize.
    fingerprint = (
        defaults,
        cfg.optimizer.cost_calibration_min_samples,
        cfg.optimizer.learning_smoothing_alpha,
        cfg.optimizer.cost_calibration_clamp,
    )
    # Throttle: a cost fit is a statistical estimate that barely moves with one more
    # sample among many, so reuse it until enough *new* feedback accrues rather than
    # re-scanning the whole op-stats history on every `collect()` (the hub version bumps
    # per recorded operator, so an exact-version cache would miss every query — turning a
    # stream of small queries into O(queries²) calibration work). Staleness only affects
    # plan *cost*, never results, so a slightly old fit is safe.
    version = hub.version
    cached = _CALIB_CACHE.get(hub)
    if (
        cached is not None
        and cached[1] == fingerprint
        and 0 <= version - cached[0] < _RECALIBRATE_AFTER
    ):
        return cached[2]
    try:
        coeffs = _calibrate(hub.op_stats_by_kind(), defaults, cfg)
    except Exception:  # pragma: no cover - calibration must never break planning
        coeffs = defaults
    _CALIB_CACHE[hub] = (version, fingerprint, coeffs)
    return coeffs


def _calibrate(
    by_kind: dict[str, list[dict]],
    defaults: CostCoefficients,
    cfg: Config,
) -> CostCoefficients:
    min_samples = cfg.optimizer.cost_calibration_min_samples
    alpha = cfg.optimizer.learning_smoothing_alpha
    clamp = max(1.0, cfg.optimizer.cost_calibration_clamp)

    # Per-family usable samples, keeping only families above the sample floor.
    usable: dict[str, list[tuple[float, float, float]]] = {}
    for kind, coeff in _KIND_COEFF.items():
        s = _samples(by_kind.get(kind, []))
        if len(s) >= min_samples and getattr(defaults, coeff, 0.0) > 0.0:
            usable[kind] = s
    if not usable:
        return defaults

    # Global anchor k (work units per ms): chosen so the default model's total work
    # over all usable samples equals their total measured ms. This keeps calibrated
    # coefficients on the same scale as the untouched defaults.
    total_default_work = 0.0
    total_ms = 0.0
    for kind, samples in usable.items():
        c0 = getattr(defaults, _KIND_COEFF[kind])
        for rin, rout, t in samples:
            total_default_work += c0 * _basis(kind, rin, rout)
            total_ms += t
    if total_default_work <= 0.0 or total_ms <= 0.0:
        return defaults
    k = total_default_work / total_ms

    updates: dict[str, float] = {}
    for kind, samples in usable.items():
        coeff = _KIND_COEFF[kind]
        c0 = getattr(defaults, coeff)
        per_row = [k * t / b for rin, rout, t in samples if (b := _basis(kind, rin, rout)) > 0.0]
        if not per_row:
            continue
        measured = _median(per_row)
        smoothed = alpha * measured + (1.0 - alpha) * c0
        updates[coeff] = _clamp(smoothed, c0, clamp)

    return dataclasses.replace(defaults, **updates) if updates else defaults


def _clamp(value: float, default: float, factor: float) -> float:
    """Bound `value` to within `factor`x of `default` (both directions)."""
    lo, hi = default / factor, default * factor
    return max(lo, min(hi, value))
