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
coefficient is then `median(k x t_ms / basis)` over its samples, **shrunk toward the
shipped default in proportion to how little evidence there is** (`_shrink`), and clamped
to within a configured factor of that default, so timing noise can never produce a
degenerate model. Families without enough samples keep their default. Pure function:
reads the hub, returns coefficients; decides nothing.

The shrinkage matters: blending a fixed fraction of the *default* into every fit (as this
did) leaves a permanent bias — a coefficient whose true value is 10x the default
converges to 5.5x it and stays there however much data arrives. Weighting by sample count
makes the fit an estimator that converges to the measurement while staying stable cold.
"""

from __future__ import annotations

import dataclasses
import math
import weakref
from statistics import median

from batcher._internal.logging import note_suppressed
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


def _samples(rows: list[dict]) -> list[tuple[float, float, float, float]]:
    """Usable `(rows_in, rows_out, t_op_ms, expr_factor)` tuples.

    The input-bound families (filter/distinct/aggregate/hash_join) fit against
    *input* rows, so an accurate `rows_in` is load-bearing: fitting a selective
    filter's per-row cost against its (small) output would overstate the coefficient.
    Prefer the directly measured `n_input`; for a row persisted before that field
    existed, reconstruct it from `n_actual / selectivity` (the inverse of how
    `selectivity` was recorded); finally fall back to `rows_out` (a source op reports
    input == output). A sample with no positive basis or no positive time is dropped.

    `expr_factor` is the per-row cost of the expressions the operator evaluated, relative
    to a plain comparison. The fit divides it out, so the coefficient measures the
    engine's per-row *overhead* rather than whatever expressions the workload contained:
    without it a regex-heavy workload fits a huge `filter_row`, which the cost model then
    multiplies by the regex's factor a second time. Absent (an older row) means 1.0.
    """
    out: list[tuple[float, float, float, float]] = []
    for r in rows:
        measured_out = r.get("n_actual")
        if measured_out is None:
            measured_out = r.get("rows_out")
        rout = float(measured_out or 0.0)
        measured_in = r.get("n_input") or r.get("rows_in")
        if measured_in:
            rin = float(measured_in)
        else:
            sel = float(r.get("selectivity", 0.0) or 0.0)
            rin = rout / sel if sel > 0.0 else rout
        t = float(r.get("t_op_ms", 0.0))
        factor = float(r.get("expr_factor") or 1.0)
        if t > 0.0 and (rin > 0.0 or rout > 0.0) and factor > 0.0:
            # A *measured* zero output is real — a fully-selective filter genuinely emitted
            # no rows — so only an **absent** measurement falls back to the other side.
            # Treating a real 0 as "unknown" substituted the input count, which for the
            # output-basis families (`scan`/`project`/`union`, see `_basis`) fits a per-row
            # coefficient against a basis orders of magnitude too large and understates it.
            out_rows = rout if measured_out is not None else (rout or rin)
            out.append((rin or rout, out_rows, t, factor))
    return out


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
        cfg.optimizer.cost_calibration_clamp,
    )
    # `learning_smoothing_alpha` is deliberately *not* in the fingerprint: neither
    # `_calibrate` nor `_measured_jit_speedup` reads it, so including it only forced a
    # whole-history op_stats re-scan on a knob that cannot change the fit.
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
    except Exception as exc:  # pragma: no cover - calibration must never break planning
        note_suppressed("kyber", "load calibrated cost coefficients", exc)
        coeffs = defaults
    _CALIB_CACHE[hub] = (version, fingerprint, coeffs)
    return coeffs


def _calibrate(
    by_kind: dict[str, list[dict]],
    defaults: CostCoefficients,
    cfg: Config,
) -> CostCoefficients:
    min_samples = cfg.optimizer.cost_calibration_min_samples
    clamp = max(1.0, cfg.optimizer.cost_calibration_clamp)
    # The shipped default carries the weight of one sample floor: at exactly `min_samples`
    # observations the fit sits halfway between the default and the measurement, and it
    # converges to the measurement as evidence accumulates.
    prior_strength = float(min_samples)

    # Per-family usable samples, keeping only families above the sample floor.
    usable: dict[str, list[tuple[float, float, float, float]]] = {}
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
        for rin, rout, t, factor in samples:
            total_default_work += c0 * _basis(kind, rin, rout) * factor
            total_ms += t
    if total_default_work <= 0.0 or total_ms <= 0.0:
        return defaults
    k = total_default_work / total_ms

    updates: dict[str, float] = {}
    for kind, samples in usable.items():
        coeff = _KIND_COEFF[kind]
        c0 = getattr(defaults, coeff)
        # `t ~= coeff x basis(rows) x expr_factor`, so the per-row coefficient is the
        # measured time divided by BOTH the row basis and the expression cost.
        per_row = [
            k * t / (b * f) for rin, rout, t, f in samples if (b := _basis(kind, rin, rout)) > 0.0
        ]
        if not per_row:
            continue
        measured = median(per_row)
        updates[coeff] = _clamp(_shrink(measured, c0, len(per_row), prior_strength), c0, clamp)

    speedup = _measured_jit_speedup(by_kind, defaults, cfg)
    if speedup is not None:
        updates["jit_speedup"] = speedup

    return dataclasses.replace(defaults, **updates) if updates else defaults


def _measured_jit_speedup(
    by_kind: dict[str, list[dict]], defaults: CostCoefficients, cfg: Config
) -> float | None:
    """Fit `jit_speedup`, the one parameter separating compiled from interpreted pricing.

    The engine tags every operator with the tier that ran its per-row work
    (`op_stats.backend`: `"jit"`, `"interp"`, or `"interp+jit"`) — measured metadata that
    nothing consumed before. Dividing an operator's wall time by its row basis *and* its
    `expr_factor` leaves a per-row residual that, if the cost model were perfect, would be
    the same constant in both tiers. It is not, and the ratio of the two residuals is the
    factor by which the model misprices compiled work relative to interpreted work.
    Scaling the prior by that ratio removes the bias.

    This is a **model** parameter fitted from data, not a hardware measurement: because
    `weights` prices interpreted expressions from a hand-written table, the residual
    absorbs that table's systematic error along with any true tier difference. That is
    the useful thing to fit — it is exactly the number `expr_cost_factor` needs to rank a
    regex against a compiled comparison correctly — but it should not be read as "the JIT
    is Nx faster".

    Fitted **per family** (a filter's per-row cost differs from a projection's) and
    combined by median, so an unbalanced tier mix across families cannot skew it.
    `"interp+jit"` rows are skipped: they blend both tiers. Returns `None` when any
    bucket is too small, leaving the prior in place.
    """
    min_samples = cfg.optimizer.cost_calibration_min_samples
    clamp = max(1.0, cfg.optimizer.cost_calibration_clamp)
    ratios: list[float] = []
    for kind in ("filter", "project"):
        if getattr(defaults, _KIND_COEFF[kind], 0.0) <= 0.0:
            continue
        by_backend: dict[str, list[dict]] = {"jit": [], "interp": []}
        for row in by_kind.get(kind, []):
            bucket = by_backend.get(row.get("backend", ""))
            if bucket is not None:
                bucket.append(row)
        residuals: dict[str, float] = {}
        for backend, rows in by_backend.items():
            per_row = [
                t / (b * f)
                for rin, rout, t, f in _samples(rows)
                if (b := _basis(kind, rin, rout)) > 0.0
            ]
            if len(per_row) < min_samples:
                continue  # this backend bucket is too thin; the `!= 2` check below declines
            residuals[backend] = median(per_row)
        if len(residuals) != 2 or min(residuals.values()) <= 0.0:
            continue
        ratios.append(residuals["interp"] / residuals["jit"])
    if not ratios:
        return None
    # `expr_factor` already divided the compiled bucket by the prior, so the residual
    # ratio *scales* the prior rather than replacing it. A compiled expression is never
    # slower than the same expression interpreted, hence the floor at 1.0.
    #
    # Deliberately NOT run through `_shrink`, unlike the absolute coefficients. Those fit an
    # absolute value that must be blended toward the shipped default; this one is already
    # *prior-relative* — the measurement is `prior x ratio`, so the prior is the anchor, and
    # each ratio is itself a median over `min_samples` rows. Shrinking would anchor it twice
    # and stop the loop learning a genuinely different engine, which is exactly the fixed-point
    # failure `_shrink` was written to remove. `_clamp` still bounds how far it may travel.
    measured = defaults.jit_speedup * median(ratios)
    return _clamp(max(1.0, measured), defaults.jit_speedup, clamp)


def _shrink(measured: float, prior: float, n_samples: int, prior_strength: float) -> float:
    """Blend a measured coefficient toward its prior, weighted by how much evidence exists.

    The previous blend was `alpha*measured + (1-alpha)*default` with a fixed `alpha = 0.5`.
    That is not an estimator: its fixed point is `0.5*measured + 0.5*default`, so a
    coefficient whose true value is 10x the shipped default converges to 5.5x it and stays
    there **no matter how many samples arrive**. The loop could not learn its own engine.

    This is the standard shrinkage form: treat the shipped default as a prior worth
    `prior_strength` pseudo-samples, so the weight on the measurement is
    `n / (n + prior_strength)`. With little evidence the default dominates (a cold or noisy
    store never degrades the model); with plenty it converges to the measurement.

    The blend is **geometric**, `prior · (measured/prior)^w`, not the arithmetic
    `w·measured + (1-w)·prior`. A cost coefficient is a positive *scale*: what matters about
    it is the ratio to its prior, and the two blends disagree about the midpoint of a ratio.
    At `w = 0.5` a measurement 10x the prior blends to 5.5x arithmetically and 3.16x
    geometrically — and only the geometric one is consistent, because a measurement at 0.1x
    must blend to the reciprocal (0.316x), which the arithmetic form puts at 0.55x. The
    asymmetry biases every coefficient upward whenever the measurements straddle the prior,
    and it is the same reason the q-error correction averages geometrically. It also matches
    `_clamp`, which already bounds the result multiplicatively.

    Args:
        measured: The coefficient the samples imply.
        prior: The shipped default.
        n_samples: How many usable samples produced `measured`.
        prior_strength: The default's weight, in pseudo-samples.

    Returns:
        The shrunk coefficient.
    """
    if n_samples <= 0:
        return prior
    weight = n_samples / (n_samples + max(0.0, prior_strength))
    if measured <= 0.0 or prior <= 0.0:
        # A non-positive coefficient has no logarithm and no meaningful ratio; fall back to
        # the arithmetic blend rather than dropping the measurement entirely.
        return weight * measured + (1.0 - weight) * prior
    return math.exp(weight * math.log(measured) + (1.0 - weight) * math.log(prior))


def _clamp(value: float, default: float, factor: float) -> float:
    """Bound `value` to within `factor`x of `default` (both directions)."""
    lo, hi = default / factor, default * factor
    return max(lo, min(hi, value))
