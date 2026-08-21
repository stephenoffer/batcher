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
overall scale — when reality matches the defaults, calibration is a no-op. `k` is the
**median** sample's ratio of default work to measured time rather than the ratio of the
two totals, because a total is set by the biggest operators in an ever-growing history
and therefore never settles (`_anchor`). Each
coefficient is then `median(k x t_ms / basis)` over its samples, **shrunk toward the
shipped default in proportion to how little evidence there is** (`shrink`), and clamped
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
from batcher._internal.mathx import clamp_factor
from batcher.config import Config, CostCoefficients, active_config
from batcher.kyber.learning import is_material_change
from batcher.metadata import MetadataHub

__all__ = ["calibrate", "live_coefficients", "shrink"]

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


def live_coefficients(hub: MetadataHub | None) -> CostCoefficients | None:
    """The coefficients currently in force for `hub`, without provoking a refit.

    `plan_cache` fingerprints these directly rather than keying on *when* they were last
    fitted. A refit counter is the wrong clock for the memo: on a query recording more
    operators than `_RECALIBRATE_AFTER`, a refit fires every execution, so a counter moves
    every execution even when the fit reproduces itself.
    """
    cached = _CALIB_CACHE.get(hub) if hub is not None else None
    return cached[2] if cached is not None else None


# Each calibratable operator `kind` (the native `ExecMetrics` tag) maps to the cost
# coefficient its dominant per-row term scales, plus the `basis(rows_in, rows_out)`
# that term multiplies. `hash_build_row` is fit from `aggregate` (the purest hash-build
# signal); `hash_probe_row` from `hash_join`. The remaining coefficients (`output_row`,
# `map_row`, `bytes_per_row`) have no clean single-family signal and keep their defaults.
#
# **The join's basis is its probe side alone, and the fit absorbs its build time anyway.**
# `_samples` prefers `n_input`, which `ExecMetrics` narrowed to the probe rows when it gained
# a separate `rows_build` (so that a join's `selectivity` means fan-out rather than nothing).
# The numerator is still `t_op_ms`, the whole operator's wall time — build *and* probe — so
# the fitted `hash_probe_row` is that whole time spread over probe rows, and
# `cost.model._join_cost` then charges `hash_build_row x build_rows` on top of it. A
# calibrated join is therefore priced above what it was measured to take, by its build term.
#
# This is stated rather than corrected because correcting it is a cost-model retune, not a
# bug fix: the honest form is a two-coefficient regression over `(build_rows, probe_rows)`
# against `t_op_ms`, which changes how every join ranks against every non-join and has to be
# measured (`python benchmarks/run.py`) rather than reasoned about. What bounds the damage
# meanwhile is that it is *uniform* — every join is over-charged by the same term — so
# join-order and build-side choices, which compare joins with joins, are largely unaffected;
# what moves is a join weighed against an aggregate or a sort. `shrink` and `clamp_factor`
# bound how far the fit can travel from the shipped default in any case.
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


def calibrate(
    hub: MetadataHub | None,
    config: Config | None = None,
    hw_fingerprint: str | None = None,
) -> CostCoefficients:
    """Fit `CostCoefficients` from the hub's measured `op_stats`.

    Returns the default coefficients unchanged when there is no hub, no measured
    data, or no family with enough samples — so a cold metadata store never degrades
    planning. Best-effort: any failure falls back to the defaults.

    Args:
        hub: The metadata hub holding the measured history.
        config: The config supplying the default coefficients and the fit's bounds.
        hw_fingerprint: The machine class whose measurements to fit from, from
            `HardwareProfile.fingerprint`. `None` fits from this process's own class, which is
            right single-node and wrong on a cluster: these coefficients are in machine units,
            the plan they rank will run on the workers, and this process is the driver. A mixed
            fleet has no single answer and passes `""`, which falls back to the local class.

    Returns:
        The fitted coefficients, or the shipped defaults when there is not enough evidence.
    """
    cfg = config or active_config()
    defaults = cfg.optimizer.cost_coeffs
    if hub is None:
        return defaults
    # Reuse the prior fit unless the hub absorbed new feedback or the relevant config
    # changed — avoids the whole-history op_stats scan on every optimize. The machine class is
    # part of the key: the same hub fits different coefficients for different target hardware,
    # and a session that plans single-node and distributed in turn must not serve one from the
    # other's cache.
    fingerprint = (
        defaults,
        cfg.optimizer.cost_calibration_min_samples,
        cfg.optimizer.cost_calibration_clamp,
        hw_fingerprint or "",
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
        coeffs = _calibrate(hub.op_stats_by_kind(hw_fingerprint), defaults, cfg)
    except Exception as exc:  # pragma: no cover - calibration must never break planning
        note_suppressed("kyber", "load calibrated cost coefficients", exc)
        coeffs = defaults
    # Only against a fit of the *same* fingerprint. One hub serves several machine classes
    # across a session (a driver planning for its workers, then for itself), and the whole
    # point of the class being in the key is that those answers are different — blending them
    # would hand each the other's measurements.
    if cached is not None and cached[1] == fingerprint:
        coeffs = _settled(cached[2], coeffs)
    _CALIB_CACHE[hub] = (version, fingerprint, coeffs)
    return coeffs


def _settled(prior: CostCoefficients, fresh: CostCoefficients) -> CostCoefficients:
    """Blend a fresh fit into the live one, and **keep the live object** when nothing moved.

    Successive fits estimate the same stationary quantity from different windows of the same
    history, so averaging them reduces the estimator's variance — which is exactly the
    treatment `learning._smooth` gives every other learned scalar, applied here for a second
    reason on top of accuracy.

    That reason is the plan cache. The coefficients *are* its key
    (`plan_cache._bucketed`), so a fit that wanders inside its own noise band is not a neutral
    event: it moves the key, the memo misses, and the query re-plans. Bucketing the key with a
    deadband was supposed to absorb that and could not, because the wander is larger than the
    bucket: measured on TPC-DS q77 at scale 1, `hash_build_row` swung between adjacent
    half-octave buckets on consecutive refits — over 40% — so the memo never hit once and the
    query paid **135 ms of optimizer time on every execution**, against 13 ms for DuckDB's
    whole query. `project_row` did the same on q5.

    Damping the estimate is the fix that addresses the cause rather than widening the tolerance
    around it: a coefficient that is genuinely moving still gets there, in a few refits instead
    of one, and a coefficient that is merely noisy stops moving at all. When the blend changes
    nothing material, the *previous object* is returned unchanged, so a caller comparing
    identity (and the key derived from it) sees a fit that has settled.

    **A move of more than an octave is taken whole, and that is the difference between damping
    and dawdling.** Damping every move alike makes the *approach* the problem: a coefficient
    whose first, cold-start fit is 30x its settled value walks toward it geometrically, crosses
    a half-octave key bucket on each of the eight refits it takes to arrive, and misses the memo
    every one of them — which is q77 again, in slow motion instead of at random. Nothing about
    a 2x-or-worse discrepancy looks like timing noise (the wander this damps was measured at
    ~40%, well inside an octave), so the estimator has no reason to disbelieve it: it lands in
    one refit and the key stops moving. Inside an octave the blend is exactly as before.
    """
    blended = dataclasses.replace(
        fresh,
        **{
            name: _tracked(getattr(prior, name), getattr(fresh, name))
            for name in _numeric_fields(fresh)
        },
    )
    if all(
        not is_material_change(getattr(prior, name), getattr(blended, name))
        for name in _numeric_fields(blended)
    ):
        return prior
    return blended


def _tracked(prior: float, fresh: float) -> float:
    """`fresh` when it disagrees with `prior` by more than an octave, else the damped blend.

    The cost of the rule, stated plainly: a family with few samples *can* jump an octave on
    noise, and then it lands whole. `sort_row` was seen moving three buckets in one refit on
    TPC-DS q77, where a query sorts a handful of rows and the fit has almost nothing to go on.
    Two things bound that — `shrink` pulls a thin sample back toward the shipped default before
    it ever reaches here, and `clamp_factor` caps how far from that default any fit can land —
    so the damage is a briefly mispriced family inside a fixed envelope, against the certainty
    of a memo that never hits while a cold-start fit walks to its settled value.
    """
    if prior > 0.0 and fresh > 0.0 and not (0.5 <= fresh / prior <= 2.0):
        return fresh
    return _ALPHA * fresh + (1.0 - _ALPHA) * prior


#: Weight the newest fit carries in the blend above. One third is a ~3-refit memory: fast
#: enough that a real change in the machine or the engine lands within a few queries, slow
#: enough that the run-to-run swing measured on `hash_build_row` damps below the bucket the
#: plan cache keys on.
_ALPHA = 1.0 / 3.0


def _numeric_fields(coeffs: CostCoefficients) -> list[str]:
    """The coefficient names that carry a real number (every field, today — but not by fiat)."""
    return [
        f.name
        for f in dataclasses.fields(coeffs)
        if isinstance(getattr(coeffs, f.name), (int, float))
        and not isinstance(getattr(coeffs, f.name), bool)
    ]


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

    k = _anchor(usable, defaults)
    if k is None:
        return defaults

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
        updates[coeff] = clamp_factor(shrink(measured, c0, len(per_row), prior_strength), c0, clamp)

    speedup = _measured_jit_speedup(by_kind, defaults, cfg)
    if speedup is not None:
        updates["jit_speedup"] = speedup

    return dataclasses.replace(defaults, **updates) if updates else defaults


def _anchor(
    usable: dict[str, list[tuple[float, float, float, float]]],
    defaults: CostCoefficients,
) -> float | None:
    """Work units per millisecond: the scale that puts a fitted coefficient beside a default one.

    Every fitted coefficient is `k x measured_ms / basis`, so `k` is what keeps a *fitted*
    family comparable to an *unfitted* one — a family below the sample floor keeps its shipped
    default, and the two are compared inside one plan cost. It is the typical sample's ratio of
    default-model work to measured time.

    **The median, not the ratio of the two totals, and that difference is the whole point.**
    A ratio of sums is a work-weighted mean, so it is set by the largest operators in the
    history — the full-table scans — whose measured time is also the most variable, and every
    execution appends more of them. Measured on TPC-DS q77 at scale 1, run after identical run
    in one session, the summed anchor read **1.78e4 -> 3.04 -> 3.67 -> 4.19 -> 4.56 -> 4.87 ->
    5.13 -> 5.32e4** and was still climbing; `hash_build_row` rode it from 2.0 to 9.8 and the
    plan-cache key moved with it, so the query re-planned on **every** execution (~200 ms of
    optimizer against DuckDB's 21 ms for the whole query). The per-sample median over the same
    history reads **9.7e3 / 9.8e3 / 9.2e3 / 9.5e3 / 9537 / 9537 / 9537** — it settles, and a
    settled fit is what lets the memo hit at all. Robustness here is not a statistical nicety;
    it is the difference between a learning loop that converges and one that walks.
    """
    ratios = [
        c0 * basis * factor / t
        for kind, samples in usable.items()
        if (c0 := getattr(defaults, _KIND_COEFF[kind])) > 0.0
        for rin, rout, t, factor in samples
        if t > 0.0 and (basis := _basis(kind, rin, rout)) > 0.0
    ]
    return median(ratios) if ratios else None


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
    # Deliberately NOT run through `shrink`, unlike the absolute coefficients. Those fit an
    # absolute value that must be blended toward the shipped default; this one is already
    # *prior-relative* — the measurement is `prior x ratio`, so the prior is the anchor, and
    # each ratio is itself a median over `min_samples` rows. Shrinking would anchor it twice
    # and stop the loop learning a genuinely different engine, which is exactly the fixed-point
    # failure `shrink` was written to remove. `clamp_factor` still bounds how far it may travel.
    measured = defaults.jit_speedup * median(ratios)
    return clamp_factor(max(1.0, measured), defaults.jit_speedup, clamp)


def shrink(measured: float, prior: float, n_samples: int, prior_strength: float) -> float:
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
    `clamp_factor`, which already bounds the result multiplicatively.

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
