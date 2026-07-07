"""Learned per-family memory model — turn measured `m_peak_bytes` into sizing.

The headline gap this closes: Core records each operator's *actual* peak memory
(`OperatorFeedback.m_peak_bytes`) into the MetadataHub's `op_stats`, but every
Carbonite sizing decision — admission, spill, reservation, per-task grant, morsel —
sizes from Kyber's PLAN ESTIMATE alone and never consults what the operator really
used. This module is the memory analog of `kyber.calibration` (which fits cost
coefficients from the same `op_stats`): it fits a per-operator-family **bytes-per-
input-row** figure from the measured peaks, so a decision can *blend* the plan
estimate toward measured reality.

Why a per-row ratio, not an absolute peak. A family's absolute peak depends on the
query's size (a 1M-row aggregate peaks far above a 10-row one), so replaying a stored
absolute byte figure onto a differently-sized plan would be wrong — exactly the
mistake `kyber.learning` avoids by learning filter *selectivity* (a ratio) rather
than an absolute row count. `bytes_per_row = median(m_peak_bytes / n_input)` is
size-general: multiply by *this* plan's estimated input rows to get a learned peak.

Everything here is pure and best-effort: it reads the hub, returns numbers, decides
nothing, and any failure or a cold store falls back to the caller's plan estimate so
a first run is byte-for-byte unchanged. Results never depend on it — only *how much
memory a decision reserves / when it spills / how big a morsel is*, which are
performance and scheduling, never correctness.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass

from batcher.config import Config, active_config
from batcher.metadata import MetadataHub

__all__ = ["LearnedMemoryModel", "learned_memory_model"]

# LogicalPlan node-class names (`PhysicalOp.kind`, e.g. "Aggregate", "Join") vs the
# native `ExecMetrics` tags feedback is recorded under (e.g. "aggregate", "hash_join").
# Both are normalized to one canonical family so a plan op finds the stats its own
# execution recorded.
_KIND_ALIASES = {"join": "hash_join"}


def _canonical_kind(kind: str) -> str:
    """Canonical memory-family key for a plan-op or feedback `kind` (case/alias-folded)."""
    k = kind.lower()
    return _KIND_ALIASES.get(k, k)


# Re-fit the model only after this many new feedback rows accrue (the hub version bumps
# once per recorded operator), mirroring `kyber.calibration._RECALIBRATE_AFTER`: a
# per-row memory figure barely moves with one more sample among many, so reuse the fit
# rather than re-scanning the whole `op_stats` history on every query. Staleness only
# affects *sizing*, never results, so a slightly old fit is safe.
_REFIT_AFTER = 64

# Per-hub memo, keyed weakly so a dropped hub (a test reset) evicts its entry. Value is
# `(version, fingerprint, model)`: reused while the hub absorbed no new feedback and the
# relevant config is unchanged.
_MODEL_CACHE: weakref.WeakKeyDictionary[MetadataHub, tuple[int, tuple, LearnedMemoryModel]] = (
    weakref.WeakKeyDictionary()
)


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


@dataclass(frozen=True, slots=True)
class LearnedMemoryModel:
    """Measured bytes-per-input-row per operator family, with plan-estimate blending.

    Built by `learned_memory_model` from the hub's `op_stats`. `bytes_per_row`
    returns the measured per-row footprint of a family (or `None` when too few
    samples), and `blend_peak` mixes a plan's per-operator byte estimate toward the
    measured reality, clamped so a noisy measurement can never wildly move sizing.
    Cold (no hub / no samples) → every method defers to the caller's plan estimate,
    so a first run is unchanged.
    """

    _bytes_per_row: dict[str, float]
    _alpha: float
    _clamp: float
    _row_bytes: int

    def bytes_per_row(self, kind: str) -> float | None:
        """Measured peak bytes per input row for `kind`'s family, or `None` if unlearned."""
        return self._bytes_per_row.get(_canonical_kind(kind))

    def max_bytes_per_row(self) -> float | None:
        """The widest measured per-row footprint across all learned families, or `None`.

        The morsel byte-budget cap uses this: a workload whose rows proved wide anywhere
        should keep a tighter row-count so its true byte working set stays bounded."""
        widths = [w for w in self._bytes_per_row.values() if w > 0]
        return max(widths) if widths else None

    def blend_peak(self, kind: str, plan_estimate: int) -> int:
        """Blend a plan's per-operator peak-byte estimate toward the measured reality.

        The plan estimate assumes `optimizer.row_bytes` bytes per row; the learned
        `bytes_per_row` is what the family actually used. Their ratio rescales the
        estimate to measured reality (size-general — it multiplies *this* plan's own
        estimate), exp-smoothed toward the measurement by `alpha` and clamped to within
        `clamp`x of the estimate so timing/measurement noise can't produce a degenerate
        size. Returns the plan estimate unchanged when the family is unlearned; when the
        plan could not size the op (`plan_estimate <= 0`) there is nothing to rescale,
        so it also abstains (an unsized op stays unsized — conservative).
        """
        bpr = self.bytes_per_row(kind)
        if bpr is None or plan_estimate <= 0 or self._row_bytes <= 0:
            return plan_estimate
        measured = plan_estimate * (bpr / self._row_bytes)
        blended = self._alpha * measured + (1.0 - self._alpha) * plan_estimate
        lo, hi = plan_estimate / self._clamp, plan_estimate * self._clamp
        return int(max(lo, min(hi, blended)))

    def plan_peak(self, plan_ops: object) -> int:
        """The plan's dominant-breaker peak, each op blended toward measured reality.

        `plan_ops` is any iterable of objects exposing `.kind` and `.bounds.m_max_bytes`
        (a `PhysicalPlan.ops`). Each op's plan estimate is blended by its own family, and
        the max is taken — the same dominant-breaker rule the plan estimator uses, only
        sharpened per family. Cold families pass through unchanged, so on a cold store
        this equals the plan's own dominant breaker exactly.
        """
        best = 0
        for op in plan_ops:  # type: ignore[attr-defined]
            kind = getattr(op, "kind", "")  # a bare-sized test double has no kind → unlearned
            best = max(best, self.blend_peak(kind, op.bounds.m_max_bytes))
        return best


def _fit(hub: MetadataHub, cfg: Config) -> LearnedMemoryModel:
    """Fit the per-family bytes-per-row model from the hub's measured `op_stats`."""
    opt = cfg.optimizer
    min_samples = max(1, opt.cost_calibration_min_samples)
    by_kind = hub.op_stats_by_kind()
    bpr: dict[str, float] = {}
    for kind, rows in by_kind.items():
        samples: list[float] = []
        for r in rows:
            peak = float(r.get("m_peak_bytes", 0) or 0.0)
            n_in = r.get("n_input") or r.get("rows_in") or 0
            if not n_in:
                # An op recorded before `n_input` existed: reconstruct input rows from
                # output / selectivity (the inverse of how selectivity was recorded).
                out = float(r.get("n_actual", 0) or 0.0)
                sel = float(r.get("selectivity", 0.0) or 0.0)
                n_in = out / sel if sel > 0.0 else out
            n_in = float(n_in)
            if peak > 0.0 and n_in > 0.0:
                samples.append(peak / n_in)
        if len(samples) >= min_samples:
            bpr[_canonical_kind(kind)] = _median(samples)
    return LearnedMemoryModel(
        _bytes_per_row=bpr,
        _alpha=opt.learning_smoothing_alpha,
        _clamp=max(1.0, opt.cost_calibration_clamp),
        _row_bytes=max(1, opt.row_bytes),
    )


def learned_memory_model(
    hub: MetadataHub | None, config: Config | None = None
) -> LearnedMemoryModel:
    """The learned per-family memory model for `hub` (an empty, pass-through model if cold).

    Reuses the prior fit unless the hub absorbed enough new feedback or the relevant
    config changed — avoiding the whole-history `op_stats` scan on every query. With no
    hub (a standalone manager, or api not yet wiring one) returns an empty model whose
    every method defers to the plan estimate, so sizing is byte-for-byte the current
    behavior. Best-effort: any failure yields the empty pass-through model.
    """
    cfg = config or active_config()
    if hub is None:
        return _empty_model(cfg)
    opt = cfg.optimizer
    fingerprint = (
        opt.learning_smoothing_alpha,
        opt.cost_calibration_min_samples,
        opt.cost_calibration_clamp,
        opt.row_bytes,
    )
    version = hub.version
    cached = _MODEL_CACHE.get(hub)
    if cached is not None and cached[1] == fingerprint and 0 <= version - cached[0] < _REFIT_AFTER:
        return cached[2]
    try:
        model = _fit(hub, cfg)
    except Exception:  # pragma: no cover - sizing must never break a query
        model = _empty_model(cfg)
    _MODEL_CACHE[hub] = (version, fingerprint, model)
    return model


def _empty_model(cfg: Config) -> LearnedMemoryModel:
    opt = cfg.optimizer
    return LearnedMemoryModel(
        _bytes_per_row={},
        _alpha=opt.learning_smoothing_alpha,
        _clamp=max(1.0, opt.cost_calibration_clamp),
        _row_bytes=max(1, opt.row_bytes),
    )
