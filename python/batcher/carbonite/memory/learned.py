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
than an absolute row count. `bytes_per_row = quantile(m_peak_bytes / n_input)` is
size-general: multiply by *this* plan's estimated input rows to get a learned peak. The
quantile is deliberately an *upper* one rather than the median — see `_FOOTPRINT_QUANTILE`
for why a median is the wrong statistic for a one-sided safety quantity.

Everything here is pure and best-effort: it reads the hub, returns numbers, decides
nothing, and any failure or a cold store falls back to the caller's plan estimate so
a first run is byte-for-byte unchanged. Results never depend on it — only *how much
memory a decision reserves / when it spills / how big a morsel is*, which are
performance and scheduling, never correctness.
"""

from __future__ import annotations

import dataclasses
import math
import weakref
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from batcher._internal.logging import note_suppressed
from batcher._internal.mathx import blend, clamp_factor
from batcher.config import Config, active_config
from batcher.metadata import MetadataHub

__all__ = ["LearnedMemoryModel", "learned_memory_model"]

# Which quantile of the measured per-row footprints becomes the family's figure.
#
# The median was the obvious first choice and is the wrong statistic for this quantity.
# Memory sizing is one-sided: over-provisioning costs a little headroom, under-provisioning
# costs the process. A median is by construction exceeded by half the runs, so fitting the
# median means every second execution of a family reserves less than it turned out to need
# — and the ones that exceed it are exactly the large, skewed, or wide-row runs that OOM.
# A high quantile keeps the same size-generality (it is still a per-row ratio) while
# landing above the bulk of the distribution.
#
# 0.8 rather than 0.95: the samples are few (a family may have ten), the tail of a memory
# distribution is fat and noisy, and `blend_peak` already clamps the result to within
# `cost_calibration_clamp`x of the plan estimate. Chasing a far tail on ten samples fits
# noise; P80 moves the fit off the coin-flip without doing that. Spill volume uses the same
# quantile, where over-estimating shards into more, smaller buckets — also the safe side.
_FOOTPRINT_QUANTILE = 0.8


def _upper_quantile(values: Sequence[float], q: float = _FOOTPRINT_QUANTILE) -> float:
    """The `q`-quantile of `values` by linear interpolation between order statistics.

    The R type-7 / `numpy.percentile` definition, written out because this module is on the
    per-query path and must not pull numpy in for one number. For a single sample it is
    that sample; for two it interpolates, so a family with the minimum sample count still
    gets a figure that moves smoothly as evidence arrives rather than jumping between
    order statistics.

    Args:
        values: The measurements, in any order. Must be non-empty.
        q: The quantile in ``[0, 1]``.

    Returns:
        The interpolated quantile.
    """
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = math.floor(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    mixed = ordered[low] * (1.0 - frac) + ordered[high] * frac
    # Clamped to the two order statistics it interpolates between. `a*(1-f) + b*f` is not
    # exactly bounded by `[a, b]` in floating point — for a == b it can land one ulp
    # outside — so without this the "quantile" of a constant sample can exceed every value
    # in it. A per-row memory figure above every measurement is a small error that feeds
    # straight into a byte budget, and it makes the function's own contract untrue.
    return min(max(mixed, ordered[low]), ordered[high])


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
    # Measured out-of-core spill volume per input row per family (from `spill_bytes`), for
    # the families that actually spilled. Sizes spill partitions from the volume that really
    # goes to disk, which is smaller than the total working-set `peak` the plan otherwise
    # shards on. Empty until a family has spilled enough times to fit.
    _spill_per_row: dict[str, float]
    # The cardinality placeholder Kyber stamps on an operator it could not size
    # (`optimizer.cardinality.unknown_rows`, ~1e12). `predicted_spill_bytes` must reject an
    # `est_rows` at or above it rather than multiply a per-row figure by the sentinel.
    # Defaults to 0.0 — "no sentinel supplied, so trust no estimate" — which makes a model
    # built without it fall back to the caller's peak-based sizing, per this module's
    # contract that anything unlearned defers to the plan estimate.
    _unknown_rows: float = 0.0
    # The widest learned per-row footprint across every family. Derived in `__post_init__`
    # rather than passed in, so it cannot disagree with `_bytes_per_row`: it is a *fact
    # about* that dict, and a constructor argument would be one more thing every builder
    # (and every test double) has to remember to keep consistent.
    #
    # It is precomputed at all because `max_bytes_per_row(None)` is asked once per credit
    # grant and once per adaptive flow-control round — the shuffle's per-fetch path — where
    # rescanning the dict is a linear pass to recover a constant of the fit.
    _widest_bytes_per_row: float | None = dataclasses.field(init=False, default=None)

    def __post_init__(self) -> None:
        positive = [w for w in self._bytes_per_row.values() if w > 0]
        object.__setattr__(self, "_widest_bytes_per_row", max(positive) if positive else None)

    def bytes_per_row(self, kind: str) -> float | None:
        """Measured peak bytes per input row for `kind`'s family, or `None` if unlearned."""
        return self._bytes_per_row.get(_canonical_kind(kind))

    def spill_bytes_per_row(self, kind: str) -> float | None:
        """Measured spill volume per input row for `kind`'s family, or `None` if it never
        spilled enough to fit. The size-general basis for predicting a plan's spill volume."""
        return self._spill_per_row.get(_canonical_kind(kind))

    def _est_input_rows(self, op: object) -> float:
        """The op's estimated row count: Kyber's own figure when usable, else recovered.

        Kyber already publishes the exact estimate at `properties.est_rows` (`annotate.py`),
        so prefer it. The fallback divides the peak-byte bound back out by `row_bytes`, which
        is only correct when the plan sized that op with the *flat* default width — since
        `annotate` moved to a byte-true `row_width`, inverting with the flat constant is off
        by `row_width / row_bytes`, an order of magnitude on the wide payloads (blobs,
        embeddings) `row_width` exists to model. It is kept solely for objects that carry
        bounds but no properties (bare-sized test doubles, as in `plan_peak`).

        Returns `0.0` for an operator Kyber could not size: `est_rows` at or above the
        `unknown_rows` placeholder is a sentinel, not a count, and multiplying a per-row
        figure by ~1e12 would swamp the total. A NaN `est_rows` is the field's *unset*
        default (`PlanProperties`), not an estimate of zero, so it falls through to the
        recovered figure rather than silently contributing nothing.
        """
        props = getattr(op, "properties", None)
        rows = getattr(props, "est_rows", None) if props is not None else None
        if rows is not None and not math.isnan(rows):
            return float(rows) if 0.0 <= rows < self._unknown_rows else 0.0
        # Recovering rows from the byte bound: divide by the width the plan actually sized
        # with (`row_size`) when it published one, and only fall back to the flat
        # `row_bytes` default when it did not — inverting a learned width with the flat
        # constant is wrong by exactly their ratio.
        width = getattr(props, "row_size", None) if props is not None else None
        divisor = float(width) if width is not None and width == width and width > 0 else 0.0
        if divisor <= 0.0:
            divisor = float(self._row_bytes)
        if divisor <= 0.0:
            return 0.0
        return op.bounds.m_max_bytes / divisor  # type: ignore[attr-defined]

    def predicted_spill_bytes(self, plan_ops: object) -> int:
        """Predicted total out-of-core spill volume for `plan_ops`, or `0` if unlearned.

        For each op with a learned spill-per-row, multiply by that op's estimated input rows
        (`_est_input_rows`) and sum. `0` when no op's family has a spill history — the caller
        then keeps its peak-based sizing.
        """
        total = 0.0
        for op in plan_ops:  # type: ignore[attr-defined]
            spr = self.spill_bytes_per_row(getattr(op, "kind", ""))
            if spr is None:
                continue
            total += spr * self._est_input_rows(op)
        return int(total)

    def max_bytes_per_row(self, kinds: Iterable[str] | None = None) -> float | None:
        """The widest measured per-row footprint, or `None` if nothing is learned.

        The morsel byte-budget cap uses this: a workload whose rows proved wide should keep
        a tighter row-count so its true byte working set stays bounded. `kinds` restricts the
        max to *this plan's* operator families (canonical or plan-class names), so a narrow
        scan-only plan is not throttled by an unrelated wide aggregate measured in an earlier
        query; `None` (the default) keeps the global widest, unchanged."""
        if kinds is None:
            return self._widest_bytes_per_row
        wanted = {_canonical_kind(k) for k in kinds}
        widths = [w for k, w in self._bytes_per_row.items() if k in wanted and w > 0]
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
        blended = blend(plan_estimate, measured, self._alpha)
        return int(clamp_factor(blended, plan_estimate, self._clamp))


def _memory_basis_rows(row: dict) -> float:
    """The row count an operator's peak memory actually scales with.

    For a **join** that is `probe + build`: this is a batch engine, so a join materializes
    *both* inputs, and `m_peak_bytes` accounts for both. `n_input` now reports the probe
    side alone (so the join's `selectivity` is a meaningful fan-out), so dividing the
    two-sided peak by it would produce a bytes-per-row that grows with the build side.

    For every other family the input rows are the basis. A row persisted before `n_input`
    existed reconstructs them from `n_actual / selectivity`, the inverse of how selectivity
    was recorded.
    """
    n_in = row.get("n_input") or row.get("rows_in") or 0
    if not n_in:
        out = float(row.get("n_actual", 0) or 0.0)
        sel = float(row.get("selectivity", 0.0) or 0.0)
        n_in = out / sel if sel > 0.0 else out
    return float(n_in) + float(row.get("n_build") or 0)


def _fit(hub: MetadataHub, cfg: Config) -> LearnedMemoryModel:
    """Fit the per-family bytes-per-row model from the hub's measured `op_stats`."""
    opt = cfg.optimizer
    min_samples = max(1, opt.cost_calibration_min_samples)
    by_kind = hub.op_stats_by_kind()
    bpr: dict[str, float] = {}
    spr: dict[str, float] = {}
    for kind, rows in by_kind.items():
        samples: list[float] = []
        spill_samples: list[float] = []
        for r in rows:
            # The true peak is the greater of the Arrow working-set estimate and the measured
            # process RSS high-water (`peak_rss_bytes`): the latter captures transient scratch,
            # allocator fragmentation, and off-pool buffers the estimate cannot see, so fitting
            # against the max sizes admission/spill against reality and never under-provisions.
            peak = max(
                float(r.get("m_peak_bytes", 0) or 0.0),
                float(r.get("peak_rss_bytes", 0) or 0.0),
            )
            basis = _memory_basis_rows(r)
            if peak > 0.0 and basis > 0.0:
                samples.append(peak / basis)
            spill = float(r.get("spill_bytes", 0) or 0.0)
            if spill > 0.0 and basis > 0.0:
                spill_samples.append(spill / basis)
        canon = _canonical_kind(kind)
        if len(samples) >= min_samples:
            bpr[canon] = _upper_quantile(samples)
        if len(spill_samples) >= min_samples:
            spr[canon] = _upper_quantile(spill_samples)
    return LearnedMemoryModel(
        _bytes_per_row=bpr,
        _alpha=opt.learning_smoothing_alpha,
        _clamp=max(1.0, opt.cost_calibration_clamp),
        _row_bytes=max(1, opt.row_bytes),
        _unknown_rows=opt.cardinality.unknown_rows,
        _spill_per_row=spr,
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
        opt.cardinality.unknown_rows,
    )
    version = hub.version
    cached = _MODEL_CACHE.get(hub)
    if cached is not None and cached[1] == fingerprint and 0 <= version - cached[0] < _REFIT_AFTER:
        return cached[2]
    try:
        model = _fit(hub, cfg)
    except Exception as exc:  # pragma: no cover - sizing must never break a query
        note_suppressed("carbonite", "load the learned memory model", exc)
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
        _unknown_rows=opt.cardinality.unknown_rows,
        _spill_per_row={},
    )
