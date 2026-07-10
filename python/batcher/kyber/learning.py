"""Cross-execution learning — the metadata feedback loop.

After a query runs, its measured output cardinality is recorded in the
MetadataHub keyed by the plan's structural signature. The next time a plan of the
same shape appears — even as a *sub-plan* of a larger query — the estimator uses
the measured size instead of a default. This is how Batcher's decisions improve
with use: knowledge from past executions sharpens future plans.
"""

from __future__ import annotations

import math
import weakref
from typing import Any

from batcher.config import active_config
from batcher.kyber.signature import plan_signature
from batcher.metadata import MetadataHub
from batcher.plan.logical import LogicalPlan

__all__ = [
    "AVG_BYTES_KEY",
    "CARDINALITY_CORRECTION_KEY",
    "MCV_KEY",
    "NDV_KEY",
    "QUANTILES_KEY",
    "load_learned_stats",
    "record_column_stats",
    "record_execution",
    "record_selectivity",
]

_NAMESPACE = "kyber.stats"
# Reserved keys inside the stats namespace. Everything else in the namespace is keyed by
# a plan signature; these hold cross-signature state the `StatsEstimator` reads. They are
# the schema of the learned store, so they live here (the writer) and are imported by the
# estimator (the reader) rather than restated as literals on both sides.
NDV_KEY = "__column_ndv__"  # per-column distinct counts
QUANTILES_KEY = "__column_quantiles__"  # per-column quantile grids
AVG_BYTES_KEY = "__column_avg_bytes__"  # per-column average byte widths
MCV_KEY = "__column_mcv__"  # per-column most-common-values (skew)
# Derived, not stored: `load_learned_stats` folds the measured q-error history into
# `{signature: correction_factor}` under this key. See `_cardinality_corrections`.
CARDINALITY_CORRECTION_KEY = "__cardinality_correction__"

# Per-hub memo of the derived corrections, keyed weakly so a dropped hub evicts its
# entry. The value is `(hub.version, fingerprint, corrections)`: valid while the hub has
# absorbed no new operator feedback and the relevant config is unchanged. Without it,
# every `optimize` re-folds the recent q-error history — the fixed per-query overhead
# that dominates a sub-millisecond query.
_CORRECTION_CACHE: weakref.WeakKeyDictionary[MetadataHub, tuple[int, tuple, dict[str, float]]] = (
    weakref.WeakKeyDictionary()
)


def _smooth(prior: float, observed: float, n_obs: int) -> float:
    """Exponentially smooth `prior` toward `observed`, with an observation-count
    floor on the step. Early observations (small `n_obs`) move fast — the effective
    weight is `max(alpha, 1/(n_obs+1))`, i.e. a running mean until enough evidence
    accrues, then the configured `alpha` — so a settled estimate is stable while a
    single anomalous early run can't anchor it."""
    alpha = max(active_config().optimizer.learning_smoothing_alpha, 1.0 / (n_obs + 1))
    return alpha * observed + (1.0 - alpha) * prior


def load_learned_stats(hub: MetadataHub | None) -> dict[str, Any]:
    """Load the learned per-signature statistics (`{sig: {"rows": float}}`).

    Reassembled from the per-key store, so the shape consumers expect is unchanged, plus
    the derived `CARDINALITY_CORRECTION_KEY` entry folded in from the measured
    per-operator q-error history (`op_stats`).
    """
    if hub is None:
        return {}
    stats = dict(hub.load_keyed_params(_NAMESPACE))
    corrections = _cardinality_corrections(hub)
    if corrections:
        stats[CARDINALITY_CORRECTION_KEY] = corrections
    return stats


def _cardinality_corrections(hub: MetadataHub) -> dict[str, float]:
    """Per-signature cardinality correction factors, from the measured q-error history.

    Core records, for every operator it runs, the rows it actually produced (`n_actual`)
    alongside the rows Kyber's *structural* estimator predicted (`n_estimated`). Their
    ratio is the q-error. Averaging it per operator signature yields the factor by which
    the structural estimator is systematically wrong for that shape, so the next plan
    starts from a corrected number instead of repeating the mistake. This is the loop
    DuckDB, Polars, and Daft do not have at all, and that Spark AQE closes only within a
    single query.

    The average is **geometric**, because q-error is multiplicative and symmetric: a 4x
    over-estimate and a 4x under-estimate must cancel to 1.0, which an arithmetic mean
    would not (it would give 2.125).

    Only the most recent `cardinality_correction_window` samples of each signature count.
    The structural estimator is not static — it sharpens as the column-statistics loop
    learns NDVs and quantiles, and the data itself drifts — so a correction fitted to the
    estimator of ten runs ago is stale. A bounded window lets a correction decay to 1.0
    once the estimator no longer needs it, which an all-history mean would do only
    asymptotically.

    Signatures below `min_samples` are skipped and every factor is clamped, so one
    anomalous run cannot distort a plan.

    Best-effort: a malformed row is skipped, and any failure yields no corrections rather
    than raising into planning.
    """
    cfg = active_config().optimizer
    min_samples = cfg.cardinality_correction_min_samples
    max_factor = cfg.cardinality_correction_max_factor
    window = cfg.cardinality_correction_window
    if min_samples <= 0 or max_factor <= 1.0 or window <= 0:
        return {}
    # `optimize` is called several times per query (the main pass, the metadata-answer
    # rewrite, and once per adaptive stage) and this fold is O(recent history). Memoize it
    # against the hub's monotonic feedback counter, so it recomputes only when a new
    # observation has actually arrived.
    fingerprint = (min_samples, max_factor, window)
    cached = _CORRECTION_CACHE.get(hub)
    if cached is not None and cached[0] == hub.version and cached[1] == fingerprint:
        return cached[2]
    try:
        samples = _q_error_samples(hub, window)
    except Exception:  # pragma: no cover - learning must never break planning
        return {}
    out: dict[str, float] = {}
    for sig, log_qs in samples.items():
        if len(log_qs) < min_samples:
            continue
        factor = math.exp(sum(log_qs) / len(log_qs))  # geometric mean of the q-errors
        factor = min(max_factor, max(1.0 / max_factor, factor))
        if factor != 1.0:
            out[sig] = factor
    _CORRECTION_CACHE[hub] = (hub.version, fingerprint, out)
    return out


def _q_error_samples(hub: MetadataHub, window: int) -> dict[str, list[float]]:
    """Per-signature `log(actual / estimated)` for the most recent `window` samples.

    Logs, not ratios, because the caller takes a geometric mean. Rows the estimator
    cannot learn from — no signature, no structural estimate (`n_estimated == 0`, which
    Kyber writes for a measured, exact, or unknown-size operator), or an empty output —
    are skipped rather than recorded as a q-error of zero.
    """
    samples: dict[str, list[float]] = {}
    for row in hub.op_stats_with_signature():  # oldest first
        sig = row.get("signature") or ""
        est = float(row.get("n_estimated") or 0.0)
        actual = float(row.get("n_actual") or 0.0)
        if not sig or est <= 0.0 or actual <= 0.0:
            continue
        bucket = samples.setdefault(sig, [])
        bucket.append(math.log(actual / est))
        if len(bucket) > window:
            bucket.pop(0)  # keep only the newest `window` observations
    return samples


def record_execution(hub: MetadataHub | None, plan: LogicalPlan, output_rows: int) -> None:
    """Record a plan's measured output cardinality. Best-effort; never raises.

    Reads and writes only this signature's own key, so a concurrent record for a
    different shape cannot clobber it (no whole-blob lost-update race).
    """
    if hub is None:
        return
    try:
        sig = plan_signature(plan)
        entry = dict(hub.get_keyed_param(_NAMESPACE, sig) or {})  # preserve sibling keys
        prior = entry.get("rows")
        entry["rows"] = (
            float(output_rows)
            if prior is None
            else _smooth(prior, float(output_rows), entry.get("n_obs", 0))
        )
        entry["n_obs"] = entry.get("n_obs", 0) + 1
        hub.put_keyed_param(_NAMESPACE, sig, entry)
    except Exception:  # pragma: no cover - learning must never break execution
        pass


def record_selectivity(
    hub: MetadataHub | None, plan: LogicalPlan, sources: list, output_rows: int
) -> None:
    """Record a filter's MEASURED selectivity (kept fraction), keyed by its signature.

    Unlike a learned absolute row count, a selectivity *ratio* generalizes across
    input sizes: a `WHERE` clause measured on one scan sharpens the estimate even
    when the same filter later runs over a differently-sized input. Only recorded
    for a filter directly over a scan, and the *full* scan size (`row_count`, cheap
    and pre-pushdown) is the denominator — so it stays correct even when the
    predicate was pushed into the source. Best-effort; never raises.
    """
    if hub is None:
        return
    try:
        flt = _filter_over_scan(plan)
        if flt is None:
            return
        full = sources[flt.input.source_id].row_count()
        if not full or full <= 0:
            return
        sel = max(0.0, min(1.0, output_rows / full))
        sig = plan_signature(flt)
        entry = dict(hub.get_keyed_param(_NAMESPACE, sig) or {})
        prior = entry.get("selectivity")
        n_obs = entry.get("sel_n_obs", 0)
        entry["selectivity"] = sel if prior is None else _smooth(prior, sel, n_obs)
        entry["sel_n_obs"] = n_obs + 1
        hub.put_keyed_param(_NAMESPACE, sig, entry)
    except Exception:  # pragma: no cover - learning must never break execution
        pass


def _filter_over_scan(plan: LogicalPlan):
    """The outermost `Filter` whose input is a `Scan`, reachable through
    row-preserving `Project`s (so the plan's output rows equal that filter's output
    rows). `None` if the plan isn't shaped that way."""
    from batcher.plan.logical import Filter, Project, Scan

    node = plan
    while isinstance(node, Project):
        node = node.input
    if isinstance(node, Filter) and isinstance(node.input, Scan):
        return node
    return None


def record_column_stats(
    hub: MetadataHub | None,
    ndv: dict[str, float],
    quantiles: dict[str, dict[str, list[float]]],
    avg_bytes: dict[str, float] | None = None,
    mcv: dict[str, dict[str, float]] | None = None,
) -> None:
    """Record measured per-column distinct counts, quantile boundaries, widths, and
    most-common-values.

    These feed the `CardinalityEstimator`'s `__column_ndv__` (equality/join
    selectivity), `__column_quantiles__` (range selectivity), `__column_avg_bytes__`
    (byte-true memory/broadcast sizing), and `__column_mcv__` (skew-aware equality
    selectivity), so a query that has seen a column's data once plans better on every
    subsequent run. Best-effort; never raises. Core measures
    (`core.column_statistics` / `core.heavy_hitters`); Kyber persists/consumes.
    """
    avg_bytes = avg_bytes or {}
    mcv = mcv or {}
    if hub is None or (not ndv and not quantiles and not avg_bytes and not mcv):
        return
    try:
        # Each reserved column key is its own backend entry, updated independently
        # so a concurrent per-signature record (or another column update) can't
        # clobber it.
        if ndv:
            col_ndv = dict(hub.get_keyed_param(_NAMESPACE, NDV_KEY) or {})
            col_ndv.update(ndv)
            hub.put_keyed_param(_NAMESPACE, NDV_KEY, col_ndv)
        if quantiles:
            col_q = dict(hub.get_keyed_param(_NAMESPACE, QUANTILES_KEY) or {})
            col_q.update(quantiles)
            hub.put_keyed_param(_NAMESPACE, QUANTILES_KEY, col_q)
        if avg_bytes:
            col_w = dict(hub.get_keyed_param(_NAMESPACE, AVG_BYTES_KEY) or {})
            col_w.update(avg_bytes)
            hub.put_keyed_param(_NAMESPACE, AVG_BYTES_KEY, col_w)
        if mcv:
            col_mcv = dict(hub.get_keyed_param(_NAMESPACE, MCV_KEY) or {})
            col_mcv.update(mcv)
            hub.put_keyed_param(_NAMESPACE, MCV_KEY, col_mcv)
    except Exception:  # pragma: no cover - learning must never break execution
        pass
