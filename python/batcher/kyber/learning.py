"""Cross-execution learning — the metadata feedback loop.

After a query runs, its measured output cardinality is recorded in the
MetadataHub keyed by the plan's structural signature. The next time a plan of the
same shape appears — even as a *sub-plan* of a larger query — the estimator uses
the measured size instead of a default. This is how Batcher's decisions improve
with use: knowledge from past executions sharpens future plans.
"""

from __future__ import annotations

from typing import Any

from batcher.config import active_config
from batcher.kyber.signature import plan_signature
from batcher.metadata import MetadataHub
from batcher.plan.logical import LogicalPlan

__all__ = [
    "load_learned_stats",
    "record_column_stats",
    "record_execution",
    "record_selectivity",
]

_NAMESPACE = "kyber.stats"
# Reserved keys inside the stats namespace that hold per-column (not per-signature)
# statistics the `CardinalityEstimator` reads: distinct counts and quantile grids.
_NDV_KEY = "__column_ndv__"
_QUANTILES_KEY = "__column_quantiles__"
_AVG_BYTES_KEY = "__column_avg_bytes__"
_MCV_KEY = "__column_mcv__"


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

    Reassembled from the per-key store, so the shape consumers expect is unchanged.
    """
    if hub is None:
        return {}
    return hub.load_keyed_params(_NAMESPACE)


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
            col_ndv = dict(hub.get_keyed_param(_NAMESPACE, _NDV_KEY) or {})
            col_ndv.update(ndv)
            hub.put_keyed_param(_NAMESPACE, _NDV_KEY, col_ndv)
        if quantiles:
            col_q = dict(hub.get_keyed_param(_NAMESPACE, _QUANTILES_KEY) or {})
            col_q.update(quantiles)
            hub.put_keyed_param(_NAMESPACE, _QUANTILES_KEY, col_q)
        if avg_bytes:
            col_w = dict(hub.get_keyed_param(_NAMESPACE, _AVG_BYTES_KEY) or {})
            col_w.update(avg_bytes)
            hub.put_keyed_param(_NAMESPACE, _AVG_BYTES_KEY, col_w)
        if mcv:
            col_mcv = dict(hub.get_keyed_param(_NAMESPACE, _MCV_KEY) or {})
            col_mcv.update(mcv)
            hub.put_keyed_param(_NAMESPACE, _MCV_KEY, col_mcv)
    except Exception:  # pragma: no cover - learning must never break execution
        pass
