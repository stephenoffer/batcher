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
from collections import deque
from dataclasses import dataclass, field
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
    "bump_generation",
    "columns_for",
    "generation",
    "is_material_change",
    "load_learned_stats",
    "qualify",
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

# Column statistics are keyed by **source, then column** — never by column name alone.
#
# A bare column name does not identify a column. Two tables both have an `id`, a `key`, a
# `date`; a flat `{name: stat}` map merges them, so whichever table was measured last
# silently answers for every other table with a column of that name — process-wide, for
# every join and group-by estimate that reads it. This repo already learned that lesson on
# the *row* side (see `StatsEstimator._estimate_uncached`: every `Scan` shares the
# signature `["scan"]`, so one table's measured 5M rows became a 1,000-row table's
# estimate, and a pruned MERGE sized its join at 2.4 TB). The column maps had the same
# defect and this is the qualifier that closes it.
#
# The key stays a flat string — `f"{source}\x1f{column}"` — so the stored shape is still
# `dict[str, value]` and every backend, the generation-bump check, and the merge logic are
# untouched. `\x1f` (ASCII unit separator) cannot occur in a column name.
_SOURCE_SEP = "\x1f"


def qualify(source_key: str, column: str) -> str:
    """The store key for `column` **of `source_key`** (see `_SOURCE_SEP`)."""
    return f"{source_key}{_SOURCE_SEP}{column}"


def columns_for(learned: dict[str, Any], stat_key: str, source_key: str | None) -> dict[str, Any]:
    """The `{column: value}` slice of a learned column map that describes `source_key`.

    Entries written *unqualified* (no separator) are treated as applying to every source.
    That is the legacy shape — a hub persisted by an older build, or a test that seeds the
    map directly — and a source-qualified entry always wins over it. Nothing on the live
    path writes unqualified any more (`record_column_stats` requires a source key), so the
    fallback is a compatibility shim, not a way back into the collision.
    """
    table = learned.get(stat_key) or {}
    prefix = f"{source_key}{_SOURCE_SEP}" if source_key is not None else None
    out: dict[str, Any] = {}
    qualified: dict[str, Any] = {}
    for key, value in table.items():
        if _SOURCE_SEP not in key:
            out[key] = value  # legacy: unqualified, applies to any source
        elif prefix is not None and key.startswith(prefix):
            qualified[key[len(prefix) :]] = value
    out.update(qualified)  # a measurement of *this* source beats a legacy global one
    return out


# Per-hub memo of the derived corrections, keyed weakly so a dropped hub evicts its
# entry. The value is `(hub.version, fingerprint, corrections)`: valid while the hub has
# absorbed no new operator feedback and the relevant config is unchanged. Without it,
# every `optimize` re-folds the recent q-error history — the fixed per-query overhead
# that dominates a sub-millisecond query.
_CORRECTION_CACHE: weakref.WeakKeyDictionary[MetadataHub, tuple[int, tuple, dict[str, float]]] = (
    weakref.WeakKeyDictionary()
)

# Cap on the distinct plan signatures whose q-error window is tracked. Each window holds
# at most `cardinality_correction_window` floats, so this bounds the fold's memory for a
# session that issues endlessly many distinct shapes.
_MAX_TRACKED_SIGNATURES = 4096


@dataclass(slots=True)
class _QErrorState:
    """The incremental q-error fold for one hub: how far it has read, and what it holds."""

    consumed: int  # `MetadataHub.signed_appends` as of the last fold
    window: int  # the configured per-signature sample window this state was built for
    samples: dict[str, deque[float]] = field(default_factory=dict)


# Per-hub incremental q-error windows, keyed weakly so a dropped hub evicts its state.
_QERROR_CACHE: weakref.WeakKeyDictionary[MetadataHub, _QErrorState] = weakref.WeakKeyDictionary()


# Bumped whenever something is learned that could change a *plan*, never for the routine
# drift of an already-converged estimate. `plan_cache` keys on it: a memoized plan stays
# valid until the feedback loop learns something worth re-planning for. This is the same
# judgement the adaptive executor makes — re-optimize when reality disagreed with the
# estimate, not merely because a smoothed average moved in its fourth decimal.
_GENERATION = 0

# A measured cardinality this far from the prior is a *material* correction: the estimate
# the last plan was chosen under was wrong by more than a factor of `1 + this`, which is
# enough to flip a build side or a join order. Smaller moves are the exponential average
# settling and must not invalidate a plan, or nothing would ever be reused.
_MATERIAL_CHANGE = 0.10


def generation() -> int:
    """A counter that advances only when the loop learns something plan-relevant."""
    return _GENERATION


def bump_generation() -> None:
    """Declare that something plan-relevant was learned, invalidating memoized plans.

    Called by every writer whose value the optimizer reads — the join-strategy bandit, the
    adaptive gate, partition sizing, and the column sketches. Only the *converged drift* of
    an already-known cardinality is exempt (`_is_material`), because that write happens on
    every single execution and gating on it is what makes memoizing a plan possible at all.
    Bumping too often only costs a re-plan; bumping too rarely leaves a stale plan in place,
    so anything uncertain should bump."""
    global _GENERATION
    _GENERATION += 1


def _bump_generation() -> None:
    bump_generation()


def is_material_change(prior: float | None, observed: float) -> bool:
    """Whether `observed` corrects `prior` by enough to be worth re-planning for.

    The threshold exists because every learned value is rewritten on every execution: an
    exponential average settles, a counter ticks. Treating that drift as news would make a
    memoized plan worthless. A correction past `_MATERIAL_CHANGE` is large enough to flip a
    build side or a join order; smaller ones are the estimate converging.
    """
    if prior is None:
        return True  # nothing was known; the next plan can only be better informed
    if prior <= 0:
        return observed > 0
    return abs(observed - prior) / prior > _MATERIAL_CHANGE


def _is_material(prior: float | None, observed: float) -> bool:
    return is_material_change(prior, observed)


def _smooth(prior: float, observed: float, n_obs: int) -> float:
    """Exponentially smooth `prior` toward `observed`, with an observation-count floor.

    The step is `max(floor, 1/(n_obs+1))`: a **running mean** while evidence is thin (so a
    single anomalous early run cannot anchor the estimate), decaying into an EWMA with a
    ~`1/floor`-observation memory once enough runs have accrued.

    The floor is `learned_scalar_alpha_floor`, not `learning_smoothing_alpha`. The latter is
    a *static blend weight* used elsewhere; at its value of 0.5 the newest run would always
    carry half the weight, so `1/(n_obs+1)` would be dominated from the second observation
    onward and the estimate would never converge.
    """
    alpha = max(active_config().optimizer.learned_scalar_alpha_floor, 1.0 / (n_obs + 1))
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


def _q_error_samples(hub: MetadataHub, window: int) -> dict[str, deque[float]]:
    """Per-signature `log(actual / estimated)` for the most recent `window` samples.

    Logs, not ratios, because the caller takes a geometric mean. Rows the estimator
    cannot learn from — no signature, no structural estimate (`n_estimated == 0`, which
    Kyber writes for a measured, exact, or unknown-size operator), or an empty output —
    are skipped rather than recorded as a q-error of zero.

    The fold is **incremental**: the per-signature windows persist across calls and only
    the feedback rows recorded since the last call are absorbed (`MetadataHub.
    signed_appends` says how many that is). Re-folding the retained history on every call
    would instead put a cost proportional to the session's cumulative query count on the
    critical path of *every* optimize — the same trap the Hub's own views avoid.
    """
    state = _QERROR_CACHE.get(hub)
    # Read the view *before* the cursor: the first read is what materializes the Hub's
    # view from the backend, and that load is what gives `signed_appends` its initial
    # value. Reading the cursor first would see 0 and absorb nothing.
    rows = hub.op_stats_with_signature()  # oldest first
    appends = hub.signed_appends
    if state is None or state.window != window or state.consumed > appends:
        # First fold for this hub, a reconfigured window, or a hub whose counter moved
        # backwards (a fresh backend behind it): rebuild from the retained history.
        state = _QErrorState(consumed=0, window=window, samples={})
        _QERROR_CACHE[hub] = state
    fresh = appends - state.consumed
    if fresh > 0:
        # The Hub's view is bounded, so a cursor left far enough behind can name more
        # rows than remain; absorb whatever is still retained.
        for row in rows[-fresh:] if fresh < len(rows) else rows:
            _absorb_q_error(state, row)
        state.consumed = appends
    return state.samples


def _absorb_q_error(state: _QErrorState, row: dict[str, Any]) -> None:
    """Fold one feedback row into its signature's bounded q-error window."""
    sig = row.get("signature") or ""
    est = float(row.get("n_estimated") or 0.0)
    actual = float(row.get("n_actual") or 0.0)
    if not sig or est <= 0.0 or actual <= 0.0:
        return
    bucket = state.samples.get(sig)
    if bucket is None:
        if len(state.samples) >= _MAX_TRACKED_SIGNATURES:
            # Oldest-inserted eviction (dicts preserve insertion order), so a session
            # issuing unboundedly many distinct plan shapes cannot grow this map forever.
            del state.samples[next(iter(state.samples))]
        bucket = state.samples[sig] = deque(maxlen=state.window)
    bucket.append(math.log(actual / est))


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
        if _is_material(prior, float(output_rows)):
            _bump_generation()
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
    source_key: str | None = None,
) -> None:
    """Record measured per-column distinct counts, quantile boundaries, widths, and
    most-common-values, as statistics **of one source**.

    These feed the `StatsEstimator`'s `__column_ndv__` (equality/join selectivity),
    `__column_quantiles__` (range selectivity), `__column_avg_bytes__` (byte-true
    memory/broadcast sizing), and `__column_mcv__` (skew-aware equality selectivity), so a
    query that has seen a column's data once plans better on every subsequent run.

    `source_key` is the source these columns were measured from (a data-stable
    `source.identity()`), and every key is qualified with it — because a column name alone
    does not identify a column, and an unqualified map lets one table's `id` answer for
    another's. Omitting it writes the legacy unqualified shape, which `columns_for` still
    honors as a global fallback; nothing on the live path does.

    Best-effort; never raises. Core measures (`core.column_statistics` /
    `core.heavy_hitters`); Kyber persists/consumes.
    """
    avg_bytes = avg_bytes or {}
    mcv = mcv or {}
    if hub is None or (not ndv and not quantiles and not avg_bytes and not mcv):
        return

    def keyed(values: dict[str, Any]) -> dict[str, Any]:
        if source_key is None:
            return values
        return {qualify(source_key, col): v for col, v in values.items()}

    try:
        # Each reserved column key is its own backend entry, updated independently
        # so a concurrent per-signature record (or another column update) can't
        # clobber it.
        if ndv:
            col_ndv = dict(hub.get_keyed_param(_NAMESPACE, NDV_KEY) or {})
            fresh = keyed(ndv)
            # A column measured for the first time can change every join and group-by
            # estimate that reads it — the one column-stat event worth re-planning for.
            if any(name not in col_ndv for name in fresh):
                _bump_generation()
            col_ndv.update(fresh)
            hub.put_keyed_param(_NAMESPACE, NDV_KEY, col_ndv)
        if quantiles:
            col_q = dict(hub.get_keyed_param(_NAMESPACE, QUANTILES_KEY) or {})
            col_q.update(keyed(quantiles))
            hub.put_keyed_param(_NAMESPACE, QUANTILES_KEY, col_q)
        if avg_bytes:
            col_w = dict(hub.get_keyed_param(_NAMESPACE, AVG_BYTES_KEY) or {})
            col_w.update(keyed(avg_bytes))
            hub.put_keyed_param(_NAMESPACE, AVG_BYTES_KEY, col_w)
        if mcv:
            col_mcv = dict(hub.get_keyed_param(_NAMESPACE, MCV_KEY) or {})
            col_mcv.update(keyed(mcv))
            hub.put_keyed_param(_NAMESPACE, MCV_KEY, col_mcv)
    except Exception:  # pragma: no cover - learning must never break execution
        pass
