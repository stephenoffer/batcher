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

from batcher._internal.logging import note_suppressed
from batcher.config import active_config
from batcher.kyber.column_tables import (
    AVG_BYTES_KEY,
    CARDINALITY_CORRECTION_KEY,
    MCV_KEY,
    NDV_KEY,
    QUANTILES_KEY,
    ROW_BYTES_KEY,
    UDF_ROW_SECONDS_KEY,
    merge_column_table,
    qualify,
)
from batcher.kyber.column_tables import (
    STATS_NAMESPACE as _NAMESPACE,
)
from batcher.kyber.correction import correction_factor
from batcher.kyber.measured_selectivity import measured_selectivities
from batcher.kyber.measured_width import measured_widths
from batcher.kyber.signature import plan_signature
from batcher.metadata import MetadataHub
from batcher.metadata.hardware_scope import local_or_planned_fingerprint
from batcher.metadata.udf_stats import load_udf_row_seconds_table
from batcher.plan.logical import LogicalPlan

__all__ = [
    "bump_generation",
    "generation",
    "is_material_change",
    "load_learned_stats",
    "q_error_window",
    "record_column_row_bytes",
    "record_column_stats",
    "record_execution",
    "record_selectivity",
]

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
    #: Signatures whose window changed since the correction factors were last summarized,
    #: so only those are re-derived. Owned by `_cardinality_corrections`, which clears it.
    dirty: set[str] = field(default_factory=set)


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


#: The assembled bundle per hub, valid while the hub's two change counters stand still **and
#: the machine class it was assembled for is the same one**. Weakly keyed so a dropped hub
#: evicts its own entry.
#:
#: The machine class belongs in the key because one of the bundle's components is scoped by it:
#: `load_udf_row_seconds_table` reads `scoped("udf.row_seconds")`, which resolves through the
#: ambient `planning_for`. Without it, a driver that plans a distributed run (the workers'
#: class) and then a local one (its own) serves each the other's measured per-row UDF costs
#: under the same `(version, params_version)` pair — the two counters do not move, so the stale
#: bundle looks current. Measured with one `fn` recorded at 1 ms/row on a worker and 1 ns/row
#: on the driver: the worker-scoped read returned the driver's figure, a millionfold error in
#: the number that decides whether a `map_batches` is priced as a trivial column map. It is the
#: same machine-class blend `metadata.hardware_scope` exists to prevent, entering through a
#: cache key rather than a namespace — the third time in this loop, after `cpu_shares` and
#: `spill_rates`.
_BUNDLE_CACHE: weakref.WeakKeyDictionary[MetadataHub, tuple[int, int, str, dict[str, Any]]] = (
    weakref.WeakKeyDictionary()
)


def load_learned_stats(hub: MetadataHub | None) -> dict[str, Any]:
    """Load the learned per-signature statistics (`{sig: {"rows": float}}`).

    Reassembled from the per-key store, so the shape consumers expect is unchanged, plus
    the derived `CARDINALITY_CORRECTION_KEY` entry folded in from the measured
    per-operator q-error history (`op_stats`).

    **The result is read-only and shared**, and reassembled only when the hub has actually
    changed — `hub.version` covers the operator history the folds below read, and
    `hub.params_version` covers the parameter store. Every consumer treats it as a lookup
    table, which is what makes sharing sound.

    Reassembling per call made a session's per-query cost grow with its own history, which
    is the opposite of what a learning loop is for. The work is O(learned signatures) with a
    fresh dict allocated per signature, and it runs several times per query — twice before
    the optimizer (the ndv seeding and the metadata-answer attempt), once inside it, and
    once in the close-out. Measured on TPC-DS at scale 1, replaying the suite in one
    session: the same probe query took **9.4 ms with nothing else run and 21.5 ms after 100
    other queries had run**, and this function was 57% of the growth. Nothing about the
    probe changed — only how much the process remembered.
    """
    if hub is None:
        return {}
    fingerprint = (hub.version, hub.params_version, local_or_planned_fingerprint())
    cached = _BUNDLE_CACHE.get(hub)
    if cached is not None and cached[:3] == fingerprint:
        return cached[3]
    stats = dict(hub.load_keyed_params(_NAMESPACE))
    corrections = _cardinality_corrections(hub)
    if corrections:
        stats[CARDINALITY_CORRECTION_KEY] = corrections
    for sig, sel in measured_selectivities(hub).items():
        # `setdefault`: an explicit `record_selectivity` entry still wins. The two agree
        # wherever both fire, so this only fills a signature that had nothing.
        entry = dict(stats.get(sig) or {})
        entry.setdefault("selectivity", sel)
        stats[sig] = entry
    for sig, width in measured_widths(hub).items():
        # The measured output width of a shape. The byte axes otherwise re-derive it by summing
        # per-column priors through every operator that reshapes a row, which is what
        # `cost.model` cites as the reason it declines to charge for width at all. Folded into
        # the same per-signature entry so the estimator reads one map rather than three.
        entry = dict(stats.get(sig) or {})
        entry.setdefault("row_bytes", width)
        stats[sig] = entry
    # Measured per-row cost of each `map_batches` callable Core has timed. Keyed by UDF
    # identity rather than plan signature (a callable costs what it costs, whatever plan it
    # sits in), so it rides under its own reserved key instead of the per-signature entries.
    udf_costs = load_udf_row_seconds_table(hub)
    if udf_costs:
        stats[UDF_ROW_SECONDS_KEY] = udf_costs
    _BUNDLE_CACHE[hub] = (*fingerprint, stats)
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
    would not (it would give 2.125). It is also recency-weighted and shrunk toward "no
    correction" by how much the samples actually agree — see `kyber.correction`, which owns
    that estimator.

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
    except Exception as exc:  # pragma: no cover - learning must never break planning
        note_suppressed("kyber", "read learned cardinality corrections", exc)
        return {}
    # Re-derive only the signatures whose q-error window actually moved. `_q_error_samples`
    # was already incremental; this summary was not, so every query re-computed a geometric
    # mean for every shape the session had *ever* run — a cost proportional to cumulative
    # history, on the critical path of each optimize. The memo above hides it only while the
    # hub stands still, and the hub moves on every query. Measured on TPC-DS at scale 1: a
    # probe query cost 9.4 ms cold and 21.5 ms after 100 other queries in the same session.
    state = _QERROR_CACHE[hub]
    stale = cached is None or cached[1] != fingerprint
    out: dict[str, float] = {} if stale else cached[2]
    for sig in samples if stale else state.dirty:
        log_qs = samples.get(sig)
        # A signature can be evicted from the window map by the tracking cap while still
        # named here; it then has no evidence left and must not keep its old factor.
        factor = 1.0 if log_qs is None else correction_factor(list(log_qs), min_samples, max_factor)
        # **A correction appearing does not move the learned generation, and it was tried.**
        # It is the one plan-relevant value that never announces itself — it lives in this
        # fold rather than in the keyed-parameter store `plan_cache.record_write` guards — so
        # a plan memoized on the first run is one chosen from the structural estimate alone
        # and kept. That is a real cost: H2O `groupby` q9 is estimated at ~100 groups
        # structurally and 10,000 once corrected, Kyber picks the executor for the shape from
        # that estimate (`MATERIALIZE_AGG_MIN_GROUPS`), and it keeps the 100-group choice —
        # 62 ms against the 44 ms the other executor takes. Bumping here fixed exactly that,
        # q9 62 -> 44 ms and q2 50 -> 39.
        #
        # It also cost more than it bought, because the generation is **global**. A mixed
        # workload meets new signatures continuously, so "bump once per signature" is still a
        # steady stream of bumps, and each one drops *every* memoized plan in the process.
        # Measured on TPC-DS at scale 1 with fifty-five other queries of the suite run first,
        # q34 went from **17 ms to 85-210** and the suite's ratio spread widened. Two h2o
        # queries at ~0.5x against one TPC-DS query at 5x is not a trade worth making.
        #
        # The fix this wants is a plan cache keyed by *the corrections its own plan reads*
        # rather than by one global counter. Until then, a correction is silent.
        if factor != 1.0:
            out[sig] = factor
        else:
            out.pop(sig, None)
    state.dirty.clear()
    _CORRECTION_CACHE[hub] = (hub.version, fingerprint, out)
    return out


def q_error_window(hub: MetadataHub, signature: str, window: int) -> deque[float] | None:
    """One signature's bounded window of `log(actual / estimated)`, or `None` if untracked.

    The narrow read `kyber.correction` needs to judge whether a shape's estimate has held
    up. Kept here because this module owns the incremental fold; exposed rather than
    reaching into `_q_error_samples` so the fold's caching contract stays internal.
    """
    return _q_error_samples(hub, window).get(signature)


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
            # The evicted signature is marked dirty so the incremental summary in
            # `_cardinality_corrections` re-derives it, finds no evidence, and drops its
            # factor — a whole-map rebuild used to do that implicitly.
            evicted = next(iter(state.samples))
            del state.samples[evicted]
            state.dirty.add(evicted)
        bucket = state.samples[sig] = deque(maxlen=state.window)
    bucket.append(math.log(actual / est))
    state.dirty.add(sig)


def record_execution(hub: MetadataHub | None, plan: LogicalPlan, output_rows: int) -> None:
    """Record a plan's measured output cardinality. Best-effort; never raises.

    Reads and writes only this signature's own key, so a concurrent record for a
    different shape cannot clobber it (no whole-blob lost-update race).

    **The generation moves on the *stored* value, not on the observation.** What a plan reads
    is the smoothed estimate; the raw count is one sample of it, and smoothing exists precisely
    so a single sample does not move the estimate by its own distance. Comparing the sample
    against the prior therefore invalidates on drift the estimate absorbed — the same mistake
    `plan_cache.record_write` documents for the bandit, in the one place that does not route
    through it.

    It is not a small effect, because a signature is *structural*: two queries of the same
    shape share one entry (see the scan-collision note in `kyber.signature`), so the estimate
    is fed alternating observations and never settles on either. Measured on TPC-DS at scale 1
    with sixty other queries of the suite already run in the session, q77's signature held 100
    while q77 measured 44 on every execution — a 56% "change" each time — so **every run of
    q77 invalidated every memoized plan in the process**, and q77 re-planned for 131 ms of its
    268 ms. The learned generation is global; one query that never settles costs the whole
    session its plan cache.
    """
    if hub is None:
        return
    try:
        sig = plan_signature(plan)
        entry = dict(hub.get_keyed_param(_NAMESPACE, sig) or {})  # preserve sibling keys
        prior = entry.get("rows")
        updated = (
            float(output_rows)
            if prior is None
            else _smooth(prior, float(output_rows), entry.get("n_obs", 0))
        )
        if _is_material(prior, updated):
            _bump_generation()
        entry["rows"] = updated
        entry["n_obs"] = entry.get("n_obs", 0) + 1
        hub.put_keyed_param(_NAMESPACE, sig, entry)
    except Exception as exc:  # pragma: no cover - learning must never break execution
        note_suppressed("kyber", "persist a learned row count", exc)


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
    except Exception as exc:  # pragma: no cover - learning must never break execution
        note_suppressed("kyber", "persist a learned selectivity", exc)


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


def record_column_row_bytes(
    hub: MetadataHub | None,
    widths: dict[str, float],
    source_key: str | None = None,
) -> None:
    """Record cheaply-measured byte widths for **every** column a query read.

    The sketched statistics (`record_column_stats`) are restricted to the columns a later plan
    could consult a *distribution* for — join keys, group keys, filtered columns — because a
    KLL grid and a Misra-Gries table cost ~56 ns a cell and a column nothing predicates on has
    no use for either.

    A byte width is not like that. `StatsEstimator.row_width` sums per-column widths over every
    **output** column, so the columns that dominate a row's size are precisely the payload ones
    no predicate mentions — the embedding, the document, the image — and those were the ones
    never measured. The result was a row width understated by orders of magnitude on exactly the
    data where it decides whether a task fits in memory: `kyber.annotate` sizes a stage from it,
    and its own table puts a 768-dim embedding at 12 GB per task under the flat prior.

    Measuring it costs nothing worth counting. Arrow already knows an array's buffer size, so
    this is `nbytes / num_rows` per column — O(columns), no sample, no sketch, no per-row work.

    Written to its own table rather than `AVG_BYTES_KEY`; see `column_tables.ROW_BYTES_KEY` for
    why that separation is load-bearing.

    Args:
        hub: The metadata hub to write to; `None` is a no-op.
        widths: `{column: bytes per row}` for the columns read.
        source_key: The source these columns belong to. `None` skips the write — a width that
            cannot be attributed to a source is a width that would be applied to the wrong one.
    """
    if hub is None or not widths or not source_key:
        return
    try:
        merge_column_table(
            hub, ROW_BYTES_KEY, {qualify(source_key, c): w for c, w in widths.items()}
        )
    except Exception as exc:  # pragma: no cover - learning must never break a query
        note_suppressed("kyber", "persist measured column row widths", exc)


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
            fresh = keyed(ndv)
            existing = hub.get_keyed_param(_NAMESPACE, NDV_KEY) or {}
            # A column measured for the first time can change every join and group-by
            # estimate that reads it — the one column-stat event worth re-planning for.
            if any(name not in existing for name in fresh):
                _bump_generation()
            merge_column_table(hub, NDV_KEY, fresh, existing)
        if quantiles:
            merge_column_table(hub, QUANTILES_KEY, keyed(quantiles))
        if avg_bytes:
            merge_column_table(hub, AVG_BYTES_KEY, keyed(avg_bytes))
        if mcv:
            merge_column_table(hub, MCV_KEY, keyed(mcv))
    except Exception as exc:  # pragma: no cover - learning must never break execution
        note_suppressed("kyber", "persist learned column statistics", exc)
