"""Learned execution-time sizing for the distributed executor — measure once, tune next run.

The distributed map/agg/shuffle path sizes several *scheduling* parameters from plan
estimates: how many partitions to split a source into, how much CPU each task reserves, how
many shuffle reducers to fan out to, how big an inference actor pool to build, how aggressive
straggler speculation should be, and how deep to read ahead on a scan. Those estimates are
cold on the first run and never improve — Ray Data has no such loop at all.

This module closes that loop the way `kyber.gpu.adaptive` closes the GPU-crossover loop:
Core **measures** the real outcome of every operator (rows in/out, per-core CPU busy fraction,
wall time, peak bytes) into the `MetadataHub` as `OperatorFeedback` — the *existing* feedback,
shared by the single-node executor and the distributed workers that ship their sub-plan metrics
back to the driver. The readers here **consume** that measured history, keyed by operator family
or source signature, and seed the next run's sizing decision so it converges to the workload
instead of rediscovering it. Two parameters that have no per-operator feedback analogue
(a source's measured total rows, a pool's served-partition count) keep a small EMA the executor
folds directly.

A cold signature returns `None` from every reader, so the caller keeps today's default and a
first run is byte-for-byte unchanged.

The one absolute rule holds throughout: every parameter here is a *scheduling* knob —
partition count only shards, CPU share only packs, reducer count only rebalances buckets (all
correct under the mergeable algebra), speculation only duplicates-then-dedupes, and readahead
only prefetches. **None of them can change a result.** Reads and writes are best-effort: a
malformed bucket or a hub error yields the default, never an exception into execution.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed
from batcher.config import active_config

if TYPE_CHECKING:
    from batcher.metadata import MetadataHub

__all__ = [
    "learned_actor_pool_size",
    "learned_cpu_weight_factor",
    "learned_partition_rows",
    "learned_shuffle_fanout",
    "learned_straggler_factor",
    "record_actor_pool_reuse",
    "record_partition_rows",
]

# Namespaces for the two directly-folded EMAs (no per-operator feedback analogue exists).
_NS_PARTITION = "dist_partition_rows"
_NS_POOL = "dist_actor_pool"

# Trust a learned value only after this many measured samples — a single distributed run is
# noisy (autoscaler warmup, cold caches, one stray straggler), so a few must agree before the
# learned value displaces the plan estimate. Matches the crossover learner's conservatism.
_MIN_SAMPLES = 3
# Target per-core busy fraction the CPU-share loop aims a family at: at this utilization the
# reserved cores are well spent, so the weight is kept; below it the family is IO/GPU-bound and
# cores sit idle, so the reservation shrinks proportionally.
_CPU_TARGET_UTIL = 0.85
_CPU_FACTOR_LO = 0.25
_CPU_FACTOR_HI = 1.0


def _alpha() -> float:
    """The exp-smoothing weight for a new observation (shared learning knob)."""
    return float(active_config().optimizer.learning_smoothing_alpha)


def _ema(hub: MetadataHub, namespace: str, key: str, value: float) -> None:
    """Fold one observation into a per-signature EMA bucket ``{ema, n}``. Best-effort."""
    if hub is None or value != value or value < 0.0:  # None hub / NaN / negative guard
        return
    try:
        s = hub.get_keyed_param(namespace, key) or {}
        prior = s.get("ema")
        a = _alpha()
        ema = float(value) if prior is None else a * float(value) + (1.0 - a) * float(prior)
        hub.put_keyed_param(namespace, key, {"ema": ema, "n": int(s.get("n", 0)) + 1})
    except Exception as exc:  # pragma: no cover - learning must never break a query
        note_suppressed("dist", "fold sizing ema", exc)
        return


def _read_ema(hub: MetadataHub | None, namespace: str, key: str) -> float | None:
    """The learned EMA for a signature once it clears `_MIN_SAMPLES`, else `None`."""
    if hub is None:
        return None
    try:
        s = hub.get_keyed_param(namespace, key) or {}
    except Exception:  # pragma: no cover
        return None
    if int(s.get("n", 0)) < _MIN_SAMPLES or "ema" not in s:
        return None
    return float(s["ema"])


def _family_samples(hub: MetadataHub | None, family: str, field: str) -> list[float]:
    """Every recorded `field` value for an operator `family` from the existing op-stats feedback.

    Reads `MetadataHub.op_stats_by_kind` — the measured runtime history Core already records for
    each native operator (and that distributed workers ship back) — so a learned decision reuses
    that feedback instead of maintaining a parallel write path. Best-effort: empty on any error."""
    if hub is None:
        return []
    try:
        rows = hub.op_stats_by_kind().get(family, [])
    except Exception:  # pragma: no cover
        return []
    out: list[float] = []
    for r in rows:
        v = r.get(field)
        if isinstance(v, (int, float)) and v == v:
            out.append(float(v))
    return out


# --- Partition count (map/agg source fan-out) -------------------------------------------
def record_partition_rows(hub: MetadataHub | None, source_id: str, rows: int) -> None:
    """Fold a run's measured total input rows for `source_id` into its EMA.

    Sized once per run on the driver from the concatenated partition outputs, so a source whose
    row count is not cheaply known from a footer (an in-memory / iterator / catalog source) can
    still drive `_adaptive_partition_count` on the next run instead of falling back to the blunt
    cluster-fill worker count."""
    if hub is not None and rows > 0:
        _ema(hub, _NS_PARTITION, source_id, float(rows))


def learned_partition_rows(hub: MetadataHub | None, source_id: str) -> int | None:
    """The learned total row count for `source_id`, or `None` when not yet learnable.

    Lets the partition-count heuristic use a *measured* size for a source with no cheap footer
    count, rather than defaulting to the worker fan-out."""
    ema = _read_ema(hub, _NS_PARTITION, source_id)
    return int(ema) if ema is not None and ema >= 1.0 else None


# --- Per-task CPU share (compute weight) -------------------------------------------------
def learned_cpu_weight_factor(hub: MetadataHub | None, family: str) -> float | None:
    """A multiplier in ``[_CPU_FACTOR_LO, _CPU_FACTOR_HI]`` for a family's compute weight, or
    `None` when the family has no measured CPU-utilization history yet.

    Reads the mean `cpu_utilization` Core recorded for the operator `family` (CPU-time / (wall x
    threads), in [0, 1]) and returns ``clamp(mean_util / _CPU_TARGET_UTIL)``: a family measured
    at the target busy fraction keeps its full weight (factor 1.0); one that used only a quarter
    of its reserved cores (IO/GPU-bound) reserves a quarter as many next time (factor 0.25). The
    unmeasured 0.0 sentinel is dropped so an older engine's blank signal never collapses the
    reservation. Purely a packing decision — the rows a task processes are unchanged."""
    samples = [u for u in _family_samples(hub, family, "cpu_utilization") if u > 0.0]
    if len(samples) < _MIN_SAMPLES:
        return None
    mean_util = sum(samples) / len(samples)
    return max(_CPU_FACTOR_LO, min(_CPU_FACTOR_HI, mean_util / _CPU_TARGET_UTIL))


# --- Shuffle fan-out (reducer count) -----------------------------------------------------
# The operator families whose distributed form is an all-to-all shuffle exchange.
_SHUFFLE_FAMILIES = ("aggregate", "hash_join", "sort", "window")


def learned_shuffle_fanout(hub: MetadataHub | None, family: str | None, workers: int) -> int | None:
    """A learned reducer count for a shuffle exchange, or `None` when not yet learnable.

    From the mean measured *input* rows Core recorded for the shuffle's operator `family`
    (`aggregate` / `hash_join` / ...) — or, when `family` is `None`, pooled across every shuffle
    family so the shared `shuffle_partitions` helper can consult it without the caller naming a
    family: ``ceil(mean_rows / target_rows_per_task)`` clamped to ``[1, workers]``. A shuffle
    whose measured volume needs fewer than `workers` reducers to keep each reducer's state in
    memory fans out to only that many (fewer, fuller buckets — less ``mappers x reducers`` stream
    overhead); a large shuffle keeps the full worker fan-out. Any reducer count is result-correct
    under the mergeable algebra, so this only shapes the exchange."""
    families = (family,) if family is not None else _SHUFFLE_FAMILIES
    rows: list[float] = []
    for fam in families:
        rows.extend(r for r in _family_samples(hub, fam, "n_input") if r > 0.0)
    if len(rows) < _MIN_SAMPLES:
        return None
    mean_rows = sum(rows) / len(rows)
    target = max(1, active_config().optimizer.target_rows_per_task)
    want = math.ceil(mean_rows / target)
    return max(1, min(int(workers), want))


# --- Inference actor-pool size (measured reuse) ------------------------------------------
def record_actor_pool_reuse(hub: MetadataHub | None, pipeline_sig: str, partitions: int) -> None:
    """Fold the number of partitions an inference pool actually served into its EMA.

    A pool sized to the worker/partition count but fed far fewer partitions per run wasted actor
    builds (each a full model load); learning the real served count right-sizes the pool next run
    so a small recurring inference job stops over-provisioning GPU actors."""
    if hub is not None and partitions > 0:
        _ema(hub, _NS_POOL, pipeline_sig, float(partitions))


def learned_actor_pool_size(hub: MetadataHub | None, pipeline_sig: str, default: int) -> int | None:
    """A learned actor-pool size for a pipeline signature, or `None` when not yet learnable.

    The measured served-partition count clamped to ``[1, default]`` — never above what the caller
    already resolved (autoscaling still grows within the run), only trimming a pool consistently
    fed fewer partitions than it had actors. Pool size is pure parallelism; the partitions and
    their rows are unchanged, so the result is identical."""
    served = _read_ema(hub, _NS_POOL, pipeline_sig)
    if served is None:
        return None
    return max(1, min(int(default), round(served)))


# --- Straggler speculation threshold (task-time variance) --------------------------------
def learned_straggler_factor(hub: MetadataHub | None, family: str) -> float | None:
    """A learned straggler-speculation factor for an operator family, or `None` when not yet
    learnable.

    From the coefficient of variation (``stddev / mean``) of the wall times Core recorded for the
    family: a family whose operators finish uniformly (low CV) gets a *higher* factor (don't waste
    backups on a tight distribution), while a heavy-tailed family (high CV — real stragglers) gets
    a lower factor so a backup fires sooner. Clamped to a band around the config default.
    Speculation only duplicates a slow task and keeps whichever copy finishes first, so the result
    is unchanged regardless of the factor."""
    times = [t for t in _family_samples(hub, family, "t_op_ms") if t > 0.0]
    if len(times) < _MIN_SAMPLES:
        return None
    mean = sum(times) / len(times)
    if mean <= 0.0:
        return None
    variance = sum((t - mean) ** 2 for t in times) / len(times)
    cv = (variance**0.5) / mean
    default = float(active_config().distributed.speculation_straggler_factor)
    factor = default * (1.0 + max(0.0, 1.0 - cv))
    return max(default * 0.75, min(default * 2.0, factor))


# --- Aggregate reducer count (learned output cardinality) --------------------------------
def aggregate_reducer_count(agg, base_reducers: int) -> int:
    """Reducer count for a keyed aggregate, sized by its LEARNED output cardinality.

    An aggregate's reduce shuffles PARTIAL-aggregated state, whose size is the group count —
    not the (far larger) scanned input. `base_reducers` (the generic one-per-worker fan-out)
    is therefore wrong for a low-cardinality group-by: a 60M-row → 4-group aggregate does not
    need one reducer per worker each fetching from every mapper (a near-empty all-to-all), it
    needs one. When a prior run measured this aggregate's output rows
    (`record_aggregate_cardinality`), size the reducers to keep each within
    `target_rows_per_task` groups. A cold signature keeps `base_reducers`; the mergeable
    algebra makes any reducer count result-identical, so this only shapes the exchange."""
    try:
        import math

        from batcher.core.runtime import default_hub
        from batcher.kyber.learned_tuning import learned_signature_rows
        from batcher.kyber.signature import plan_signature

        rows = learned_signature_rows(default_hub(), plan_signature(agg))
    except Exception:  # learning is best-effort; a miss keeps the default fan-out
        return base_reducers
    if rows is None or rows <= 0:
        return base_reducers
    target = max(1, active_config().optimizer.target_rows_per_task)
    return max(1, min(base_reducers, math.ceil(rows / target)))


def record_aggregate_cardinality(agg, output_rows: int) -> None:
    """Record a distributed aggregate's measured output rows so the next run sizes its shuffle
    reducers to the real group count — reconnecting the "Core measures → Kyber decides" loop the
    distributed aggregate path (unmetered `execute_plan`) otherwise leaves cold. Best-effort;
    keyed by the agg's own signature so `aggregate_reducer_count` reads back what was written,
    and so Kyber's next estimate for this aggregate is LEARNED, not a cold input-row fallback."""
    try:
        from batcher.core.runtime import default_hub
        from batcher.kyber import record_execution

        record_execution(default_hub(), agg, output_rows)
    except Exception:  # pragma: no cover - learning must never break a query
        return
