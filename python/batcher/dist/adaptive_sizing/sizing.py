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
from batcher.metadata.hardware_scope import scoped
from batcher.plan.feedback import oversubscribed

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
    "row_shuffle_reducer_count",
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
    """Fold one observation into a per-signature EMA bucket ``{ema, n}``. Best-effort.

    Scoped to the machine class: every value stored here is a partition row count or a worker
    pool size, chosen against the node's memory and cores. A partition sized for a 244 GiB
    worker is an OOM on a 16 GiB one, and averaging the two teaches a size that is wrong on
    both. An autoscaling group that mixes instance types is the ordinary case, not the exotic
    one, so the scoping is on by default rather than a cluster-mode flag.
    """
    if hub is None or value != value or value < 0.0:  # None hub / NaN / negative guard
        return
    try:
        s = hub.get_keyed_param(scoped(namespace), key) or {}
        prior = s.get("ema")
        a = _alpha()
        ema = float(value) if prior is None else a * float(value) + (1.0 - a) * float(prior)
        hub.put_keyed_param(scoped(namespace), key, {"ema": ema, "n": int(s.get("n", 0)) + 1})
    except Exception as exc:  # pragma: no cover - learning must never break a query
        note_suppressed("dist", "fold sizing ema", exc)
        return


def _read_ema(hub: MetadataHub | None, namespace: str, key: str) -> float | None:
    """The learned EMA for a signature once it clears `_MIN_SAMPLES`, else `None`."""
    if hub is None:
        return None
    try:
        s = hub.get_keyed_param(scoped(namespace), key) or {}
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
    reservation. Purely a packing decision — the rows a task processes are unchanged.

    **A contended family is not a low-utilization family**, and this is the one place the
    distinction has to be made. A family whose threads were repeatedly evicted from their
    cores, or whose pages were being fetched back from disk, measures exactly as low as one
    that never wanted the cores — and the response the reading above would give is the
    opposite of the right one. Shrinking a contended family's reservation lets Ray pack more
    of its tasks onto the cores they are already fighting over, which lowers utilization
    further, which shrinks the reservation again: a loop that tightens itself with no step in
    it that looks wrong. `oversubscribed` reads the preemption and major-fault history Core
    already records and suppresses the learned factor, so the caller keeps its planned weight.

    Deliberately suppression rather than a *raise*: reserving more than planned under
    contention would relieve it, but it also over-provisions a whole cluster off a signal
    measured per operator family, and the loop this closes is the amplifying one."""
    samples = [u for u in _family_samples(hub, family, "cpu_utilization") if u > 0.0]
    if len(samples) < _MIN_SAMPLES:
        return None
    if _family_oversubscribed(hub, family):
        return None
    mean_util = sum(samples) / len(samples)
    return max(_CPU_FACTOR_LO, min(_CPU_FACTOR_HI, mean_util / _CPU_TARGET_UTIL))


def _family_oversubscribed(hub: MetadataHub | None, family: str) -> bool:
    """Whether this family's measured history shows it competing for the machine.

    Best-effort: any failure reads as "no evidence", which keeps the caller's prior behavior.
    """
    if hub is None:
        return False
    try:
        return oversubscribed(hub.op_stats_by_kind().get(family, []))
    except Exception as exc:  # pragma: no cover - a learned read must never break a query
        note_suppressed("dist", "read family contention", exc)
        return False


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
    target = max(1, active_config().optimizer.target_rows_per_task)
    want = math.ceil(_sizing_quantile(rows) / target)
    return max(1, min(int(workers), want))


# Which point of a family's measured input-volume history to size the exchange from. The
# errors here are not symmetric, which is what picks the statistic:
#
#   * This count only ever *reduces* the fan-out (it is clamped to `[1, workers]`), so an
#     under-estimate gives each reducer more state than it can hold and spills, while an
#     over-estimate is capped at the worker count the query would have used anyway.
#   * A family's history is routinely multi-modal — the same aggregate runs over a day and
#     over a year — and a mean sits between the modes, describing neither.
#
# So the sizing point is a high quantile: fan out for the larger runs the family actually
# sees. The `1 - 1/n` guard keeps a short history from picking anything but its maximum,
# which is the conservative reading when there is little to go on.
_SIZING_QUANTILE = 0.9


def _sizing_quantile(values: list[float]) -> float:
    """The `_SIZING_QUANTILE` point of `values` by nearest-rank; `0.0` when empty.

    Nearest-rank rather than an interpolating quantile so the result is always a volume the
    family was actually measured at, never a number between two modes that no run produced.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil(_SIZING_QUANTILE * len(ordered))
    return ordered[min(len(ordered), max(1, rank)) - 1]


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
def _estimated_rows(node, sources) -> float | None:
    """Kyber's estimate of `node`'s output rows, or `None` if it cannot be had.

    The cold-start half of every reducer sizing. `learned_signature_rows` only answers once
    the shape has run at least once, and the run it cannot help is the one that most needs
    it: a first execution at full scale, where guessing one reducer per worker is what makes
    the reduce spill. Kyber has already estimated this node's output cardinality from source
    statistics and its ndv sketches, and that estimate is a far better prior than the
    cluster's shape — it is at least *about* the data.

    For an aggregate the estimate is its group count (what the reduce holds); for a raw-row
    exchange — a join, sort, window, or distinct — it is the row count itself. Both callers
    want the same number for the same reason, which is why this is one function.

    Best-effort by construction, and only ever a fan-out: a wrong estimate costs a
    badly-shaped exchange, never a wrong answer.

    **It is only as good as the source's column statistics, and for some sources there are
    none.** An aggregate whose group keys have no measured `ndv` falls back in
    `StatsEstimator._estimate_aggregate` to a flat `rows x 0.1`, which is a constant rather
    than an estimate: measured cold in a fresh process over 4 M rows, it reads 400,000 whether
    the true group count is 100 or 3,147,395. That is survivable *here* only because the
    caller floors the reducer count at the worker count — a low estimate therefore lands on
    exactly the pre-existing one-reducer-per-worker behaviour and cannot make the exchange
    worse. It does mean the cold-start scaling this fallback exists for is real only for
    sources that carry statistics (a Parquet footer, a learned `__column_ndv__`), and that
    giving in-memory sources cheap key statistics would be what makes it general.

    The learned path above needs none of this: it is exact from the second run on.
    """
    if not sources:
        return None
    try:
        cardinality = active_config().optimizer.cardinality
        from batcher.kyber.cardinality import CardinalityEstimator

        est = CardinalityEstimator(list(sources), {}, cardinality)
        rows = est.estimate(node).rows
    except Exception as exc:
        note_suppressed("dist", "estimate reducer count", exc)
        return None
    if not rows or rows <= 0:
        return None
    # `unknown_rows` (1e12) is Kyber's *placeholder* for a relation nothing could size — its
    # own estimator calls it "not an estimate at all" and refuses to reason from it. A sizing
    # that took it at face value would read a source with no statistics (an iterator, a
    # connector with no catalog) as the largest table imaginable and open the maximum number
    # of near-empty streams for it, which is the low-end waste the reducer counts exist to
    # avoid. No evidence must look like no evidence.
    return None if rows >= cardinality.unknown_rows else rows


def _sizing_rows(node, sources) -> float | None:
    """How many rows the exchange below `node` carries — measured, else estimated, else None.

    The one place the two reducer sizings agree on where their number comes from: the
    measured history for this exact shape first, because it is exact from the second run on,
    and Kyber's estimate only when there is none. Both were written out twice before, and the
    two copies are what a divergence would hide — a sizing that consults history in one
    exchange and not another is invisible until a cluster is under it.

    Every failure lands on None, which every caller reads as "keep the fan-out you had". That
    is the module's standing contract made mechanical rather than incidental: a scheduling
    knob must never raise into execution, and the estimate path reaches the optimizer, the
    metadata hub, and the config to produce its answer.

    Args:
        node: The plan node whose output volume is being sized.
        sources: The sources `node` reads, for the cold-start estimate. Must be narrowed to
            match the node's own source ids — a relabeled map plan reads source 0.

    Returns:
        A positive row count, or None when neither history nor an estimate can supply one.
    """
    rows = None
    try:
        from batcher.core.runtime import default_hub
        from batcher.kyber.learned_tuning import learned_signature_rows
        from batcher.kyber.signature import plan_signature

        rows = learned_signature_rows(default_hub(), plan_signature(node))
    except Exception as exc:  # learning is best-effort; a miss keeps the default fan-out
        note_suppressed("dist", "read learned reducer count", exc)
    if rows is None or rows <= 0:
        try:
            rows = _estimated_rows(node, sources)
        except Exception as exc:  # pragma: no cover - the estimator guards itself too
            note_suppressed("dist", "estimate reducer count", exc)
            rows = None
    return rows if rows is not None and rows > 0 else None


def aggregate_reducer_count(agg, base_reducers: int, floor: int = 1, sources=None) -> int:
    """Reducer count for a keyed aggregate, sized by its LEARNED output cardinality.

    An aggregate's reduce shuffles PARTIAL-aggregated state, whose size is the group count —
    not the (far larger) scanned input. `base_reducers` (the generic one-per-worker fan-out)
    is therefore the wrong number in *both* directions, and the group count is what fixes it.

    Too many, at the low end: a 60M-row to 4-group aggregate does not need one reducer per
    worker each fetching from every mapper (a near-empty all-to-all), it needs one.

    Too few, at the high end, which is the one that breaks scaling. One reducer per worker
    fixes the reduce fan-out to the *cluster*, so each reducer's group table grows with the
    data: double the rows on the same cluster and every reducer's hash table doubles, until
    it stops fitting and the reduce spills. Wall time then grows faster than the input, which
    is exactly the superlinearity the mergeable algebra exists to avoid — the algebra allows
    any number of partial states to be merged independently, and pinning that number to the
    node count throws the property away. Sizing to `target_rows_per_task` groups per reducer
    instead keeps each reducer's state bounded at any scale; Ray queues the surplus tasks
    across the same workers, so more reducers cost scheduling, not memory.

    The count is capped by `distributed.max_shuffle_partitions`, because an exchange opens
    `mappers x reducers` streams and that product, not the reducer count, is what a very
    large cluster cannot afford.

    `floor` is the count below which trimming stops paying — the worker count, because a
    bucket is reduced by exactly ONE worker, so fewer buckets than workers leaves the rest
    idle for the whole reduce phase. The group count alone cannot see that: 5 M groups
    against a 4 M-row target asks for 2 reducers, which on an 8-worker cluster sat six
    workers out and made the reduce *slower* the more workers were added — measured on a
    9-node cluster at 0.65 s (2 workers), 1.05 s (4) and 4.47 s (8), against a map barrier
    that scaled normally over the same runs. The floor is itself capped by `rows`, so the
    low-cardinality trim above still reaches 1 for an aggregate that really does produce
    fewer groups than there are workers — that near-empty all-to-all is real.

    A cold signature falls back to Kyber's *estimated* group count
    (`_estimated_rows`) so a first run at scale is still sized by cardinality rather
    than by the cluster's shape; only when there is no estimate either does it keep
    `base_reducers`. The mergeable algebra makes any reducer count result-identical, so this
    only shapes the exchange.
    """
    rows = _sizing_rows(agg, sources)
    if rows is None:
        return base_reducers
    target = max(1, active_config().optimizer.target_rows_per_task)
    # `rows` is a learned EMA and therefore a float; the floor is capped by it, so it must be
    # truncated to an int or the whole count becomes a float and `partition_batches` — whose
    # `num_partitions` is a Rust `usize` — raises at the FFI boundary on the worker.
    want = max(1, math.ceil(rows / target), min(floor, int(rows)))
    cap = active_config().distributed.max_shuffle_partitions
    return min(want, cap) if cap > 0 else want


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
    except Exception as exc:  # pragma: no cover - learning must never break a query
        note_suppressed("dist", "record aggregate cardinality", exc)
        return


def row_shuffle_reducer_count(
    map_plan, base_reducers: int, sources=None, source_id: int = 0
) -> int:
    """Reducer count for an exchange that shuffles **raw rows** — join, sort, window, distinct.

    `aggregate_reducer_count` sizes the one exchange whose shuffled volume is *smaller* than
    its input: an aggregate exchanges partial state, so its reduce is sized by the group
    count. Every other shuffle exchanges the rows themselves, and until now every one of them
    took `shuffle_partitions(workers)` — a count that consults only *learned* history and, on
    a cold store, is exactly one bucket per worker.

    One bucket per worker is the wrong shape at scale for a reason that has nothing to do
    with parallelism. A bucket is the unit a reducer holds at once: a join builds its hash
    table from one, a sort sorts one, a window materializes one partition-run of one. Fixing
    the bucket count to the *cluster* makes that working set grow with the data — double the
    rows on the same cluster and every reducer's working set doubles, until it stops fitting
    and the operator spills. Wall time then grows faster than the input, which is the
    superlinearity the mergeable algebra exists to remove; pinning the bucket count to the
    node count is how it gets put back. The first run at full scale is both the one that
    suffers most and the one a learned count cannot help, so this reads Kyber's estimate
    when no history exists.

    **This may only raise the count, never lower it.** Below `base_reducers` a bucket-per-
    worker floor is already in force for a reason (fewer buckets than workers idles workers
    for the whole reduce phase), and unlike an aggregate a raw-row shuffle has no
    low-cardinality case where trimming is right — every input row lands in some bucket. The
    surplus buckets cost scheduling, not memory: Ray queues them across the same workers.

    The count is capped by `distributed.max_shuffle_partitions`, because an exchange opens
    `mappers x reducers` streams and that product is what a very large cluster cannot afford.

    Args:
        map_plan: The map-side plan whose output is exchanged. Its estimated or measured
            row count is the volume being divided. It has been relabeled to read source 0.
        base_reducers: The generic fan-out (`shuffle_partitions`), used as the floor.
        sources: The query's bound sources, for Kyber's cold-start estimate.
        source_id: Which of them `map_plan` originally read. The narrowing happens **here**
            rather than at the call site so an out-of-range id costs the estimate and not the
            query — every caller relabels its plan, and every one of them had to remember to
            pass `[sources[its own id]]` or silently size the exchange from whichever table
            happened to be source 0.

    Returns:
        The reducer count, at least `base_reducers`. Any count is result-identical under the
        mergeable algebra, so this only shapes the exchange.
    """
    try:
        narrowed = None if sources is None else [sources[source_id]]
    except (IndexError, TypeError) as exc:
        note_suppressed("dist", "narrow the sources for a shuffle estimate", exc)
        narrowed = None
    rows = _sizing_rows(map_plan, narrowed)
    if rows is None:
        return base_reducers
    target = max(1, active_config().optimizer.target_rows_per_task)
    want = max(base_reducers, math.ceil(rows / target))
    cap = active_config().distributed.max_shuffle_partitions
    return min(want, cap) if cap > 0 else want
