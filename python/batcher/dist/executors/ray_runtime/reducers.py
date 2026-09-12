"""How finely a shuffle divides its work — on both sides of the exchange.

`shuffle_partitions` sizes the reduce side (how many hash buckets), `map_partitions` the map
side (how many source partitions). They are the same *kind* of question and deliberately live
together, but they answer it for opposite reasons: buckets exist to bound a reducer's memory
and keep every worker busy, while map partitions exist to make the unit of scheduling and of
recovery smaller than a whole node's share of the input.

Split from `scaling`, which measures the *cluster*: how many nodes there are, how many cores
they have, and how much of that a query may use. The partition counts are a different
question — they are about the shuffle's own shape, and are answered from the operator's
measured history rather than from the topology. Keeping the two apart is also what keeps
`scaling` inside the module size limit.

The public import path is unchanged; `batcher.dist.executors.ray_runtime` re-exports this as
it always did.
"""

from __future__ import annotations

from batcher._internal.logging import note_suppressed
from batcher.config import active_config

__all__ = ["map_partitions", "shuffle_partitions"]


def shuffle_partitions(workers: int) -> int:
    """The number of shuffle partitions (reducers / hash buckets) for an all-to-all
    exchange over `workers` mappers.

    One bucket per worker, raised toward `workers x
    distributed.shuffle_partition_multiplier` when the shuffle's measured volume needs more,
    and capped by `distributed.max_shuffle_partitions`.

    **The worker count is a floor, not a target.** Reducers are not only how the reduce
    bounds its per-bucket memory, they *are* its parallelism: a bucket is reduced by exactly
    one worker, so fewer buckets than workers leaves workers idle for the whole reduce
    phase. Sizing purely for memory ignored that, and it fired on ordinary shapes — a 100 M
    row `GROUP BY` producing 5 M groups needs `ceil(5M / target_rows_per_task)` = 2 reducers
    to stay inside the memory target, so on an 8-worker cluster **six workers sat out the
    reduce**. Measured on a 9-node cluster: the reduce took 4.63 s of a 6.05 s query, and it
    got *slower* as workers were added (0.65 s at 2 workers, 1.05 s at 4, 4.47 s at 8) while
    the map barrier scaled normally. The learned volume may therefore only raise the count.

    Above the floor, more buckets lower each reducer's memory and make the work units finer.
    They do **not** fix skew, which is the usual reason given for the Spark-style
    many-partitions-per-executor default: a hash bucket is the unit a key cannot be split
    below, so a single dominant key stays on one reducer however fine the hash. Measured on
    12.5M rows with 40% on one key, max/mean bucket load goes from 3.8 at 8 buckets to 51.8
    at 128 — the hot bucket is the same size, only the mean shrinks. Splitting one key
    across reducers is `dist/skew.py`'s salting, not this. The multiplier bounds how far the
    count goes, because an exchange opens `mappers x reducers` streams — already O(nodes^2)
    at one reducer per worker, which is what the cap exists to bound at 10k+ nodes — and
    past a few thousand buckets the extra Flight fetches buy no parallelism at all.

    A cold store (no measured history) stays at one bucket per worker: full parallelism at
    the smallest stream count, and no guess. Any reducer count is result-correct under the
    mergeable algebra, so this only affects scaling. Always at least 1; the cap is disabled
    when the config value is 0.
    """
    cfg = active_config().distributed
    workers = max(1, workers)
    ceiling = max(workers, workers * max(1, cfg.shuffle_partition_multiplier))
    # `_learned_shuffle_fanout` answers "how many buckets does the measured volume need",
    # clamped to `[1, ceiling]`, or None with nothing measured. Take it only where it asks
    # for MORE than one bucket per worker; below that the floor wins.
    learned = _learned_shuffle_fanout(ceiling)
    n = min(ceiling, max(workers, learned if learned is not None else workers))
    cap = cfg.max_shuffle_partitions
    return n if cap <= 0 else max(1, min(n, cap))


def buckets_for_envelope(hint: int, source) -> int:
    """Raise `hint` until one *ordered* bucket is expected to fit the memory envelope.

    `shuffle_partitions` sizes the reduce for parallelism -- one bucket per worker, raised
    only by a **learned** shuffle volume -- which is right for a hash exchange, where a
    reducer that finds its bucket too big still has grace re-partitioning underneath it. The
    ordered global window has none: there is one partition, the kernel refuses an oversized
    bucket outright (`MemoryBudgetExceededError: window without PARTITION BY cannot spill`),
    and on a cold store the count was exactly `workers` whatever the input weighed. The
    single-node twin has sized from the data since `_buckets_for_staged`; scheduling the same
    algebra across machines does not change what a bucket costs to open.

    Size comes from the source's declared `byte_size` where it has one, else from
    `rows x schema_row_bytes` -- the same rows-times-width estimate `dist.gpu.shards` uses,
    and the only one available for an in-memory relation, whose `SourceStatistics` carries a
    row count but no `byte_size`. (Populating that field instead would feed the *width*
    estimator a sharper number, which `SourceStatistics.content_byte_size` documents as a
    measured plan regression and a separate, benchmark-driven change.)

    Only ever *raises* the count, for the reason `shuffle_partitions` gives at length: the
    worker count is a floor, and going below it idles workers for the whole reduce. Capped by
    `distributed.max_shuffle_partitions`, like every other reducer count.

    Args:
        hint: The parallelism-derived bucket count, used as a floor.
        source: The bound source, read for its size.

    Returns:
        The bucket count to cut, always `>= hint`.
    """
    from batcher.dist.spill.buckets import bucket_envelope

    envelope = bucket_envelope()
    if envelope <= 0:
        return hint
    total = _source_bytes(source)
    if total <= 0:
        return hint  # nothing measurable: keep the parallelism floor rather than guess
    cap = active_config().distributed.max_shuffle_partitions
    n = max(hint, -(-int(total) // envelope))
    return n if cap <= 0 else max(1, min(n, cap))


def _source_bytes(source) -> float:
    """A source's total size in bytes, or `0` when nothing about it can be measured."""
    from batcher.plan.types.widths import schema_row_bytes

    try:
        stats = source.statistics()
    except Exception:  # pragma: no cover - a connector that cannot answer costs the floor
        return 0.0
    if stats is None:
        return 0.0
    if stats.byte_size:
        return float(stats.byte_size)
    rows = stats.row_count
    if not rows:
        return 0.0
    try:
        return float(rows) * schema_row_bytes(source.schema())
    except Exception:  # pragma: no cover - unreadable schema: fall back to the floor
        return 0.0


def map_partitions(workers: int) -> int:
    """The number of map partitions a shuffle divides its input into — its **task unit**.

    One partition per worker is the smallest count that keeps every worker busy, and it was
    the only count for a long time. It also makes the task unit a *node's share of the
    input*, and that is what a coarse-grained engine pays for at the tail: a worker that
    runs half speed still holds a full partition, so the map barrier waits on it; a worker
    that dies loses a full partition, and one survivor replays the whole thing. Neither cost
    is about how much data there is — both are about the unit being indivisible.

    Cutting the input into `workers x distributed.map_partition_multiplier` pieces instead
    makes the unit that much smaller, and `map_barrier` deals them out as actors go idle, so
    a slow worker simply takes fewer. This is the same reason Spark runs 10k-100k tasks per
    stage. What it does **not** buy is skew tolerance within a key: partitions divide the
    *input*, and a shuffle's imbalance lives in its hash buckets, which is `shuffle_partitions`
    and, for a single dominant key, `dist/skew.py`'s salting.

    **The multiplier is spent only on a fleet that can use it.** Its benefit is the dynamic
    re-deal, and that requires the barrier to have partitions left over after its initial fill
    — which on a *uniform* fleet it never does, because `_idle_pool` is `workers x
    map_slots_per_worker()` deep and both factors are 4, so all `workers x 4` partitions go out
    in the first deal and the go-idle path never runs. Verified by construction on a 100-worker
    fleet: 400 partitions, a 400-deep pool, four per worker, nothing to re-deal.

    Its cost is not zero, and it lands after the barrier. Measured on 100 x 4-core nodes at
    TPC-H sf100, forward and reversed (`BENCHMARK_RESULTS.md`, 2026-09-08): a 20 M-group
    `GROUP BY` ran **3,192 ms at 400 partitions and 2,093 ms at 100**, and a five-aggregate one
    **5,674 ms against 3,236** — 34% and 43%, entirely in the phase after the map barrier
    (443 ms against 428, unchanged) and not in the tree, whose task count was held fixed and
    showed nothing. What is left scaling with the multiplier there is how many distinct sources
    each reducer fetches from: `mappers x reducers`, 40,000 against 10,000.

    So on an *unequal* fleet the multiplier stays, because there `_idle_pool` weights the
    initial deal by each worker's cores and that is worth having — measured at a per-core
    spread of 8.0x against 1.2x. What it costs on a uniform fleet is recovery *spread*, not
    recovery: a lost worker's share is `1/workers` either way, but one survivor replays it
    instead of four, which is ~340 ms once on a failure against ~1,100 ms saved on every query.

    The count is a ceiling, not a target. The caller passes it to `partition_descriptors` as
    `max_partitions`, and a source that cannot yield that many splits produces fewer — an
    input of ten row-groups on an eight-worker cluster is ten partitions, not thirty-two, so
    a small source never pays for empty tasks. Capped by `max_shuffle_partitions` for the
    same O(nodes²) reason the reduce side is: the exchange opens `mappers x reducers`
    streams, and both factors are in that product.

    **Not `map._adaptive_partition_count`**, which answers the same-sounding question for a
    *stateless* map and answers it from data volume and cluster cores. The two differ in what
    an extra partition costs. A stateless map task's output goes straight to the driver or the
    next stage, so its count is free to follow the data up to the core count. A shuffle
    mapper's output is `n_reducers` published buckets, so every extra map partition multiplies
    the exchange's stream count — the cost the reduce side is already capped for. Sizing the
    shuffle's map stage by cores would inherit a policy tuned where that term does not exist.

    Args:
        workers: The shuffle's worker fan-out.

    Returns:
        The maximum number of map partitions, at least `workers`.
    """
    cfg = active_config().distributed
    workers = max(1, workers)
    multiplier = max(1, cfg.map_partition_multiplier)
    if 1 < multiplier <= _map_slots() and _fleet_is_uniform(workers):
        multiplier = 1
    n = workers * multiplier
    cap = cfg.max_shuffle_partitions
    return max(workers, n if cap <= 0 else min(n, cap))


def _map_slots() -> int:
    """How many map partitions one actor may hold in flight, as the barrier will deal them.

    This is the second half of the condition above, and stating it as a *number* rather than
    as an assumption is what makes the rule adapt instead of being tuned to one configuration.
    The multiplier buys exactly two things. A **dynamic re-deal**, which requires the barrier to
    still hold partitions after its initial fill — that is `workers x multiplier` against a pool
    `workers x map_slots_per_worker()` deep, so it needs `multiplier > slots` and nothing about
    the fleet's shape. And a **core-weighted initial deal**, which only means anything when the
    workers differ, which is what `_fleet_is_uniform` asks.

    At the shipped defaults both are 4, so the re-deal is unreachable and a uniform fleet gets
    no value from the multiplier at all — which is the 34-43% recorded above. But the earlier
    form of this check tested only the fleet's shape, so raising `map_partition_multiplier` to
    8 against 4 slots would have collapsed it to 1 and thrown away a re-deal that had just
    become reachable. Reading the slot count keeps the two reasons independent.

    Falls back to the multiplier's own value when the count cannot be read, which makes the
    comparison false and leaves the multiplier alone: not knowing is a reason to change nothing.

    Returns:
        The per-worker map slot count, or a value that disables the collapse.
    """
    try:
        from batcher.dist.executors.ray_runtime.scheduling import map_slots_per_worker

        return max(1, int(map_slots_per_worker()))
    except Exception as exc:  # pragma: no cover
        note_suppressed("dist", "read the map slot count for the partition multiplier", exc)
        return 0


def _fleet_is_uniform(workers: int) -> bool:
    """Whether the fleet about to take `workers` groups of work holds one core grant per worker.

    The two states `_idle_pool` deals evenly on, and only those: an envelope carrying **no**
    per-worker grants — which is what `SchedulingEnvelope` documents a homogeneous cluster and
    every explicitly-sized fan-out as producing — or one carrying a grant per worker that is
    flat. Both are its `caps` guard read from this side, and the two have to agree about which
    fleet this is: a multiplier spent on a deal that weighs every worker equally buys nothing,
    and a multiplier withheld from a deal that weighs by cores takes away what it weights.

    **No ambient envelope is not a uniform fleet.** It is a caller outside a distributed
    execution — a unit test, a sizing question asked before a fleet exists — which knows nothing
    about the deal, so it answers `False` and the multiplier behaves exactly as it always did
    there. A grant list of the wrong length is the same kind of "don't know": `fleet_worker_cpus`
    declines it because a positional grant would then name the wrong worker.

    Not routed through `fleet_worker_cpus` itself, even though `_idle_pool` calls it, because it
    folds all three of those into one `None` and the first of them is the case this must say yes
    to.

    Args:
        workers: The fleet's width, as `map_partitions` was given it.

    Returns:
        `True` for a known-uniform fleet, `False` otherwise.
    """
    try:
        from batcher.dist.executors.ray_runtime.scheduling import current_envelope

        env = current_envelope()
    except Exception as exc:  # pragma: no cover - a sizing hint never fails a query
        note_suppressed("dist", "read the fleet shape for the map partition count", exc)
        return False
    if env is None:
        return False
    caps = env.worker_cpus
    return not caps or (len(caps) == workers and max(caps) == min(caps))


def _learned_shuffle_fanout(ceiling: int) -> int | None:
    """The learned reducer count for a shuffle, in `[1, ceiling]`, or `None` if unmeasured.

    Best-effort read of the process-wide MetadataHub's measured shuffle-family input volume;
    any failure (no hub, cold store) returns `None`, which the caller reads as "no evidence"
    rather than as a count. Returning `ceiling` instead — the previous shape — made a cold
    store indistinguishable from one that had measured a shuffle needing the full ceiling,
    so the caller could not tell a guess from a measurement."""
    try:
        from batcher.core import default_hub
        from batcher.dist.adaptive_sizing import learned_shuffle_fanout

        return learned_shuffle_fanout(default_hub(), None, ceiling)
    except Exception as exc:  # pragma: no cover - learning is best-effort
        note_suppressed("dist", "read learned shuffle fan-out", exc)
        return None
