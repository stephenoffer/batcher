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
    n = workers * max(1, cfg.map_partition_multiplier)
    cap = cfg.max_shuffle_partitions
    return max(workers, n if cap <= 0 else min(n, cap))


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
