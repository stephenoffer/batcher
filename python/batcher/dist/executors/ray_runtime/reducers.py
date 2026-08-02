"""How many reducers a shuffle fans out to.

Split from `scaling`, which measures the *cluster*: how many nodes there are, how many cores
they have, and how much of that a query may use. The reducer count is a different question —
it is about the shuffle's own shape, and it is answered from the operator's measured history
rather than from the topology. Keeping the two apart is also what keeps `scaling` inside the
module size limit.

The public import path is unchanged; `batcher.dist.executors.ray_runtime` re-exports this as
it always did.
"""

from __future__ import annotations

from batcher._internal.logging import note_suppressed
from batcher.config import active_config

__all__ = ["shuffle_partitions"]


def shuffle_partitions(workers: int) -> int:
    """The number of shuffle partitions (reducers / hash buckets) for an all-to-all
    exchange over `workers` mappers — capped by `distributed.max_shuffle_partitions`.

    An exchange creates `mappers * reducers` streams; leaving the reducer count equal to the
    worker fan-out (one per node) makes it O(nodes^2), which collapses at 10k+ nodes. The
    reducer count only needs to balance keys and keep each reducer's state in memory, so it
    is capped: regular clusters (≤ the cap) are unchanged, huge clusters stay bounded.

    When prior runs have measured the shuffle families' real input volume, a learned reducer
    count (`learned_shuffle_fanout`) trims the fan-out for a shuffle whose measured data needs
    fewer, fuller buckets than one-per-worker — never above `workers`, so it only ever reduces
    the stream count. A cold store (no measured history) keeps the worker fan-out unchanged. Any
    reducer count is result-correct under the mergeable algebra, so this only affects scaling.
    Always at least 1; the cap is disabled when the config value is 0.
    """
    cap = active_config().distributed.max_shuffle_partitions
    n = max(1, workers)
    n = _learned_shuffle_fanout(n)
    return n if cap <= 0 else min(n, cap)


def _learned_shuffle_fanout(workers: int) -> int:
    """The learned reducer count for a shuffle over `workers` mappers, else `workers`.

    Best-effort read of the process-wide MetadataHub's measured shuffle-family input volume; any
    failure (no hub, cold store) returns `workers` unchanged."""
    try:
        from batcher.core import default_hub
        from batcher.dist.adaptive_sizing import learned_shuffle_fanout

        learned = learned_shuffle_fanout(default_hub(), None, workers)
        return learned if learned is not None else workers
    except Exception as exc:  # pragma: no cover - learning is best-effort
        note_suppressed("dist", "read learned shuffle fan-out", exc)
        return workers
