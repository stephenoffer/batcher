"""Worker-loss recovery for the distributed top-N fold.

`execute_topn_flight` is the one mergeable path that gathered its per-worker results with a
bare `ray.get`, so a worker lost mid-fold raised `ActorDiedError` and took the query with it.
Measured on a 72M-row `ORDER BY ... LIMIT`: killing a `_FlightWorker` **mid-method** lost the
query 5 times out of 5, while the same fault against an *idle* fleet actor was survived 9 times
out of 9 -- the fleet reforms fine, it was the in-flight partition that had nowhere to go. Its
sibling `execute_sort_flight` already documents "worker loss is survived in every phase"; this
path simply never got the same treatment.

Recovering it is the cheapest case in the engine, which is why the gap is worth closing rather
than documenting. A top-N partition is computed from a **durable source split** and ships only
`k` rows; there is no shuffle state to reconstruct, so a lost partition is recomputable on any
survivor and the recomputed rows are the same rows. That is the mergeable algebra doing what it
exists to do -- `partial` on a survivor combines exactly as it would have on the dead worker.

Two properties the retry must not break, both of them load-bearing in `execute_topn_flight`:

*The fold stays in worker order.* `LIMIT k` over rows that tie at the k-th place may return any
of them, so an arrival-ordered fold would return different rows run to run on the same data. A
recomputed partition is therefore folded back at its **original index**, not appended.

*The driver's peak stays at 2k.* The caller folds one partition at a time; this helper returns
one partition's batches and never accumulates the fleet's worth.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

from ._faults import _reduce_failure_is_worker_loss, recovery_policy

__all__ = ["topn_partition"]


def topn_partition(
    refs: list[Any], index: int, actors: list, local_ir: str, parts: list, dead: set[int]
) -> list:
    """One top-N partition's non-empty batches, recomputed on a survivor if its worker died.

    Args:
        refs: The per-worker result handles; `refs[index]` is the one being collected.
        index: Which partition to collect. Also the worker that was asked for it first.
        actors: The fleet, indexed by worker id -- recompute targets are drawn from it.
        local_ir: The per-worker plan (map prefix + local top-N heap) to re-run.
        parts: Partition descriptors over the durable source, indexed like `refs`.
        dead: Workers already known lost. Mutated as further losses are found, so one fold
            does not retry a worker a previous partition already buried.

    Returns:
        The partition's batches with empty ones dropped, ready to fold.

    Raises:
        BaseException: The original failure, when it is a deterministic bug rather than a lost
            worker -- retrying that only re-runs it on another host and buries the traceback.
        ResourceError: When every surviving worker has been tried and the partition is still
            unrecoverable.
    """
    import ray

    try:
        return _nonempty(ray.get(refs[index]))
    except Exception as exc:
        # Broad on purpose, then classified on the next line: worker loss arrives as several
        # unrelated Ray types and a deterministic bug must still be re-raised untouched.
        if not _reduce_failure_is_worker_loss(exc):
            raise
        dead.add(index)
        return _recompute_elsewhere(index, exc, actors, local_ir, parts, dead)


def _nonempty(batches: Iterable) -> list:
    return [b for b in batches if b.num_rows > 0]


def _recompute_elsewhere(
    index: int, original: BaseException, actors: list, local_ir: str, parts: list, dead: set[int]
) -> list:
    """Re-run partition `index` on surviving workers until one answers or none is left."""
    import ray

    policy = recovery_policy()
    workers = len(actors)
    last: BaseException = original
    for attempt in range(max(1, int(policy.max_attempts))):
        for target in range(workers):
            if target in dead:
                continue
            try:
                ref = actors[target].local_topn.remote(local_ir, parts[index])
                return _nonempty(ray.get(ref))
            except Exception as exc:
                # A deterministic failure fails identically on every host, so surfacing it
                # here keeps the real traceback instead of spending the budget and reporting
                # a resource error for a Python bug.
                if not _reduce_failure_is_worker_loss(exc):
                    raise
                dead.add(target)
                last = exc
        if len(dead) >= workers:
            break
        time.sleep(float(policy.backoff_base_s) * (2**attempt))

    from batcher._internal.errors import ResourceError

    raise ResourceError(
        f"top-N partition {index} lost its worker and could not be recomputed: "
        f"{len(dead)} of {workers} workers are gone. The partition reads from a durable "
        "source, so this is a fleet-wide loss rather than an unrecoverable partition."
    ) from last
