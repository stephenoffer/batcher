"""Config-driven fault-tolerance, recovery, and skew policies for the distributed
executor.

These are pure ``active_config()`` → policy/option builders plus the two map-stage
resilience helpers (``gather_map_results``, ``draining_workers``). They hold no Ray
lifecycle state, so they import nothing from the rest of the package.
"""

from __future__ import annotations

import contextlib
from collections import deque

from batcher.config import active_config

# Fallback in-flight cap for the map submission window when the live cluster size is
# unreadable (Ray down / test stubs) — bounds a high-fan-out job without a topology read.
_DEFAULT_PENDING_WINDOW = 1024


def speculation_policy():
    """Build the straggler-speculation policy from the active config.

    Default `max_backups=0` (speculation off → the barrier is a plain `ray.get`),
    so distributed results are unchanged unless a config explicitly enables it.

    The straggler *factor* (how many multiples of the median task time before a backup fires) is
    tuned from the measured task-time spread of prior shuffle stages (`_learned_straggler_factor`):
    a stage whose tasks finish uniformly gets a higher factor (rarely back up), a heavy-tailed one
    a lower factor (back up sooner). A cold store keeps the config default. Speculation only
    duplicates a slow task and keeps whichever copy finishes first, so the result is unchanged
    regardless of the factor.
    """
    from batcher.carbonite.resilience import SpeculationPolicy

    d = active_config().distributed
    return SpeculationPolicy(
        straggler_factor=_learned_straggler_factor(d.speculation_straggler_factor),
        min_finished_frac=d.speculation_min_finished_frac,
        max_backups=d.speculation_max_backups,
    )


def _learned_straggler_factor(default: float) -> float:
    """The learned straggler factor from measured shuffle-family task-time variance, else
    `default`. Best-effort read of the process-wide MetadataHub; any failure returns `default`."""
    try:
        from batcher.core import default_hub
        from batcher.dist.adaptive_sizing import learned_straggler_factor

        for family in ("aggregate", "hash_join", "sort", "window"):
            learned = learned_straggler_factor(default_hub(), family)
            if learned is not None:
                return learned
    except Exception:  # pragma: no cover - learning is best-effort
        pass
    return default


def skew_join_salt() -> tuple[int, float]:
    """Return ``(salt_buckets, hot_fraction)`` for skew-aware join salting.

    ``salt_buckets == 0`` (default) means salting is off → the shuffle join is the
    plain co-partition, so single-node==distributed is bit-identical.
    """
    d = active_config().distributed
    return int(d.skew_join_salt), float(d.skew_join_fraction)


def runtime_bloom_join() -> bool | str:
    """The runtime bloom-filter join policy: ``True``/``False``/``"auto"``.

    When engaged, a shuffle join prunes the probe side by a bloom built over the build
    side's keys before shuffling. Always correct; a network-volume optimization for
    selective joins. ``"auto"`` (the default) defers the per-join decision to a
    cardinality estimate (see `dist.executors.join._bloom_beneficial`); ``True``/
    ``False`` force it on/off (see `DistributedConfig.runtime_bloom_join`)."""
    return active_config().distributed.runtime_bloom_join


def recovery_policy():
    """Build the shuffle recompute-on-worker-loss recovery policy from the config.

    Bounds the recompute→retry rounds and the exponential backoff between them, so a
    larger cluster's failure rate is tunable rather than a hardcoded 3 attempts.
    """
    from batcher.carbonite.resilience import RecoveryPolicy

    d = active_config().distributed
    return RecoveryPolicy(
        max_attempts=d.recovery_max_attempts,
        backoff_base_s=d.recovery_backoff_base_s,
    )


def draining_workers(actors, workers: int) -> set[int]:
    """Worker ids under a spot-preemption drain notice, for proactive migration.

    Queries each worker's `is_draining()` once, in parallel, at a stage boundary. A
    draining worker will be reclaimed shortly, so the caller migrates its shuffle
    output to a survivor *before* it dies (a zero-loss proactive recompute) instead of
    paying a reactive recompute after a failed fetch. A worker that errors on the ping
    is already gone, so it is reported as draining (it needs migrating regardless).

    Active under the spot profile (which a spot deployment gets automatically — see
    `config.profiles.detect_spot_environment`). Empty otherwise — the monitors are not
    started off the spot profile, so a stable cluster pays nothing and skips the query.
    """
    if active_config().distributed.resilience != "spot":
        return set()
    import ray

    refs = [actors[i].is_draining.remote() for i in range(workers)]
    out: set[int] = set()
    for i, ref in enumerate(refs):
        try:
            if ray.get(ref):
                out.add(i)
        except Exception:
            out.add(i)  # unreachable already ⇒ migrate it proactively
    return out


def fault_options() -> dict:
    """Ray task fault-tolerance kwargs from config — the first line of defense.

    `max_retries` reruns a failed shuffle task (deterministic, recomputed from a
    durable source, so a rerun is safe) so a transient node/connection failure
    self-heals before the heavier app-level recompute loop engages. With
    `retry_on_transient`, retries also cover application exceptions, not just worker
    death; a deterministic failure still re-fails and surfaces once retries exhaust.
    """
    d = active_config().distributed
    opts: dict = {"max_retries": int(d.task_max_retries)}
    if d.retry_on_transient:
        opts["retry_exceptions"] = True
    return opts


def actor_fault_options() -> dict:
    """Ray actor fault-tolerance kwargs from config for compute actors (the map /
    inference pool): respawn a crashed actor (`max_restarts`) and rerun its in-flight
    call on the respawned actor (`max_task_retries`).

    Not applied to the Flight shuffle-server actors — their loss is recovered by the
    lineage recompute loop, and letting Ray restart them out from under it would race
    that recovery.
    """
    d = active_config().distributed
    return {
        "max_restarts": int(d.actor_max_restarts),
        "max_task_retries": int(d.actor_max_task_retries),
    }


def _pending_window() -> int:
    """The max map tasks kept in flight at once — a submit-ahead cap.

    Submitting every partition task up front floods Ray's scheduler / object store at high
    fan-out (the "too many pending tasks" anti-pattern). The window is `max_pending_tasks`
    when the user pinned one, else `pending_window_factor x` the cluster's schedulable
    cores — generous enough that ordinary fan-outs still submit everything at once (the
    caller clamps to `min(window, n)`, so `n <= window` is the unchanged fast path) while a
    100k-partition job stays bounded. Never below 1; falls back to `_DEFAULT_PENDING_WINDOW`
    when the cluster size is unreadable (Ray down / test stubs), so the cap still engages.
    """
    d = active_config().distributed
    if d.max_pending_tasks > 0:
        return max(1, d.max_pending_tasks)
    cores = 0.0
    with contextlib.suppress(Exception):
        import ray

        cores = float(ray.cluster_resources().get("CPU", 0.0))
    if cores <= 0:
        return max(1, _DEFAULT_PENDING_WINDOW)
    return max(1, int(cores) * max(1, d.pending_window_factor))


def gather_map_results(submit, n: int, policy=None, *, max_pending: int | None = None) -> list:
    """Gather `n` partition results, resubmitting any whose task died to preemption.

    `submit(idx)` launches partition `idx` and returns a Ray ``ObjectRef``; it is
    called again to resubmit a partition whose attempt raised a worker/node-loss fault
    — Ray reschedules the resubmission onto surviving capacity. Bounded by the recovery
    policy's `max_attempts` resubmissions per partition; a deterministic application
    error (`RayTaskError`) re-raises immediately rather than wasting attempts on a
    fault a rerun cannot fix.

    Submissions are bounded to an in-flight **window** (`max_pending` when given, else
    `_pending_window()`) so a high-fan-out stage does not flood Ray's scheduler /
    object store with pending tasks: at most `window` tasks are outstanding, and a slot
    is refilled from the queue each time one completes. When `n <= window` every task is
    submitted before the first wait — byte-identical to the old submit-all behavior, so
    ordinary queries are unchanged. A preempted partition is requeued at the front so it
    keeps priority for a slot and cannot be starved past `max_attempts`.

    This is the map/inference analogue of the shuffle recompute loop
    (`ShuffleRecovery`): a stateless map partition has no published lineage, but it
    *is* its own lineage — a map/inference UDF recomputes idempotently from its durable
    partition descriptor, so a resubmit neither loses nor duplicates output. Without
    this loop a single preemption fails the whole stage (a plain ``ray.get`` raises).
    Returns results in partition order (assembly is index-addressed, so the submit
    order never affects the output).
    """
    import ray
    from ray.exceptions import RayError, RayTaskError

    policy = policy or recovery_policy()
    if n <= 0:
        return []
    window = max_pending if (max_pending and max_pending > 0) else _pending_window()
    window = max(1, min(window, n))
    results: list = [None] * n
    attempts = [0] * n
    pending: deque[int] = deque(range(n))  # indices awaiting (re)submission, in order
    inflight: dict = {}  # ref -> idx

    def _fill() -> None:
        # Top the window back up. When window >= n this submits every partition before
        # the first wait (the unchanged fast path); otherwise it keeps <= window in flight.
        while pending and len(inflight) < window:
            idx = pending.popleft()
            inflight[submit(idx)] = idx

    _fill()
    while inflight:
        done, _ = ray.wait(list(inflight), num_returns=1)
        ref = done[0]
        idx = inflight.pop(ref)
        try:
            results[idx] = ray.get(ref)
        except RayTaskError:
            raise  # a deterministic UDF error — resubmitting cannot help
        except RayError:
            # Worker / actor / node loss (preemption). Requeue at the front so the
            # survivor-resubmit keeps priority for the next free slot.
            attempts[idx] += 1
            if attempts[idx] > policy.max_attempts:
                raise
            pending.appendleft(idx)
        _fill()
    return results
