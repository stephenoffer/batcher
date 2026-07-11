"""Config-driven fault-tolerance, recovery, and skew policies for the distributed
executor.

These are pure ``active_config()`` → policy/option builders plus the map-stage
resilience helpers (``gather_map_results``, ``map_barrier``, ``draining_workers``). They
hold no Ray lifecycle state, so they import nothing from the rest of the package.
"""

from __future__ import annotations

import contextlib
from collections import deque

from batcher.config import active_config

# Fallback in-flight cap for the map submission window when the live cluster size is
# unreadable (Ray down / test stubs) — bounds a high-fan-out job without a topology read.
_DEFAULT_PENDING_WINDOW = 1024

# `RayError` subclasses that are NOT worker loss and that a retry cannot fix. A recovery
# loop that reads these as a dead worker blames a perfectly healthy host, marks the whole
# fleet dead one retry at a time, and finally reports "no surviving worker" instead of the
# real cause. Deliberately narrow: an `OutOfMemoryError` (Ray's memory monitor killing a
# task under node pressure) stays retryable, because rescheduling it onto a less-loaded
# node genuinely can succeed. Matched by name so a Ray version lacking one — or a test's
# stub `ray.exceptions` — degrades to the old catch-all instead of failing at import.
_FATAL_RAY_ERROR_NAMES = (
    "RuntimeEnvSetupError",  # the worker environment is broken — every retry re-breaks it
    "TaskCancelledError",  # we cancelled it (e.g. a speculation loser); not a death
    "GetTimeoutError",  # a caller-imposed deadline, not a death
)


def _is_fatal_ray_error(exc: BaseException) -> bool:
    """Whether `exc` is a Ray error that a worker-loss retry must NOT absorb."""
    try:
        import ray.exceptions as ray_exc
    except Exception:  # pragma: no cover - ray optional
        return False
    fatal = tuple(
        t for t in (getattr(ray_exc, n, None) for n in _FATAL_RAY_ERROR_NAMES) if t is not None
    )
    return bool(fatal) and isinstance(exc, fatal)


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


def gather_map_results(
    submit, n: int, policy=None, *, max_pending: int | None = None, on_lost=None, on_done=None
) -> list:
    """Gather `n` partition results, resubmitting any whose task died to preemption.

    `submit(idx)` launches partition `idx` and returns a Ray ``ObjectRef``; it is
    called again to resubmit a partition whose attempt raised a worker/node-loss fault
    — Ray reschedules the resubmission onto surviving capacity. Bounded by the recovery
    policy's `max_attempts` resubmissions per partition; a deterministic application
    error (`RayTaskError`) re-raises immediately rather than wasting attempts on a
    fault a rerun cannot fix.

    `on_lost(idx)`, when given, is called with the failed partition *before* it is
    requeued, and `on_done(idx)` after one completes. Stateless tasks need neither (Ray
    reschedules them anywhere), but a barrier over pinned **actors** must record which
    worker died so `submit` can retarget the retry at a survivor, and which workers have
    proven themselves alive so it retargets at one of *those* — see `map_barrier`. When
    `on_lost` returns truthy the failure *revealed a newly-dead worker*, and the retry is
    not charged to `max_attempts` (discovering the cluster is not the partition's fault).

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
            if on_done is not None:
                on_done(idx)
        except RayTaskError:
            raise  # a deterministic UDF error — resubmitting cannot help
        except RayError as exc:
            # Worker / actor / node loss (preemption). Requeue at the front so the
            # survivor-resubmit keeps priority for the next free slot. A `RayError` that
            # is *not* a death (broken runtime_env, OOM, cancellation) is re-raised: it
            # would otherwise be retried onto healthy workers and, via `on_lost`, blame
            # each of them in turn until the fleet looked entirely dead.
            if _is_fatal_ray_error(exc):
                raise
            # A failure that taught us a worker is dead is *progress*, not a wasted try:
            # in a correlated preemption wave a retry can land on a host that is already
            # gone but not yet observed, and charging that to the partition's budget can
            # exhaust it while survivors still exist. Progress is bounded (each worker is
            # discovered dead at most once), and `submit` raises once none are left.
            progressed = bool(on_lost(idx)) if on_lost is not None else False
            if not progressed:
                attempts[idx] += 1
                if attempts[idx] > policy.max_attempts:
                    raise
            pending.appendleft(idx)
        _fill()
    return results


def map_barrier(workers: int, launch, policy=None, dead: set[int] | None = None) -> tuple:
    """Run a shuffle MAP barrier under worker-loss recovery. Returns `(results, dead)`.

    `launch(host, src)` must issue source `src`'s map-publish on actor `host` and return
    an ``ObjectRef`` resolving to whatever the barrier collects — the address of the
    Flight server the buckets landed on, or (for a sampling barrier) the sample itself.
    On a clean run every source maps to its own actor (`host == src`), so the returned
    `results[src]` is exactly the fleet's address list and behavior is unchanged.

    `dead` seeds (and is mutated with) the known-lost workers, so a stage with several
    barriers — the sort's sample, then its range-publish — shares one view of the fleet
    instead of rediscovering each loss.

    A worker preempted *during* the barrier is the common spot failure — the map phase
    reads the source from object storage and is usually the longest part of a query — and
    a bare ``ray.get`` over pinned actors would fail the whole query there. Instead the
    lost worker is recorded in `dead` and its source is republished on a survivor under
    the **same `src`**, so the reducers' `(stage, src, bucket)` tickets still resolve. The
    map partition is a deterministic function of its durable descriptor, so the
    regenerated buckets are byte-identical: recovery changes *where* a partial lives,
    never *what* it holds. Bounded by the recovery policy's `max_attempts` per source;
    with every worker gone it raises `ResourceError` rather than looping.

    The returned `dead` set must be threaded into the reduce stage so it never hosts a
    reducer on a worker known to be gone.
    """
    from batcher._internal.errors import ResourceError

    # slot -> the worker its latest attempt was launched on. Initially `src`, but a
    # relocated source diverges, and it is the *host* that died, not the source id.
    assigned: dict[int, int] = {}
    dead = set() if dead is None else dead
    # Hosts that have *completed* a source. In a correlated preemption wave several
    # workers are already gone but only the ones whose slot has failed are known; a
    # relocation onto an unobserved-dead host burns one of the source's `max_attempts`
    # and can exhaust the budget. Retargeting onto a host that just returned a result
    # proves liveness at the moment we choose it, so a wave costs one attempt per source.
    confirmed: set[int] = set()
    rotation = 0

    def _pick_live() -> int:
        nonlocal rotation
        live = [i for i in confirmed if i not in dead] or [
            i for i in range(workers) if i not in dead
        ]
        if not live:
            raise ResourceError("no surviving worker to recompute the lost map partition on")
        rotation += 1
        return sorted(live)[rotation % len(live)]  # spread relocations, don't pile on one host

    def _submit(src: int):
        host = src if src not in dead else _pick_live()
        assigned[src] = host
        return launch(host, src)

    def _on_lost(src: int) -> bool:
        host = assigned.get(src, src)  # the HOST died; `src` may be a relocated slot
        newly_dead = host not in dead
        dead.add(host)
        confirmed.discard(host)  # a host that completed earlier can still be preempted
        return newly_dead  # progress: don't charge this retry to `src`'s budget

    def _on_done(src: int) -> None:
        confirmed.add(assigned[src])

    results = gather_map_results(_submit, workers, policy, on_lost=_on_lost, on_done=_on_done)
    return results, dead
