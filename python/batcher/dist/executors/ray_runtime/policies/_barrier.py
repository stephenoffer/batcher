"""The map-stage barrier: gather partition results under worker-loss recovery.

A stateless map partition has no published lineage, but it *is* its own lineage — it
recomputes idempotently from its durable partition descriptor — so a preempted task can
simply be resubmitted. Without that a single preemption fails the whole stage, because a
plain `ray.get` raises. Split from the policy builders because this is a control loop,
not a policy.
"""

from __future__ import annotations

import contextlib
from collections import deque

from batcher._internal import events
from batcher.config import active_config

from ._drain import draining_workers  # noqa: F401  (re-exported for the façade)
from ._faults import (
    _DEFAULT_PENDING_WINDOW,
    _is_fatal_ray_error,
    _is_transient_udf_error,
    recovery_policy,
)

__all__ = ["gather_map_results", "map_barrier"]


def _pending_window(task_cpus: float = 1.0) -> int:
    """The max map tasks kept in flight at once — a submit-ahead cap.

    Submitting every partition task up front floods Ray's scheduler / object store at high
    fan-out (the "too many pending tasks" anti-pattern). The window is `max_pending_tasks`
    when the user pinned one, else `pending_window_factor x` the tasks the cluster's
    schedulable cores can actually hold — generous enough that ordinary fan-outs still
    submit everything at once (the caller clamps to `min(window, n)`, so `n <= window` is
    the unchanged fast path) while a 100k-partition job stays bounded. Never below 1; falls
    back to `_DEFAULT_PENDING_WINDOW` when the cluster size is unreadable (Ray down / test
    stubs), so the cap still engages.

    `task_cpus` is what one of *these* tasks requests, which is the difference between
    counting cores and counting tasks. The map path sizes each task from its own partition
    (`map._adaptive_task_cpus`, floor `_MIN_TASK_CPU` = 0.125), so a many-small-files stage
    fits eight tasks per core — and a window derived from cores alone throttled it to an
    eighth of the concurrency the cluster could hold, on exactly the workload the adaptive
    share exists to speed up. Clamped to 1.0 from above: a multi-core grant means fewer
    tasks per core, but the flood this guards against is a *count* problem, so a wide stage
    keeps the same generous window it had.

    It must be passed by the caller rather than read from the ambient
    `SchedulingEnvelope`: the envelope carries the *fleet* grant (a whole node's cores after
    `_even_cpu_share`), not what an individual map task asked for, so reading it there
    yields ~1.0 and silently does nothing.
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
    share = min(1.0, max(float(task_cpus), 1e-3))
    return max(1, int(cores / share) * max(1, d.pending_window_factor))


def gather_map_results(
    submit,
    n: int,
    policy=None,
    *,
    max_pending: int | None = None,
    on_lost=None,
    on_done=None,
    task_cpus: float = 1.0,
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
    window = max_pending if (max_pending and max_pending > 0) else _pending_window(task_cpus)
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
        except RayTaskError as exc:
            # A deterministic UDF error fails the same way everywhere, so resubmitting cannot
            # help — surface it immediately. But a CUDA OOM, a throttled model endpoint, or a
            # network timeout also arrives as a `RayTaskError`, and those DO clear on a retry.
            # Failing the whole job on one used to discard hours of completed inference.
            if not _is_transient_udf_error(exc):
                raise
            attempts[idx] += 1
            if attempts[idx] > policy.max_attempts:
                raise
            pending.appendleft(idx)
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
        if newly_dead:
            # The first moment anything in the engine knows this worker is gone. Published
            # only on the transition, so a host that fails ten sources is one event.
            events.publish(
                events.RECOVERY,
                event="worker_lost",
                worker=host,
                src=src,
                dead_total=len(dead),
                workers=workers,
            )
        return newly_dead  # progress: don't charge this retry to `src`'s budget

    def _on_done(src: int) -> None:
        confirmed.add(assigned[src])

    results = gather_map_results(_submit, workers, policy, on_lost=_on_lost, on_done=_on_done)
    return results, dead
