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

from ..capacity import fleet_worker_cpus
from ..scheduling import map_slots_per_worker
from ._drain import draining_workers  # noqa: F401  (re-exported for the façade)
from ._faults import (
    _DEFAULT_PENDING_WINDOW,
    _is_fatal_ray_error,
    _is_transient_udf_error,
    check_results_trusted,
    is_recoverable_task_failure,
    node_ledger,
    recovery_policy,
    retry_budget,
)

__all__ = ["gather_map_results", "map_barrier"]

#: How often the map barrier wakes while nothing has completed, so a stage that cannot be
#: scheduled can be *reported* rather than waited on in silence. Only the reporting cadence:
#: a wake with no result costs one `ray.wait` round-trip and changes no scheduling decision,
#: so it is well below the two-minute warning threshold and well above a spin.
_STALL_POLL_S = 5.0


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


def _stall_diagnosis(task_cpus: float, outstanding: int) -> str | None:
    """Why this barrier's outstanding tasks have not been placed, or `None`.

    The map path sizes each task's CPU share from its own partition, so the ask that is
    actually pending is `task_cpus` — not the ambient fleet envelope, which describes a
    whole worker. The rest of the envelope (GPUs, the memory grant, a custom accelerator)
    is still the fleet's, so it is folded in around that share.
    """
    try:
        from batcher.dist.executors.ray_runtime.capacity import Demand, describe_pending_demand
        from batcher.dist.executors.ray_runtime.scheduling import current_envelope

        demand = Demand.from_envelope(current_envelope(), count=outstanding)
        return describe_pending_demand(
            Demand(
                num_cpus=max(float(task_cpus), 1e-3),
                num_gpus=demand.num_gpus,
                memory_bytes=demand.memory_bytes,
                resources=demand.resources,
                count=outstanding,
            )
        )
    except Exception as exc:  # a diagnostic must never be the thing that fails the query
        from batcher._internal.logging import note_suppressed

        note_suppressed("dist", "diagnose the stalled barrier", exc)
        return None


def _relieve_stall(task_cpus: float, *, pinned: bool) -> bool:
    """Hand the idle session fleet's cores back, so this stage's pending tasks can place.

    A warm fleet is a placement-group reservation of nearly the whole cluster, held for
    `distributed.session_fleet_idle_s` after the query that used it finished. That is what
    makes a *second* Batcher query cheap, and it is also what a stage of plain Ray tasks
    submitted outside the reservation waits on. Measured on a 27-node fleet: the uniform
    fan-out reserved 384 of 384 cores, so such a stage had no core to run on at all and
    waited out the idle timer.

    `yield_session_fleet` already knew how to resolve that and had one caller
    (`map._placeable_scheduling`), which covers the map and write stages and nothing else.
    Calling it here covers every stage that gathers through this barrier, and does so
    *reactively* — the fleet is kept warm right up to the point something else needs its
    cores, which is the whole value of keeping it.

    Args:
        task_cpus: What one pending task requests, the figure that has to become free.
        pinned: Whether this barrier's work runs on the fleet's own actors. Then the fleet
            must NOT be released: tearing it down would kill the actors mid-stage, turning a
            slow query into a failed one. `on_lost` is the caller's own marker for that —
            a stateless-task barrier passes none, an actor-pinned one must.

    Returns:
        Whether the fleet was released. `False` leaves the caller exactly where it was.
    """
    import logging

    from batcher._internal.logging import get_logger, log_kv, note_suppressed

    if pinned:
        return False
    try:
        from batcher.dist.fleet import yield_session_fleet

        if not yield_session_fleet(task_cpus):
            return False
    except Exception as exc:  # pragma: no cover - relief must never fail the stage
        note_suppressed("dist", "yield the idle fleet for a stalled stage", exc)
        return False
    log_kv(
        get_logger("dist"),
        logging.INFO,
        "released the idle shuffle fleet so this stage could be scheduled",
        task_cpus=task_cpus,
    )
    return True


def gather_map_results(
    submit,
    n: int,
    policy=None,
    *,
    max_pending: int | None = None,
    on_lost=None,
    on_done=None,
    sink=None,
    task_cpus: float = 1.0,
    budget=None,
    stage: str = "",
) -> list:
    """Gather `n` partition results, resubmitting any whose task died to preemption.

    `submit(idx)` launches partition `idx` and returns a Ray ``ObjectRef``; it is
    called again to resubmit a partition whose attempt raised a worker/node-loss fault
    — Ray reschedules the resubmission onto surviving capacity. Bounded by the recovery
    policy's `max_attempts` resubmissions per partition; a deterministic application
    error (`RayTaskError`) re-raises immediately rather than wasting attempts on a
    fault a rerun cannot fix.

    `on_lost(idx, exc)`, when given, is called with the failed partition and the failure
    that lost it, *before* it is
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

    Retries are additionally drawn from a **job-wide budget** (`budget`, or the configured
    one). The per-partition `max_attempts` above bounds each partition and bounds nothing
    about the stage: at a hundred thousand partitions it authorizes hundreds of thousands of
    retries, and a fleet that is broken in some way no probe catches will spend the whole run
    using them — finishing hours later with whatever error happened to be last rather than
    with the first one, which said exactly what was wrong. When the budget is spent the next
    failure is raised with its own traceback instead of being retried.

    `sink(idx, value)`, when given, **consumes each result as it lands** instead of the
    barrier retaining it, and the returned list is all-`None`. That is what a caller whose
    next step is a mergeable fold wants, and the difference is asymptotic rather than
    tidiness: retaining every result makes the driver hold `W` partial states at once and
    then fold them in a line *after* the barrier, so both its peak memory and a Θ(W) serial
    tail grow with the cluster. Folding on arrival makes the peak one running state, and
    hides the fold inside the map phase it overlaps — the same Θ(W) total work, off the
    critical path. A `sink` that raises fails the stage, so it must be a pure accumulate.

    This is the map/inference analogue of the shuffle recompute loop
    (`ShuffleRecovery`): a stateless map partition has no published lineage, but it
    *is* its own lineage — a map/inference UDF recomputes idempotently from its durable
    partition descriptor, so a resubmit neither loses nor duplicates output. Without
    this loop a single preemption fails the whole stage (a plain ``ray.get`` raises).
    `stage` labels the `PARTITION` event published as each partition lands. This barrier
    holds both halves of "N of M" -- the width of the stage and the slot that just finished
    -- and is therefore the only place that can answer it *while* the stage runs. Without
    it a long map or inference stage reported nothing at all until it returned: the shuffle
    barrier (`carbonite.resilience.gather_with_backups`) has published this since it
    existed, and the map path, which is where a multi-hour job actually spends its time,
    did not. The live progress line falls back to an indeterminate sweep with no ETA when
    nothing publishes it.

    Returns results in partition order (assembly is index-addressed, so the submit
    order never affects the output).
    """
    import ray
    from ray.exceptions import RayError, RayTaskError

    policy = policy or recovery_policy()
    if n <= 0:
        return []
    budget = retry_budget() if budget is None else budget
    budget.record_attempt(n)
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
    # A map/inference stage that cannot be scheduled looks exactly like one that is merely
    # slow, and an unbounded `ray.wait` reports neither: the query sits in this loop with no
    # output for as long as the cluster stays full. That is the common shape on a shared
    # cluster — a shuffle fleet's placement group holds every core, and the map tasks
    # submitted outside it wait for a core that never comes free — and it presents as a hung
    # job with idle devices.
    #
    # On the first stall the barrier now tries to *fix* that rather than only report it
    # (`_relieve_stall`): the warm fleet is handed back, and the tasks already pending place
    # themselves on the cores it was holding. Reporting remains the answer for every other
    # cause, because a legitimately slow first task is indistinguishable from a stuck one.
    import time

    from batcher.carbonite.resilience import STALL_WARN_AFTER_S, warn_barrier_stalled

    barrier_started = time.monotonic()
    stall_warnings = 0
    relieved = False
    finished = 0
    completed = 0
    while inflight:
        done, _ = ray.wait(list(inflight), num_returns=1, timeout=_STALL_POLL_S)
        if not done:
            waited = time.monotonic() - barrier_started
            if not finished and waited > STALL_WARN_AFTER_S * (stall_warnings + 1):
                stall_warnings += 1
                if not relieved:
                    # Once per barrier: a second attempt cannot help (the fleet is gone) and
                    # would only spend the stall window re-reading the cluster.
                    relieved = True
                    if _relieve_stall(task_cpus, pinned=on_lost is not None):
                        continue
                warn_barrier_stalled(waited, n, _stall_diagnosis(task_cpus, len(inflight)))
            continue
        finished += 1
        ref = done[0]
        idx = inflight.pop(ref)
        try:
            value = ray.get(ref)
            # Consumed on arrival, or retained — never both, so the driver's peak is the
            # sink's running state rather than every partition's result at once.
            if sink is not None:
                sink(idx, value)
            else:
                results[idx] = value
            del value
            if on_done is not None:
                on_done(idx)
            completed += 1
            # After the result is safely handled, so a partition is only ever counted when
            # it really landed -- `finished` above counts *wakeups*, including the ones that
            # turn out to be a transient failure and get resubmitted.
            events.publish(
                events.PARTITION, name=stage or "stage", total=n, slot=idx, done=completed
            )
        except RayTaskError as exc:
            # A deterministic UDF error fails the same way everywhere, so resubmitting cannot
            # help — surface it immediately. But a CUDA OOM, a throttled model endpoint, or a
            # network timeout also arrives as a `RayTaskError`, and those DO clear on a retry.
            # Failing the whole job on one used to discard hours of completed inference.
            #
            # `is_recoverable_task_failure` is the second of those, and it was missing. The
            # comment in `_faults` reads "a map task that fails reports worker loss as a *Ray*
            # error", which was true until a map task could **read a Flight intermediate**: a
            # stage scanning what a previous stage published fetches from a peer inside the
            # task, so a lost peer arrives here as a `RetryableShuffleError` wrapped in a
            # `RayTaskError` — the transport's own word for "retry me" — and was re-raised.
            # Observed as a windowed rank over a hot key dying with `transport error` while
            # the identical query on a uniform key passed, because only the skewed one moved
            # a bucket big enough for the fetch to break.
            if not (_is_transient_udf_error(exc) or is_recoverable_task_failure(exc)):
                raise
            # Almost every failure loses work, which is what a retry is for. A device that
            # took an uncontained ECC fault did something else: it kept running and returned
            # a wrong number, so the partitions that already *succeeded* on it are suspect
            # too. Retrying past that produces a job that completes and writes out
            # corruption, which is worse than the crash it avoided.
            check_results_trusted(exc)
            attempts[idx] += 1
            if attempts[idx] > policy.max_attempts or not budget.try_consume():
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
            progressed = bool(on_lost(idx, exc)) if on_lost is not None else False
            if not progressed:
                attempts[idx] += 1
                # Charged to the job-wide budget for the same reason it is charged to the
                # partition's: a retry that taught us nothing is a retry, and a cluster
                # losing workers faster than the stage can finish must fail on the loss
                # rather than resubmit into it indefinitely.
                if attempts[idx] > policy.max_attempts or not budget.try_consume():
                    raise
            pending.appendleft(idx)
        _fill()
    return results


def _idle_pool(workers: int, slots: int) -> deque[int]:
    """The pre-filled idle pool, each worker appearing in proportion to the cores it holds.

    The pool is what the barrier deals sources from, and it used to be filled with every worker
    exactly `slots` times. On a uniform fleet that is right. On an unequal one it is a *static*
    even deal wearing a dynamic barrier's clothes, because `map_partitions` sizes the source
    count at `workers x slots` — exactly the pool's depth — so every source is handed out from
    the initial fill and the go-idle path that would have corrected the imbalance never runs.

    Measured on the 28-node / 384-core mixed cluster, 128 sources over 32 workers:

        cores  workers  partitions  per worker  per core
           24        5          20        4.00       0.167
           15        8          32        4.00       0.267
            3       16          64        4.00       1.333

    Every worker took exactly four regardless of size, so a 3-core worker carried eight times
    the per-core load of a 24-core one and the whole stage waited on the small machines. That
    is the straggler being the *assignment* rather than the machine — the same failure
    `assignment._balance` documents for split packing, on the other side of the barrier.

    Dealing in proportion to cores fixes the initial deal without touching the dynamic one: a
    worker that finishes early still returns to the pool and still takes more. Ordering is
    round-robin rather than blocked, so the first `workers` sources still reach every worker
    and the extra slots land on the big machines afterwards — a blocked fill would hand the
    first twenty-four sources to one worker and idle the rest.

    Assignment is a scheduling concern only: a source is identified by its `src` id and its
    partition is a deterministic function of its durable descriptor, so which worker computes
    it never changes the result.

    Args:
        workers: The fleet's width.
        slots: How many sources each worker may hold in flight, on average.

    Returns:
        Worker ids to deal from, `workers * slots` deep.
    """
    total = max(1, slots) * workers
    caps = fleet_worker_cpus(workers)
    if not caps or len(caps) != workers or min(caps) <= 0 or max(caps) == min(caps):
        return deque(h for _ in range(slots) for h in range(workers))

    share = total / sum(caps)
    counts = [max(1, round(c * share)) for c in caps]
    # Rounding drifts off `total`; settle it on the largest workers, which is where a slot is
    # worth the most and where a rounding loss would otherwise land systematically.
    order = sorted(range(workers), key=lambda h: caps[h], reverse=True)
    drift = total - sum(counts)
    position = 0
    while drift != 0 and position < 4 * workers:
        host = order[position % workers]
        if drift > 0:
            counts[host] += 1
            drift -= 1
        elif counts[host] > 1:
            counts[host] -= 1
            drift += 1
        position += 1

    pool: deque[int] = deque()
    left = list(counts)
    while any(left):
        for host in range(workers):
            if left[host]:
                pool.append(host)
                left[host] -= 1
    return pool


def map_barrier(
    sources: int,
    launch,
    policy=None,
    dead: set[int] | None = None,
    workers: int | None = None,
    placement=None,
) -> tuple:
    """Run a shuffle MAP barrier under worker-loss recovery. Returns `(results, dead)`.

    `launch(host, src)` must issue source `src`'s map-publish on actor `host` and return
    an ``ObjectRef`` resolving to whatever the barrier collects — the address of the
    Flight server the buckets landed on, or (for a sampling barrier) the sample itself.
    On a clean run every source maps to its own actor (`host == src`), so the returned
    `results[src]` is exactly the fleet's address list and behavior is unchanged.

    **More sources than workers** (`workers` given and smaller than `sources`) is how a
    shuffle decouples its task granularity from its node count. The barrier then keeps
    exactly `workers` tasks in flight and hands each new source to whichever actor just
    went idle, so the assignment is *dynamic*: a slow node takes fewer partitions instead
    of holding the barrier open on the one oversized partition it was statically dealt,
    and a lost worker's outstanding partitions are re-dealt across every survivor rather
    than replayed whole onto one. `workers is None` (the default) means one source per
    worker, which pins `host == src` and is byte-identical to the pinned behavior.

    The generalization is safe because a source is identified by its `src` id, never by
    the host that happens to compute it: the buckets are published under `(stage, src,
    bucket)` tickets and the partition is a deterministic function of its durable
    descriptor. That is the same property relocation already relied on, applied from the
    first attempt rather than only after a death.

    `placement` (a `SourcePlacement`), when given, is filled in with where each source
    actually landed. The reduce stage needs it to answer "what did this worker's death
    lose", and once the assignment is dynamic the barrier is the only thing that knows —
    the answer is no longer the source id.

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

    `dead` is per-stage, and that is the gap the **fault ledger** fills. A worker that failed
    every source of the previous shuffle is not in *this* barrier's `dead` set, so relocation
    would pick it again, discover it again, and pay another attempt for the privilege — once
    per barrier, for the whole query. The ledger remembers across stages, so a host that has
    been failing is deprioritized from the start. It is only ever a *preference*: when every
    live worker is quarantined the barrier still uses them, because a stage that cannot place
    work is worse than a stage placed badly, and the ledger's own blast-radius cap means that
    state is already telling us the fault is systemic.
    """
    from batcher._internal.errors import ResourceError

    pinned = workers is None or workers >= sources
    workers = sources if workers is None else workers
    ledger = node_ledger()
    if ledger is not None:
        # The fleet size the blast-radius cap is measured against. Without it the cap sees
        # only the workers that have already failed — "one of one is blocked" — and engages
        # on the first fault instead of on a systemic one.
        ledger.observe([str(i) for i in range(workers)])

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
        if ledger is not None:
            # Prefer hosts the ledger has nothing against. `or live` is the whole safety
            # argument: when every survivor is quarantined the barrier proceeds anyway,
            # because failing to place work is worse than placing it on a suspect host — and
            # a fleet in that state is one the ledger's cap has already declared systemic.
            live = [i for i in live if not ledger.is_blocked(str(i))] or live
        rotation += 1
        return sorted(live)[rotation % len(live)]  # spread relocations, don't pile on one host

    # Actors free to take the next source, in the order they went idle. Only consulted
    # when there are more sources than workers; the pinned barrier never touches it.
    #
    # Each actor appears `slots` times, so `slots` of its sources are in flight at once.
    # A map task reads its partition from object storage and then folds it, and at one
    # slot per actor the node does neither while doing the other — the same gap
    # `FLEET_CONCURRENCY` was introduced to close on the *reduce* side, left open on the
    # map side, which is where a scan-heavy query spends nearly all of its time. Measured
    # on a 64 x 16-core fleet at TPC-H sf100, the map phase held the cluster at 7-14% of
    # its cores while the barrier waited on 64 single-threaded S3 reads.
    slots = map_slots_per_worker() if not pinned else 1
    idle: deque[int] = _idle_pool(workers, slots) if not pinned else deque(range(workers))

    def _next_idle() -> int:
        while idle:
            host = idle.popleft()
            if host not in dead:
                return host
        # The pool is empty when every actor is busy or has failed without returning its
        # slot (a transient task error frees no host). Falling back to any survivor keeps
        # the barrier making progress; the window bounds how many pile onto it.
        return _pick_live()

    def _submit(src: int):
        host = (src if src not in dead else _pick_live()) if pinned else _next_idle()
        assigned[src] = host
        if placement is not None:
            placement.relocate(src, host)
        return launch(host, src)

    def _on_lost(src: int, exc: BaseException) -> bool:
        host = assigned.get(src, src)  # the HOST died; `src` may be a relocated slot
        newly_dead = host not in dead
        dead.add(host)
        if ledger is not None:
            # The failure's own category, not a hardcoded `worker_lost`. The ledger weights
            # blame by category precisely so a machine is quarantined for being unhealthy
            # rather than for being given work, and every loss arriving as the heaviest
            # non-hardware category defeated that: a **preemption scores 0.0** — the table's
            # own words are "a planned reclamation says nothing about the node's health" —
            # and was being charged 1.0, so a spot fleet quarantined its own nodes for
            # behaving exactly as spot nodes do. An `ActorUnavailableError` is charged half,
            # because Ray defines it as *temporarily* inaccessible (restarting, a network
            # blip, or a death not yet reported) and tells callers to ping rather than to
            # declare death; a confirmed `ActorDiedError` still charges in full.
            #
            # This is only the *blame* weight. Whether to retry is decided above by
            # exception type, and is unchanged.
            from batcher.carbonite.resilience import classify_failure

            ledger.record_failure(str(host), classify_failure(exc).name)
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
        if not pinned and assigned[src] not in dead:
            idle.append(assigned[src])  # this actor is free for the next source
        if ledger is not None:
            # The only evidence that clears a quarantine. Without recording successes the
            # ledger is a one-way record of everything that ever went wrong, and the fleet it
            # describes only ever shrinks.
            ledger.record_success(str(assigned[src]))

    # An over-partitioned barrier runs exactly `workers * slots` tasks at a time: the window
    # IS the actor pool, so a completion both frees a host and releases the slot that refills
    # it. The window must stay equal to the pool — a *wider* one would queue several sources
    # behind one actor and give the assignment back to Ray's arrival order, which is the
    # static dealing this exists to avoid, and a narrower one would leave slots unusable.
    results = gather_map_results(
        _submit,
        sources,
        policy,
        max_pending=None if pinned else workers * slots,
        on_lost=_on_lost,
        on_done=_on_done,
    )
    return results, dead
