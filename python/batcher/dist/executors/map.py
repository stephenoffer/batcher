"""Distributed `map_batches` (batch inference) — the Ray Data competitor path.

A linear scan / filter / project / map_batches chain is embarrassingly parallel:
each worker reads its partition (a split-manifest it reads directly from storage, or
a shipped batch list) and runs the full chain (preprocessing + model UDF) locally —
no shared filesystem required, so it runs unchanged on a real multi-node cluster.

Two scheduling shapes:

* **Stateless tasks** (default): one Ray task per partition, optionally reserving
  GPUs (`num_gpus`). Best when the UDF holds no expensive state.
* **Stateful actor pool**: when the pipeline asks for `concurrency` actors or uses
  a class (factory) UDF, a fixed pool of long-lived actors each build the model
  *once* and stream partitions through it — the GPU-inference pattern (load the
  model once, reuse across many batches), with `num_gpus` reserved per actor. This
  is the heterogeneous CPU+GPU pipeline Ray Data specializes in.
"""

from __future__ import annotations

import contextlib
import contextvars

import pyarrow as pa

from batcher.dist.executors.partition_io import partition_descriptors
from batcher.dist.executors.plan_analysis import _relabel_single_source
from batcher.dist.executors.ray_runtime import _ensure_ray
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan, MapBatches

__all__ = ["resident_inference_pools", "stream_distributed_map"]

# A query-lifetime registry of inference actor pools, keyed by pipeline signature, so a
# `map_batches`/inference pipeline's model loads ONCE per query and is reused across
# stages (and repeated terminal calls) instead of rebuilt every distributed-map call —
# the GPU-saturation win (a cold model reload between stages starves the GPU). `None`
# (the default, outside a `resident_inference_pools()` scope) keeps the per-call pool
# with its autoscaling + preemption recovery, so the default path is unchanged.
_INFERENCE_POOLS: contextvars.ContextVar[dict[tuple, list] | None] = contextvars.ContextVar(
    "batcher_inference_pools", default=None
)


@contextlib.contextmanager
def resident_inference_pools():
    """Keep inference actor pools resident for the duration of the block.

    A `map_batches` pipeline run more than once inside the block (across adaptive stages
    or repeated terminals) reuses the same model-loaded actors instead of rebuilding the
    model each time — so the GPU stays fed. All pools are torn down on exit.
    """
    token = _INFERENCE_POOLS.set({})
    try:
        yield
    finally:
        registry = _INFERENCE_POOLS.get() or {}
        _shutdown_pools(registry)
        _INFERENCE_POOLS.reset(token)


def _shutdown_pools(registry: dict[tuple, list]) -> None:
    import ray

    for actors in registry.values():
        for actor in actors:
            with contextlib.suppress(Exception):
                ray.kill(actor)
    registry.clear()


def _pipeline_signature(plan0: LogicalPlan) -> tuple:
    """A reuse key for `plan0`'s inference pool: the identities of its `map_batches`
    functions (a class factory is one stable object), so the same model maps to the same
    resident pool and a different model gets its own."""
    fns: list[int] = []
    node: LogicalPlan | None = plan0
    while node is not None:
        if isinstance(node, MapBatches):
            fns.append(id(node.fn))
        node = getattr(node, "input", None)
    return tuple(fns)


def _new_map_actor(plan0: LogicalPlan, opts: dict):
    """Spawn one model-loaded `_MapActor` (the single actor-creation point, so residency
    reuse and tests can account for every build)."""
    cls = _MapActor.options(**opts) if opts else _MapActor
    return cls.remote(plan0)


def _resident_pool_for(plan0: LogicalPlan, opts: dict, size: int) -> list:
    """The resident actor pool for `plan0` (built once on first use, reused after).

    Caller guarantees a `resident_inference_pools()` scope is active. The model is built
    in each actor's `__init__`, so reuse means it loads once per query, not per stage."""
    registry = _INFERENCE_POOLS.get()
    sig = _pipeline_signature(plan0)
    pool = registry.get(sig)
    if pool is None:
        pool = [_new_map_actor(plan0, opts) for _ in range(max(1, size))]
        registry[sig] = pool
    return pool


def _run_resident_pool(plan0, partitions, opts, size):
    """Map `partitions` through the query-resident pool for `plan0` (model loaded once),
    preserving submission order. Returns ``(ordered_results, peak_gpu_util)``."""
    import ray
    from ray.util.actor_pool import ActorPool

    actors = _resident_pool_for(plan0, opts, size)
    pool = ActorPool(actors)
    results = list(pool.map(lambda actor, part: actor.run.remote(part), list(partitions)))
    samples = [s for s in ray.get([a.gpu_stats.remote() for a in actors]) if s is not None]
    return results, (max(samples) if samples else None)


def _map_resources(plan: LogicalPlan) -> tuple[float, bool, object, str | None]:
    """GPU reservation, whether an actor pool is needed, its size spec, and the
    accelerator type to pin GPU actors/tasks to.

    The size spec is an `int` (fixed pool), a ``(min, max)`` tuple (autoscale to the
    workload), or `None` (default to the worker count)."""
    num_gpus = 0.0
    wants_pool = False
    concurrency: object = None
    accelerator_type: str | None = None
    node: LogicalPlan | None = plan
    while node is not None:
        if isinstance(node, MapBatches):
            num_gpus = max(num_gpus, node.num_gpus)
            if node.accelerator_type is not None:
                accelerator_type = node.accelerator_type
            # An explicit concurrency, or a class (factory) UDF that must build the
            # model once per worker, both require long-lived actors.
            if node.concurrency is not None or isinstance(node.fn, type):
                wants_pool = True
                if node.concurrency is not None:
                    concurrency = _merge_concurrency(concurrency, node.concurrency)
        node = getattr(node, "input", None)
    return num_gpus, wants_pool, concurrency, accelerator_type


def _merge_concurrency(a: object, b: object) -> object:
    """Combine two actor-pool size specs from stacked map stages (take the larger)."""
    if a is None:
        return b
    if b is None:
        return a
    ta = a if isinstance(a, tuple) else (a, a)
    tb = b if isinstance(b, tuple) else (b, b)
    return (max(ta[0], tb[0]), max(ta[1], tb[1]))


def _resolve_pool_size(spec: object, num_partitions: int, default: int) -> int:
    """Resolve a size spec to a concrete actor count.

    `None` → `default`; an `int` → itself; a ``(min, max)`` tuple → the workload size
    (partition count) clamped into ``[min, max]`` (the autoscaling-to-demand rule)."""
    if spec is None:
        return default
    if isinstance(spec, tuple):
        lo, hi = spec
        return max(lo, min(hi, num_partitions))
    return int(spec)


def _distributed_map(
    plan: LogicalPlan,
    sources: list[Source],
    workers: int,
    hub=None,
) -> pa.Table:
    """Run a linear map/inference pipeline across Ray workers, one partition each.

    When a `hub` is supplied and the pipeline used a GPU actor pool, the measured
    GPU utilization is recorded so the next run's `num_gpus` request can adapt."""
    _ensure_ray(workers)
    plan0, sid = _relabel_single_source(plan)
    num_gpus, wants_pool, concurrency, accelerator_type = _map_resources(plan)
    # Carbonite's scheduling envelope carries the *adapted* GPU request (the raw
    # `map_batches(num_gpus=...)` tag tuned by measured utilization). When present it
    # is authoritative, so the per-task `.options(num_gpus=...)` uses the adapted value.
    from batcher.dist.executors.ray_runtime import (
        current_envelope,
        gather_map_results,
        recovery_policy,
    )

    env = current_envelope()
    if env is not None and num_gpus > 0:
        num_gpus = env.num_gpus
    if env is not None and env.accelerator_type is not None:
        accelerator_type = env.accelerator_type

    partitions = partition_descriptors(sources[sid], workers)

    opts = _gpu_options(num_gpus, accelerator_type)
    if wants_pool:
        if isinstance(concurrency, tuple):
            lo, hi = concurrency
        else:
            from batcher.ml.gpu import gpu_aware_pool_default

            default_pool = gpu_aware_pool_default(
                num_gpus, workers, len(partitions), accelerator_type
            )
            lo = hi = _resolve_pool_size(concurrency, len(partitions), default_pool)
        # A query-resident pool (model loaded once, reused across stages) when a
        # `resident_inference_pools()` scope is active; otherwise the per-call pool with
        # autoscaling + preemption recovery.
        if _INFERENCE_POOLS.get() is not None:
            results, gpu_util = _run_resident_pool(plan0, partitions, opts, hi)
        else:
            results, gpu_util = _drive_actor_pool(
                plan0, partitions, opts, lo, hi, recovery_policy()
            )
        _record_gpu_feedback(hub, plan, gpu_util)
    else:
        task = _map_udf_task.options(**opts) if opts else _map_udf_task
        results = gather_map_results(
            lambda idx: task.remote(plan0, partitions[idx]), len(partitions)
        )

    batches: list[pa.RecordBatch] = []
    for r in results:
        if r:
            batches.extend(r)
    return pa.Table.from_batches(batches) if batches else pa.table({})


def stream_distributed_map(plan: LogicalPlan, sources: list[Source], workers: int):
    """Yield a breaker-free scan/filter/project pipeline's output one partition at a time.

    Like `_distributed_map`'s stateless-task path, but each worker's output is yielded
    *as it completes* (`ray.wait`) instead of all being collected into one driver table —
    so a huge distributed scan's *result* streams back with the driver holding only one
    partition's output at a time. For pure breaker-free relational pipelines (no
    `map_batches` ⇒ no actor pool / GPU state); the caller guarantees that shape.
    """
    import ray

    _ensure_ray(workers)
    plan0, sid = _relabel_single_source(plan)
    num_gpus, _wants_pool, _concurrency, accelerator_type = _map_resources(plan)
    from batcher.dist.executors.ray_runtime import current_envelope

    env = current_envelope()
    if env is not None and env.accelerator_type is not None:
        accelerator_type = env.accelerator_type

    partitions = partition_descriptors(sources[sid], workers)
    opts = _gpu_options(num_gpus, accelerator_type)
    task = _map_udf_task.options(**opts) if opts else _map_udf_task
    pending = [task.remote(plan0, p) for p in partitions]
    # Collect one finished partition at a time so the driver holds a single partition's
    # output, not the whole result — the bounded-memory way to pull a large scan.
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        out = ray.get(done[0])
        if out:
            yield from out


def _record_gpu_feedback(hub, plan: LogicalPlan, gpu_util: float | None) -> None:
    """Persist the pipeline's observed GPU utilization for next-run adaptation."""
    if hub is None or gpu_util is None:
        return
    from batcher.ml.gpu import gpu_feedback_key, record_gpu_utilization

    record_gpu_utilization(hub, gpu_feedback_key(plan), gpu_util)


def _gpu_options(num_gpus: float, accelerator_type: str | None) -> dict:
    """Ray `.options(...)` GPU kwargs: reserve GPUs only when positive, pin the model
    only when both a GPU and an `accelerator_type` are requested."""
    opts: dict = {}
    if num_gpus:
        opts["num_gpus"] = num_gpus
        if accelerator_type:
            opts["accelerator_type"] = accelerator_type
    return opts


def _autoscale_action(
    pending: int, n_actors: int, n_idle: int, min_size: int, max_size: int
) -> str:
    """Decide whether to grow/shrink the actor pool — ``"up"`` / ``"down"`` / ``"hold"``.

    Grow when there is queued work and headroom below `max_size`; shrink an idle actor
    when the backlog has drained and the pool is above `min_size`; otherwise hold. Pure
    so the policy is unit-testable without Ray."""
    if pending > 0 and n_actors < max_size:
        return "up"
    if pending == 0 and n_idle > 0 and n_actors > min_size:
        return "down"
    return "hold"


def _drive_actor_pool(plan0, partitions, opts, min_size, max_size, policy):
    """Stream partitions through an actor pool that scales in ``[min_size, max_size]``
    and **replaces an actor lost to preemption**, reassigning its partition.

    Each actor builds the model once (`_MapActor`) and reserves the GPU `opts`. The
    pool starts at `min_size` (so a slow model load doesn't block every replica at
    once), grows toward `max_size` while partitions queue, and reaps idle actors once
    the backlog drains — demand-driven autoscaling (the `concurrency=(min, max)`
    contract); a fixed pool is ``min_size == max_size``.

    The fault-tolerance part: on a `RayActorError` (a preempted GPU node) the dead
    actor is dropped, its in-flight partition requeued, and the pool respawned toward
    the floor — so the stage heals instead of crashing on the first loss (the old
    `ActorPool.map` / unguarded `ray.get` raised). Bounded by `policy.max_attempts`
    per partition; a deterministic UDF error (`RayTaskError`) surfaces immediately.
    A map/inference UDF recomputes idempotently from its durable partition descriptor,
    so a reassigned partition is neither lost nor duplicated. Returns
    ``(ordered_results, peak_gpu_util)``.
    """
    from collections import deque

    import ray
    from ray.exceptions import RayError, RayTaskError

    def _spawn():
        cls = _MapActor.options(**opts) if opts else _MapActor
        return cls.remote(plan0)

    parts = list(partitions)
    hi = max(1, min(max_size, len(parts)))
    lo = max(1, min(min_size, hi))
    actors = [_spawn() for _ in range(lo)]
    idle = deque(actors)
    pending = deque(range(len(parts)))  # partition indices awaiting assignment
    inflight: dict = {}  # ref -> (actor, idx)
    results: list = [None] * len(parts)
    attempts = [0] * len(parts)
    peak_util: float | None = None
    try:
        while pending or inflight:
            while pending and idle:
                idx = pending.popleft()
                actor = idle.popleft()
                inflight[actor.run.remote(parts[idx])] = (actor, idx)
            action = _autoscale_action(len(pending), len(actors), len(idle), lo, hi)
            if action == "up":
                new = _spawn()
                actors.append(new)
                idle.append(new)
                continue  # assign the new actor on the next loop
            if action == "down":
                victim = idle.pop()
                peak_util = _max_opt(peak_util, _drain_gpu_stat(victim))
                actors.remove(victim)
                ray.kill(victim)
            if not inflight:
                continue
            ready, _ = ray.wait(list(inflight), num_returns=1)
            ref = ready[0]
            actor, idx = inflight.pop(ref)
            try:
                results[idx] = ray.get(ref)
                idle.append(actor)  # the producing actor is free again
            except RayTaskError:
                raise  # a deterministic UDF error — resubmitting cannot help
            except RayError:
                # The actor was lost (preemption). Drop it, requeue its partition, and
                # respawn toward the floor so the pool heals instead of only shrinking.
                if actor in actors:
                    actors.remove(actor)
                attempts[idx] += 1
                if attempts[idx] > policy.max_attempts:
                    raise
                pending.appendleft(idx)
                if len(actors) < lo:
                    new = _spawn()
                    actors.append(new)
                    idle.append(new)
        for a in actors:
            peak_util = _max_opt(peak_util, _drain_gpu_stat(a))
        return results, peak_util
    finally:
        for a in actors:
            ray.kill(a)


def _drain_gpu_stat(actor) -> float | None:
    """The actor's peak GPU utilization (best-effort; `None` if unavailable)."""
    import ray

    try:
        return ray.get(actor.gpu_stats.remote())
    except Exception:  # pragma: no cover - feedback must never break execution
        return None


def _max_opt(a: float | None, b: float | None) -> float | None:
    """`max` over two optional floats, ignoring `None`."""
    vals = [v for v in (a, b) if v is not None]
    return max(vals) if vals else None


def _prebuild_factories(node: LogicalPlan) -> LogicalPlan:
    """Instantiate every class (factory) UDF in a linear plan once, so the model loads a
    single time per actor and is reused across batches. Thin alias for the shared
    `core.udf.prebuild_factories` (the streaming micro-batch loop reuses the same)."""
    from batcher.core.udf import prebuild_factories

    return prebuild_factories(node)


class _MapActor:
    """A long-lived worker that builds its model once and maps many partitions.

    It also samples GPU utilization while running so the scheduler can adapt the
    `num_gpus` request on the next run (the feedback half of GPU scheduling)."""

    def __init__(self, plan0: LogicalPlan) -> None:
        # Build the (class) UDFs locally, once — the model load happens here.
        self._plan = _prebuild_factories(plan0)
        self._gpu_util_max: float | None = None

    def run(self, partition: dict):
        from batcher import core
        from batcher.dist.executors.partition_io import read_partition_descriptor
        from batcher.io.source import InMemorySource
        from batcher.ml.gpu import sample_gpu_utilization

        rows = read_partition_descriptor(partition)
        if not rows:
            return None
        out = core.execute_with_udfs(self._plan, [InMemorySource(rows)])
        # Sample GPU load right after the forward pass (None on a GPU-less host).
        util = sample_gpu_utilization()
        if util is not None:
            prev = self._gpu_util_max
            self._gpu_util_max = util if prev is None else max(prev, util)
        if not out or sum(b.num_rows for b in out) == 0:
            return None
        return out

    def run_split(self, addr: str, ticket):
        """Map one prior-stage bucket fetched in place from `(addr, ticket)`, so a
        resident inference pool is fed directly from upstream output (a co-located
        bucket reads via shared memory / direct memory — no driver round-trip) instead
        of waiting for the driver to hand it a materialized partition."""
        from batcher import core
        from batcher.carbonite.transfer.server import fetch
        from batcher.io.source import InMemorySource
        from batcher.ml.gpu import sample_gpu_utilization

        rows = fetch(addr, ticket)
        if not rows:
            return None
        out = core.execute_with_udfs(self._plan, [InMemorySource(rows)])
        util = sample_gpu_utilization()
        if util is not None:
            prev = self._gpu_util_max
            self._gpu_util_max = util if prev is None else max(prev, util)
        if not out or sum(b.num_rows for b in out) == 0:
            return None
        return out

    def gpu_stats(self) -> float | None:
        """The peak GPU utilization this actor observed, or `None` if no GPU."""
        return self._gpu_util_max


def _map_udf_task(plan0, partition):
    from batcher import core
    from batcher.dist.executors.partition_io import read_partition_descriptor
    from batcher.io.source import InMemorySource

    rows = read_partition_descriptor(partition)
    if not rows:
        return None
    out = core.execute_with_udfs(plan0, [InMemorySource(rows)])
    if not out or sum(b.num_rows for b in out) == 0:
        return None
    return out
