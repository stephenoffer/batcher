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
import os

import pyarrow as pa

from batcher.dist.executors.partition_io import descriptor_rows, partition_descriptors
from batcher.dist.executors.plan_analysis import _relabel_single_source
from batcher.dist.executors.ray_runtime import _ensure_ray
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan, MapBatches

# Smallest CPU share a task may request: a tiny partition gets a fraction of a core so
# Ray packs many such tasks per core (high parallelism over many small files) instead of
# each reserving a whole core. 1/8 core by default. Env-overridable.
_MIN_TASK_CPU = max(0.01, float(os.environ.get("BATCHER_MIN_TASK_CPU", "0.125")))
# How much heavier a per-batch UDF / inference stage is per row than a plain scan/filter.
# A `map_batches` partition gets this many times the CPU a same-sized scan would — the
# plan-level compute-skew factor (data skew is handled per-partition by `descriptor_rows`).
_MAP_COMPUTE_WEIGHT = max(1.0, float(os.environ.get("BATCHER_MAP_COMPUTE_WEIGHT", "4.0")))

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

    # Data/compute-driven task count: a tiny source → a few tasks; a large one → ~one task
    # per core; a (single-threaded) UDF → more tasks (the way to parallelize it). The GPU
    # actor-pool path keeps the worker count (its pool size, not the CPU task count).
    n_parts = workers if wants_pool else _adaptive_partition_count(sources[sid], plan, workers)
    partitions = partition_descriptors(sources[sid], n_parts)

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
        # Skew-aware adaptive CPU: each stateless task requests a CPU share sized to its
        # own partition's data (x the plan's compute weight) — fractional for a tiny
        # partition (packed many-per-core), several cores for a large one. A heavier
        # (skewed) partition therefore gets proportionally more CPU than its peers.
        shares = _adaptive_task_cpus(partitions, plan)
        # SPREAD so tasks distribute one-per-node across the cluster even though each now
        # reserves only its right-sized (often < a node) CPU share — without it Ray would
        # pack several small-share tasks onto one node, idling the rest. (The old whole-
        # node reservation forced this spread implicitly by exhausting each node.)
        sched = _spread_options()

        def _launch(idx):
            return _map_udf_task.options(**{**opts, "num_cpus": shares[idx], **sched}).remote(
                plan0, partitions[idx]
            )

        results = gather_map_results(_launch, len(partitions))

    batches: list[pa.RecordBatch] = []
    for r in results:
        if r:
            batches.extend(r)
    return pa.Table.from_batches(batches) if batches else pa.table({})


def _spread_options() -> dict:
    """Ray `.options(...)` that SPREAD tasks across nodes, or `{}` if unavailable.

    With right-sized (often sub-node) per-task CPU, the default scheduler would pack
    several tasks onto one node and idle the rest; SPREAD keeps the fan-out covering the
    cluster. Best-effort: returns `{}` if the strategy import fails (older Ray)."""
    try:
        return {"scheduling_strategy": "SPREAD"}
    except Exception:
        return {}


def _placeable_node_cores() -> float:
    """Max CPUs a single task may request and still be placeable on every node — the
    smallest alive node's core count (so a multi-CPU task fits anywhere). Falls back to
    the driver's cpu count when topology is unavailable."""
    try:
        import ray

        cores = [
            float(n.get("Resources", {}).get("CPU", 0.0)) for n in ray.nodes() if n.get("Alive")
        ]
        cores = [c for c in cores if c > 0]
        if cores:
            return float(int(min(cores)))
    except Exception:
        pass
    return float(os.cpu_count() or 4)


def _cluster_cores() -> float:
    """Total schedulable CPUs across alive nodes (the cap on useful task parallelism)."""
    try:
        import ray

        return float(int(ray.cluster_resources().get("CPU", 0.0))) or float(os.cpu_count() or 4)
    except Exception:
        return float(os.cpu_count() or 4)


def _source_total_rows(source) -> int | None:
    """Total rows of a splittable source from footer-derived split counts (no data I/O),
    or `None` when it can't be known cheaply (an in-memory/iterator source)."""
    try:
        splits = source.splits()
    except Exception:
        return None
    total = 0
    for s in splits:
        rows = getattr(s, "rows", None)
        if rows is None:
            return None  # a split with no cheap count → don't guess
        total += rows
    return total if splits else None


def _adaptive_partition_count(source, plan, fallback: int) -> int:
    """How many tasks to split a map/scan source into — data- and compute-driven.

    `ceil(total_rows x compute_weight / rows_per_cpu)`, clamped to `[1, cluster_cores]`
    and to the number of splits. So a tiny source runs as a few (even one) tasks while a
    large one fans out to ~one task per core — and a per-batch UDF (weight > 1), being
    single-threaded per task, fans out to MORE tasks (the way to parallelize it) rather
    than reserving idle cores on fewer tasks. Falls back to `fallback` (the cluster-fill
    worker count) when the row total isn't cheaply known."""
    import math

    from batcher.config import active_config
    from batcher.core.udf import has_map_batches

    total = _source_total_rows(source)
    if total is None:
        return fallback
    weight = _MAP_COMPUTE_WEIGHT if has_map_batches(plan) else 1.0
    rows_per_cpu = max(1, active_config().optimizer.target_rows_per_task // 2)
    want = math.ceil((total * weight) / rows_per_cpu)
    n = max(1, min(want, int(_cluster_cores())))
    with contextlib.suppress(Exception):
        n = min(n, max(1, len(source.splits())))  # never more tasks than splits
    return n


def _adaptive_task_cpus(partitions, plan) -> list[float]:
    """A per-task CPU share for each partition, sized to its data x the plan's compute
    weight (see `_MIN_TASK_CPU` / `_MAP_COMPUTE_WEIGHT`).

    Small partition → a fraction of a core (Ray packs many such tasks per core, so many
    small files run with high parallelism instead of each grabbing a whole core); large
    partition → multiple cores (up to one node). Because the share is per-partition, a
    skewed (heavier) partition is given more CPU than its lighter peers — adaptive to
    both data skew (row count) and plan-level compute skew (a `map_batches`/UDF stage is
    weighted heavier per row than a plain scan)."""
    from batcher.config import active_config
    from batcher.core.udf import has_map_batches

    node_cores = _placeable_node_cores()
    # Rows one core processes in a reasonable slice — half the breaker target (which sizes
    # a whole multi-core task), so a full target-sized partition asks for ~2 cores.
    rows_per_cpu = max(1, active_config().optimizer.target_rows_per_task // 2)
    weight = _MAP_COMPUTE_WEIGHT if has_map_batches(plan) else 1.0
    shares = []
    for p in partitions:
        want = (descriptor_rows(p) * weight) / rows_per_cpu
        shares.append(round(max(_MIN_TASK_CPU, min(node_cores, want)), 3))
    return shares


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


def _map_agg_task(plan0, partition, group_keys_json, aggregates_json):
    """Run the map/UDF prefix on a partition, then PARTIAL-aggregate its output.

    The map (the expensive UDF) runs on the worker over its own partition, and only the
    small partial-aggregate state leaves the worker — the driver does the cross-partition
    `combine_finalize`. This distributes a `map_batches → aggregate` pipeline (Ray Data's
    bread and butter) instead of running the whole UDF single-node on the driver."""
    import batcher._native as nat
    from batcher import core
    from batcher.dist.executors.partition_io import read_partition_descriptor
    from batcher.io.source import InMemorySource

    rows = read_partition_descriptor(partition)
    if not rows:
        return None
    out = core.execute_with_udfs(plan0, [InMemorySource(rows)])
    if not out or sum(b.num_rows for b in out) == 0:
        return None
    return nat.partial_aggregate(group_keys_json, aggregates_json, out)


def _distributed_map_aggregate(above, agg, sources, workers):
    """Distribute an aggregate over a linear `map_batches`/UDF pipeline.

    Each worker maps its source partition through the UDF prefix and partial-aggregates
    the result; the driver `combine_finalize`s the partials (mergeable two-phase) and
    applies anything above the aggregate. The UDF — the costly part — runs across the
    cluster, not single-node on the driver."""
    import json

    import pyarrow as pa

    import batcher._native as nat
    from batcher.dist.executors.partition_io import _apply_above, _empty_agg_table
    from batcher.dist.executors.ray_runtime import gather_map_results

    _ensure_ray(workers)
    map_plan, sid = _relabel_single_source(agg.input)
    gk = json.dumps([{"expr": k.expr.to_ir(), "alias": k.alias} for k in agg.group_keys])
    aj = json.dumps([s.agg.to_ir(s.alias) for s in agg.aggregates])
    n_parts = _adaptive_partition_count(sources[sid], agg.input, workers)
    partitions = partition_descriptors(sources[sid], n_parts)
    # Skew-aware adaptive CPU per task (sized to the partition that runs the UDF here),
    # SPREAD across nodes so the right-sized (sub-node) tasks still cover the cluster.
    shares = _adaptive_task_cpus(partitions, agg.input)
    sched = _spread_options()

    def _launch(idx):
        return _map_agg_task.options(num_cpus=shares[idx], **sched).remote(
            map_plan, partitions[idx], gk, aj
        )

    partials = gather_map_results(_launch, len(partitions))
    flat = [p for p in partials if p is not None]
    if not flat:
        table = _empty_agg_table(agg)
    else:
        out = nat.combine_finalize(gk, aj, flat)
        table = pa.Table.from_batches([out]) if out is not None else _empty_agg_table(agg)
    return table if not above else _apply_above(above, table)
