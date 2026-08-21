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

import atexit
import contextlib
import contextvars
import os
from collections import deque

import pyarrow as pa

from batcher._internal.hardware import INFERENCE_INFLIGHT_DEPTH_MAX, available_cpu_count
from batcher._internal.logging import get_logger, note_suppressed
from batcher._internal.native import engine
from batcher.dist.executors.partition_io import (
    descriptor_rows,
    partition_descriptors,
    source_pushdown,
)
from batcher.dist.executors.plan_analysis import _relabel_single_source
from batcher.dist.executors.ray_runtime import (
    _ensure_ray,
    create_worker_placement,
    current_envelope,
    engine_config_json,
    placement_actor_options,
    release_placement,
)
from batcher.io.source import Source
from batcher.plan.ir_specs import agg_spec_json
from batcher.plan.logical import LogicalPlan, MapBatches
from batcher.plan.visitor import scanned_source_ids

# Smallest CPU share a task may request: a tiny partition gets a fraction of a core so
# Ray packs many such tasks per core (high parallelism over many small files) instead of
# each reserving a whole core. 1/8 core by default. Env-overridable.
_MIN_TASK_CPU = max(0.01, float(os.environ.get("BATCHER_MIN_TASK_CPU", "0.125")))
# How much heavier a per-batch UDF / inference stage is per row than a plain scan/filter.
# A `map_batches` partition gets this many times the CPU a same-sized scan would — the
# plan-level compute-skew factor (data skew is handled per-partition by `descriptor_rows`).
#
# **The CPU a partition reserves, and nothing else.** It was also multiplied into the
# *partition count*, which made every task four times smaller for no gain — a task's
# intra-task `num_workers` is derived from this same share, so the wider task runs the UDF
# just as many ways, and it does so with a quarter of the per-task overhead. Measured 1.4-2.0x
# slower the other way; see `_adaptive_partition_count`. Keep it out of the count.
_MAP_COMPUTE_WEIGHT = max(1.0, float(os.environ.get("BATCHER_MAP_COMPUTE_WEIGHT", "4.0")))
# Hard ceiling on the per-actor submit-ahead depth (partitions an inference actor keeps in
# flight). Single source in the neutral `_internal.hardware`, shared with the ML autobatcher
# — `dist` cannot import `ml`, so the constant lives below both rather than being pasted twice.
_MAP_INFLIGHT_MAX = INFERENCE_INFLIGHT_DEPTH_MAX
# Bound on the pool-wide liveness probe. Unchanged in value from the old per-actor
# `ray.get(timeout=10)`, but now paid ONCE for the whole pool rather than per actor (see
# `_healthy_actors`), so a 200-actor pool costs 10s of worst-case probing, not 2000s.
_POOL_PROBE_TIMEOUT_S = 10.0

_log = get_logger("dist")

__all__ = ["release_inference_pools", "resident_inference_pools", "stream_distributed_map"]

# A query-lifetime registry of inference actor pools, keyed by pipeline signature, so a
# `map_batches`/inference pipeline's model loads ONCE per query and is reused across
# stages (and repeated terminal calls) instead of rebuilt every distributed-map call —
# the GPU-saturation win (a cold model reload between stages starves the GPU). `None`
# (the default, outside a `resident_inference_pools()` scope) keeps the per-call pool
# with its autoscaling + preemption recovery, so the default path is unchanged.
_INFERENCE_POOLS: contextvars.ContextVar[dict[tuple, list] | None] = contextvars.ContextVar(
    "batcher_inference_pools", default=None
)

# A SESSION-lifetime registry (module-global, not scoped) of warm inference pools, used
# when `distributed.warm_inference_pools` is on and no `resident_inference_pools()` scope is
# active. Keyed by pipeline signature so the same model loads once per session and is reused
# across every `collect()` — the long-lived-actor win (Ray Data respawns the pool per
# execution, paying the ~20x-first-batch cold start each time). Torn down at process exit or
# by `release_inference_pools()`; a pool whose actors died is rebuilt on next use.
_SESSION_POOLS: dict[tuple, list] = {}

# Pins the `fn` objects whose `id()` a live pool's key was built from. Without this the key
# is a bare address: once a pipeline's model callable is freed (its `Dataset` went out of
# scope while the warm pool outlived it), CPython may hand that same address to a *different*
# model, whose pipeline then matches the old key and silently runs inference on the wrong
# model. `kyber.plan_cache` pins its source objects for exactly this reason. Keyed by the
# same signature as the registries, and dropped in step with them.
_POOL_KEEPALIVE: dict[tuple, tuple] = {}


def release_inference_pools() -> None:
    """Tear down all session-warm inference actor pools and free their GPUs.

    Warm pools (``distributed.warm_inference_pools``, on by default) keep a model's actors
    alive across ``collect()`` calls so it loads once per session. Call this to release those
    GPUs before other GPU work, or when done with inference; it also runs automatically at
    process exit. A no-op when no pools are warm. The next inference `collect()` rebuilds the
    pool (paying the one-time load again)."""
    _shutdown_pools(_SESSION_POOLS)


# Free any session-warm GPU actors at process exit so a finished batch job never leaves GPUs
# held on the cluster (best-effort — Ray may already be shutting down).
atexit.register(release_inference_pools)


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
    _unpin_pool_keys(registry)
    registry.clear()


def _pipeline_functions(plan0: LogicalPlan) -> tuple:
    """`plan0`'s `map_batches` callables, outermost first (a class factory is one object)."""
    fns: list[object] = []
    node: LogicalPlan | None = plan0
    while node is not None:
        if isinstance(node, MapBatches):
            fns.append(node.fn)
        node = getattr(node, "input", None)
    return tuple(fns)


def _pipeline_signature(plan0: LogicalPlan) -> tuple:
    """A reuse key for `plan0`'s inference pool: the identities of its `map_batches`
    functions (a class factory is one stable object), so the same model maps to the same
    resident pool and a different model gets its own.

    The key is only as stable as those addresses, so whoever stores a pool under it must
    also pin the callables in `_POOL_KEEPALIVE` — see `_pin_pool_key`."""
    return tuple(id(fn) for fn in _pipeline_functions(plan0))


def _pin_pool_key(sig: tuple, plan0: LogicalPlan) -> None:
    """Hold `plan0`'s callables alive for as long as a pool is cached under `sig`."""
    _POOL_KEEPALIVE[sig] = _pipeline_functions(plan0)


def _unpin_pool_keys(sigs) -> None:
    """Release the callables pinned for `sigs` (their pools are gone)."""
    for sig in list(sigs):
        _POOL_KEEPALIVE.pop(sig, None)


def _new_map_actor(plan0: LogicalPlan, opts: dict):
    """Spawn one model-loaded `_MapActor` (the single actor-creation point, so residency
    reuse and tests can account for every build)."""
    cls = _MapActor.options(**opts) if opts else _MapActor
    return cls.remote(plan0)


def _pool_key(plan0: LogicalPlan, opts: dict) -> tuple:
    """The registry key for a resident pool: its pipeline **and its resource request**.

    Keying on the pipeline alone makes two actors interchangeable when they are not. The
    adaptive loop re-sizes `num_gpus` between runs, and a pool keyed only by UDF identity
    then *grows* to the new replica count instead of being replaced — so four whole-GPU
    actors from the previous run stay alive holding every device while the eight half-GPU
    actors that replaced them wait for a GPU that will never come free. The query does not
    fail; it hangs, with the cluster fully reserved and nothing running.
    """
    resources = tuple(sorted((k, repr(v)) for k, v in (opts or {}).items()))
    return (_pipeline_signature(plan0), resources)


def _evict_stale_configurations(key: tuple, registry: dict) -> None:
    """Kill any pool for the same pipeline built against a *different* resource request.

    Its actors cannot serve this run — they were placed against the old fraction — and they
    hold exactly the devices the new pool needs, so leaving them warm is a deadlock rather
    than a wasted reservation.
    """
    import ray

    stale = [k for k in registry if k[0] == key[0] and k != key]
    for k in stale:
        for actor in registry.pop(k, []):
            with contextlib.suppress(Exception):
                ray.kill(actor)


def _resident_pool_for(
    plan0: LogicalPlan, opts: dict, size: int, registry: dict, devices: int = 0
) -> list:
    """The resident actor pool for `plan0` in `registry` (built once, reused after).

    The model is built in each actor's `__init__`, so reuse means it loads once per
    registry lifetime (a query scope, or the whole session for the warm registry). A pool
    whose actors have died (preemption between reuses) is healed — dead actors are dropped
    and respawned to the requested size — so a session-warm pool survives node churn.

    `devices` (non-zero only on a *first*, unmeasured run) asks for the cold-start fill: the
    pool starts at one actor per device, and once the models have loaded their footprint is
    read and a second actor per device is added if the device shows room. That is what gives
    run 0 the packing the measured loop would otherwise only reach on run 1 — and it cannot
    over-pack, because it grows on a measurement rather than on a guess.
    """
    sig = _pool_key(plan0, opts)
    _evict_stale_configurations(sig, registry)
    pool = registry.get(sig)
    pool = _healthy_actors(pool) if pool else []
    if len(pool) < max(1, size):
        pool = pool + [_new_map_actor(plan0, opts) for _ in range(max(1, size) - len(pool))]
    if devices > 0:
        want = devices * _cold_start_density(pool)
        if want > len(pool):
            pool = pool + [_new_map_actor(plan0, opts) for _ in range(want - len(pool))]
    registry[sig] = pool
    _pin_pool_key(sig, plan0)
    return pool


#: A partition ceiling high enough that `gpu_aware_pool_default`'s "never more actors than
#: partitions" clamp cannot bind while we are asking it how many actors the devices want.
_UNCLAMPED_PARTITIONS = 1 << 30


def _pool_partition_count(
    workers: int,
    num_gpus: float,
    accelerator_type: str | None,
    resources: dict[str, float] | None,
    concurrency: object,
) -> int:
    """Partitions for an accelerator actor pool: at least one per actor the devices want.

    The pool is sized by devices and then clamped to the partition count so no actor sits
    idle — which inverts the causality, because the partition count is sized from *data*. A
    2.4 GB image corpus is under one `target_bytes_per_task`, so it took the four-partition
    floor, and four partitions is what decided that a twelve-actor pool would run four
    actors. The data was choosing how many accelerators were allowed to work.

    One partition per actor, not more: an inference partition carries its own dispatch, lazy
    source and result gather, so over-sharding costs real time — sizing these at
    `replicas x submit-depth` instead measured **1,287 img/s against 2,576** for the same
    work. The aim is that no device is idle, not that every device is oversubscribed.

    Never below the caller's `workers`, and a partition count only shards, so the merged
    result is identical for any value.
    """
    if concurrency is not None:  # the caller sized their own pool; don't second-guess it
        return workers
    from batcher.ml.gpu import gpu_aware_pool_default

    replicas = gpu_aware_pool_default(
        num_gpus, workers, _UNCLAMPED_PARTITIONS, accelerator_type, resources=resources
    )
    return max(workers, replicas)


def _cold_start_devices(hub, plan: LogicalPlan, concurrency: object, num_gpus: float) -> int:
    """Accelerators to fill on a first, unmeasured run — 0 when the cold fill does not apply.

    Applies only when every one of these holds: the stage asked for an accelerator, it left
    `concurrency` to the engine, and nothing has been measured for this pipeline yet. The
    last is what keeps this a *cold start* rather than a competing policy: from the first
    recorded utilization onward, `recommend_num_gpus` owns the density and this returns 0.
    """
    if concurrency is not None or num_gpus <= 0:
        return 0
    try:
        import ray

        from batcher.ml.gpu import gpu_feedback_key, load_gpu_utilization

        if load_gpu_utilization(_learning_hub(hub), gpu_feedback_key(plan)) is not None:
            return 0
        return int(float(ray.cluster_resources().get("GPU", 0.0)))
    except Exception as exc:  # pragma: no cover - sizing must never break a query
        note_suppressed("dist", "size the cold-start accelerator fill", exc)
        return 0


def _cold_start_density(pool: list) -> int:
    """Actors per device this pool's *measured* model footprint leaves room for.

    Reads one loaded actor rather than all of them: they run the same model on the same
    device class, so the first answer is the answer. Falls back to 1 — unpacked, the previous
    behaviour — if nothing can be measured, so a device is never packed on a guess.
    """
    import ray

    from batcher.ml.gpu import cold_start_actors_per_device

    if not pool:
        return 1
    try:
        return cold_start_actors_per_device(ray.get(pool[0].loaded_vram.remote()))
    except Exception as exc:  # pragma: no cover - sizing must never break a query
        note_suppressed("dist", "probe the loaded model's VRAM for cold-start packing", exc)
        return 1


def _healthy_actors(pool: list) -> list:
    """Drop actors that are no longer alive (a cheap liveness ping), keeping the survivors —
    so a warm pool reused after a preemption doesn't dispatch to a dead actor.

    The probes are issued together and awaited in **one** bounded `ray.wait`, not one
    blocking `ray.get(timeout=...)` per actor. Serially, a pool that lost a node paid the
    full timeout for every dead actor before any work started — 200 actors x 10s = up to
    2000s of dead time on exactly the recovery path that most needs to be fast. Concurrently
    it is one timeout for the whole pool regardless of size. An actor whose probe is still
    unresolved when the wait expires is treated as dead, which is the same verdict the serial
    version reached, just reached once.
    """
    import ray

    if not pool:
        return []
    probes = {a: a.gpu_stats.remote() for a in pool}
    refs = list(probes.values())
    ready, _ = ray.wait(refs, num_returns=len(refs), timeout=_POOL_PROBE_TIMEOUT_S)
    resolved = set(map(id, ready))
    alive = []
    for actor, ref in probes.items():
        try:
            if id(ref) not in resolved:
                raise TimeoutError("liveness probe did not resolve")
            ray.get(ref)  # already resolved — surfaces a dead-actor error without blocking
            alive.append(actor)
        except Exception:  # dead / unreachable — drop it (a replacement is spawned to size)
            with contextlib.suppress(Exception):
                ray.kill(actor)
    return alive


def _actor_inflight_depth() -> int:
    """Per-actor submit-ahead depth for the inference actor pool.

    The envelope's adapted depth (raised from measured GPU utilization by the conductor)
    when a scheduling envelope is in force, else the config floor. Always in
    ``[1, _MAP_INFLIGHT_MAX]``. Depth 1 is the historical one-partition-at-a-time behavior.
    """
    from batcher.config import active_config
    from batcher.dist.executors.ray_runtime import current_envelope

    env = current_envelope()
    depth = env.inflight_depth if env is not None and env.inflight_depth > 0 else None
    if depth is None:
        depth = active_config().distributed.map_inflight_depth
    return max(1, min(_MAP_INFLIGHT_MAX, int(depth)))


def _emptiest_actor(actors, slots: dict):
    """The actor with the most free in-flight slots, or `None` when the pool is full.

    Both actor-pool drivers used to take "the first actor with a free slot", which fills
    actor 0 to its submit depth before actor 1 receives anything. Whenever the partition
    count is at or below ``len(actors) x depth`` — the ordinary case for an inference stage,
    whose partitions are few and wide — the tail of the pool never runs at all.

    The deeper the submit-ahead, the worse it gets, and the depth is raised by exactly the
    signal that means "this GPU is starved" (`recommend_inflight_depth`). So the lever meant
    to keep one device fed took work away from the others: measured on four T4s with four
    partitions, two GPUs sat at 0% at depth 2, and at depth 4 a single actor ran the whole
    stage while three GPUs idled — at a throughput high enough to look healthy.

    `max` returns the first of equal keys, so an idle pool fills round-robin (every actor
    gets its first partition before any gets its second) and the depth still stacks once
    every actor is busy, which is what it is for.
    """
    if not actors:
        return None
    actor = max(actors, key=lambda a: slots[a])
    return actor if slots[actor] > 0 else None


def _pipeline_actor_pool(actors, partitions, depth: int) -> list:
    """Run `partitions` through a FIXED pool of `actors`, up to `depth` in flight per actor,
    preserving partition order.

    No spawn / heal / autoscale — the caller owns the pool lifetime (the resident / warm
    path, whose actors are health-checked before reuse). At `depth == 1` this is one
    partition per actor at a time (the `ActorPool.map` behavior it replaces); a higher depth
    keeps each actor fed across the dispatch/gather round-trip. Results are index-addressed,
    so the output is identical to depth 1 for any depth. Returns results in partition order.
    """
    import ray

    parts = list(partitions)
    n = len(parts)
    results: list = [None] * n
    if not actors or n == 0:
        return results
    slots = {a: max(1, depth) for a in actors}
    inflight: dict = {}  # ref -> (actor, idx)
    pending = deque(range(n))

    def _assign() -> None:
        while pending:
            actor = _emptiest_actor(actors, slots)
            if actor is None:
                break
            idx = pending.popleft()
            inflight[actor.run.remote(parts[idx], idx)] = (actor, idx)
            slots[actor] -= 1

    _assign()
    while inflight:
        ready, _ = ray.wait(list(inflight), num_returns=1)
        ref = ready[0]
        actor, idx = inflight.pop(ref)
        results[idx] = ray.get(ref)
        slots[actor] += 1
        _assign()
    return results


def _run_resident_pool(plan0, partitions, opts, size, registry, devices: int = 0):
    """Map `partitions` through the resident pool for `plan0` in `registry` (model loaded
    once), preserving submission order.

    Returns ``(ordered_results, gpu_util, peak_vram)``. The utilization is the *most loaded*
    actor's **sustained** figure: packing keyed on the fleet mean would oversubscribe a device
    that is already busy while its idle neighbours pull the average down, so the binding
    device decides. VRAM stays the peak across actors, for the same capacity reason.
    """
    import ray

    actors = _resident_pool_for(plan0, opts, size, registry, devices)
    results = _pipeline_actor_pool(actors, partitions, _actor_inflight_depth())
    samples = [s for s in ray.get([a.gpu_stats.remote() for a in actors]) if s is not None]
    vram = [v for v in (_drain_gpu_vram(a) for a in actors) if v is not None]
    return results, (max(samples) if samples else None), (max(vram) if vram else None)


def _run_scoped_pool(plan0, partitions, opts, lo, hi, scope, devices: int = 0):
    """Run `partitions` through the query-resident pool for `plan0`, healing a lost pool.

    The `resident_inference_pools()` scope reuses one model-loaded pool across a query's
    stages and repeated terminals. That pool had no preemption recovery: a GPU node lost
    mid-run (after the pool was cached, so past any liveness check) raised `RayError` and
    failed the whole query — exactly the multi-hour inference job the scope exists to keep
    fed. On a loss, evict the dead pool from the scope and re-run on a fresh recovering
    per-call pool; the scope rebuilds its resident pool on the next call. Symmetric to
    `_run_warm_pool` for the session-warm registry. Returns
    ``(ordered_results, peak_gpu_util, peak_vram)``."""
    from ray.exceptions import RayError

    from batcher.dist.executors.ray_runtime import recovery_policy

    try:
        return _run_resident_pool(plan0, partitions, opts, hi, scope, devices)
    except RayError:
        _evict_scoped_pool(plan0, scope)
        return _drive_actor_pool(plan0, partitions, opts, lo, hi, recovery_policy())


def _evict_scoped_pool(plan0, scope) -> None:
    """Drop (and kill) the resident pool for `plan0` from a `resident_inference_pools` scope."""
    _evict_pipeline_pools(plan0, scope)


def _run_warm_pool(plan0, partitions, opts, lo, hi, devices: int = 0):
    """Run `partitions` through the SESSION-warm pool for `plan0`, healing a lost pool.

    On the rare case the warm pool loses actors mid-run (a node preempted after the liveness
    check), it evicts the pool and re-runs on a fresh recovering per-call pool, so a warm
    pool never turns a preemption into a failed query. The next inference call rebuilds the
    warm pool. Returns ``(ordered_results, peak_gpu_util, peak_vram)``."""
    from ray.exceptions import RayError

    from batcher.dist.executors.ray_runtime import recovery_policy

    try:
        return _run_resident_pool(plan0, partitions, opts, hi, _SESSION_POOLS, devices)
    except RayError:
        _evict_session_pool(plan0)
        return _drive_actor_pool(plan0, partitions, opts, lo, hi, recovery_policy())


def _evict_session_pool(plan0) -> None:
    """Drop (and kill) the session-warm pool for `plan0` — after a preemption or on demand."""
    _evict_pipeline_pools(plan0, _SESSION_POOLS)


def _evict_pipeline_pools(plan0, registry: dict) -> None:
    """Kill every pool `registry` holds for `plan0`, whatever resource request built it.

    A pipeline can have more than one entry, because the key carries the resource request
    (`_pool_key`) — so an eviction that matched only one exact key would leave the other
    configuration's actors alive holding their devices.
    """
    import ray

    sig = _pipeline_signature(plan0)
    keys = [k for k in list(registry) if k[0] == sig]
    for key in keys:
        for actor in registry.pop(key, []):
            with contextlib.suppress(Exception):
                ray.kill(actor)
    _unpin_pool_keys(keys)


def _map_resources(
    plan: LogicalPlan,
) -> tuple[float, bool, object, str | None, dict[str, float]]:
    """GPU reservation, whether an actor pool is needed, its size spec, the accelerator type
    to pin GPU actors/tasks to, and the custom Ray resources the fused stages ask for.

    The size spec is an `int` (fixed pool), a ``(min, max)`` tuple (autoscale to the
    workload), or `None` (default to the worker count)."""
    num_gpus = 0.0
    wants_pool = False
    concurrency: object = None
    accelerator_type: str | None = None
    resources: dict[str, float] = {}
    node: LogicalPlan | None = plan
    while node is not None:
        if isinstance(node, MapBatches):
            num_gpus = max(num_gpus, node.num_gpus)
            if node.accelerator_type is not None:
                accelerator_type = node.accelerator_type
            # Stacked map stages fuse into one task, so their accelerator needs combine:
            # take the max per resource name, matching how `num_gpus` merges above. A
            # stage asking for 4 TPU chips and one asking for 2 must get 4, not 2.
            for name, amount in node.resources:
                resources[name] = max(resources.get(name, 0.0), amount)
            # An explicit concurrency, or a class (factory) UDF that must build the
            # model once per worker, both require long-lived actors.
            if node.concurrency is not None or isinstance(node.fn, type):
                wants_pool = True
                if node.concurrency is not None:
                    concurrency = _merge_concurrency(concurrency, node.concurrency)
        node = getattr(node, "input", None)
    return num_gpus, wants_pool, concurrency, accelerator_type, resources


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
    *,
    preserve_order: bool = False,
    write_spec: dict | None = None,
    cluster_by: tuple[str, ...] | None = None,
):
    """Run a linear map/inference pipeline across Ray workers, one partition each.

    With `write_spec` each worker writes its own output shard to the sink and returns only
    `WrittenFile` locators, so the function returns a merged `WriteManifest` instead of a
    `pa.Table` and the post-inference result never lands on the driver. Without it the
    per-partition batches are gathered and concatenated as before.

    When a `hub` is supplied and the pipeline used a GPU actor pool, the measured
    GPU utilization is recorded so the next run's `num_gpus` request can adapt.

    `preserve_order` partitions the source into contiguous source-ordered runs so the
    partition-index-assembled output reproduces the source's global row order. Callers that
    slice or number that output (distributed `LIMIT`, `with_row_index`) require it; the
    default load-balanced assignment is fine for every order-independent map/scan.

    `cluster_by` assigns whole clustering groups, so no two splits sharing a value in those
    columns land on different partitions. That is what lets `plan` be a **breaker** -- an
    `Aggregate`, a `Distinct`, a partitioned `Window` -- rather than only a map: with every
    group whole inside one partition, running the breaker per partition and concatenating is
    the complete result, and no exchange is needed. See
    `dist/executor.py::_partition_aligned_aggregate`."""
    _ensure_ray(workers)
    plan0, sid = _relabel_single_source(plan)
    num_gpus, wants_pool, concurrency, accelerator_type, resources = _map_resources(plan)
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
    # actor-pool path keeps the worker count (its pool size, not the CPU task count) —
    # `partition_descriptors` row-balances those partitions across every worker, so a source
    # arriving as one large batch fans out evenly (one balanced slice per GPU actor) instead
    # of landing whole on worker 0.
    n_parts = (
        _pool_partition_count(workers, num_gpus, accelerator_type, resources, concurrency)
        if wants_pool
        else _adaptive_partition_count(sources[sid], plan, workers, hub)
    )
    proj, pred = _scan_pushdown(plan0)
    partitions = partition_descriptors(
        sources[sid],
        n_parts,
        projection=proj,
        predicate=pred,
        preserve_order=preserve_order,
        cluster_by=cluster_by,
    )
    if write_spec is not None:
        # A `num_files` layout names a total across the whole write, so each shard needs to
        # know how many shards it is one of before it can take its share of that budget.
        # Only knowable here, where the partition count is decided.
        write_spec = {**write_spec, "shards": len(partitions)}

    opts = _gpu_options(num_gpus, accelerator_type, resources)
    if wants_pool:
        if isinstance(concurrency, tuple):
            lo, hi = concurrency
        else:
            from batcher.ml.gpu import gpu_aware_pool_default

            default_pool = gpu_aware_pool_default(
                num_gpus, workers, len(partitions), accelerator_type, resources=resources
            )
            lo = hi = _resolve_pool_size(concurrency, len(partitions), default_pool)
            # A recurring inference pipeline that has consistently served fewer partitions than
            # it built actors right-sizes its (auto) pool from that measured reuse, so a small
            # job stops over-provisioning GPU actors. Only trims an auto-resolved size, never an
            # explicit `concurrency`; pool size is pure parallelism, so the result is identical.
            if concurrency is None:
                from batcher.dist.adaptive_sizing import learned_actor_pool_size

                learned = learned_actor_pool_size(hub, _pipeline_signature(plan0), hi)
                if learned is not None:
                    lo = hi = learned
        _record_actor_pool_reuse(hub, plan0, len(partitions))
        # A first, unmeasured run starts at one actor per device and fills the devices from
        # the model's measured footprint once it has loaded (`_resident_pool_for`). Without
        # this, run 0 of every new pipeline is the unpacked configuration the measured loop
        # exists to replace — which for a single-shot job is the only configuration it ever
        # gets. Zero (no cold fill) as soon as anything has been measured, and for an
        # explicit `concurrency`, which is the caller sizing their own pool.
        cold_devices = _cold_start_devices(hub, plan, concurrency, num_gpus)
        if cold_devices:
            lo = hi = max(1, cold_devices)
        # Pick the pool lifetime: an explicit `resident_inference_pools()` scope (query
        # lifetime) wins; else the SESSION-warm registry when `warm_inference_pools` is on
        # (model loads once per session, reused across `collect()`s — the 2x win on repeated
        # / iterative / cold-start-bound inference); else the per-call pool with autoscaling +
        # preemption recovery (spawned and killed each call, the historical default).
        from batcher.config import active_config

        scope = _INFERENCE_POOLS.get()
        warm = active_config().distributed.warm_inference_pools
        if write_spec is not None:
            # A writing stage builds its actors with the sink bound in, so it cannot borrow a
            # pool from the session-warm / resident registries (those actors were built to
            # RETURN batches). Use the per-call pool, which also carries preemption recovery.
            results, gpu_util, gpu_vram = _drive_actor_pool(
                plan0, partitions, opts, lo, hi, recovery_policy(), write_spec
            )
        elif scope is not None:
            results, gpu_util, gpu_vram = _run_scoped_pool(
                plan0, partitions, opts, lo, hi, scope, cold_devices
            )
        elif warm:
            results, gpu_util, gpu_vram = _run_warm_pool(
                plan0, partitions, opts, lo, hi, cold_devices
            )
        else:
            results, gpu_util, gpu_vram = _drive_actor_pool(
                plan0, partitions, opts, lo, hi, recovery_policy()
            )
        _record_gpu_feedback(hub, plan, gpu_util, gpu_vram, _actors_per_device(hi, num_gpus))
    else:
        # Skew-aware adaptive CPU: each stateless task requests a CPU share sized to its
        # own partition's data (x the plan's compute weight) — fractional for a tiny
        # partition (packed many-per-core), several cores for a large one. A heavier
        # (skewed) partition therefore gets proportionally more CPU than its peers.
        shares = _adaptive_task_cpus(partitions, plan, hub)
        # Resolve SPREAD vs Ray's locality-aware DEFAULT against the live cluster: SPREAD
        # only where these right-sized (often sub-node) tasks would otherwise pack onto one
        # node and idle the rest, DEFAULT (restoring argument locality) when packing isn't a
        # risk or the cluster is large enough that DEFAULT's balancing suffices. On a
        # heterogeneous cluster this also keeps a CPU-only map fleet off GPU nodes.
        sched = _map_scheduling_options(env, shares)
        plan_ref = _shared_arg(plan0)
        cfg_for = _engine_config_cache()

        def _launch(idx):
            # Intra-task workers = this task's own CPU share (>=1); cluster-wide
            # parallelism is the many tasks, not a full-width pool inside each one.
            workers = max(1, round(shares[idx]))
            # Ship the DRIVER's engine config, with the engine's rayon width pinned to this
            # task's own CPU grant. Every other distributed operator does this; the map path
            # did not, so a worker fell back to its own `active_config()` — whose
            # `parallelism: 0` means "use every core on this box". Dozens of concurrent map
            # tasks on a 16-core node each opened a 16-thread pool: hundreds of threads
            # thrashing 16 cores, and the session config (morsel size, memory budget)
            # silently ignored on every distributed scan.
            # The skew-adaptive share is the CPU-only task's grant. An *accelerator* task's
            # grant comes from `opts` instead (zero — see `_gpu_options`), so `opts` is
            # applied after it rather than before: spelling the share last overrode the
            # accelerator zero and put a custom-resource stage (a TPU/Trainium task, which
            # takes this branch because it wants no actor pool) back behind the CPU
            # reservation the zero exists to escape.
            return _map_udf_task.options(**{"num_cpus": shares[idx], **opts, **sched}).remote(
                plan_ref, partitions[idx], workers, cfg_for(shares[idx]), write_spec, idx
            )

        # `task_cpus` is the smallest share any of these tasks asks for, so the
        # submit-ahead window counts tasks rather than cores. A stage of many small
        # partitions runs several tasks per core, and a cores-derived window would cap
        # its concurrency at a fraction of what the cluster can hold.
        results = gather_map_results(
            _launch, len(partitions), task_cpus=min(shares) if shares else 1.0
        )

    if write_spec is not None:
        # Only locators came back. Merging them is a commutative concat, so partition order
        # does not matter and a recomputed shard replaces its own deterministic file.
        from batcher.io.manifest import WriteManifest

        return WriteManifest(tuple(f for shard in results if shard for f in shard))

    batches: list[pa.RecordBatch] = []
    for r in results:
        if r:
            batches.extend(r)
    _record_source_rows(hub, sources[sid], sum(b.num_rows for b in batches))
    if not batches:
        # A pipeline whose filter matched nothing still has a schema, and the single-node
        # path returns it. Returning a *column-less* table here made `distributed ==
        # single-node` false for every empty result — and broke any caller that went on to
        # select a column or concat with a non-empty batch. Fall back to the column-less
        # table only when the plan cannot state its schema (a UDF whose output type is
        # unknown until it runs).
        schema = plan.available_schema()
        return pa.table({}) if schema is None else pa.Table.from_batches([], schema=schema.arrow)
    # Reconcile a UDF whose output schema drifts across partitions (e.g. one partition's
    # rows carry extra fields) to one union schema, so the gather concatenates instead of
    # failing — the same schema-drift tolerance the single-node path gives.
    from batcher.io.schema.evolution import reconcile_batches

    return pa.Table.from_batches(reconcile_batches(batches))


def _shared_arg(value):
    """`value` as one object-store entry every task of this stage shares, not per-task bytes.

    A Ray task argument is serialized **once per `.remote()` call**. The map plan is the same
    object for every partition and it is not small — it carries the cloudpickled UDF, which for
    an ML stage closes over a tokenizer, a config, or a model handle. So the driver paid a full
    pickle of it per partition, on the one thread that also has to submit every task: measured
    at 128 partitions on this cluster, 233 ms of a 2,244 ms query (10%) went into submission
    alone, and that term is linear in the fan-out. The byte-bounded partitioning above
    deliberately produces *thousands* of partitions on a large scan, where the same term would
    dominate the query outright.

    Putting it once hands every task a reference instead: one serialization, one object-store
    copy fetched at most once per node, and Ray dereferences it into the task's argument so no
    task body changes. Falls back to the value itself if `ray.put` is unavailable (a stubbed
    Ray in a unit test), which is exactly the previous behaviour.

    **The size of the payload is the whole justification, so do not extend this to the shuffle
    map barrier**, which passes a JSON `map_ir` per source rather than a pickled UDF. Measured
    on this cluster at 512 submissions of a 210-byte argument: by-value 95 / 123 / 69 ms
    against by-reference 87 / 124 / 91 ms over three alternating rounds — indistinguishable.
    A first, unrepeated measurement showed 201 ms against 71 ms and looked like a 2.8x win; it
    was the first batch of submissions paying one-time costs, and running the two orders
    alternately is what showed it. The win here is real because a cloudpickled tokenizer is
    orders of magnitude larger, not because `.remote()` dislikes arguments.
    """
    try:
        import ray

        return ray.put(value)
    except Exception as exc:  # pragma: no cover - an optimization, never a requirement
        note_suppressed("dist", "share the map plan through the object store", exc)
        return value


def _engine_config_cache():
    """A memoized `engine_config_json(share)` for one stage's task submissions.

    The map path sizes each task's CPU grant from its own partition, so the config has to be
    rebuilt per *distinct share* — but not per *task*. `engine_config_json` re-serializes the
    active config and round-trips it through `json.loads`/`json.dumps` on every call, and the
    shares repeat heavily (every same-sized partition asks for the same grant; on a balanced
    scan they are all identical). Caching within the stage keeps the config exactly as
    per-task-correct as before while paying for it once per distinct value.
    """
    cache: dict[float, str] = {}

    def cfg_for(share: float) -> str:
        cached = cache.get(share)
        if cached is None:
            cached = cache[share] = engine_config_json(share)
        return cached

    return cfg_for


def _record_source_rows(hub, source, rows: int) -> None:
    """Persist a run's measured total rows for `source` so the next run's partition count can
    seed from it when the footer count is unknown. Best-effort; never breaks a query.

    Noted rather than suppressed: a failed write is indistinguishable from a source that has
    never run, so the partition count silently keeps falling back to the blunt cluster-fill
    worker count on every future run, forever, with nothing saying why."""
    try:
        from batcher.dist.adaptive_sizing import record_partition_rows

        record_partition_rows(_learning_hub(hub), source.identity(), rows)
    except Exception as exc:  # pragma: no cover - a learned write must never break a query
        note_suppressed("dist", "record measured source rows", exc)


def _record_actor_pool_reuse(hub, plan0, partitions: int) -> None:
    """Persist how many partitions this inference pool served, so a recurring pipeline can
    right-size its actor pool next run. Best-effort; never breaks a query.

    Noted rather than suppressed, for the reason `_record_source_rows` gives: silence here
    reads as "this pipeline has never run", and every later run keeps over-provisioning GPU
    actors — each one a full model load."""
    try:
        from batcher.dist.adaptive_sizing import record_actor_pool_reuse

        record_actor_pool_reuse(_learning_hub(hub), _pipeline_signature(plan0), partitions)
    except Exception as exc:  # pragma: no cover - a learned write must never break a query
        note_suppressed("dist", "record inference actor-pool reuse", exc)


def _spread_helps(shares: list[float]) -> bool:
    """Whether SPREAD genuinely beats Ray's locality-aware DEFAULT for these map tasks.

    True only when packing is a real risk on a modest cluster: below `map_spread_node_cap`
    nodes (past it SPREAD's per-node scan is itself the scheduler bottleneck, and DEFAULT's
    utilization balancing already spreads load), more tasks than nodes (so stacking would
    actually idle nodes), and a mean per-task CPU share small enough
    (< `map_spread_pack_share`) that DEFAULT would pack many onto one node. A single-node
    cluster returns False (SPREAD == DEFAULT there — skip the spread bookkeeping). Falls
    back to True on an unreadable topology, preserving the prior unconditional-SPREAD
    behavior when the cluster shape can't be read.
    """
    from batcher.config import active_config
    from batcher.dist.executors.ray_runtime.scaling import alive_node_count

    dc = active_config().distributed
    nodes = alive_node_count()  # snapshot-aware; 0 on unreadable topology
    if nodes == 0:
        return True  # topology unreadable → keep the old SPREAD behavior
    if nodes <= 1 or nodes >= dc.map_spread_node_cap:
        return False
    n_tasks = len(shares)
    if n_tasks <= nodes:
        return False  # at most one task per node under DEFAULT anyway — no packing risk
    mean_share = sum(shares) / n_tasks if shares else 1.0
    return mean_share < dc.map_spread_pack_share


def _map_scheduling_options(env, shares: list[float]) -> dict:
    """Ray `.options(...)` scheduling-strategy fragment for stateless map/agg tasks.

    Resolves SPREAD vs Ray's locality-aware DEFAULT against the live cluster (config
    `map_spread`): ``"always"`` forces SPREAD (the historical unconditional behavior),
    ``"never"`` forces DEFAULT, ``"auto"`` (default) keeps SPREAD only where
    `_spread_helps` finds packing to be a real risk and otherwise uses DEFAULT
    (locality-aware: prefers nodes already holding the task's args, then low utilization).
    An explicit ``STRICT_SPREAD`` envelope preference always forces SPREAD. When the
    envelope asks to stay off GPU nodes, a hard CPU-only node selector is merged in (a
    no-op unless the cluster opts in and can host the fleet). Returns `{}` for DEFAULT
    with no selector. Placement never changes which rows a partition holds, so the result
    is identical for any choice.
    """
    from batcher.config import active_config
    from batcher.dist.executors.ray_runtime import node_class_selector

    mode = active_config().distributed.map_spread
    if env is not None and env.placement_strategy == "STRICT_SPREAD":
        opts: dict = {"scheduling_strategy": "SPREAD"}
    elif mode == "always":
        opts = {"scheduling_strategy": "SPREAD"}
    elif mode == "never":
        opts = {}
    else:
        opts = {"scheduling_strategy": "SPREAD"} if _spread_helps(shares) else {}
    if env is not None and shares:
        mean_share = sum(shares) / len(shares)
        sel = node_class_selector(env.prefer_cpu_only_nodes, len(shares), mean_share)
        if sel:
            opts["resources"] = {**opts.get("resources", {}), **sel["resources"]}
    return opts


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
    except Exception as exc:
        note_suppressed("dist", "read cluster node CPU counts", exc)
    return float(available_cpu_count())


def _cluster_cores() -> float:
    """Total schedulable CPUs across alive nodes (the cap on useful task parallelism)."""
    try:
        import ray

        return float(int(ray.cluster_resources().get("CPU", 0.0))) or float(available_cpu_count())
    except Exception:
        return float(available_cpu_count())


def _learning_hub(hub=None):
    """The MetadataHub to read learned sizing from — the one threaded in, else the
    process-wide default (the same store Core records feedback to). Best-effort: any
    failure to reach a hub yields `None`, so a learned read simply falls back to the
    plan default."""
    if hub is not None:
        return hub
    try:
        from batcher.core import default_hub

        return default_hub()
    except Exception as exc:  # pragma: no cover - learning is best-effort
        note_suppressed("dist", "resolve the learning hub", exc)
        return None


def _plan_family(plan: LogicalPlan) -> str:
    """The operator family key a plan's compute is dominated by, for learned per-family
    sizing: a `map_batches`/UDF pipeline vs a plain relational scan."""
    from batcher.core.udf import has_map_batches

    return "map_batches" if has_map_batches(plan) else "scan"


def _learned_weight_factor(plan: LogicalPlan, hub=None) -> float:
    """A learned CPU-reservation multiplier for `plan`'s dominant family, or 1.0 when cold.

    Reads the family's measured per-core busy fraction (`learned_cpu_weight_factor`): a family
    that historically left its reserved cores idle (IO/GPU-bound) reserves proportionally fewer
    cores next run, so the CPU share and task fan-out track how CPU-bound the work actually is —
    a pure packing decision that never changes which rows a task holds."""
    from batcher.dist.adaptive_sizing import learned_cpu_weight_factor

    try:
        factor = learned_cpu_weight_factor(_learning_hub(hub), _plan_family(plan))
    except Exception as exc:  # pragma: no cover - learning is best-effort
        note_suppressed("dist", "read learned task-weight factor", exc)
        factor = None
    return factor if factor is not None else 1.0


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


def _scan_pushdown(plan0: LogicalPlan) -> tuple[list[str] | None, dict | None]:
    """The projection + predicate a relabeled map plan's scan can read straight from storage.

    `_relabel_single_source` rewrites the sub-plan's scan to source 0, so the analysis is
    keyed on 0. The shuffle operators (join, aggregate, sort) have always pushed this down;
    the map/UDF path did not, so every task read **every column** of its partition and let the
    plan's `Project` throw the rest away. For the pipeline the ML/inference path is built on —
    a UDF over one or two columns of a wide table — that is the whole table pulled from object
    storage, per task: on TPC-H sf10 `lineitem`, 16 columns fetched to use 1.

    `(None, None)` (the analysis can't run) keeps the read-everything behavior, which is
    always correct — this only ever narrows what is read, never what is computed.
    """
    return source_pushdown(plan0, 0)


# When a shuffle-free plan is worth the parallelism it gives up. Grouping a partition's splits
# into one assignable unit is the only thing the aligned plan costs, and it costs nothing at all
# for a reader that already splits one-per-partition; it bites for a file-per-split reader
# (Delta, Iceberg), where a partition is many files and the group is one.
#
# **Two conditions, because one is not enough**, and the second was found by measuring rather
# than reasoning. `benchmarks/internals/partition_aligned.py --sweep` varies the partition count
# against a fixed 8-worker fleet, forcing the aligned path so the losing cases are visible.
# Writing `A` for the aligned plan's tasks, `min(groups, workers)`, and `S` for what the shuffle
# would have had, `min(splits, workers)`:
#
#     reader                parts=1        2        4        8       16
#     Hive tree (S == A)      0.95x    1.33x    2.35x    4.44x    4.10x
#     Delta, 4 files/part     0.62x    1.30x    2.44x    2.69x    2.64x
#
# The two losses are the two rows where **A is 1**: the whole query runs on one worker. And a
# ratio cannot see that, which is the trap. Delta at one partition has `A/S = 1/4` and loses
# 1.6x; Delta at two has `A/S = 2/8`, the *same* quarter, and wins 1.3x. A rule written on the
# ratio alone accepts the regression.
#
# So: at least two tasks, and at least a quarter of what the shuffle would have run. The first
# rules out the serial plan; the second keeps a two-partition table off a five-hundred-worker
# fleet, which the sweep's fixed width cannot reach but arithmetic can. Every measured point
# above is decided correctly by the pair. (An earlier rule demanded full retention, `A >= S`,
# which threw away the measured 2.35x and every Hive case; the one before that was the ratio
# alone, which accepted the 0.62x.)
_MIN_ALIGNED_TASKS = 2
_MIN_PARALLELISM_RETENTION = 4


def _source_clustering_columns(source) -> tuple[str, ...]:
    """What `source` says its splits will hold constant, without planning a read.

    The source-level form of the split protocol's `clustering_columns`, and deliberately the
    same name: a source answers what its own splits will declare. Note this is *not*
    `SourceStatistics.partition_keys`, which reports every partition column a table has —
    a nested ``year=/month=`` tree answers ``(year, month)`` there and ``(year,)`` here,
    because the year is what a split holds constant.

    Duck-typed rather than required of `Source`: a connector that cannot answer cheaply simply
    does not declare the method, and is treated as unclustered — which costs a missed
    optimization, never a wrong answer.
    """
    fn = getattr(source, "clustering_columns", None)
    if not callable(fn):
        return ()
    try:
        return tuple(fn())
    except Exception as exc:  # a source that cannot answer declares nothing
        note_suppressed("dist", "read the source's declared clustering", exc)
        return ()


def scan_clustering_for(plan: LogicalPlan, sources, workers: int, hub=None) -> tuple[str, ...]:
    """The columns the split set a `_distributed_map(plan, ...)` read would use is exactly
    value-partitioned by — the storage layout's own answer to "what is this already
    partitioned on".

    A partitioned table — a Hive directory tree, a Delta or Iceberg table — reads splits that
    each hold their partition columns constant. Grouping those splits by value and assigning
    whole groups (`cluster_by`) puts every row with a given value on one worker, which is
    exactly what a shuffle by those columns would have arranged. This function reports the
    columns for which that holds; `io.splits.declared_clustering` is what checks that every
    split in the set agrees on them, because the failure mode of over-claiming is a group
    reported twice as two partial finals — a wrong answer at cluster scale and green on one
    node.

    The split set is planned with the **same** arguments `_distributed_map` will use — the
    same partition count from `_adaptive_partition_count`, the same pushed projection and
    predicate — so the set inspected here is the set the read will get, not a lookalike. That
    is what makes this a fact rather than a forecast; do not simplify it to a fixed count.

    Args:
        plan: The single-source sub-plan the map stage would run per partition.
        sources: The query's bound sources.
        workers: The fleet width. Bounds both plans' parallelism, so it is what the
            retention test is measured against.
        hub: The metadata hub, only so the partition count matches the executor's.
        min_splits: Report no clustering below this many splits. A caller skipping an
            exchange trades the shuffle away for the read's own parallelism, and a table
            whose whole content sits in three directories has three tasks' worth of it — so
            a fleet wider than the layout is better served by shuffling. One (the default)
            means "any layout will do"; a scheduler passes its fleet width.

    Returns:
        The clustering columns, or an empty tuple when the read guarantees none.
    """
    from batcher.dist.executors.partition_io._sources import _scan_splits
    from batcher.io.splits import declared_clustering, group_by_clustering

    ids = scanned_source_ids(plan)
    if len(ids) != 1:
        return ()
    sid = next(iter(ids))
    if sid >= len(sources):
        return ()
    # Ask the source whether a layout exists at all, BEFORE planning a read to find out. A
    # split plan over a large dataset is real driver time -- measured at 25 ms for 500 flat
    # Parquet files, and it scales with the file count -- and on a source that is not
    # partitioned every millisecond of it is thrown away. Every source answers this from
    # metadata it has already loaded: one memoized directory listing, the replayed Delta log,
    # the Iceberg table's own spec. A necessary condition only; the split set is still checked.
    if not _source_clustering_columns(sources[sid]):
        return ()
    plan0, _ = _relabel_single_source(plan)
    proj, pred = _scan_pushdown(plan0)
    n_parts = _adaptive_partition_count(sources[sid], plan, workers, hub)
    try:
        splits = _scan_splits(sources[sid], n_parts, pred, proj)
    except Exception as exc:  # a source that cannot plan splits guarantees no clustering
        note_suppressed("dist", "read the source's split layout", exc)
        return ()
    groups = group_by_clustering(splits)
    if groups is None:
        return ()
    # The GROUP count, not the split count, is the parallelism a clustered read can supply: a
    # value cannot be in two places, so however many files a partition holds, it is one task's
    # worth of assignment. Weigh what that gives up against what the shuffle would have had --
    # both capped by the fleet, since neither plan can use more workers than exist -- and
    # decline when too much of it goes. See `_MIN_PARALLELISM_RETENTION` for the measurements.
    fleet = max(1, workers)
    aligned_tasks = min(len(groups), fleet)
    shuffled_tasks = min(len(splits), fleet)
    if aligned_tasks < min(_MIN_ALIGNED_TASKS, shuffled_tasks):
        return ()  # a serial plan, where the shuffle had more than one task to spend
    if aligned_tasks * _MIN_PARALLELISM_RETENTION < shuffled_tasks:
        return ()
    return declared_clustering(splits)


def _adaptive_partition_count(source, plan, fallback: int, hub=None) -> int:
    """How many tasks to split a map/scan source into — data- and compute-driven.

    Two independent terms, taken as a **maximum**, because they answer different questions
    and only one of them may be traded away for the other:

    * *Parallelism* — `ceil(total_rows / rows_per_cpu)`, clamped to the cluster's core count.
      A tiny source runs as a few (even one) tasks; a large one fans out until a task holds
      about `rows_per_cpu` rows. More tasks than cores buys nothing, so the clamp is right.
    * *Memory* — `_byte_partition_count`, the count that holds one task's input to
      `target_bytes_per_task`. This one is a **bound, not a preference**, and the core clamp
      must not apply to it. It used to: `min(max(rows_term, bytes_term), cluster_cores)`
      discarded the byte term for any source bigger than `cores x target_bytes_per_task`,
      which is precisely the range it exists for. A 1 TB scan on this 128-core cluster asked
      for 4,096 partitions and got 128 — **8 GiB per task against a 256 MiB budget, growing
      linearly with the data**. That is not a slow query; it is an OOM, and it arrives exactly
      when the shape the byte term was added for (a wide multimodal scan) gets large.

    `min(n, len(splits))` still caps both: a task with no split to read is not a task.

    **Neither compute weight belongs in this count** — not `_MAP_COMPUTE_WEIGHT` and not the
    learned CPU factor. Both describe how *heavy* a row is, which decides how much CPU a task
    should reserve (`_adaptive_task_cpus`), not how many tasks there are. Multiplying the count
    by them instead made every task correspondingly *smaller*, and the two errors compounded:
    the learned factor's [0.25, 1.0] range cancelled the fixed 4.0 outright, so a map measured
    as IO-bound quietly ran a quarter of the tasks next run.

    Fewer, wider tasks are also simply faster here, which is the part that is easy to get
    backwards. The old docstring justified the multiplier by claiming a per-batch UDF is
    "single-threaded per task" — but `_map_udf_task` sets its intra-task `num_workers` from
    the very CPU share this weight inflates, so a 4-core task runs the UDF four ways. The two
    arrangements buy the same total threads; one buys them with a quarter of the dispatch,
    descriptor decoding, engine setup and worker acquisition. Measured on this cluster over
    64 M rows, best of three, wall-clock:

    | UDF cost/row | 32 tasks x 4 CPU | 64 x 2 | 128 x 1 |
    |---|--:|--:|--:|
    | light   |   **532 ms** |   884 ms |  524 ms |
    | medium  |   **642 ms** | 1,158 ms | 1,215 ms |
    | heavy   | **1,554 ms** | 2,157 ms | 2,047 ms |

    and a GIL-bound pure-Python UDF — the case where intra-task *threads* cannot help and the
    multiplier looked most justified — preferred fewer tasks hardest of all (730 ms at 8 tasks
    against 1,787 ms at 32), because a wider task also gets a wider `map_batches` process pool.
    Dropping the multiplier lands this workload on 32 tasks of 4 CPUs, which is the winning
    column at every weight.

    When the source's row total isn't cheaply known from a footer, a *measured* total row
    count learned from a prior run of the same source (`learned_partition_rows`) seeds the
    fan-out instead of the blunt `fallback` worker count; a genuinely-cold source still
    falls back. Partition count only shards rows, so any count is result-identical."""
    import math

    from batcher.config import active_config
    from batcher.dist.adaptive_sizing import learned_partition_rows

    total = _source_total_rows(source)
    if total is None:
        try:
            total = learned_partition_rows(_learning_hub(hub), source.identity())
        except Exception as exc:  # pragma: no cover - a learned read must never break a query
            # The read half of the same loop. `None` here is "never measured", which is what a
            # broken read also produces — and the caller then takes the `fallback`, so a
            # persistently failing read looks exactly like a source nothing has learned about.
            note_suppressed("dist", "read the learned source row count", exc)
    if total is None:
        return fallback
    rows_per_cpu = max(1, active_config().optimizer.target_rows_per_task // 2)
    by_rows = math.ceil(total / rows_per_cpu)
    n = max(1, min(by_rows, int(_cluster_cores())))
    n = max(n, _byte_partition_count(source, plan, total, hub))
    with contextlib.suppress(Exception):
        n = min(n, max(1, len(source.splits())))  # never more tasks than splits
    return n


def _byte_partition_count(source, plan, total_rows: int, hub=None) -> int:
    """Task count implied by the source's *bytes*, not its rows.

    Row count alone is the wrong unit whenever rows are wide: 4M rows of 4 KB thumbnails
    and 4M rows of 200 MB videos size to the same fan-out, and the second lands ~800 GB on
    one task. This is the ordinary shape of a multimodal scan, not an edge case.

    `auto_num_partitions` (`api/tuning/decisions.py`) already shards *shuffle* partitions
    this way, on the same `target_bytes_per_task` budget and the same estimator; the
    map/scan fan-out simply never adopted it. Taking the larger of the row- and
    byte-derived counts means a narrow source is unaffected and a wide one fans out.

    Returns 1 — i.e. defers entirely to the row-derived count — when the width cannot be
    estimated, since a task count only shards rows and is result-identical either way.
    """
    try:
        import math

        from batcher.config import active_config

        opt = active_config().optimizer
        total_bytes = _source_total_bytes(source)
        if total_bytes is None:
            from batcher.kyber import load_learned_stats
            from batcher.kyber.cardinality import CardinalityEstimator

            learned = load_learned_stats(_learning_hub(hub))
            width = CardinalityEstimator(sources=[source], learned=learned).row_width(
                plan, opt.row_bytes
            )
            total_bytes = total_rows * width
        return max(1, math.ceil(total_bytes / max(1, opt.target_bytes_per_task)))
    except Exception as exc:  # pragma: no cover - sizing must never break a query
        note_suppressed("dist", "size the byte-based partition count", exc)
        return 1


def _source_total_bytes(source) -> float | None:
    """The source's own byte total, when its statistics report one.

    Preferred over `rows x estimated width` because the estimator's width for a
    variable-length column is a *type prior* — 36 bytes for `binary`, whatever the column
    actually holds. On a media corpus that prior is wrong by four orders of magnitude, and
    it is exactly the case where byte-based sizing matters, so a source that knows its real
    size (a listing gives it for free) must be believed over the prior.
    """
    try:
        stats = source.statistics()
    except Exception:
        return None
    return float(stats.byte_size) if stats is not None and stats.byte_size else None


def _adaptive_task_cpus(partitions, plan, hub=None) -> list[float]:
    """A per-task CPU share for each partition, sized to its data x the plan's compute
    weight (see `_MIN_TASK_CPU` / `_MAP_COMPUTE_WEIGHT`).

    Small partition → a fraction of a core (Ray packs many such tasks per core, so many
    small files run with high parallelism instead of each grabbing a whole core); large
    partition → multiple cores (up to one node). Because the share is per-partition, a
    skewed (heavier) partition is given more CPU than its lighter peers — adaptive to
    both data skew (row count) and plan-level compute skew (a `map_batches`/UDF stage is
    weighted heavier per row than a plain scan).

    The plan-level weight is further scaled by a *measured* per-core busy fraction learned for
    this family (`_learned_weight_factor`): a family that ran CPU-underutilized reserves fewer
    cores next run. Reserving fewer/more cores only changes packing, never the rows a task
    processes, so the result is identical."""
    from batcher.config import active_config
    from batcher.core.udf import has_map_batches

    node_cores = _placeable_node_cores()
    # Rows one core processes in a reasonable slice — half the breaker target (which sizes
    # a whole multi-core task), so a full target-sized partition asks for ~2 cores.
    rows_per_cpu = max(1, active_config().optimizer.target_rows_per_task // 2)
    weight = _MAP_COMPUTE_WEIGHT if has_map_batches(plan) else 1.0
    weight *= _learned_weight_factor(plan, hub)
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
    num_gpus, _wants_pool, _concurrency, accelerator_type, resources = _map_resources(plan)
    from batcher.dist.executors.ray_runtime import current_envelope

    env = current_envelope()
    if env is not None and env.accelerator_type is not None:
        accelerator_type = env.accelerator_type

    proj, pred = _scan_pushdown(plan0)
    partitions = partition_descriptors(sources[sid], workers, projection=proj, predicate=pred)
    opts = _gpu_options(num_gpus, accelerator_type, resources)
    task = _map_udf_task.options(**opts) if opts else _map_udf_task
    plan_ref = _shared_arg(plan0)  # one object-store copy for the stage — see `_shared_arg`
    pending = [task.remote(plan_ref, p) for p in partitions]
    # Collect one finished partition at a time so the driver holds a single partition's
    # output, not the whole result — the bounded-memory way to pull a large scan.
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        out = ray.get(done[0])
        if out:
            yield from out


def _record_gpu_feedback(
    hub,
    plan: LogicalPlan,
    gpu_util: float | None,
    gpu_vram: float | None = None,
    actors_per_device: float | None = None,
) -> None:
    """Persist the pipeline's observed GPU utilization, peak VRAM, and packing density.

    Utilization adapts `num_gpus`; the peak VRAM fraction adapts how many inference actors
    pack onto one device (`actors_per_gpu_from_learned_vram`). `actors_per_device` is the
    density that *produced* this utilization, without which the reading cannot be acted on:
    78% means "add an actor" at one per device and "leave it alone" at two, and a loop that
    cannot tell those apart alternates between them forever. All keyed by the pipeline's
    stable identity; best-effort (each recorder no-ops on `None`)."""
    if hub is None:
        return
    from batcher.ml.gpu import (
        gpu_feedback_key,
        record_gpu_actors_per_device,
        record_gpu_peak_vram,
        record_gpu_utilization,
    )

    key = gpu_feedback_key(plan)
    record_gpu_utilization(hub, key, gpu_util)
    record_gpu_peak_vram(hub, key, gpu_vram)
    record_gpu_actors_per_device(hub, key, actors_per_device)


def _actors_per_device(pool_size: int, num_gpus: float) -> float | None:
    """How many actors of this pool shared one device, or `None` when it is not a GPU stage.

    Derived from the *request*, not from a placement probe: an actor asking for `num_gpus`
    of a device is one of `1 / num_gpus` that fit on it, and Ray honors that by construction.
    A whole-GPU request is one actor per device.
    """
    if num_gpus <= 0 or pool_size <= 0:
        return None
    return max(1.0, 1.0 / num_gpus)


def _gpu_options(
    num_gpus: float,
    accelerator_type: str | None,
    resources: dict[str, float] | None = None,
) -> dict:
    """Ray `.options(...)` accelerator kwargs: GPUs, a device-model pin, and custom
    accelerator resources.

    `num_gpus` covers every accelerator Ray reports as the ``GPU`` resource — NVIDIA, AMD,
    Intel, MetaX. The rest are *custom resources* instead (``TPU``, ``neuron_cores``,
    ``HPU``, ``NPU``, ...), which is what `resources` carries. Passing the dict straight
    through rather than enumerating vendors is deliberate: it also covers a private on-prem
    resource an operator defined on their own Ray cluster, which no vendor list could.

    `accelerator_type` is applied whenever it is set, **not** only alongside `num_gpus`.
    Gating it on GPUs silently dropped the pin on exactly the hardware that needs it most:
    a TPU or Trainium node has `num_gpus == 0`, so a job asking for a specific device model
    got no pin at all and could land on any node in the cluster.

    An accelerator stage additionally reserves **no CPU**, which is not a detail: Ray's rule
    is that an actor asking for *any* resource takes one core for its whole lifetime
    (`DEFAULT_ACTOR_CREATION_CPU_SPECIFIED`), and naming `num_gpus` is asking for one. So an
    inference pool that named only its devices still queued behind a core — and the shuffle
    fleet takes its workers in a placement group holding the cluster's whole CPU capacity, so
    on any pipeline that shuffles before it infers (a `group_by`/`join`/`sort` feeding
    `ds.ml.map_batches`, the documented heterogeneous CPU+GPU shape) that core never comes
    free. It does not fail: the pool never places, every device sits idle, and `ray status`
    reports a fully reserved cluster, which reads as busy rather than stuck. This is the same
    deadlock `bc`'s GPU *relational* path fixed in `gpu_task_options`, on the *inference* path.

    Zero is also the honest request, for the reason it is there: the work is on the device,
    the host thread submits kernels and moves buffers, and concurrency is already bounded by
    `num_gpus` — the device share is what is being contended, not the core."""
    opts: dict = {}
    if num_gpus:
        opts["num_gpus"] = num_gpus
    if resources:
        opts["resources"] = dict(resources)
    if accelerator_type and (num_gpus or resources):
        opts["accelerator_type"] = accelerator_type
    if opts:
        opts["num_cpus"] = 0
    return opts


def _pool_placement_envelope(env, opts: dict):
    """`env` with its per-bundle CPU grant reduced to what an accelerator actor actually asks.

    A placement-group bundle reserves resources; an actor then requests them again from its
    bundle. Those two numbers have to agree, and for an inference pool they did not. The
    device-tiled fan-out grants each worker a whole accelerator node's cores divided by its
    devices (`_accelerator_fill_workers`) — on a 4-node, 8-core, 1-device-per-node cluster
    that is 8 cores per actor, so a 4-actor pool reserved all 32 cores in the cluster. The
    actors themselves now ask for none (`_gpu_options`), so the reservation was for cores
    nothing would use, and it is the reservation that fails: anything else holding a single
    core makes the gang unsatisfiable, and the pool spends the whole placement timeout before
    degrading to the default scheduling that would have worked immediately.

    Left alone for a CPU-only stage, where the core *is* the resource being reserved and the
    bundle is already honest.
    """
    if env is None or "num_cpus" not in opts:
        return env
    import dataclasses

    return dataclasses.replace(env, num_cpus=float(opts["num_cpus"]))


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


def _drive_actor_pool(plan0, partitions, opts, min_size, max_size, policy, write_spec=None):
    """Stream partitions through an actor pool that scales in ``[min_size, max_size]``
    and **replaces an actor lost to preemption**, reassigning its partition.

    Each actor builds the model once (`_MapActor`) and reserves the GPU `opts`. The
    pool starts at `min_size` (so a slow model load doesn't block every replica at
    once), grows toward `max_size` while partitions queue, and reaps idle actors once
    the backlog drains — demand-driven autoscaling (the `concurrency=(min, max)`
    contract); a fixed pool is ``min_size == max_size``.

    Each actor may keep up to `_actor_inflight_depth()` partitions in flight (submit-ahead)
    so a GPU stays fed across the dispatch/gather round-trip; an actor is free to take work
    while any of its slots is open and fully idle (all slots free) when a candidate to reap.
    Depth 1 is the historical one-at-a-time behavior.

    The fault-tolerance part: on a `RayActorError` (a preempted GPU node) the dead actor is
    dropped and **every** partition it had in flight (up to `depth`) is requeued, and the
    pool respawned toward the floor — so the stage heals instead of crashing on the first
    loss (the old `ActorPool.map` / unguarded `ray.get` raised). Bounded by
    `policy.max_attempts` per partition; a deterministic UDF error (`RayTaskError`) surfaces
    immediately. A map/inference UDF recomputes idempotently from its durable partition
    descriptor, so a reassigned partition is neither lost nor duplicated. Returns
    ``(ordered_results, peak_gpu_util)``.
    """
    import ray
    from ray.exceptions import RayError, RayTaskError

    parts = list(partitions)
    depth = max(1, min(_actor_inflight_depth(), len(parts) or 1))
    hi = max(1, min(max_size, len(parts)))
    lo = max(1, min(min_size, hi))

    # Gang-schedule the pool. Sized to `hi` (the autoscaling ceiling) so growing the pool
    # never has to find capacity that was not reserved up front; a bundle sits idle until
    # the pool scales into it. Without this an inference pool acquired its GPUs one actor
    # at a time and could HALF-PLACE AND STALL — some actors running, the rest pending
    # forever on a cluster that will never free the remainder. It is also the only way a
    # `gpu_collective` stage reaches STRICT_PACK: the strategy is resolved inside
    # `create_worker_placement`, so a path that never called it could not express NCCL
    # co-location at all.
    env = _pool_placement_envelope(current_envelope(), opts)
    # Reserving the group is an optimization, never a correctness requirement: the pool
    # runs correctly under default scheduling. So a reservation that times out (returns
    # None) or outright fails (raises — no placement API, a Ray version without it) must
    # degrade to default scheduling rather than take the whole stage down with it.
    try:
        pg = create_worker_placement(hi, env)
        reason = "the cluster could not satisfy the reservation in time"
    except Exception as exc:  # placement is best-effort; never fail the stage on it
        pg = None
        reason = f"placement was unavailable ({type(exc).__name__}: {exc})"
    if pg is None and hi > 1:
        # Not fatal, but a real capacity signal, and it silently forfeits gang scheduling.
        # Degrading without a word is how a stalled or badly-placed GPU pool becomes
        # unexplainable after the fact.
        _log.warning(
            "placement group for %d inference actors was not granted (%s); falling back "
            "to default scheduling: no gang scheduling, actors may place unevenly",
            hi,
            reason,
        )
    free_bundles = deque(range(hi)) if pg is not None else deque()
    bundle_of: dict = {}

    def _spawn():
        actor_opts = opts
        bundle = free_bundles.popleft() if free_bundles else None
        if pg is not None and bundle is not None:
            actor_opts = placement_actor_options(pg, bundle, opts)
        cls = _MapActor.options(**actor_opts) if actor_opts else _MapActor
        actor = cls.remote(plan0, write_spec)
        if bundle is not None:
            bundle_of[actor] = bundle
        return actor

    def _release_bundle(actor) -> None:
        """Return a dead/reaped actor's bundle so its replacement reuses the reservation."""
        bundle = bundle_of.pop(actor, None)
        if bundle is not None:
            free_bundles.append(bundle)

    actors = [_spawn() for _ in range(lo)]
    slots = dict.fromkeys(actors, depth)  # free in-flight slots per actor
    pending = deque(range(len(parts)))  # partition indices awaiting assignment
    inflight: dict = {}  # ref -> (actor, idx)
    results: list = [None] * len(parts)
    attempts = [0] * len(parts)
    peak_util: float | None = None
    peak_vram: float | None = None

    def _assign() -> None:
        while pending:
            actor = _emptiest_actor(actors, slots)
            if actor is None:
                break
            idx = pending.popleft()
            inflight[actor.run.remote(parts[idx], idx)] = (actor, idx)
            slots[actor] -= 1

    try:
        while pending or inflight:
            _assign()
            # An actor is a reap candidate only when *fully* idle (no in-flight partition).
            fully_idle = [a for a in actors if slots[a] == depth]
            action = _autoscale_action(len(pending), len(actors), len(fully_idle), lo, hi)
            if action == "up":
                new = _spawn()
                actors.append(new)
                slots[new] = depth
                continue  # assign the new actor on the next loop
            if action == "down" and fully_idle:
                victim = fully_idle[-1]
                peak_util = _max_opt(peak_util, _drain_gpu_stat(victim))
                peak_vram = _max_opt(peak_vram, _drain_gpu_vram(victim))
                actors.remove(victim)
                slots.pop(victim, None)
                _release_bundle(victim)
                ray.kill(victim)
            if not inflight:
                continue
            ready, _ = ray.wait(list(inflight), num_returns=1)
            ref = ready[0]
            actor, idx = inflight.pop(ref)
            try:
                results[idx] = ray.get(ref)
                slots[actor] += 1  # the producing actor has a slot free again
            except RayTaskError as exc:
                # Deterministic UDF errors surface immediately, but a transient one (CUDA OOM
                # under a concurrency spike, a throttled model endpoint) clears on a retry —
                # and killing a multi-hour inference job on one discards every completed
                # partition. Charged to the same per-partition attempt budget as a preemption.
                from batcher.dist.executors.ray_runtime.policies import (
                    _is_transient_udf_error,
                    check_results_trusted,
                )

                if not _is_transient_udf_error(exc):
                    raise
                # A device that took an uncontained ECC fault kept running and returned a
                # wrong number, so the partitions this actor already completed are suspect
                # too. That is the one failure a retry must not absorb: it would finish the
                # job successfully and write the corruption out.
                check_results_trusted(exc)
                attempts[idx] += 1
                if attempts[idx] > policy.max_attempts:
                    raise
                slots[actor] += 1
                pending.appendleft(idx)
            except RayError:
                # The actor was lost (preemption). Drop it and reclaim EVERY partition it
                # had in flight — with depth>1 one death orphans up to `depth` of them —
                # then respawn toward the floor so the pool heals instead of only shrinking.
                orphaned = [i for (a, i) in inflight.values() if a is actor]
                orphaned.append(idx)
                for r in [r for r, (a, _) in inflight.items() if a is actor]:
                    del inflight[r]
                if actor in actors:
                    actors.remove(actor)
                slots.pop(actor, None)
                _release_bundle(actor)
                for i in orphaned:
                    attempts[i] += 1
                    if attempts[i] > policy.max_attempts:
                        raise
                    pending.appendleft(i)
                while len(actors) < lo:
                    new = _spawn()
                    actors.append(new)
                    slots[new] = depth
        for a in actors:
            peak_util = _max_opt(peak_util, _drain_gpu_stat(a))
            peak_vram = _max_opt(peak_vram, _drain_gpu_vram(a))
        return results, peak_util, peak_vram
    finally:
        for a in actors:
            ray.kill(a)
        # The placement group is a cluster-wide reservation; leaking it strands the pool's
        # GPUs for the life of the job.
        release_placement(pg)


def _drain_gpu_stat(actor) -> float | None:
    """The actor's peak GPU utilization (best-effort; `None` if unavailable)."""
    import ray

    try:
        return ray.get(actor.gpu_stats.remote())
    except Exception as exc:  # pragma: no cover - feedback must never break execution
        note_suppressed("dist", "drain a GPU utilization sample", exc)
        return None


def _drain_gpu_vram(actor) -> float | None:
    """The actor's peak VRAM fraction (best-effort; `None` if the actor predates the method
    — e.g. a test double — or has no GPU)."""
    import ray

    stat = getattr(actor, "gpu_vram_stats", None)
    if stat is None:
        return None
    try:
        return ray.get(stat.remote())
    except Exception as exc:  # pragma: no cover - feedback must never break execution
        note_suppressed("dist", "drain a GPU VRAM sample", exc)
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


def _sustained_utilization():
    """A utilization window for one resident actor, sampled on a timer.

    Every resident stage actor — the terminal `_MapActor` here and the middle
    `RelayActor` in `dist.streaming.relay` — needs the same window, for the same
    reason (see `SustainedUtilization`: a post-forward reading calls a 13%-busy stage
    86% busy, which holds the packing and submit-depth levers shut).

    It is a function rather than a second `from batcher.ml.gpu import ...` at the
    relay's call site because that import is one of the six ratcheted `dist -> ml`
    exemptions in `pyproject.toml`. The class belongs in `interop`, and moving it
    there needs a GPU and a recorded `gpu_shadow_verify` run
    (`.claude/rules/device-tier.md`); until then the edge stays at the sites already
    recorded rather than growing a seventh.
    """
    from batcher.ml.gpu import SustainedUtilization

    return SustainedUtilization()


def _lazy_partition_source(partition: dict):
    """A lazy `IteratorSource` over a partition descriptor, or None if it is empty.

    Yields the descriptor's batches one at a time (`iter_partition_descriptor`) — reading a
    split manifest's row-groups from storage incrementally, or iterating a shipped batch list
    — so a streaming plan overlaps the read with downstream compute. The schema is taken from
    the manifest / first batch without a full read. The materializing path calls `.read()`,
    which still lists everything, so non-streaming execution is byte-for-byte unchanged.
    """
    from batcher.dist.executors.partition_io import iter_partition_descriptor
    from batcher.io.source import IteratorSource

    schema = _descriptor_schema(partition)
    if schema is None:
        return None
    return IteratorSource(lambda: iter_partition_descriptor(partition), schema)


def _descriptor_schema(desc: dict):
    """The Arrow schema a partition descriptor yields (projected), or None if it is empty."""
    if "splits" in desc:
        splits = desc.get("splits") or []
        if not splits:
            return None
        schema = splits[0].schema()
        projection = desc.get("projection")
        if projection is not None:
            schema = pa.schema([schema.field(c) for c in projection])
        return schema
    batches = desc.get("batches") or []
    return batches[0].schema if batches else None


class _MapActor:
    """A long-lived worker that builds its model once and maps many partitions.

    It also samples GPU utilization while running so the scheduler can adapt the
    `num_gpus` request on the next run (the feedback half of GPU scheduling)."""

    def __init__(self, plan0: LogicalPlan, write_spec: dict | None = None) -> None:
        # Build the (class) UDFs locally, once — the model load happens here. The pool's
        # size is the parallelism, so each actor runs its UDF serially (workers=1) rather
        # than spawning a full-width intra-actor pool that would oversubscribe the node.
        self._plan = _with_inference_workers(_prebuild_factories(plan0))
        self._write_spec = write_spec
        self._gpu_vram_max: float | None = None
        # Sustained utilization, sampled on a timer for the actor's working window — NOT the
        # post-forward reading this used to take. See `SustainedUtilization`: sampling right
        # after a forward pass reads the device at its busiest, which reported 86% for a stage
        # whose true sustained figure was 13%, and that is above every threshold the packing
        # and submit-depth levers trigger on. The measurement held its own fix shut.
        self._util = _sustained_utilization()

    def run(self, partition: dict, idx: int = 0):
        from batcher import core
        from batcher.ml.gpu import sample_gpu_vram_fraction

        # A LAZY source over the descriptor: the scan reads its splits (storage) / iterates
        # its shipped batches incrementally, so `stream_linear_chain` overlaps reading chunk
        # k+1 with the GPU forward on chunk k instead of the device idling through an eager
        # whole-partition read (the ~60%->higher-util fix for a scan->GPU inference).
        source = _lazy_partition_source(partition)
        if source is None:
            return None
        self._util.begin_call()
        out = core.execute_with_udfs(self._plan, [source])
        self._util.end_call()
        # VRAM stays a PEAK: it is a capacity constraint, and the largest footprint the run
        # ever reached is what the next run must fit. Utilization is a rate, so it is a mean.
        self._observe_gpu(sample_gpu_vram_fraction())
        if not out or sum(b.num_rows for b in out) == 0:
            return [] if self._write_spec is not None else None
        # Writing stage: this actor writes its own inference output straight to the sink and
        # returns only `WrittenFile` locators. That is what keeps a batch-inference job whose
        # RESULT is larger than the driver (a 2B-row embedding write) from OOMing the driver
        # — the post-inference rows never leave the worker that produced them.
        if self._write_spec is not None:
            return _write_udf_output(out, self._write_spec, idx)
        return out

    def node_host(self) -> str:
        """The host this actor landed on, in the same form a shuffle address carries.

        Lets the streaming driver hand a morsel to a consumer on the *producer's* node, so
        the fetch below stays on loopback instead of crossing the cluster network. Matches
        `host_of(addr)` by construction — both are this node's advertised IP — so the two
        sides of that comparison cannot drift apart.
        """
        import os

        import ray

        return os.environ.get("BATCHER_ADVERTISE_HOST") or ray.util.get_node_ip_address()

    def run_split(self, addr: str, ticket):
        """Map one prior-stage bucket fetched in place from `(addr, ticket)`, so a
        resident inference pool is fed directly from upstream output instead of waiting
        for the driver to hand it a materialized partition.

        The fetch is Flight in every case: this actor runs no shuffle server of its own,
        so it has neither a local partition store to read from nor the shared-memory
        reader that reaching one would need. Locality is bought on the *driver* side
        instead, by handing a morsel to a consumer on the producer's node (`node_host`),
        which keeps the transfer on loopback rather than the cluster network.
        """
        from batcher import core
        from batcher.carbonite.transfer.lifecycle import process_client
        from batcher.io.source import InMemorySource
        from batcher.ml.gpu import sample_gpu_vram_fraction

        # The *pooled* client, not the one-shot `server.fetch`. This runs once per morsel
        # on the GPU consumer of the streaming pipeline — the hottest fetch in the engine —
        # and the one-shot form opens a fresh gRPC channel every time, so the stage paid a
        # full connection handshake ahead of each morsel it was supposed to be overlapping
        # with the previous one's forward pass. `server.fetch`'s own docstring names the
        # pooled client as the form for repeated fetches; this was the repeated fetch.
        rows = process_client().fetch(addr, str(ticket))
        if not rows:
            return None
        self._util.begin_call()
        out = core.execute_with_udfs(self._plan, [InMemorySource(rows)])
        self._util.end_call()
        self._observe_gpu(sample_gpu_vram_fraction())
        if not out or sum(b.num_rows for b in out) == 0:
            return None
        return out

    def _observe_gpu(self, vram: float | None) -> None:
        """Fold one post-forward VRAM sample into this actor's running peak."""
        self._gpu_vram_max = _max_opt(self._gpu_vram_max, vram)

    def gpu_stats(self) -> float | None:
        """This actor's **sustained** GPU utilization, or `None` if the device reports none.

        A mean over the actor's working window, not the peak — that is the quantity
        `recommend_num_gpus` and `recommend_inflight_depth` are defined against, and a peak
        reading silently keeps both of them from ever firing (see `SustainedUtilization`).

        Draining rather than reading: the window is scoped to a *run*. A warm actor outlives
        the run that used it, and a window left open keeps accumulating the idle time between
        runs — which reported 25.7% for a pool that had just held four T4s at 92.7%.

        Doubles as this pool's liveness probe (`_live_actors`), so it must stay cheap and
        must never raise: it reads two accumulated counters. The probe runs immediately
        before a run dispatches, which is exactly where the window should open.
        """
        return self._util.drain()

    def gpu_vram_stats(self) -> float | None:
        """The peak VRAM fraction this actor observed, or `None` if no GPU — the memory
        twin of `gpu_stats`, sized so the next run packs actors by measured footprint."""
        return self._gpu_vram_max

    def loaded_vram(self) -> float | None:
        """VRAM the device holds *now* — after the model load, before any batch has run.

        The one fact about an opaque model that is available without a prior execution, and
        therefore the only thing a *first* run can size its packing from. Ray queues actor
        methods behind the constructor, so simply calling this also waits for the load to
        finish, which is what makes the answer meaningful rather than a race.
        """
        from batcher.ml.gpu import sample_gpu_vram_fraction

        return sample_gpu_vram_fraction()


# Threads a CPU (preprocess/decode) stage runs inside a GPU inference actor, so it keeps a
# fast GPU stage fed (the guides' 2-4:1 CPU:GPU ratio). GPU stages always stay at 1 (one CUDA
# context). Modest so several fractional-GPU actors per node don't grossly oversubscribe the
# cores; a decode/normalize `fn` releases the GIL (PIL/cv2/NumPy/torch) so threads scale.
_INFERENCE_CPU_WORKERS = max(1, int(os.environ.get("BATCHER_INFERENCE_CPU_WORKERS", "4")))


def _with_inference_workers(plan):
    """Set each map stage's `num_workers` for a GPU inference actor: a GPU stage keeps 1 (one
    CUDA context), a CPU stage gets `_INFERENCE_CPU_WORKERS` so its decode/preprocess fans
    across the node's spare cores and stays ahead of the GPU stage — the fix for a fast/small
    model whose single-threaded decode would otherwise starve the device (util < 50%)."""
    import dataclasses

    from batcher.plan.logical import MapBatches
    from batcher.plan.visitor import children, with_children

    if isinstance(plan, MapBatches):
        plan = dataclasses.replace(
            plan, num_workers=1 if plan.num_gpus > 0 else _INFERENCE_CPU_WORKERS
        )
    kids = children(plan)
    if kids:
        return with_children(plan, [_with_inference_workers(c) for c in kids])
    return plan


def _with_map_workers(plan, n: int):
    """Return `plan` with every `map_batches` stage's `num_workers` set to `n`.

    Distributed parallelism comes from running many partition tasks across the cluster,
    so a task must run its UDF with intra-task workers sized to *its own* CPU share — not
    the driver-resolved all-local-cores count, which would make every task on every node
    spawn a full-width thread/process pool and oversubscribe the cluster many times over.
    """
    import dataclasses

    from batcher.plan.logical import MapBatches
    from batcher.plan.visitor import children, with_children

    if isinstance(plan, MapBatches):
        plan = dataclasses.replace(plan, num_workers=n)
    kids = children(plan)
    if kids:
        return with_children(plan, [_with_map_workers(c, n) for c in kids])
    return plan


def _write_udf_output(batches: list, write_spec: dict, idx: int) -> list:
    """Write one worker's UDF output to the sink, returning only `WrittenFile` locators.

    Shared by the actor-pool path (`_MapActor.run`) and the stateless-task path
    (`_map_udf_task`) so a distributed inference write has exactly one write semantics.
    The shard name is deterministic (`part-{idx}`), so a partition recomputed after a
    preemption overwrites its own partial file rather than orphaning it.

    The spec's `layout` is resolved here, against this shard's own rows: a batch-inference
    write is the case where the driver has *never* seen the output, so the row cap behind
    `repartition(num_files=...)` / `repartition(target_size_mb=...)` can only be computed
    on the worker. `resume` likewise has to arrive here to mean anything.
    """
    from batcher.dist.executors.write import _shard_rows_per_file
    from batcher.io.sink import SINKS

    table = pa.Table.from_batches(batches)
    sink = SINKS.get(write_spec["fmt"])(**(write_spec.get("sink_kwargs") or {}))
    layout = write_spec.get("layout")
    if layout is not None:
        layout = layout.for_shard(idx, int(write_spec.get("shards", 1)))
    return sink.write_partitioned(
        table,
        write_spec["path"],
        partition_by=write_spec.get("partition_by"),
        file_index=idx,
        resume=bool(write_spec.get("resume", False)),
        max_rows_per_file=_shard_rows_per_file(table, layout),
    )


def _map_udf_task(
    plan0,
    partition,
    workers: int = 1,
    cfg_json: str | None = None,
    write_spec: dict | None = None,
    idx: int = 0,
):
    from batcher import core
    from batcher.dist.executors.partition_io import read_partition_descriptor
    from batcher.io.source import InMemorySource

    rows = read_partition_descriptor(partition)
    if not rows:
        return [] if write_spec is not None else None
    out = core.execute_with_udfs(
        _with_map_workers(plan0, workers), [InMemorySource(rows)], engine_config=cfg_json
    )
    if not out or sum(b.num_rows for b in out) == 0:
        return [] if write_spec is not None else None
    # Write in place so the post-UDF rows never travel back through the driver.
    if write_spec is not None:
        return _write_udf_output(out, write_spec, idx)
    return out


def _map_agg_task(plan0, partition, group_keys_json, aggregates_json, workers: int = 1):
    """Run the map/UDF prefix on a partition, then PARTIAL-aggregate its output.

    The map (the expensive UDF) runs on the worker over its own partition, and only the
    small partial-aggregate state leaves the worker — the driver does the cross-partition
    `combine_finalize`. This distributes a `map_batches → aggregate` pipeline (Ray Data's
    bread and butter) instead of running the whole UDF single-node on the driver."""
    nat = engine()
    from batcher import core
    from batcher.dist.executors.partition_io import read_partition_descriptor
    from batcher.io.source import InMemorySource

    rows = read_partition_descriptor(partition)
    if not rows:
        return None
    out = core.execute_with_udfs(_with_map_workers(plan0, workers), [InMemorySource(rows)])
    if not out or sum(b.num_rows for b in out) == 0:
        return None
    return nat.partial_aggregate(group_keys_json, aggregates_json, out)


def _distributed_map_aggregate(above, agg, sources, workers):
    """Distribute an aggregate over a linear `map_batches`/UDF pipeline.

    Each worker maps its source partition through the UDF prefix and partial-aggregates
    the result; the driver `combine_finalize`s the partials (mergeable two-phase) and
    applies anything above the aggregate. The UDF — the costly part — runs across the
    cluster, not single-node on the driver."""

    import pyarrow as pa

    nat = engine()
    from batcher.dist.executors.partition_io import _apply_above
    from batcher.dist.executors.plan_analysis import _empty_agg_table
    from batcher.dist.executors.ray_runtime import current_envelope, gather_map_results

    _ensure_ray(workers)
    map_plan, sid = _relabel_single_source(agg.input)
    gk, aj = agg_spec_json(agg)
    n_parts = _adaptive_partition_count(sources[sid], agg.input, workers)
    proj, pred = _scan_pushdown(map_plan)
    partitions = partition_descriptors(sources[sid], n_parts, projection=proj, predicate=pred)
    # Skew-aware adaptive CPU per task (sized to the partition that runs the UDF here);
    # placement resolves SPREAD vs locality-aware DEFAULT against the live cluster.
    shares = _adaptive_task_cpus(partitions, agg.input)
    sched = _map_scheduling_options(current_envelope(), shares)
    # One object-store copy of the map prefix for the whole stage — see `_shared_arg`.
    plan_ref = _shared_arg(map_plan)

    def _launch(idx):
        workers = max(1, round(shares[idx]))
        return _map_agg_task.options(num_cpus=shares[idx], **sched).remote(
            plan_ref, partitions[idx], gk, aj, workers
        )

    # Fold each partition's partial into a running state **as it lands**, rather than
    # holding all of them and folding after the barrier. `combine` is associative and
    # commutative, so the running fold is the same state; what changes is that the driver's
    # peak is one partial instead of `partitions` of them, and the Θ(partitions) fold
    # overlaps the map phase instead of forming a serial tail behind it — the term that
    # would otherwise grow as the cluster does.
    #
    # The fold is in **arrival** order, where the gathered list was in partition order, and
    # that is a deliberate trade rather than an oversight: it puts a float reduction's
    # summation order under the scheduler, so a sum can move in its last bits between two
    # runs over the same data. That is the reassociation the distributed contract already
    # allows (partition count moves it too, and `bc-runtime`'s Neumaier compensation is what
    # bounds it either way) — but it was previously only *across* configurations, and here
    # it is also within one. Integer and min/max/count aggregates are unaffected; a caller
    # needing bit-repeatable float sums has the same recourse it always had, which is to fix
    # the partition count.
    running: list = [None]

    def _fold(_idx, partial):
        if partial is None:
            return
        running[0] = partial if running[0] is None else nat.combine(gk, aj, [running[0], partial])

    gather_map_results(
        _launch, len(partitions), task_cpus=min(shares) if shares else 1.0, sink=_fold
    )
    if running[0] is None:
        table = _empty_agg_table(agg)
    else:
        out = nat.combine_finalize(gk, aj, [running[0]])
        table = pa.Table.from_batches([out]) if out is not None else _empty_agg_table(agg)
    return table if not above else _apply_above(above, table)
