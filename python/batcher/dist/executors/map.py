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

# Smallest CPU share a task may request: a tiny partition gets a fraction of a core so
# Ray packs many such tasks per core (high parallelism over many small files) instead of
# each reserving a whole core. 1/8 core by default. Env-overridable.
_MIN_TASK_CPU = max(0.01, float(os.environ.get("BATCHER_MIN_TASK_CPU", "0.125")))
# How much heavier a per-batch UDF / inference stage is per row than a plain scan/filter.
# A `map_batches` partition gets this many times the CPU a same-sized scan would — the
# plan-level compute-skew factor (data skew is handled per-partition by `descriptor_rows`).
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


def _resident_pool_for(plan0: LogicalPlan, opts: dict, size: int, registry: dict) -> list:
    """The resident actor pool for `plan0` in `registry` (built once, reused after).

    The model is built in each actor's `__init__`, so reuse means it loads once per
    registry lifetime (a query scope, or the whole session for the warm registry). A pool
    whose actors have died (preemption between reuses) is healed — dead actors are dropped
    and respawned to the requested size — so a session-warm pool survives node churn."""
    sig = _pipeline_signature(plan0)
    pool = registry.get(sig)
    pool = _healthy_actors(pool) if pool else []
    if len(pool) < max(1, size):
        pool = pool + [_new_map_actor(plan0, opts) for _ in range(max(1, size) - len(pool))]
    registry[sig] = pool
    _pin_pool_key(sig, plan0)
    return pool


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
            actor = next((a for a in actors if slots[a] > 0), None)
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


def _run_resident_pool(plan0, partitions, opts, size, registry):
    """Map `partitions` through the resident pool for `plan0` in `registry` (model loaded
    once), preserving submission order. Returns ``(ordered_results, peak_gpu_util, peak_vram)``."""
    import ray

    actors = _resident_pool_for(plan0, opts, size, registry)
    results = _pipeline_actor_pool(actors, partitions, _actor_inflight_depth())
    samples = [s for s in ray.get([a.gpu_stats.remote() for a in actors]) if s is not None]
    vram = [v for v in (_drain_gpu_vram(a) for a in actors) if v is not None]
    return results, (max(samples) if samples else None), (max(vram) if vram else None)


def _run_scoped_pool(plan0, partitions, opts, lo, hi, scope):
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
        return _run_resident_pool(plan0, partitions, opts, hi, scope)
    except RayError:
        _evict_scoped_pool(plan0, scope)
        return _drive_actor_pool(plan0, partitions, opts, lo, hi, recovery_policy())


def _evict_scoped_pool(plan0, scope) -> None:
    """Drop (and kill) the resident pool for `plan0` from a `resident_inference_pools` scope."""
    import ray

    sig = _pipeline_signature(plan0)
    for actor in scope.pop(sig, []):
        with contextlib.suppress(Exception):
            ray.kill(actor)
    _unpin_pool_keys([sig])


def _run_warm_pool(plan0, partitions, opts, lo, hi):
    """Run `partitions` through the SESSION-warm pool for `plan0`, healing a lost pool.

    On the rare case the warm pool loses actors mid-run (a node preempted after the liveness
    check), it evicts the pool and re-runs on a fresh recovering per-call pool, so a warm
    pool never turns a preemption into a failed query. The next inference call rebuilds the
    warm pool. Returns ``(ordered_results, peak_gpu_util, peak_vram)``."""
    from ray.exceptions import RayError

    from batcher.dist.executors.ray_runtime import recovery_policy

    try:
        return _run_resident_pool(plan0, partitions, opts, hi, _SESSION_POOLS)
    except RayError:
        _evict_session_pool(plan0)
        return _drive_actor_pool(plan0, partitions, opts, lo, hi, recovery_policy())


def _evict_session_pool(plan0) -> None:
    """Drop (and kill) the session-warm pool for `plan0` — after a preemption or on demand."""
    import ray

    sig = _pipeline_signature(plan0)
    for actor in _SESSION_POOLS.pop(sig, []):
        with contextlib.suppress(Exception):
            ray.kill(actor)
    _unpin_pool_keys([sig])


def _map_resources(plan: LogicalPlan) -> tuple[float, bool, object, str | None]:
    """GPU reservation, whether an actor pool is needed, its size spec, and the
    accelerator type to pin GPU actors/tasks to.

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
    default load-balanced assignment is fine for every order-independent map/scan."""
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
    n_parts = workers if wants_pool else _adaptive_partition_count(sources[sid], plan, workers, hub)
    proj, pred = _scan_pushdown(plan0)
    partitions = partition_descriptors(
        sources[sid], n_parts, projection=proj, predicate=pred, preserve_order=preserve_order
    )

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
            results, gpu_util, gpu_vram = _run_scoped_pool(plan0, partitions, opts, lo, hi, scope)
        elif warm:
            results, gpu_util, gpu_vram = _run_warm_pool(plan0, partitions, opts, lo, hi)
        else:
            results, gpu_util, gpu_vram = _drive_actor_pool(
                plan0, partitions, opts, lo, hi, recovery_policy()
            )
        _record_gpu_feedback(hub, plan, gpu_util, gpu_vram)
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
            return _map_udf_task.options(**{**opts, "num_cpus": shares[idx], **sched}).remote(
                plan0, partitions[idx], workers, engine_config_json(shares[idx]), write_spec, idx
            )

        results = gather_map_results(_launch, len(partitions))

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


def _record_source_rows(hub, source, rows: int) -> None:
    """Persist a run's measured total rows for `source` so the next run's partition count can
    seed from it when the footer count is unknown. Best-effort; never breaks a query."""
    with contextlib.suppress(Exception):
        from batcher.dist.adaptive_sizing import record_partition_rows

        record_partition_rows(_learning_hub(hub), source.identity(), rows)


def _record_actor_pool_reuse(hub, plan0, partitions: int) -> None:
    """Persist how many partitions this inference pool served, so a recurring pipeline can
    right-size its actor pool next run. Best-effort; never breaks a query."""
    with contextlib.suppress(Exception):
        from batcher.dist.adaptive_sizing import record_actor_pool_reuse

        record_actor_pool_reuse(_learning_hub(hub), _pipeline_signature(plan0), partitions)


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
    except Exception:  # pragma: no cover - learning is best-effort
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
    except Exception:  # pragma: no cover - learning is best-effort
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


def _adaptive_partition_count(source, plan, fallback: int, hub=None) -> int:
    """How many tasks to split a map/scan source into — data- and compute-driven.

    `ceil(total_rows x compute_weight / rows_per_cpu)`, clamped to `[1, cluster_cores]`
    and to the number of splits. So a tiny source runs as a few (even one) tasks while a
    large one fans out to ~one task per core — and a per-batch UDF (weight > 1), being
    single-threaded per task, fans out to MORE tasks (the way to parallelize it) rather
    than reserving idle cores on fewer tasks.

    When the source's row total isn't cheaply known from a footer, a *measured* total row
    count learned from a prior run of the same source (`learned_partition_rows`) seeds the
    fan-out instead of the blunt `fallback` worker count; a genuinely-cold source still
    falls back. Partition count only shards rows, so any count is result-identical."""
    import math

    from batcher.config import active_config
    from batcher.core.udf import has_map_batches
    from batcher.dist.adaptive_sizing import learned_partition_rows

    total = _source_total_rows(source)
    if total is None:
        with contextlib.suppress(Exception):
            total = learned_partition_rows(_learning_hub(hub), source.identity())
    if total is None:
        return fallback
    weight = _MAP_COMPUTE_WEIGHT if has_map_batches(plan) else 1.0
    weight *= _learned_weight_factor(plan, hub)
    rows_per_cpu = max(1, active_config().optimizer.target_rows_per_task // 2)
    want = math.ceil((total * weight) / rows_per_cpu)
    want = max(want, _byte_partition_count(source, plan, total, hub))
    n = max(1, min(want, int(_cluster_cores())))
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
    except Exception:  # pragma: no cover - sizing must never break a query
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
    pending = [task.remote(plan0, p) for p in partitions]
    # Collect one finished partition at a time so the driver holds a single partition's
    # output, not the whole result — the bounded-memory way to pull a large scan.
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        out = ray.get(done[0])
        if out:
            yield from out


def _record_gpu_feedback(
    hub, plan: LogicalPlan, gpu_util: float | None, gpu_vram: float | None = None
) -> None:
    """Persist the pipeline's observed GPU utilization *and* peak VRAM for next-run adaptation.

    Utilization adapts `num_gpus`; the peak VRAM fraction adapts how many inference actors
    pack onto one device (`actors_per_gpu_from_learned_vram`). Both keyed by the pipeline's
    stable identity; best-effort (each recorder no-ops on `None`)."""
    if hub is None:
        return
    from batcher.ml.gpu import gpu_feedback_key, record_gpu_peak_vram, record_gpu_utilization

    key = gpu_feedback_key(plan)
    record_gpu_utilization(hub, key, gpu_util)
    record_gpu_peak_vram(hub, key, gpu_vram)


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
    got no pin at all and could land on any node in the cluster."""
    opts: dict = {}
    if num_gpus:
        opts["num_gpus"] = num_gpus
    if resources:
        opts["resources"] = dict(resources)
    if accelerator_type and (num_gpus or resources):
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
    env = current_envelope()
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
            actor = next((a for a in actors if slots[a] > 0), None)
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
                from batcher.dist.executors.ray_runtime.policies import _is_transient_udf_error

                if not _is_transient_udf_error(exc):
                    raise
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
    except Exception:  # pragma: no cover - feedback must never break execution
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
        self._gpu_util_max: float | None = None
        self._gpu_vram_max: float | None = None

    def run(self, partition: dict, idx: int = 0):
        from batcher import core
        from batcher.ml.gpu import sample_gpu_utilization, sample_gpu_vram_fraction

        # A LAZY source over the descriptor: the scan reads its splits (storage) / iterates
        # its shipped batches incrementally, so `stream_linear_chain` overlaps reading chunk
        # k+1 with the GPU forward on chunk k instead of the device idling through an eager
        # whole-partition read (the ~60%->higher-util fix for a scan->GPU inference).
        source = _lazy_partition_source(partition)
        if source is None:
            return None
        out = core.execute_with_udfs(self._plan, [source])
        # Sample GPU load + VRAM right after the forward pass (None on a GPU-less host).
        self._observe_gpu(sample_gpu_utilization(), sample_gpu_vram_fraction())
        if not out or sum(b.num_rows for b in out) == 0:
            return [] if self._write_spec is not None else None
        # Writing stage: this actor writes its own inference output straight to the sink and
        # returns only `WrittenFile` locators. That is what keeps a batch-inference job whose
        # RESULT is larger than the driver (a 2B-row embedding write) from OOMing the driver
        # — the post-inference rows never leave the worker that produced them.
        if self._write_spec is not None:
            return _write_udf_output(out, self._write_spec, idx)
        return out

    def run_split(self, addr: str, ticket):
        """Map one prior-stage bucket fetched in place from `(addr, ticket)`, so a
        resident inference pool is fed directly from upstream output (a co-located
        bucket reads via shared memory / direct memory — no driver round-trip) instead
        of waiting for the driver to hand it a materialized partition."""
        from batcher import core
        from batcher.carbonite.transfer.server import fetch
        from batcher.io.source import InMemorySource
        from batcher.ml.gpu import sample_gpu_utilization, sample_gpu_vram_fraction

        rows = fetch(addr, ticket)
        if not rows:
            return None
        out = core.execute_with_udfs(self._plan, [InMemorySource(rows)])
        self._observe_gpu(sample_gpu_utilization(), sample_gpu_vram_fraction())
        if not out or sum(b.num_rows for b in out) == 0:
            return None
        return out

    def _observe_gpu(self, util: float | None, vram: float | None) -> None:
        """Fold one post-forward GPU sample into this actor's running peaks (util + VRAM)."""
        self._gpu_util_max = _max_opt(self._gpu_util_max, util)
        self._gpu_vram_max = _max_opt(self._gpu_vram_max, vram)

    def gpu_stats(self) -> float | None:
        """The peak GPU utilization this actor observed, or `None` if no GPU."""
        return self._gpu_util_max

    def gpu_vram_stats(self) -> float | None:
        """The peak VRAM fraction this actor observed, or `None` if no GPU — the memory
        twin of `gpu_stats`, sized so the next run packs actors by measured footprint."""
        return self._gpu_vram_max


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
    """
    from batcher.io.sink import SINKS

    table = pa.Table.from_batches(batches)
    sink = SINKS.get(write_spec["fmt"])(**(write_spec.get("sink_kwargs") or {}))
    return sink.write_partitioned(
        table, write_spec["path"], partition_by=write_spec.get("partition_by"), file_index=idx
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

    def _launch(idx):
        workers = max(1, round(shares[idx]))
        return _map_agg_task.options(num_cpus=shares[idx], **sched).remote(
            map_plan, partitions[idx], gk, aj, workers
        )

    partials = gather_map_results(_launch, len(partitions))
    flat = [p for p in partials if p is not None]
    if not flat:
        table = _empty_agg_table(agg)
    else:
        out = nat.combine_finalize(gk, aj, flat)
        table = pa.Table.from_batches([out]) if out is not None else _empty_agg_table(agg)
    return table if not above else _apply_above(above, table)
