"""Execution strategies and their registry (the conductor's wiring).

`core.base` defines the neutral `Executor` Protocol + `ExecutionContext`; this
module supplies the concrete strategies and the registry that selects between
them. It lives in `api` — not `core` — because the strategies cross subsystem
boundaries the independence contract forbids Core from crossing: the native path
orchestrates Kyber + Carbonite, and the distributed path lives in `dist`. The
conductor is the one layer allowed to import all of them.

Selection (`select`) encodes exactly the dispatch that previously lived as an
if/elif/else in `terminal._collect`: distributed when requested, else the UDF
orchestrator when the plan contains `map_batches`, else the single-node native
engine. New tiers (morsel/JIT/LLVM/GPU) register here instead of growing that
chain.
"""

from __future__ import annotations

from collections.abc import Callable

import pyarrow as pa

from batcher._internal.hardware import available_cpu_count
from batcher._internal.logging import note_suppressed
from batcher._internal.registry import Registry
from batcher.api._join_helpers import _empty_result_schema
from batcher.api.orchestration import run_relational
from batcher.api.terminal._metadata import collect_source_metadata
from batcher.core import ExecutionContext, Executor
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan

__all__ = ["DistributedExecutor", "LocalNativeExecutor", "UdfExecutor", "select"]


class DistributedExecutor:
    """Run a plan across Ray workers (the `distributed=True` path).

    Same contract loop as single-node (via `run_relational`): Kyber optimizes,
    Carbonite turns the per-operator bounds into a scheduling envelope, Core
    executes across workers, and metadata is recorded back. The opaque
    `map_batches` pipeline is the one shape Kyber can't size relationally, so it
    gets a GPU-aware envelope built straight from the plan instead.
    """

    def execute(self, plan: LogicalPlan, sources: list[Source], ctx: ExecutionContext) -> pa.Table:
        from batcher.core.udf import has_map_batches

        if has_map_batches(plan):
            from batcher import dist

            # map/inference pipeline: Kyber doesn't size it relationally, so build a
            # GPU-aware envelope straight from the plan's `map_batches` resource tags,
            # adapted by any GPU utilization measured on a previous run.
            envelope = _map_scheduling_envelope(plan, ctx.num_workers, ctx.hub)
            table = dist.execute_distributed(
                plan,
                sources,
                ctx.num_workers,
                transport=ctx.transport,
                envelope=envelope,
                hub=ctx.hub,
            )
            collect_source_metadata(ctx.hub, sources)
            record_udf_cardinality(ctx.hub, plan, table.num_rows)
            return table
        # Relational distributed result — deterministic and identical to single-node,
        # so it shares the same result cache (`Dataset.cache()`).
        return _cached_or_run(
            plan, sources, ctx, lambda: run_relational(plan, sources, ctx, distributed=True)[0]
        )


class UdfExecutor:
    """Orchestrate a pipeline containing `map_batches` (Python/ML UDFs).

    `map_batches` is opaque to Kyber, so the pipeline runs as authored; but the
    scanned inputs still feed the metadata loop, so the *relational* queries that
    follow get sketch-driven cardinality from data this pipeline touched.
    """

    def execute(self, plan: LogicalPlan, sources: list[Source], ctx: ExecutionContext) -> pa.Table:
        from batcher import core, kyber

        # Kyber decides which columns each source must supply; Core executes with them. A UDF
        # that declared `input_columns` prunes the scan to those (plus what the plan above
        # needs); an undeclared one still reads everything, because the `fn` is a black box.
        # Hand the profiling sink down when this run is being measured, so an ML pipeline
        # gets a per-stage `stats()` tree instead of the "no per-operator metrics" refusal.
        recorder = getattr(ctx.profile, "stage_recorder", None) if ctx.profile else None
        batches = core.execute_with_udfs(
            plan, sources, kyber.required_columns_per_source(plan), recorder=recorder
        )
        schema = batches[0].schema if batches else _empty_result_schema(plan, ctx.columns)
        table = pa.Table.from_batches(batches, schema=schema)
        collect_source_metadata(ctx.hub, sources)
        record_udf_cardinality(ctx.hub, plan, table.num_rows)
        return table


class LocalNativeExecutor:
    """Single-node native execution: Kyber → Carbonite → Core, with feedback."""

    def execute(self, plan: LogicalPlan, sources: list[Source], ctx: ExecutionContext) -> pa.Table:
        return _cached_or_run(
            plan, sources, ctx, lambda: run_relational(plan, sources, ctx, distributed=False)[0]
        )


def record_udf_cardinality(hub, plan: LogicalPlan, out_rows: int) -> None:
    """Teach the estimator what this UDF pipeline's fan-out actually was.

    The estimator already *expects* to learn this. `MapBatches` is in its `_CORRECTABLE`
    set, keyed by UDF identity, with the reasoning spelled out there: a UDF may filter,
    explode, or pass rows through 1:1, and which one is a property of the *code*, not the
    plan, so no structural rule can derive it. Absent a measurement it assumes 1:1 and tags
    the result `Provenance.DEFAULT`.

    Nothing was supplying the measurement. Both UDF routes — this single-node orchestrator
    and the distributed `map_batches` branch — bypass `run_relational`, which is the one
    place `record_cardinality_outcome` is called, so a pipeline's measured output count was
    thrown away on every run. The correction machinery was in place and had nothing to
    correct from, which is the same shape as a rule that is written, tested, and never
    reached: indistinguishable from absent.

    What it costs to be wrong is not small on this shape. An inference stage that explodes
    one row into N detections, or a classifier that keeps two percent, is mis-sized by
    orders of magnitude — and everything downstream is sized from it: a join after the
    inference, the partition count, the admission envelope, the output file sizing.

    Only the *cardinality* loop is closed here, not `record_selectivity`: that one attributes
    a measured ratio to a `Filter` over a scan, and a UDF pipeline's output count says
    nothing about any predicate in it.

    Best-effort, like every other write on this path — a learning failure must not break a
    query that has already produced its answer.

    Args:
        hub: The metadata hub; a `None` hub is a no-op.
        plan: The pre-optimization plan, whose signature carries the UDF's identity.
        out_rows: Rows the pipeline actually produced.
    """
    from batcher import kyber

    try:
        kyber.record_execution(hub, plan, out_rows)
    except Exception as exc:  # pragma: no cover - learning must never break a completed run
        note_suppressed("api", "record UDF pipeline cardinality", exc)


def _cached_or_run(
    plan: LogicalPlan,
    sources: list[Source],
    ctx: ExecutionContext,
    run: Callable[[], pa.Table],
) -> pa.Table:
    """Serve `plan`'s result from the process result cache, else compute it via `run`
    and store it — the shared `Dataset.cache()` path for the relational executors.

    A no-op wrapper around `run()` unless `ctx.cache`. Only relational results are
    cached (the UDF path is opaque to Kyber and may be non-deterministic, so it never
    routes here); the key is shared across the single-node and distributed paths
    (mergeable algebra makes their results identical), so a result cached one way is
    served the other. An oversized result is simply not cached (the store's size guard).
    """
    if not ctx.cache:
        return run()
    import time

    from batcher import carbonite

    cache = carbonite.result_cache()
    key = _result_cache_key(plan, sources)
    hit = cache.get(key)
    if hit is not None:
        return hit
    started = time.perf_counter()
    table = run()
    cost = time.perf_counter() - started
    # Pin the source objects as the entry's keep-alive: the key uses their object
    # identity, so holding them alive makes a collision with a reused id impossible.
    # `cost` (recompute seconds) lets eviction keep expensive results over cheap ones.
    cache.put(key, table, keepalive=tuple(sources), cost=cost)
    return table


def _result_cache_key(plan: LogicalPlan, sources: list[Source]) -> str:
    """A correctness-safe result-cache key: plan signature, inputs, tenant, and viewer.

    Object identity (`id`) distinguishes inputs that share a shape — `Source.identity()`
    is shape-based for in-memory data (``mem:schema:rows``), so two *different* in-memory
    datasets with the same shape would otherwise collide and return each other's result.
    The cache pins the source objects (keep-alive) for the entry's lifetime, so the
    `id` cannot be reused by another object while the entry is live.

    The cache is **process-global**, so two workloads in one process share it. Without the
    last two components they would share *results*:

    - **Tenant.** Two tenants issuing the same query over the same path collided by
      construction, and the second got the first's rows.
    - **Viewer.** A governed read is rewritten by `enforce` before it reaches here, so a
      masked and an unmasked read *happened* to produce different plan signatures. That is
      an accident of the rewrite, not a guarantee — a future rewrite that normalized the
      masked form would silently start serving unmasked rows to a principal who may not
      see them. Folding the catalog and principal in makes it a guarantee instead.

    Both components are empty outside a `tenant()`/`security()` block, so an un-tenanted,
    ungoverned deployment produces byte-identical keys to before.
    """
    from batcher.kyber.signature import plan_signature

    inputs = "|".join(f"{id(s)}:{s.identity()}" for s in sources)
    return f"{plan_signature(plan)}|{inputs}|{_cache_scope()}"


def _cache_scope() -> str:
    """The tenant and viewer this result belongs to, as a cache-key component.

    Empty when neither a tenant nor a security context is active, which is what keeps this
    from perturbing a deployment that uses neither.
    """
    from batcher.api.security import current_security
    from batcher.config import active_config

    tenant_id = active_config().tenant.tenant_id
    context = current_security()
    if context is None:
        return f"t={tenant_id}" if tenant_id else ""
    # The catalog is part of the identity too: the same principal under a *different*
    # policy may legitimately see different rows, so a cached result is only reusable
    # under the catalog that produced it.
    viewer = f"{context.principal.name}:{sorted(context.principal.roles)}:{id(context.catalog)}"
    return f"t={tenant_id}|v={viewer}"


def _gpu_device_count() -> int:
    """Devices the cluster reports, or this host's when there is no cluster.

    Deliberately *not* fused with the VRAM read below into one topology call. That fusion was
    tried and was wrong twice over: the binding-VRAM figure has its own accessor
    (`cluster_gpu_memory_gb`) whose semantics are "the smallest device in the fleet", and
    routing around it packed a fraction derived from the driver's device onto a smaller
    worker — the exact OOM that accessor exists to prevent. The cost the fusion was chasing is
    not there either: topology reads inside a scheduling phase are served from a snapshot.

    `0` means "no GPU visible", which makes the fan-out clamp a no-op rather than a refusal: a
    stage that declared `num_gpus` on a fleet whose inventory cannot be read keeps what it
    asked for.
    """
    from batcher._internal.hardware import gpu_inventory

    try:
        from batcher.dist.executors.ray_runtime.scaling import cluster_hardware_profile

        profile = cluster_hardware_profile()
        if profile is not None and profile.gpu_count > 0:
            return profile.gpu_count
    except Exception:  # pragma: no cover - an unreadable cluster falls back to this host
        pass
    return len(gpu_inventory())


def _map_scheduling_envelope(plan: LogicalPlan, num_workers: int | None, hub):
    """Build a GPU- and memory-aware `SchedulingEnvelope` for a `map_batches` pipeline.

    Kyber doesn't size map pipelines relationally, so the conductor builds the envelope
    from the plan's `map_batches` resource tags (the one layer allowed to read both
    `ml.gpu` and Carbonite):

    * **`num_gpus`** — the largest declared request, *VRAM-packed* when the stage gives
      a `model_memory_gb` and the cluster's `gpu_memory_gb` is known (several small
      models share one GPU via a fraction; a large model gets a whole device), then
      *adapted* by the GPU utilization measured on a prior run.
    * **`memory_bytes`** — the model's host footprint (1.5x for weights + activations)
      so Ray reserves it per worker and won't pack more model-loading actors onto a
      node than fit — the OOM protection Carbonite gives the relational path.
    * **`accelerator_type`** — pins GPU actors to a device model when requested.
    """
    from batcher.config import active_config
    from batcher.dist.executors.ray_runtime.accelerators import cluster_gpu_memory_gb
    from batcher.ml.gpu import (
        actors_per_gpu_from_learned_vram,
        gpu_feedback_key,
        gpu_vram_gb,
        load_gpu_peak_vram,
        load_gpu_utilization,
        recommend_gpu_fraction,
        recommend_inflight_depth,
        recommend_num_gpus,
    )
    from batcher.plan.resource import SchedulingEnvelope
    from batcher.plan.visitor import walk

    cfg = active_config()
    stages = [n for n in walk(plan) if getattr(n, "num_gpus", 0.0) or hasattr(n, "model_memory_gb")]
    requested_gpus = max((getattr(n, "num_gpus", 0.0) for n in stages), default=0.0)
    model_gb = max((getattr(n, "model_memory_gb", 0.0) for n in stages), default=0.0)
    accelerator_type = next(
        (n.accelerator_type for n in stages if getattr(n, "accelerator_type", None)), None
    )
    # Custom accelerator resources (TPU / neuron_cores / HPU / an on-prem resource), taken
    # from the first stage that names any — the same first-wins rule as `accelerator_type`.
    resources = next((tuple(n.resources) for n in stages if getattr(n, "resources", ())), ())
    # Auto-pin a large-model GPU stage the user left unpinned to a device class that fits it, so
    # a heterogeneous cluster never schedules it onto a GPU too small to hold the model (an OOM on
    # load). A no-op on a homogeneous cluster, when every device fits, or when the user pinned a
    # type. GPU only — a custom-accelerator stage is placed by its `resources`, not a GPU model.
    if accelerator_type is None and requested_gpus > 0 and model_gb > 0:
        from batcher.dist.executors.ray_runtime.accelerators import recommend_accelerator_type

        accelerator_type = recommend_accelerator_type(model_gb)

    # Cold-start GPU request: VRAM-pack a small model onto a fraction when we know both
    # the model size and the GPU's VRAM (auto-detected); otherwise honor the declared
    # count. A GPU-less driver can't detect VRAM, so packing is skipped (safe).
    #
    # The CLUSTER's binding (smallest) device first, and only then this process's own. A
    # single fraction is applied to every actor in the fleet, so deriving it from the driver's
    # device is an OOM on any mixed fleet: a 0.25 computed against an 80 GB A100 packs four
    # actors onto a 16 GB T4. The cluster figure is the smallest device precisely so the
    # fraction is valid on every node the actor might land on.
    base_gpus = requested_gpus
    vram = None
    if model_gb > 0 and requested_gpus >= 1.0:
        vram = cluster_gpu_memory_gb() or gpu_vram_gb()
    if vram:
        base_gpus = recommend_gpu_fraction(model_gb, vram)
    key = gpu_feedback_key(plan)
    util = load_gpu_utilization(hub, key)
    num_gpus = recommend_num_gpus(util, base_gpus)
    # Refine packing from the MEASURED peak VRAM a prior run of this pipeline actually used,
    # not just the declared model size: if the model really consumed more VRAM than declared,
    # pack fewer actors per GPU (a larger num_gpus fraction) so it doesn't OOM. `max` means
    # the measurement only ever tightens toward safety — it never packs looser than declared,
    # so this can prevent an OOM but never cause one. Result-invariant (a scheduling hint).
    peak_vram = load_gpu_peak_vram(hub, key)
    packed = actors_per_gpu_from_learned_vram(peak_vram)
    if packed is not None and num_gpus > 0:
        num_gpus = min(1.0, max(num_gpus, round(1.0 / packed, 2)))

    # Per-actor submit-ahead depth: raise it from a prior low-utilization measurement so a
    # starved GPU is kept fed across the dispatch/gather round-trip (the ml layer owns the
    # heuristic; `dist` only turns the number into pipeline slots). Adaptation only ever
    # increases the configured floor, so a first run is unchanged — and a VRAM-tight pipeline
    # (learned peak VRAM) keeps the shallow depth so deep submission can't OOM the device.
    dc = cfg.distributed
    inflight_depth = (
        recommend_inflight_depth(util, dc.map_inflight_depth, peak_vram)
        if dc.map_inflight_adaptive
        else max(1, dc.map_inflight_depth)
    )

    n_tasks = num_workers or (cfg.execution.parallelism or available_cpu_count())
    # A GPU stage's fan-out is bounded by devices, not by cores. Asking for one actor per CPU
    # on an eight-GPU cluster leaves most of them holding a GPU request the cluster cannot
    # satisfy — pending, not failing, which is the shape that looks like a hang. Carbonite
    # owns that clamp (it protects), and applies three ceilings at once: the devices that
    # exist, the devices the power budget can run, and the devices that are healthy enough to
    # schedule on. Only the first is on by default, so an unbudgeted fleet with health checking
    # off gets exactly the inventory clamp and nothing else.
    if num_gpus > 0:
        from batcher.carbonite.policies.scheduling import DefaultSchedulingPolicy

        n_tasks = DefaultSchedulingPolicy.gpu_envelope(
            num_gpus=num_gpus,
            n_tasks=n_tasks,
            gpu_count=_gpu_device_count(),
            accelerator_type=accelerator_type,
        ).n_tasks
    # A CPU-only map stage (no GPU) is usually IO/decode-bound preprocessing — request
    # a fractional CPU so more actors pack per node, mirroring the GPU-fraction packing
    # above. A GPU stage keeps a full CPU (the GPU is the binding resource there).
    num_cpus = cfg.execution.cpus_per_task if num_gpus > 0 else cfg.execution.cpu_share_io
    return SchedulingEnvelope(
        num_cpus=num_cpus,
        memory_bytes=int(model_gb * 1.5 * (1 << 30)),
        num_gpus=num_gpus,
        n_tasks=max(1, n_tasks),
        credits=cfg.flow_control.default_credits,
        accelerator_type=accelerator_type,
        resources=resources,
        inflight_depth=inflight_depth,
    )


_REGISTRY: Registry[Executor] = Registry("executor")
_REGISTRY.add("local", LocalNativeExecutor())
_REGISTRY.add("udf", UdfExecutor())
_REGISTRY.add("distributed", DistributedExecutor())


def select(plan: LogicalPlan, *, distributed: bool) -> Executor:
    """Choose the execution strategy for `plan`, mirroring the prior dispatch.

    Distributed when requested; otherwise the UDF orchestrator for plans with
    `map_batches`; otherwise the single-node native engine.
    """
    from batcher import core

    if distributed:
        return _REGISTRY.get("distributed")
    if core.has_map_batches(plan):
        return _REGISTRY.get("udf")
    return _REGISTRY.get("local")
