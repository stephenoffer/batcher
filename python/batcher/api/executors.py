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

import pyarrow as pa

from batcher._internal.registry import Registry
from batcher.api._join_helpers import _empty_schema
from batcher.api.orchestration import _collect_source_metadata, run_relational
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
            _collect_source_metadata(ctx.hub, sources)
            return table
        return run_relational(plan, sources, ctx, distributed=True)[0]


class UdfExecutor:
    """Orchestrate a pipeline containing `map_batches` (Python/ML UDFs).

    `map_batches` is opaque to Kyber, so the pipeline runs as authored; but the
    scanned inputs still feed the metadata loop, so the *relational* queries that
    follow get sketch-driven cardinality from data this pipeline touched.
    """

    def execute(self, plan: LogicalPlan, sources: list[Source], ctx: ExecutionContext) -> pa.Table:
        from batcher import core

        batches = core.execute_with_udfs(plan, sources)
        schema = batches[0].schema if batches else _empty_schema(ctx.columns)
        table = pa.Table.from_batches(batches, schema=schema)
        _collect_source_metadata(ctx.hub, sources)
        return table


class LocalNativeExecutor:
    """Single-node native execution: Kyber → Carbonite → Core, with feedback."""

    def execute(self, plan: LogicalPlan, sources: list[Source], ctx: ExecutionContext) -> pa.Table:
        return run_relational(plan, sources, ctx, distributed=False)[0]


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
    import os

    from batcher.config import active_config
    from batcher.ml.gpu import (
        gpu_feedback_key,
        gpu_vram_gb,
        load_gpu_utilization,
        recommend_gpu_fraction,
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

    # Cold-start GPU request: VRAM-pack a small model onto a fraction when we know both
    # the model size and the GPU's VRAM (auto-detected); otherwise honor the declared
    # count. A GPU-less driver can't detect VRAM, so packing is skipped (safe).
    base_gpus = requested_gpus
    vram = gpu_vram_gb() if model_gb > 0 and requested_gpus >= 1.0 else None
    if vram:
        base_gpus = recommend_gpu_fraction(model_gb, vram)
    num_gpus = recommend_num_gpus(load_gpu_utilization(hub, gpu_feedback_key(plan)), base_gpus)

    n_tasks = num_workers or (cfg.execution.parallelism or os.cpu_count() or 4)
    return SchedulingEnvelope(
        num_cpus=cfg.execution.cpus_per_task,
        memory_bytes=int(model_gb * 1.5 * (1 << 30)),
        num_gpus=num_gpus,
        n_tasks=max(1, n_tasks),
        credits=cfg.flow_control.default_credits,
        accelerator_type=accelerator_type,
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
