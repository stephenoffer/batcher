"""GPU-vs-CPU backend policy — Kyber's cost-based decision of *where* a plan runs.

Choosing GPU vs CPU (and, on GPU, single-device vs sharded across the cluster) is an
optimization decision, so it lives in Kyber, not in the executor: **Kyber decides, Core
executes.** The decision is a pure function of the plan's estimated cardinality and width
(via Kyber's `CardinalityEstimator`), the live GPU count, and the per-GPU memory budget — no
execution and no I/O beyond the footer stats the estimator already reads.

It exists because a GPU is not free: host<->device transfer, kernel launch, and the first-touch
cuDF import are a fixed overhead that a small query never amortizes, and one GPU's memory is
finite. A single `backend="gpu"` flag that ships *any* matching shape to *one* GPU therefore
mispredicts two whole regimes — tiny inputs (CPU wins) and inputs larger than one GPU (a
single-dispatch OOMs). `decide_gpu_backend` covers those: below `gpu_min_rows` → CPU; fits one
GPU → single-dispatch; exceeds one GPU but fits the cluster's GPUs (and is shardable) →
distributed; exceeds them all → the spillable CPU engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from batcher.config import active_config

if TYPE_CHECKING:
    from batcher.io.source import Source
    from batcher.metadata import MetadataHub
    from batcher.plan.logical import LogicalPlan

__all__ = ["GpuDecision", "GpuMapParams", "decide_gpu_backend", "decide_gpu_map_params"]

# The GPU-packing quanta a model's memory fraction is rounded UP to, so Ray can co-locate
# several light inference stages on one GPU (a 3 GB model on a 12 GB GPU → 0.25 → 4 per GPU)
# instead of each wasting a whole device. A model larger than one GPU reserves whole GPUs.
_PACK_QUANTA = (0.25, 0.5, 1.0)


@dataclass(frozen=True, slots=True)
class GpuDecision:
    """Kyber's verdict for one plan: whether to use the GPU, and if so whether to shard it.

    `distributed` is only meaningful when `use_gpu` is True: True means the working set exceeds
    one GPU's memory so the run must fan out across GPUs (the mergeable distributed aggregate);
    False means a single-device dispatch fits. `reason` is a short human string for the decision
    log / `explain()`."""

    use_gpu: bool
    distributed: bool
    reason: str


def _estimate(plan: LogicalPlan, sources: list[Source], hub: MetadataHub | None):
    """`(rows, working_set_gb)` estimated by Kyber's cardinality model, or `(None, None)` when
    the size is unknown (an estimator failure or an unbounded source)."""
    from batcher.kyber import load_learned_stats
    from batcher.kyber.cardinality import CardinalityEstimator

    try:
        learned = load_learned_stats(hub) if hub is not None else None
        est = CardinalityEstimator(sources=sources, learned=learned)
        rows = int(est.estimate(plan).rows)
        width = est.row_width(plan, active_config().optimizer.row_bytes)
    except Exception:
        return None, None
    if rows <= 0:
        return 0, 0.0
    return rows, rows * max(width, 1) / 1e9


def decide_gpu_backend(
    plan: LogicalPlan,
    sources: list[Source],
    hub: MetadataHub | None = None,
    *,
    gpu_count: int,
    force: bool = False,
) -> GpuDecision:
    """Decide whether `plan` should run on the GPU, and single-device vs sharded.

    `gpu_count` is the live cluster's GPU count (0 → always CPU). `force=True` (an explicit
    `backend="gpu"`) honors the user past the small-input threshold but still routes by memory —
    so even a forced request avoids a single-dispatch OOM on data larger than one GPU. The
    memory budget per GPU is `distributed.gpu_memory_gb`; the whole-cluster budget is that times
    `gpu_count`."""
    if gpu_count < 1:
        return GpuDecision(False, False, "no GPU on the cluster")

    dc = active_config().distributed
    rows, ws_gb = _estimate(plan, sources, hub)

    # Unknown size: honor a forced request (single-dispatch, with its own CPU fallback on OOM);
    # otherwise stay on the CPU engine rather than gamble the GPU overhead on an unknown input.
    if rows is None:
        return (
            GpuDecision(True, False, "forced; size unknown")
            if force
            else GpuDecision(False, False, "size unknown; GPU overhead not justified")
        )

    if not force and rows < dc.gpu_min_rows:
        return GpuDecision(
            False, False, f"{rows} rows < gpu_min_rows={dc.gpu_min_rows}: CPU wins on overhead"
        )

    one_gpu_gb = max(dc.gpu_memory_gb, 1e-9)
    if ws_gb <= one_gpu_gb:
        return GpuDecision(True, False, f"~{ws_gb:.1f}GB fits one GPU ({one_gpu_gb:.0f}GB)")
    if ws_gb <= one_gpu_gb * gpu_count:
        return GpuDecision(
            True, True, f"~{ws_gb:.1f}GB exceeds one GPU: shard across {gpu_count} GPUs"
        )
    return GpuDecision(
        False, False, f"~{ws_gb:.1f}GB exceeds all {gpu_count} GPUs: CPU engine (spillable)"
    )


@dataclass(frozen=True, slots=True)
class GpuMapParams:
    """Kyber's resource sizing for one GPU `map_batches` (inference) stage: how much of a GPU to
    reserve per worker and the initial batch size to seed the online throughput controller."""

    num_gpus: float
    batch_size: int | None
    reason: str


def decide_gpu_map_params(
    model_memory_gb: float,
    num_gpus: float,
    batch_size: int | None,
) -> GpuMapParams:
    """Size a GPU inference stage from the model's memory footprint vs one GPU's memory.

    A user-set `num_gpus` (>0) or `batch_size` is always honored — this only *fills* what the
    user left unset, and only when `model_memory_gb` is known (the signal that the stage is a GPU
    model, since an unset `num_gpus` defaults to 0/CPU). The GPU-fraction packs light models
    (fraction rounded up to a packing quantum, so several share a device → higher utilization)
    and reserves whole GPUs for a model larger than one. The batch-size seed spends the VRAM left
    after the model on activations at `gpu_activation_bytes_per_row`, clamped — the online
    `ThroughputController` refines it from measured throughput/VRAM, but a memory-aware start
    beats a fixed 256 (too small for a light model, an instant OOM for a heavy one)."""
    dc = active_config().distributed
    gpu_gb = max(dc.gpu_memory_gb, 1e-9)
    cap = 0.85  # leave VRAM headroom for activations + fragmentation (the guides' ~80-85%)

    if model_memory_gb <= 0.0:
        return GpuMapParams(num_gpus, batch_size, "model memory unknown; left as given")

    out_gpus = num_gpus
    if num_gpus <= 0.0:  # user left it unset → decide the packing fraction
        frac = model_memory_gb / (gpu_gb * cap)
        if frac <= 1.0:
            out_gpus = next((q for q in _PACK_QUANTA if q >= frac), 1.0)
        else:
            out_gpus = float(math.ceil(frac))

    out_bs = batch_size
    if batch_size is None:  # seed the throughput controller from the VRAM headroom
        per_gpu = out_gpus if out_gpus >= 1.0 else 1.0  # a packed fraction still sees one device
        headroom_gb = max(gpu_gb * per_gpu * cap - model_memory_gb, gpu_gb * 0.05)
        act = max(dc.gpu_activation_bytes_per_row, 1)
        out_bs = int(min(max(headroom_gb * 1e9 / act, 1), 65_536))
    return GpuMapParams(
        out_gpus,
        out_bs,
        f"model {model_memory_gb:.1f}GB → num_gpus={out_gpus}, batch_size={out_bs}",
    )
