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

from dataclasses import dataclass
from typing import TYPE_CHECKING

from batcher.config import active_config

if TYPE_CHECKING:
    from batcher.io.source import Source
    from batcher.metadata import MetadataHub
    from batcher.plan.logical import LogicalPlan

__all__ = ["GpuDecision", "decide_gpu_backend"]


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
