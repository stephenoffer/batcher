"""SELECTION-phase rule — size a GPU inference stage's resources.

Kyber decides *how much GPU* a `map_batches` inference stage reserves and the batch size it
starts at, from the model's declared memory footprint (`model_memory_gb`) vs one GPU's memory
(`distributed.gpu_memory_gb`). This is the "Kyber decides resource bounds → Carbonite/Core
enforce them" hand-off applied to accelerators.

A user-set `num_gpus` (>0) or `batch_size` is always honored; the rule only fills what was left
unset, and only when `model_memory_gb` is known — the signal that the stage is a GPU model (an
unset `num_gpus` defaults to 0/CPU, so without the footprint there is nothing to size). It packs
light models several to a GPU (fraction rounded to a packing quantum → utilization), reserves
whole GPUs for a heavy one (no OOM), and seeds the online `ThroughputController` with a
VRAM-aware batch size instead of a blind constant. The heavy lifting is the pure
`policy.decide_gpu_map_params`; this rule just applies it to the node.
"""

from __future__ import annotations

import dataclasses

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase, RuleCategory
from batcher.plan.logical import LogicalPlan, MapBatches

__all__ = ["size_gpu_map_batches"]


@rule(
    name="size_gpu_map_batches",
    phase=Phase.SELECTION,
    matches=(MapBatches,),
    category=RuleCategory.SELECTION,
)
def size_gpu_map_batches(node: LogicalPlan, ctx: OptimizerContext) -> LogicalPlan | None:
    """Fill `num_gpus` / `batch_size` on a GPU `map_batches` from its model footprint."""
    assert isinstance(node, MapBatches)
    if node.model_memory_gb <= 0.0:
        return None  # not a declared GPU model → nothing to size
    if node.num_gpus > 0.0 and node.batch_size is not None:
        return None  # user pinned both → honor them
    from batcher.kyber.gpu.policy import decide_gpu_map_params

    params = decide_gpu_map_params(node.model_memory_gb, node.num_gpus, node.batch_size)
    if params.num_gpus == node.num_gpus and params.batch_size == node.batch_size:
        return None
    ctx.notes.setdefault("gpu_resource_sizing", []).append(params.reason)
    return dataclasses.replace(node, num_gpus=params.num_gpus, batch_size=params.batch_size)
