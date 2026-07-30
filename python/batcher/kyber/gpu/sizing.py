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
    """Fill `num_gpus` / `batch_size` on an accelerator `map_batches` from its model footprint."""
    if not isinstance(node, MapBatches):
        return None  # the registry dispatches by node type; decline anything else
    if node.model_memory_gb <= 0.0:
        return None  # not a declared accelerator model → nothing to size
    if node.num_gpus > 0.0 and node.batch_size is not None:
        return None  # user pinned both → honor them
    from batcher.kyber.gpu.policy import decide_gpu_map_params

    # A stage that requests a *custom* accelerator resource (TPU / Trainium / Inferentia / Gaudi)
    # carries `num_gpus == 0` and must keep it: assigning a GPU fraction would request a device
    # the accelerator fleet hasn't got, and on a GPU-less fleet that gang never schedules. Its
    # batch size is still worth seeding — from the accelerator's own HBM, recovered from the pinned
    # `accelerator_type`. A GPU stage sizes against the cluster's binding VRAM as before.
    is_non_gpu_accel = node.num_gpus <= 0.0 and bool(node.resources)
    if is_non_gpu_accel:
        from batcher._internal.accelerators import accelerator_memory_bytes

        device_gb = accelerator_memory_bytes(node.accelerator_type) / (1 << 30) or None
    else:
        # The cluster's binding (smallest) device when the topology could report it, else `None`
        # → the policy's local probe. Sizing against `ctx.hardware` rather than the driver's own
        # devices is the point of threading a profile into the optimizer: the driver is routinely
        # a CPU-only head node whose probe finds nothing.
        device_gb = ctx.hardware.gpu_memory_bytes / (1 << 30) or None
    # The width of one input row, so the batch seed charges the input tensor's own VRAM and
    # not only the activation prior. This is the estimator Kyber already threads through the
    # context, so the rule pays nothing extra for it; an estimator that abstains yields `0.0`,
    # which is the previous activation-only budget exactly.
    input_row_bytes = _input_row_bytes(node, ctx)
    params = decide_gpu_map_params(
        node.model_memory_gb,
        node.num_gpus,
        node.batch_size,
        gpu_memory_gb=device_gb,
        assign_num_gpus=not is_non_gpu_accel,
        input_row_bytes=input_row_bytes,
    )
    if params.num_gpus == node.num_gpus and params.batch_size == node.batch_size:
        return None
    ctx.notes.setdefault("gpu_resource_sizing", []).append(params.reason)
    return dataclasses.replace(node, num_gpus=params.num_gpus, batch_size=params.batch_size)


def _input_row_bytes(node: MapBatches, ctx: OptimizerContext) -> float:
    """Estimated Arrow bytes of one row entering `node`, or `0.0` when unknowable.

    The batch a GPU stage dispatches occupies the device twice over — the input rows and the
    activations derived from them — and only the second was ever budgeted. On a decoded image
    or video column the input tensor is the larger of the two by a wide margin, so the seed
    was asking the device for several times the VRAM it has.

    Never raises: a batch-size *seed* is an optimization, and an estimator that abstains must
    leave it at the previous activation-only budget rather than fail the plan.

    Args:
        node: The accelerator `map_batches` being sized.
        ctx: The optimizer context, carrying the shared estimator.

    Returns:
        Bytes per input row, or `0.0` when no estimate is available.
    """
    from batcher._internal.logging import note_suppressed

    try:
        return float(ctx.estimator.row_width(node.input, ctx.config.optimizer.row_bytes))
    except Exception as exc:  # pragma: no cover - sizing must never break a plan
        note_suppressed("kyber", "size gpu batch from input width", exc)
        return 0.0
