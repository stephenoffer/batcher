"""GPU decisions — Kyber's cost-based accelerator choices, grouped as one family.

`policy` decides *where* a plan runs (GPU vs CPU, single-device vs sharded) and how to size a
GPU inference stage (`num_gpus` / initial `batch_size`); `sizing` is the SELECTION-phase rule
that applies the map-stage sizing to the plan. Kept in one subpackage so the accelerator policy
lives together and the parent `kyber/` stays within its file budget.
"""

from __future__ import annotations

from batcher.kyber.gpu.policy import (
    GpuDecision,
    GpuMapParams,
    decide_gpu_backend,
    decide_gpu_map_params,
)
from batcher.kyber.gpu.sizing import size_gpu_map_batches

__all__ = [
    "GpuDecision",
    "GpuMapParams",
    "decide_gpu_backend",
    "decide_gpu_map_params",
    "size_gpu_map_batches",
]
