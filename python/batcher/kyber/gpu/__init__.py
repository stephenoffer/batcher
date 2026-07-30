"""GPU decisions — Kyber's cost-based accelerator choices, grouped as one family.

`policy` decides *where* a plan runs (GPU vs CPU, single-device vs sharded) and how to size a
GPU inference stage (`num_gpus` / initial `batch_size`); `sizing` is the SELECTION-phase rule
that applies the map-stage sizing to the plan; `adaptive` learns the GPU/CPU crossover from
measured runs so the `where` decision self-corrects to the hardware; `energy` decides the same
questions in watts rather than seconds, which is the binding axis on a power-capped fleet. Kept
in one subpackage so the accelerator policy lives together and the parent `kyber/` stays within
its file budget.
"""

from __future__ import annotations

from batcher.kyber.gpu.adaptive import learned_gpu_min_rows, record_backend_timing
from batcher.kyber.gpu.energy import (
    EnergyAdvice,
    device_energy_advice,
    power_bounded_devices,
    select_device_class,
    stage_joules,
)
from batcher.kyber.gpu.policy import (
    GpuDecision,
    GpuMapParams,
    decide_gpu_backend,
    decide_gpu_map_params,
)
from batcher.kyber.gpu.sizing import size_gpu_map_batches

__all__ = [
    "EnergyAdvice",
    "GpuDecision",
    "GpuMapParams",
    "decide_gpu_backend",
    "decide_gpu_map_params",
    "device_energy_advice",
    "learned_gpu_min_rows",
    "power_bounded_devices",
    "record_backend_timing",
    "select_device_class",
    "size_gpu_map_batches",
    "stage_joules",
]
