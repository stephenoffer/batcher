"""Carbonite's decisions about the device a worker is already running on.

The *facts* — which devices this process may use, how full each is, what a given
out-of-memory means — are hardware readings and live in
`batcher._internal.hardware.devices`, so the ML inference path can consult them without
importing a subsystem. What is left here is the part that is a decision: how PyTorch's
caching allocator should be configured for this worker, sized from the same VRAM headroom
the admission pool holds back and divided by how many actors share the board — and which of
the process's *other* device allocators should be folded into the one RMM governs, because a
RAPIDS worker has three of them and only one was ever configured.
"""

from __future__ import annotations

from batcher.carbonite.accel.device.rmm_adopt import adopt_rmm_everywhere, install_rmm_resource
from batcher.carbonite.accel.device.torch_alloc import (
    TorchAllocatorPlan,
    configure_torch_allocator,
    plan_torch_allocator,
    reset_torch_allocator_state,
    torch_allocator_state,
)

__all__ = [
    "TorchAllocatorPlan",
    "adopt_rmm_everywhere",
    "configure_torch_allocator",
    "install_rmm_resource",
    "plan_torch_allocator",
    "reset_torch_allocator_state",
    "torch_allocator_state",
]
