"""This process's own accelerators: which ones are its, how full they are, what an OOM means.

`hardware.nvml` reports what every device on the host is doing. This narrower package answers
the questions a *worker* asks about the device it is running on, and it sits in the neutral
layer because both the ML inference path and Carbonite need the same answers — Carbonite to
decide, `ml` to measure — and neither may import the other.

* `scope` — which physical devices this process may use (honoring the UUID and MIG forms a
  scheduler pins with, not just ordinals) and how much is really free on each, including what
  a co-tenant holds.
* `torch_memory` — reading and configuring PyTorch's caching allocator, whose fragmentation
  is what makes a stage die at 60% VRAM reporting plenty free.
* `oom` — telling the three device out-of-memory failures apart, and releasing memory in the
  order that actually recovers it.

Every entry point degrades to `None`/empty rather than raising, so a CPU-only host, a
container with no driver mounted, and a build without `pynvml` all leave callers on whatever
default they had.
"""

from __future__ import annotations

from batcher._internal.hardware.devices.oom import (
    OomKind,
    OomVerdict,
    classify_oom,
    is_device_oom,
    release_device_cache,
)
from batcher._internal.hardware.devices.scope import (
    VISIBLE_DEVICE_ENVS,
    DeviceScope,
    current_physical_index,
    device_free_bytes,
    device_scope,
    min_visible_capacity_bytes,
    visible_device_indices,
)
from batcher._internal.hardware.devices.torch_memory import (
    ALLOC_CONF_ENV,
    FRAGMENTATION_THRESHOLD,
    allocator_initialized,
    fragmentation_ratio,
    set_alloc_conf,
    set_memory_fraction,
)

__all__ = [
    "ALLOC_CONF_ENV",
    "FRAGMENTATION_THRESHOLD",
    "VISIBLE_DEVICE_ENVS",
    "DeviceScope",
    "OomKind",
    "OomVerdict",
    "allocator_initialized",
    "classify_oom",
    "current_physical_index",
    "device_free_bytes",
    "device_scope",
    "fragmentation_ratio",
    "is_device_oom",
    "min_visible_capacity_bytes",
    "release_device_cache",
    "set_alloc_conf",
    "set_memory_fraction",
    "visible_device_indices",
]
