"""Making this process's device work legible to the tools that measure devices.

Everything under `hardware` reads *from* the machine. This package writes *to* it: the markers
an external profiler needs to attribute a kernel to the operator that issued it, and the device
events needed to time that kernel at all.

* `nvtx` — the profiler-annotation shim, over the `nvtx` package, `torch.cuda.nvtx` (which is
  ROCTX on a ROCm build, so AMD is covered by the same call), and CuPy's binding. A no-op when
  none is present, which is the normal case outside a profiling run.
* `ranges` — the brackets callers actually use: an operator range for a capture, and a CUDA
  event pair for measuring device time, which wall-clock cannot do across an asynchronous
  launch.

Off unless `accelerator.profiling` is set, and free when off.

A neutral utility: any layer may import `_internal`.
"""

from __future__ import annotations

from batcher._internal.instrument.nvtx import (
    device_range,
    nvtx_backend,
    pop_range,
    push_range,
    range_decorator,
    reset_nvtx_backend,
)
from batcher._internal.instrument.ranges import (
    DeviceTiming,
    operator_range,
    profiling_enabled,
    time_device_work,
)

__all__ = [
    "DeviceTiming",
    "device_range",
    "nvtx_backend",
    "operator_range",
    "pop_range",
    "profiling_enabled",
    "push_range",
    "range_decorator",
    "reset_nvtx_backend",
    "time_device_work",
]
