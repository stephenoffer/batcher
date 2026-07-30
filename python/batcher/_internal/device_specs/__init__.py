"""Datacenter accelerator specifications — the hardware facts a cluster cannot report.

`accelerators.py` answers "how much VRAM does this model have?" because Ray reports a device
*count* and a *model name* and nothing else. Running inside a GPU datacenter needs more of the
same shape of fact, and for the same reason: nothing in the runtime reports a device's power
draw, its memory bandwidth, the width of its NVLink domain, whether it can be partitioned, or
how fast it reaches host memory. Every one of those decides something real —

* **power** decides energy-aware placement and the tokens-per-joule a datacenter bills on;
* **memory bandwidth** decides whether a scan-shaped stage gains anything from a device;
* **the host link** decides whether it gains anything *at all*, because a relational stage's
  bytes cross that link before a kernel sees them and on PCIe it is slower than host memory;
* **the NVLink domain** bounds how wide a collective can go before it leaves the fast path;
* **MIG partitionability** decides whether a small model can share a device.

This package is that table, keyed by the same `ray.util.accelerators` model names, and it is
the single source of device memory too — `accelerator_memory_bytes` reads it rather than
keeping a second copy.

**The numbers are vendor nameplate figures for the dense tensor path**, without the 2x
structured-sparsity multiplier: peak half precision (BF16 where the generation has it, FP16 on
Turing and older) and FP8 where an FP8 unit exists. They are used as *ratios*, so consistency
of basis matters more than absolute accuracy. The host-link figures are the exception and are
deliberately *effective* rather than theoretical, because a PCIe 5.0 x16 link rates 64 GB/s and
delivers around 50 on a pinned copy — and it is the delivered number that decides a plan.

**Unknown stays unknown.** Where one Ray name covers several configurations the smallest
shipping variant is recorded, and an unrecognized name yields `None` from `device_spec` and a
zero from every scalar accessor, so a caller falls back to whatever default it had.

* `table` — the rows, and the host link per part.
* `accessors` — one function per fact, plus `resolve_device_name`, which maps what a driver
  calls a device onto what this table calls it.
"""

from __future__ import annotations

from batcher._internal.device_specs.accessors import (
    device_arithmetic_intensity,
    device_fp8_tflops,
    device_generation,
    device_half_tflops,
    device_host_link,
    device_host_link_gbps,
    device_idle_watts,
    device_memory_bandwidth_gbps,
    device_mig_slices,
    device_nvlink_domain,
    device_nvlink_gbps,
    device_spec,
    device_tdp_watts,
    device_tflops_per_watt,
    device_vendor,
    devices_by_generation,
    host_transfer_seconds,
    known_device_names,
    rank_devices_by_efficiency,
    resolve_device_name,
)
from batcher._internal.device_specs.table import DeviceSpec

__all__ = [
    "DeviceSpec",
    "device_arithmetic_intensity",
    "device_fp8_tflops",
    "device_generation",
    "device_half_tflops",
    "device_host_link",
    "device_host_link_gbps",
    "device_idle_watts",
    "device_memory_bandwidth_gbps",
    "device_mig_slices",
    "device_nvlink_domain",
    "device_nvlink_gbps",
    "device_spec",
    "device_tdp_watts",
    "device_tflops_per_watt",
    "device_vendor",
    "devices_by_generation",
    "host_transfer_seconds",
    "known_device_names",
    "rank_devices_by_efficiency",
    "resolve_device_name",
]
