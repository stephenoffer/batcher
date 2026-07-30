"""Effective hardware detection — what this process's machine really is and really allows.

Every API the operating system offers describes the *host*, and inside a container the host is
not what the process gets. A Kubernetes pod sees every core and every byte of the node it
landed on while being throttled to a slice of both. This package resolves the difference, and
goes further: it describes the machine in enough detail that the engine can adapt to it rather
than assume it, because the hardware Batcher runs on spans small ARM containers to many-socket
NUMA hosts with accelerators, and no constant is right across that range.

Organized by the question each module answers:

* `cgroup` — the container's own limits, and the kernel's pressure and throttling counters.
* `cpu` — the effective core budget, and how much of it something else is taking.
* `cache` — the cache hierarchy every blocking and morsel-sizing decision is measured against.
* `memory` — the memory ceiling, page geometry, and whether swap exists.
* `topology` — NUMA nodes and SMT siblings: which cores are actually independent.
* `isa` — CPU identity and vector width.
* `storage` — the block device behind a directory, and what spilling to it will cost.
* `profile` — one assembled record of all of it, plus the fingerprint that names this
  machine class so learned parameters do not blend across unlike hardware.

Device *inventory* — what accelerator is attached and how to reach it — lives one module over
in `accelerators`, beside the model-to-VRAM table it belongs with, and is re-exported here so
every existing caller and probe-reset hook keeps its import path.

A neutral utility: any layer may import `_internal`.
"""

from __future__ import annotations

from batcher._internal.accelerators import (
    accelerator_backend,
    gpu_devices_absent,
    gpu_inventory,
)
from batcher._internal.hardware.cache import (
    cache_hierarchy,
    l3_cache_bytes,
)
from batcher._internal.hardware.cgroup import (
    cgroup_pressure,
    cgroup_v2_dirs,
    read_cgroup_bytes,
)
from batcher._internal.hardware.cpu import (
    INFERENCE_INFLIGHT_DEPTH_MAX,
    available_cpu_count,
    cpu_contention,
    cpu_oversubscription,
    process_start_method_context,
)
from batcher._internal.hardware.isa import cpu_features, cpu_model_name, cpu_vendor, simd_width_bits
from batcher._internal.hardware.memory import (
    machine_memory_bytes,
    page_size_bytes,
)
from batcher._internal.hardware.nvml import (
    DeviceTelemetry,
    device_telemetry,
    nvml_available,
    total_power_watts,
)
from batcher._internal.hardware.probes import reset_hardware_probes
from batcher._internal.hardware.profile import HardwareProfile, fingerprint, hardware_profile
from batcher._internal.hardware.storage import (
    device_class,
)
from batcher._internal.hardware.topology import (
    cpus_per_numa_node,
    numa_node_count,
    physical_core_count,
)

__all__ = [
    "INFERENCE_INFLIGHT_DEPTH_MAX",
    "DeviceTelemetry",
    "HardwareProfile",
    "accelerator_backend",
    "available_cpu_count",
    "cache_hierarchy",
    "cgroup_pressure",
    "cgroup_v2_dirs",
    "cpu_contention",
    "cpu_features",
    "cpu_model_name",
    "cpu_oversubscription",
    "cpu_vendor",
    "cpus_per_numa_node",
    "device_class",
    "device_telemetry",
    "fingerprint",
    "gpu_devices_absent",
    "gpu_inventory",
    "hardware_profile",
    "l3_cache_bytes",
    "machine_memory_bytes",
    "numa_node_count",
    "nvml_available",
    "page_size_bytes",
    "physical_core_count",
    "process_start_method_context",
    "read_cgroup_bytes",
    "reset_hardware_probes",
    "simd_width_bits",
    "total_power_watts",
]
