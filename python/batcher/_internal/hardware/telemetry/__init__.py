"""Everything a running accelerator will tell you about itself, beyond the five obvious numbers.

`hardware.nvml` reads the five figures every tool reads: power, temperature, SM utilization,
memory residency, ECC. They are the right five to start with and they are nowhere near enough to
diagnose a GPU stage, because none of them can distinguish a device that is the bottleneck from
one that is waiting on the bus, on its own memory, on a clamp, or on the pipeline feeding it.

Organized by the question each module answers:

* `dcgm` — hardware performance counters: real occupancy, tensor-pipe and DRAM activity.
* `throughput` — what the PCIe and NVLink wires carry, and whether a link trained low.
* `clocks` — clocks against their ceilings, and the counters that catch *intermittent* clamping.
* `engines` — the NVDEC/NVENC/NVJPG/OFA blocks, which no `sm_utilization` figure includes.
* `energy` — the driver's integrated joule counter, and the power envelope around it.
* `memory` — the driver's own division of the framebuffer, and the host-mappable aperture.
* `identity` — compute capability, cores, and bus width read *per device*.
* `processes` — who on a shared device is using it.
* `sampler` — the bounded accumulator that turns snapshots into a window.
* `bottleneck` — one verdict per device, in the vocabulary the fix is written in.

**This façade carries the record types and the one reader per module.** The derived helpers each
module offers — `transfer_bound_devices`, `throttle_fraction`, `half_precision_dtype`,
`tensor_cores_idle`, and the rest — are reached at their own module path, because re-exporting
forty names here would make the façade longer than several of the modules behind it.

Everything degrades to empty or zero rather than raising: absent driver, unmounted container,
consumer part, MIG instance, and per-field refusal all read as "not reported". Where "not
reported" and "zero" would mean different things, the record carries a `readable` flag and the
distinction is documented on it — a caller must never read an unreadable device as a healthy one.

A neutral utility: any layer may import `_internal`.
"""

from __future__ import annotations

from batcher._internal.hardware.telemetry.bottleneck import (
    VERDICT_ADVICE,
    Bottleneck,
    classify_device,
    fleet_verdict,
)
from batcher._internal.hardware.telemetry.clocks import DeviceClocks, device_clocks
from batcher._internal.hardware.telemetry.dcgm import (
    DcgmProfile,
    dcgm_available,
    device_profiles,
)
from batcher._internal.hardware.telemetry.energy import DeviceEnergy, device_energy
from batcher._internal.hardware.telemetry.engines import EngineUtilization, device_engines
from batcher._internal.hardware.telemetry.identity import DeviceIdentity, device_identity
from batcher._internal.hardware.telemetry.memory import DeviceMemory, device_memory
from batcher._internal.hardware.telemetry.processes import (
    ProcessUtilization,
    device_process_utilization,
)
from batcher._internal.hardware.telemetry.sampler import (
    MetricSummary,
    TelemetrySampler,
    saturation_shape,
)
from batcher._internal.hardware.telemetry.throughput import LinkThroughput, device_throughput

__all__ = [
    "VERDICT_ADVICE",
    "Bottleneck",
    "DcgmProfile",
    "DeviceClocks",
    "DeviceEnergy",
    "DeviceIdentity",
    "DeviceMemory",
    "EngineUtilization",
    "LinkThroughput",
    "MetricSummary",
    "ProcessUtilization",
    "TelemetrySampler",
    "classify_device",
    "dcgm_available",
    "device_clocks",
    "device_energy",
    "device_engines",
    "device_identity",
    "device_memory",
    "device_process_utilization",
    "device_profiles",
    "device_throughput",
    "fleet_verdict",
    "saturation_shape",
]
