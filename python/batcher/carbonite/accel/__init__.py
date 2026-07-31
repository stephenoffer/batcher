"""Accelerator resource management: device memory, partitioning, KV cache, and health.

Carbonite protects a run from the resources it does not have. On a GPU fleet the scarcest of
those is device memory, with no swap behind it and a failure mode that kills a worker rather
than degrading it; the second scarcest is a healthy device. Nine modules, one question each:

* `vram` — device memory as a pool: reserve, release, headroom, and what another tenant is
  already holding, the binding figure when sizing a stage on a shared device. `allocator` —
  how those admitted allocations are served: a suballocated pool in front of the driver, host
  spilling, and the measured high-water mark.
* `mig` — partitioning one device into isolated instances, so a small model stops holding a
  large device. `kv_cache` — the LLM cache budget that sets an inference stage's concurrency,
  and `parallelism` — how to spread a model too large for one device, and what that costs.
* `health` — live telemetry as a schedule/derate/quarantine verdict, and `amd_health` — the
  same verdict for a vendor NVML cannot see.
* `power` — the deployment's power envelope as an admission decision, with a counter-offer.
  `affinity` — putting a device's host-side work on the cores next to it.

Nothing here allocates device memory or touches a tensor: these are the control-plane
decisions, and the framework doing the allocating carries them out.
"""

from __future__ import annotations

from batcher.carbonite.accel.affinity import (
    MIN_BOUND_CPUS,
    bind_host_threads_to_device,
    device_affinity_summary,
    feeder_cpus_for_device,
    mps_active,
    mps_client_share,
)
from batcher.carbonite.accel.allocator import (
    MIN_POOL_BYTES,
    AllocatorPlan,
    configure_device_memory,
    device_allocator_state,
    plan_allocator,
    prepare_device_memory,
    reset_device_allocator,
)
from batcher.carbonite.accel.amd_health import amd_verdicts
from batcher.carbonite.accel.health import (
    HealthThresholds,
    HealthVerdict,
    assess_device,
    assess_faults,
    assess_fleet,
    configured_thresholds,
    device_reset_candidates,
    fault_reasons,
    schedulable_device_count,
    schedulable_devices,
    xid_verdicts,
)
from batcher.carbonite.accel.kv_cache import (
    KvCacheBudget,
    kv_bytes_per_token,
    kv_cache_bytes,
    max_concurrent_sequences,
)
from batcher.carbonite.accel.mig import (
    MigPlan,
    MigProfile,
    mig_plan,
    mig_profiles,
    mig_supported,
    smallest_profile_for,
)
from batcher.carbonite.accel.parallelism import ParallelPlan, plan_parallelism
from batcher.carbonite.accel.power import (
    devices_within_budget,
    validate_fleet_power,
)
from batcher.carbonite.accel.vram import DEFAULT_HEADROOM, VramPool, VramReservation

__all__ = [
    "DEFAULT_HEADROOM",
    "MIN_BOUND_CPUS",
    "MIN_POOL_BYTES",
    "AllocatorPlan",
    "HealthThresholds",
    "HealthVerdict",
    "KvCacheBudget",
    "MigPlan",
    "MigProfile",
    "ParallelPlan",
    "VramPool",
    "VramReservation",
    "amd_verdicts",
    "assess_device",
    "assess_faults",
    "assess_fleet",
    "bind_host_threads_to_device",
    "configure_device_memory",
    "configured_thresholds",
    "device_affinity_summary",
    "device_allocator_state",
    "device_reset_candidates",
    "devices_within_budget",
    "fault_reasons",
    "feeder_cpus_for_device",
    "kv_bytes_per_token",
    "kv_cache_bytes",
    "max_concurrent_sequences",
    "mig_plan",
    "mig_profiles",
    "mig_supported",
    "mps_active",
    "mps_client_share",
    "plan_allocator",
    "plan_parallelism",
    "prepare_device_memory",
    "reset_device_allocator",
    "schedulable_device_count",
    "schedulable_devices",
    "smallest_profile_for",
    "validate_fleet_power",
    "xid_verdicts",
]
