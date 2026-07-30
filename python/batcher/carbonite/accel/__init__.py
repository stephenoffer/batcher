"""Accelerator resource management: device memory, partitioning, KV cache, and health.

Carbonite protects a run from the resources it does not have. On a GPU fleet the scarcest of
those is device memory — 80 GiB with no swap behind it and a failure mode that kills a worker
rather than degrading it — and the second scarcest is a healthy device.

Six modules, one question each:

* `vram` — device memory as a pool: reserve, release, headroom, and what another tenant is
  already holding, which is the binding figure when sizing an inference stage on a shared
  device.
* `allocator` — how the allocations `vram` admitted are actually served: a suballocated pool
  in front of the driver, host spilling, and the measured high-water mark.
* `mig` — partitioning one device into isolated instances, so a small model stops holding a
  large device.
* `kv_cache` — the LLM cache budget that actually sets an inference stage's concurrency.
* `health` — turning live telemetry into a schedule/derate/quarantine verdict.
* `power` — the deployment's power envelope as an admission decision, with a device-count
  counter-offer rather than a refusal.

Nothing here allocates device memory or touches a tensor: these are the control-plane
decisions, and the framework doing the allocating carries them out.
"""

from __future__ import annotations

from batcher.carbonite.accel.allocator import (
    MIN_POOL_BYTES,
    AllocatorPlan,
    configure_device_memory,
    device_allocator_state,
    plan_allocator,
    prepare_device_memory,
    reset_device_allocator,
)
from batcher.carbonite.accel.health import (
    HealthThresholds,
    HealthVerdict,
    assess_device,
    assess_fleet,
    configured_thresholds,
    schedulable_device_count,
    schedulable_devices,
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
from batcher.carbonite.accel.power import (
    devices_within_budget,
    validate_fleet_power,
)
from batcher.carbonite.accel.vram import DEFAULT_HEADROOM, VramPool, VramReservation

__all__ = [
    "DEFAULT_HEADROOM",
    "MIN_POOL_BYTES",
    "AllocatorPlan",
    "HealthThresholds",
    "HealthVerdict",
    "KvCacheBudget",
    "MigPlan",
    "MigProfile",
    "VramPool",
    "VramReservation",
    "assess_device",
    "assess_fleet",
    "configure_device_memory",
    "configured_thresholds",
    "device_allocator_state",
    "devices_within_budget",
    "kv_bytes_per_token",
    "kv_cache_bytes",
    "max_concurrent_sequences",
    "mig_plan",
    "mig_profiles",
    "mig_supported",
    "plan_allocator",
    "prepare_device_memory",
    "reset_device_allocator",
    "schedulable_device_count",
    "schedulable_devices",
    "smallest_profile_for",
    "validate_fleet_power",
]
