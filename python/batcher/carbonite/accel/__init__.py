"""Accelerator resource management: device memory, partitioning, KV cache, and health.

Carbonite protects a run from the resources it does not have. On a GPU fleet the scarcest of
those is device memory — 80 GiB with no swap behind it and a failure mode that kills a worker
rather than degrading it — and the second scarcest is a healthy device.

Four modules, one question each:

* `vram` — device memory as a pool: reserve, release, headroom, and what another tenant is
  already holding.
* `mig` — partitioning one device into isolated instances, so a small model stops holding a
  large device.
* `kv_cache` — the LLM cache budget that actually sets an inference stage's concurrency.
* `health` — turning live telemetry into a schedule/derate/quarantine verdict.

Nothing here allocates device memory or touches a tensor: these are the control-plane
decisions, and the framework doing the allocating carries them out.
"""

from __future__ import annotations

from batcher.carbonite.accel.health import (
    HealthThresholds,
    HealthVerdict,
    assess_device,
    assess_fleet,
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
from batcher.carbonite.accel.vram import DEFAULT_HEADROOM, VramPool, VramReservation

__all__ = [
    "DEFAULT_HEADROOM",
    "HealthThresholds",
    "HealthVerdict",
    "KvCacheBudget",
    "MigPlan",
    "MigProfile",
    "VramPool",
    "VramReservation",
    "assess_device",
    "assess_fleet",
    "kv_bytes_per_token",
    "kv_cache_bytes",
    "max_concurrent_sequences",
    "mig_plan",
    "mig_profiles",
    "mig_supported",
    "schedulable_devices",
    "smallest_profile_for",
]
