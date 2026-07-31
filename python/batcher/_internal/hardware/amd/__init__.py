"""AMD accelerators, read from the driver rather than from ROCm.

A package rather than a module so the vendor's probes have somewhere to grow without widening
`hardware/`, which is already at its file ceiling. `devices` is the whole surface today.

The implementation module is the one to patch in a test: this facade re-exports names, so
rebinding a constant here would leave the reader inside `devices` looking at the original.

A neutral utility: any layer may import `_internal`.
"""

from __future__ import annotations

from batcher._internal.hardware.amd.devices import (
    AMD_PCI_VENDOR,
    AMDGPU_SYSFS_ROOT,
    MEMORY_RAS_BLOCKS,
    AmdDevice,
    RasCounts,
    amd_devices,
    amd_power_watts,
    amd_present,
    ecc_faulted_amd_devices,
    readable,
    reset_amd_probe,
    throttled_amd_devices,
)

__all__ = [
    "AMDGPU_SYSFS_ROOT",
    "AMD_PCI_VENDOR",
    "MEMORY_RAS_BLOCKS",
    "AmdDevice",
    "RasCounts",
    "amd_devices",
    "amd_power_watts",
    "amd_present",
    "ecc_faulted_amd_devices",
    "readable",
    "reset_amd_probe",
    "throttled_amd_devices",
]
