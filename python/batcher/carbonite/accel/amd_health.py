"""The same admission decision, for a vendor NVML cannot see.

`health` judges an NVIDIA device from NVML. This judges an AMD one from the `amdgpu` driver's
own sysfs tree, using the operator's same thresholds and the same verdict type — a fleet should
not need a second health policy for a second vendor, and a dashboard written against one should
work on the other.

Split from `health` for size rather than for design: the two are one decision with two sources,
and `assess_fleet` reaches here directly when NVML finds nothing.
"""

from __future__ import annotations

from batcher.carbonite.accel.health import _THERMAL_MARGIN_C, HealthThresholds, HealthVerdict

__all__ = ["amd_verdicts"]


def amd_verdicts(thresholds: HealthThresholds | None = None) -> tuple[HealthVerdict, ...]:
    """Verdicts for this host's AMD accelerators, from the driver's own sysfs tree.

    The same questions `assess_device` asks, against the sources AMD publishes, and with the
    operator's same thresholds — a fleet does not need a second health policy for a second
    vendor. Where a reason means the same thing on both vendors it carries the same code
    (`hot`, `memory_full`), so a dashboard written against one works on the other.

    Two codes have no NVIDIA counterpart. `hbm_uncorrectable` is an unrepairable error in the
    memory controller, which is what a fatal Xid means and carries the same consequence: the
    board is condemned, not derated. `engine_uncorrectable` is the same class of error in a
    compute block, which can come from one bad command and clears on a reset, so it derates a
    board and never takes it out of the fleet.

    Args:
        thresholds: Threshold set to apply; the permissive defaults when omitted.

    Returns:
        One verdict per AMD device, empty when there are none or sysfs is unreadable.
    """
    from batcher._internal.hardware.amd import amd_devices

    th = thresholds or HealthThresholds()
    verdicts: list[HealthVerdict] = []
    for device in amd_devices():
        reasons: list[str] = []
        derate = 1.0
        if device.memory_uncorrectable_errors > 0:
            reasons.append("hbm_uncorrectable")
            derate = 0.0
        elif device.uncorrectable_errors > 0:
            reasons.append("engine_uncorrectable")
            derate = min(derate, 0.5)
        # The board's own critical point where it publishes one, the configured ceiling
        # otherwise, and the lower of the two when both are known — the same rule the NVIDIA
        # path applies to a device's published slowdown point, for the same reason.
        hot_at = th.max_temperature_c
        if device.temperature_limit_c > 0.0:
            hot_at = min(hot_at, device.temperature_limit_c - _THERMAL_MARGIN_C)
        if device.temperature_c > hot_at:
            reasons.append("hot")
            derate = min(derate, 0.75)
        if device.memory_total_bytes > 0:
            resident = device.memory_used_bytes / device.memory_total_bytes
            if resident > th.max_memory_fraction:
                reasons.append("memory_full")
                derate = min(derate, 0.25)
        if not reasons:
            verdicts.append(HealthVerdict(device_index=device.index, uuid=device.unique_id))
            continue
        state = "quarantine" if derate <= th.quarantine_below_derate else "degraded"
        verdicts.append(
            HealthVerdict(
                device_index=device.index,
                uuid=device.unique_id,
                state=state,
                reasons=tuple(reasons),
                derate=0.0 if state == "quarantine" else derate,
            )
        )
    return tuple(verdicts)
