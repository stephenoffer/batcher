"""Device health as an admission decision — Carbonite protecting a run from a sick GPU.

At fleet scale a GPU rarely fails by disappearing. It fails by staying present and getting
slower or wronger: the driver clamps its clocks to a third of nominal because an inlet
temperature rose, its power limit was set below what the workload needs, or its memory starts
reporting uncorrectable ECC errors — which means a tensor that read back is not the tensor that
was written. All three are invisible from inside a task, whose only symptom is that it took
longer than the others, and all three are readable from telemetry.

The split this module holds to: `_internal.hardware.nvml` **measures**, this **decides**. A
verdict is a pure function of a reading and a threshold set, so it is testable without a
device, and the thresholds live in config rather than in the code that applies them.

Three verdicts, because the actions differ:

* **healthy** — schedule normally.
* **degraded** — schedule, but expect less: the device is thermally or power clamped, so its
  share of work should shrink rather than its slot disappear. Removing a clamped device from a
  power-constrained fleet is usually wrong, since the clamp is often the fleet's own power cap
  doing its job.
* **quarantine** — do not schedule. Reserved for correctness risks (uncorrectable ECC) and for
  clamping severe enough that the device is contributing almost nothing while still drawing
  most of its idle power.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher._internal.hardware.nvml import DeviceTelemetry

__all__ = [
    "HealthThresholds",
    "HealthVerdict",
    "assess_device",
    "assess_fleet",
    "schedulable_devices",
]


@dataclass(frozen=True, slots=True)
class HealthThresholds:
    """When a reading stops being normal.

    Defaults are deliberately permissive: a false quarantine takes a working device out of a
    fleet, which costs more than a slow one stays in. An operator with a stricter SLA tightens
    them through config rather than by editing this.

    Attributes:
        max_temperature_c: Above this the device is degraded even without a driver clamp,
            because it is about to be clamped.
        quarantine_below_derate: Derate at or below which a degraded device stops being
            scheduled at all. A device contributing a quarter of a healthy one while drawing
            most of its power is worth taking out; one contributing half is not.
        max_ecc_uncorrected: Uncorrectable ECC errors tolerated before quarantine. Zero:
            an uncorrectable error means data was already returned wrong, and no throughput
            argument outweighs that.
        max_memory_fraction: Resident device memory above which the device is treated as
            full, so a stage is not admitted onto a device another tenant has filled.
    """

    max_temperature_c: float = 87.0
    quarantine_below_derate: float = 0.3
    max_ecc_uncorrected: int = 0
    max_memory_fraction: float = 0.95


@dataclass(frozen=True, slots=True)
class HealthVerdict:
    """One device's schedulability, with the reasons that produced it.

    Attributes:
        device_index: NVML index of the device.
        uuid: Device UUID, the identifier health history should be keyed on.
        state: `"healthy"`, `"degraded"`, or `"quarantine"`.
        reasons: Short machine-readable reason codes (`"thermal_throttle"`, `"ecc"`,
            `"power_clamp"`, `"hot"`, `"memory_full"`), empty when healthy.
        derate: Fraction of a healthy device's throughput this one should be given work for,
            in [0, 1]. `1.0` when healthy, `0.0` when quarantined.
    """

    device_index: int
    uuid: str = ""
    state: str = "healthy"
    reasons: tuple[str, ...] = ()
    derate: float = 1.0

    @property
    def schedulable(self) -> bool:
        """Whether work may be placed on this device at all."""
        return self.state != "quarantine"


#: Throttle reasons that indicate a *hardware* problem rather than a policy cap. A power cap is
#: the datacenter's own limit doing exactly what it was set to do; a hardware thermal slowdown
#: is the device protecting itself, which means cooling has already failed.
_HARDWARE_CLAMPS = {"thermal", "hw_slowdown"}


def assess_device(
    telemetry: DeviceTelemetry,
    thresholds: HealthThresholds | None = None,
) -> HealthVerdict:
    """Decide whether one device should be scheduled, and at what derate.

    Args:
        telemetry: A `DeviceTelemetry` reading.
        thresholds: Threshold set to apply; the permissive defaults when omitted.

    Returns:
        The verdict. A device whose telemetry reported nothing at all reads healthy: absent
        telemetry must never quarantine a fleet, which is the failure mode that would take an
        entire cluster offline the day `pynvml` stops being installed.
    """
    th = thresholds or HealthThresholds()
    reasons: list[str] = []
    derate = 1.0

    if telemetry.ecc_uncorrected > th.max_ecc_uncorrected:
        return HealthVerdict(
            device_index=telemetry.index,
            uuid=telemetry.uuid,
            state="quarantine",
            reasons=("ecc",),
            derate=0.0,
        )

    clamps = set(telemetry.throttle_reasons)
    if clamps & _HARDWARE_CLAMPS:
        reasons.append("thermal_throttle")
        derate = min(derate, 0.5)
    if "power" in clamps or "sw_thermal" in clamps:
        reasons.append("power_clamp")
        derate = min(derate, 0.75)
    if telemetry.temperature_c > th.max_temperature_c:
        reasons.append("hot")
        derate = min(derate, 0.75)
    if telemetry.memory_total_bytes > 0:
        resident = telemetry.memory_used_bytes / telemetry.memory_total_bytes
        if resident > th.max_memory_fraction:
            reasons.append("memory_full")
            derate = min(derate, 0.25)

    if not reasons:
        return HealthVerdict(device_index=telemetry.index, uuid=telemetry.uuid)
    state = "quarantine" if derate <= th.quarantine_below_derate else "degraded"
    return HealthVerdict(
        device_index=telemetry.index,
        uuid=telemetry.uuid,
        state=state,
        reasons=tuple(reasons),
        derate=0.0 if state == "quarantine" else derate,
    )


def assess_fleet(
    readings: Sequence[DeviceTelemetry] | None = None,
    thresholds: HealthThresholds | None = None,
) -> tuple[HealthVerdict, ...]:
    """Verdicts for every device on this host.

    Args:
        readings: Telemetry records to judge, or `None` to read them live.
        thresholds: Threshold set to apply.

    Returns:
        One verdict per device, empty when telemetry is unavailable.
    """
    if readings is None:
        from batcher._internal.hardware.nvml import device_telemetry

        readings = device_telemetry()
    return tuple(assess_device(r, thresholds) for r in readings)


def schedulable_devices(
    readings: Sequence[DeviceTelemetry] | None = None,
    thresholds: HealthThresholds | None = None,
) -> tuple[int, ...]:
    """Indices of the devices work may be placed on, in index order.

    Args:
        readings: Telemetry records to judge, or `None` to read them live.
        thresholds: Threshold set to apply.

    Returns:
        Device indices excluding quarantined ones. An empty tuple means either that telemetry
        is unavailable or that every device is quarantined — a caller that cannot tell those
        apart should call `assess_fleet` instead, because the correct response differs.
    """
    return tuple(v.device_index for v in assess_fleet(readings, thresholds) if v.schedulable)
