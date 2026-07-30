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

    from batcher._internal.hardware.faults import DeviceFaults
    from batcher._internal.hardware.nvml import DeviceTelemetry

__all__ = [
    "HealthThresholds",
    "HealthVerdict",
    "assess_device",
    "assess_faults",
    "assess_fleet",
    "configured_thresholds",
    "device_reset_candidates",
    "fault_reasons",
    "schedulable_device_count",
    "schedulable_devices",
    "xid_verdicts",
]


@dataclass(frozen=True, slots=True)
class HealthThresholds:
    """When a reading stops being normal.

    Defaults are deliberately permissive: a false quarantine takes a working device out of a
    fleet, which costs more than a slow one stays in. An operator with a stricter SLA tightens
    them through config rather than by editing this.

    Attributes:
        max_temperature_c: Above this the device is degraded even without a driver clamp,
            because it is about to be clamped. A *ceiling*: where the driver publishes the
            part's own slowdown point, the lower of the two applies, so a part that clamps at
            83 is not judged against a fleet-wide 87 it never reaches.
        quarantine_below_derate: Derate at or below which a degraded device stops being
            scheduled at all. A device contributing a quarter of a healthy one while drawing
            most of its power is worth taking out; one contributing half is not.
        max_ecc_uncorrected: Uncorrectable ECC errors tolerated before quarantine. Zero:
            an uncorrectable error means data was already returned wrong, and no throughput
            argument outweighs that.
        max_memory_fraction: Resident device memory above which the device is treated as
            full, so a stage is not admitted onto a device another tenant has filled.
        quarantine_on_remap_failure: Stop scheduling onto a device whose memory row remapping
            has failed. On by default, and unlike every other condition here it does not
            recover: the spare rows are gone and no reset brings them back.
        drain_on_reset_pending: Stop scheduling onto a device holding a repair that only its
            next reset will apply. Off by default, because such a device is still returning
            correct results and draining it mid-run costs more than waiting for a boundary.
    """

    max_temperature_c: float = 87.0
    quarantine_below_derate: float = 0.3
    max_ecc_uncorrected: int = 0
    max_memory_fraction: float = 0.95
    quarantine_on_remap_failure: bool = True
    drain_on_reset_pending: bool = False


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


#: How far below a device's own slowdown point it is called hot. Five degrees is roughly a
#: minute of headroom on a device under load, which is the difference between a scheduler
#: that can shed work before the clamp and one that learns about it afterwards.
_THERMAL_MARGIN_C = 5.0

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
    # The device's own slowdown point where it published one, and the configured ceiling
    # otherwise. A constant cannot serve a mixed fleet: the threshold differs by tens of
    # degrees across parts, so one figure is too strict on some and too lax on others — and
    # "too lax" means the warning arrives after the clamp it existed to precede. A margin
    # below the slowdown point is what makes this a warning rather than a restatement of
    # `thermal_throttle`, which already fires once the clamp is on.
    hot_at = th.max_temperature_c
    if telemetry.slowdown_temperature_c > 0.0:
        hot_at = min(hot_at, telemetry.slowdown_temperature_c - _THERMAL_MARGIN_C)
    if telemetry.temperature_c > hot_at:
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
    faults: Sequence[DeviceFaults] | None = None,
) -> tuple[HealthVerdict, ...]:
    """Verdicts for every device on this host.

    Args:
        readings: Telemetry records to judge, or `None` to read them live.
        thresholds: Threshold set to apply.
        faults: Fault counters to fold in, or `None` to read them live alongside the
            telemetry. Pass `()` to judge on telemetry alone, which is what a caller on a hot
            path wants: the counters cost a second round of NVML calls per device.

    Returns:
        One verdict per device, empty when telemetry is unavailable. Fault counters are joined
        by device index, and a device the counters do not cover keeps its telemetry verdict.
    """
    if readings is None:
        from batcher._internal.hardware.nvml import device_telemetry

        readings = device_telemetry()
    if faults is None:
        from batcher._internal.hardware.faults import device_faults

        faults = device_faults()
    by_index = {f.index: f for f in faults}
    verdicts = []
    for reading in readings:
        verdict = assess_device(reading, thresholds)
        fault = by_index.get(reading.index)
        verdicts.append(assess_faults(verdict, fault, thresholds) if fault else verdict)
    # The driver's own error log last, because it is the only source that can condemn a
    # device every other reading calls healthy.
    return xid_verdicts(tuple(verdicts), tuple(faults))


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


def configured_thresholds() -> HealthThresholds:
    """The threshold set the active configuration asks for.

    Keeps the mapping from `accelerator.health` to thresholds in one place, so a caller does
    not restate it and the two cannot drift. `quarantine_on_ecc=False` is expressed as an
    effectively unreachable ECC ceiling rather than as a second code path.

    Returns:
        The thresholds to apply on this deployment.
    """
    from batcher.config import active_config

    health = active_config().accelerator.health
    return HealthThresholds(
        max_temperature_c=health.max_temperature_c,
        quarantine_below_derate=health.quarantine_below_derate,
        max_ecc_uncorrected=0 if health.quarantine_on_ecc else 1 << 62,
        max_memory_fraction=health.max_memory_fraction,
        quarantine_on_remap_failure=health.quarantine_on_remap_failure,
        drain_on_reset_pending=health.drain_on_reset_pending,
    )


def fault_reasons(
    faults: DeviceFaults,
    thresholds: HealthThresholds | None = None,
) -> tuple[tuple[str, float], ...]:
    """Reason codes and derates implied by one device's fault counters.

    The counters say something telemetry cannot: a device can be cool, unclamped, and idle
    while holding a memory row it has already lost, or while running on a link that has been
    retransmitting for a week. Each reason carries the derate it justifies, so the caller
    combines them the same way it combines the thermal ones.

    Args:
        faults: One device's counters.
        thresholds: Threshold set to apply; the permissive defaults when omitted.

    Returns:
        `(reason, derate)` pairs, empty when the counters are clean or unreadable. A derate of
        `0.0` means the device should not be scheduled at all.
    """
    if not faults.readable:
        return ()  # unreadable counters are not evidence of a fault, and must never quarantine
    th = thresholds or HealthThresholds()
    out: list[tuple[str, float]] = []
    if faults.remap_failure and th.quarantine_on_remap_failure:
        # No repair is left. Unlike a clamp, this does not recover on its own.
        out.append(("row_remap_failed", 0.0))
    if faults.remapped_uncorrectable > 0:
        out.append(("row_remap_uncorrectable", 0.5))
    if faults.needs_reset:
        # The repair applies at the next reset, so the faulty row is still mapped in. Worth
        # draining at a boundary, not worth stopping a running stage for — unless the operator
        # has said their boundaries are too far apart for that to be true.
        out.append(("reset_pending", 0.0 if th.drain_on_reset_pending else 0.75))
    return tuple(out)


def assess_faults(
    verdict: HealthVerdict,
    faults: DeviceFaults,
    thresholds: HealthThresholds | None = None,
) -> HealthVerdict:
    """Fold one device's fault counters into the verdict its telemetry produced.

    Kept as a separate step rather than folded into `assess_device`, because the two readings
    come from different NVML calls with different costs and different availability: a fleet
    where the counters are refused still gets a thermal verdict, and one where telemetry is
    refused still gets a fault verdict.

    Args:
        verdict: The verdict from `assess_device`.
        faults: The same device's counters.
        thresholds: Threshold set to apply; the permissive defaults when omitted.

    Returns:
        The verdict with any fault reasons merged in and the derate taken to the lowest of the
        two. Returned unchanged when the counters are clean or unreadable.
    """
    th = thresholds or HealthThresholds()
    reasons = fault_reasons(faults, th)
    if not reasons:
        return verdict
    derate = min([verdict.derate, *(d for _, d in reasons)])
    merged = tuple(dict.fromkeys([*verdict.reasons, *(r for r, _ in reasons)]))
    state = "quarantine" if derate <= th.quarantine_below_derate else "degraded"
    return HealthVerdict(
        device_index=verdict.device_index,
        uuid=verdict.uuid or faults.uuid,
        state=state,
        reasons=merged,
        derate=0.0 if state == "quarantine" else derate,
    )


def xid_verdicts(
    verdicts: tuple[HealthVerdict, ...],
    faults: Sequence[DeviceFaults],
    events: dict[str, tuple[int, ...]] | None = None,
) -> tuple[HealthVerdict, ...]:
    """Quarantine any device the driver has reported a *fatal* Xid against.

    The one fault class neither telemetry nor the NVML counters can see. A device that has
    fallen off the bus (Xid 79), halted its micro-controller (62), or taken an uncontained ECC
    error (95) still enumerates, still reports a temperature, and still accepts work — and
    every task placed on it fails the same way, so the retries walk the whole queue onto the
    one bad device. Reading the driver's own error log turns that into a single lost task.

    Joined by PCI address, because that is the only identifier an Xid line carries; the
    counters supply the address-to-device mapping.

    Args:
        verdicts: Verdicts so far, one per device.
        faults: The same devices' counters, for their PCI addresses.
        events: Fatal codes per address, or `None` to read the kernel log.

    Returns:
        The verdicts with any Xid-quarantined device replaced. Unchanged when the log holds
        nothing fatal *and* when it cannot be read at all — a container without the host log
        must not quarantine a fleet, so silence is never treated as a signal.
    """
    if events is None:
        from batcher._internal.hardware.faults import recent_xid_events, xid_fatal, xid_readable

        if not xid_readable():
            return verdicts
        events = xid_fatal(recent_xid_events())
    if not events:
        return verdicts
    by_index = {f.index: f.pci_address for f in faults if f.pci_address}
    out: list[HealthVerdict] = []
    for verdict in verdicts:
        codes = events.get(by_index.get(verdict.device_index, ""), ())
        if not codes:
            out.append(verdict)
            continue
        from batcher._internal.hardware.faults import describe_xid

        reasons = tuple(dict.fromkeys([*verdict.reasons, *(f"xid_{c}" for c in codes)]))
        out.append(
            HealthVerdict(
                device_index=verdict.device_index,
                uuid=verdict.uuid,
                state="quarantine",
                reasons=reasons,
                derate=0.0,
            )
        )
        from batcher._internal.logging import get_logger

        get_logger("carbonite").warning(
            "device %s quarantined: %s",
            verdict.uuid or verdict.device_index,
            ", ".join(describe_xid(code) for code in codes),
        )
    return tuple(out)


def device_reset_candidates(
    faults: Sequence[DeviceFaults] | None = None,
) -> tuple[str, ...]:
    """UUIDs of devices holding a repair that only a reset will apply.

    The drain list an operator acts on between jobs. Distinct from the quarantine list: these
    devices are producing correct results now, and they will keep doing so until the faulty
    row is touched, so taking them out mid-stage costs more than it saves.

    Args:
        faults: Records to inspect, or `None` to read them live.

    Returns:
        Device UUIDs in index order, empty when nothing is pending or the counters are
        unreadable.
    """
    if faults is None:
        from batcher._internal.hardware.faults import device_faults

        faults = device_faults()
    return tuple(f.uuid for f in faults if f.readable and f.needs_reset and f.uuid)


def schedulable_device_count() -> int | None:
    """Devices on this host that are safe to schedule on, or `None` when it cannot be told.

    A device rarely fails by disappearing. It stays present and reports uncorrectable ECC
    errors, or the driver clamps it to a fraction of its clock, and a pool sized to the device
    *count* keeps feeding it either way. This is the count after those verdicts.

    `None` rather than a number when telemetry is unavailable, which a caller must treat as
    "keep the device count you already had": an absent probe is not evidence that a fleet is
    unhealthy, and turning it into one would take a cluster offline the day `pynvml` stopped
    being installed.

    Returns:
        The schedulable device count, or `None` when no telemetry could be read.
    """
    from batcher._internal.hardware.nvml import device_telemetry

    readings = device_telemetry()
    if not readings:
        return None
    return len(schedulable_devices(readings, configured_thresholds()))
