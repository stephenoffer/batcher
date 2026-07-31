"""What the *hardware* is doing, as gauges a fleet alerts on.

`metrics` counts what the engine did — queries, rows, spills, inference batches. This module
reads the machine instead, on each scrape, and the difference matters on a GPU fleet: the
engine's own GPU series only exist while an inference stage is running, so a node that is hot,
power-capped, on a degraded link, or holding a failing device reports nothing at all through
them. Those are exactly the conditions that leave every query correct and a fraction as fast.

**Facts only, never verdicts.** `observe` is a neutral layer and may not ask a subsystem what
it *decided*: whether a device is schedulable is Carbonite's answer, and exporting it from a
scrape endpoint would put observability on the wrong side of the independence contract. What
appears here is what the hardware reports. What to do about it is read from
`bt.accelerators()`, which is the conductor's to assemble.

Both vendors, in one set of series. A dashboard that has to know which vendor produced a
series is a dashboard per vendor, so the readings are normalized to watts, degrees, and a
ratio before they leave here.
"""

from __future__ import annotations

from batcher._internal.logging import note_suppressed

__all__ = [
    "NODE_CONDITION_HELP",
    "device_gauges",
    "device_readings",
    "node_conditions",
]


#: The per-device hardware readings exported as gauges: the metric suffix, its help line, and
#: how to pull it off one reading. Watts and degrees rather than a vendor's raw units, because
#: a dashboard that has to know which vendor produced a series is a dashboard per vendor.
_DEVICE_GAUGES = (
    ("power_watts", "Instantaneous board power draw", lambda d: d[1]),
    ("power_limit_watts", "The enforced power cap the board is running under", lambda d: d[2]),
    ("temperature_celsius", "Board temperature", lambda d: d[3]),
    ("utilization_ratio", "Fraction of the device's compute in use", lambda d: d[4]),
)


def device_readings() -> tuple[tuple[str, float, float, float, float], ...]:
    """`(label, watts, cap_watts, celsius, utilization)` for every local device, both vendors.

    Distinct from the `gpu.devices` series above, which is reported *by the engine* while an
    inference stage runs and so is absent on a node that is merely slow. These are read from
    the hardware on each scrape, which is the only way a fleet sees a device that is hot,
    capped, or idle while a job is running somewhere else on it.

    The label is the device's own identifier where it publishes one and its index otherwise,
    so a series survives a reboot renumbering the devices.
    """
    out: list[tuple[str, float, float, float, float]] = []
    try:
        from batcher._internal.hardware.amd import amd_devices
        from batcher._internal.hardware.nvml import device_telemetry

        for reading in device_telemetry():
            out.append(
                (
                    reading.uuid or str(reading.index),
                    reading.power_watts,
                    reading.power_limit_watts,
                    reading.temperature_c,
                    reading.sm_utilization,
                )
            )
        if not out:
            for device in amd_devices():
                out.append(
                    (
                        device.unique_id or device.card,
                        device.power_watts,
                        device.power_cap_watts,
                        device.temperature_c,
                        device.busy_percent / 100.0,
                    )
                )
    except Exception as exc:  # pragma: no cover - a scrape must never fail a process
        note_suppressed("observe", "read the node's device telemetry", exc)
    return tuple(out)


def device_gauges() -> list[str]:
    """The per-device gauge lines, or `[]` on a host with no readable device.

    A reading of exactly zero is emitted rather than skipped: for these four figures zero is a
    real value an operator wants to see — an idle device draws watts and reports a
    temperature — and a series that disappears when it goes quiet breaks every rate and
    average built on it. A host with no devices emits nothing at all, which is different.
    """
    readings = device_readings()
    if not readings:
        return []
    lines: list[str] = []
    for suffix, help_text, pull in _DEVICE_GAUGES:
        lines.append(f"# HELP batcher_device_{suffix} {help_text}")
        lines.append(f"# TYPE batcher_device_{suffix} gauge")
        for reading in readings:
            lines.append(f'batcher_device_{suffix}{{device="{reading[0]}"}} {pull(reading)}')
    return lines


#: The node conditions exported as gauges, with the one-line help each carries into a
#: dashboard. Gauges rather than counters because every one of them is a *state* an operator
#: acts on now — "three devices are on a degraded link" is actionable; the number of times
#: that has been true is not.
NODE_CONDITION_HELP = {
    "degraded_links": "Devices whose host PCIe link negotiated below its capability",
    "faulted_devices": "Devices with a memory fault: a pending repair or exhausted spares",
    "nvlink_down_devices": "Devices with one or more NVLink links not up",
    "fabric_errors": "Summed RDMA port error counters on this node",
    "fabric_ports_down": "Cabled RDMA ports that are not carrying traffic",
    "throttled_devices": "Devices whose clocks the driver is currently clamping",
    "transfer_bound_devices": "Devices whose host link is saturated while their SMs are not",
    "power_capped_devices": "Devices an operator has capped below their default power limit",
    "bar1_pressured_devices": "Devices whose host-mappable aperture is close to exhausted",
    "clock_limited_devices": "Devices held below their clock ceiling, transiently or by pinning",
}


def node_conditions() -> dict[str, int]:
    """The hardware conditions on this node, as gauges an alert can be written against.

    These are the failures that never reach a counter: a host link at quarter width, memory
    repairing itself, an NVLink fabric that dropped, a port accumulating symbol errors. Each
    leaves every query correct and a fraction as fast, so a fleet finds them by scraping for
    them or does not find them at all.

    Facts only, never verdicts. `observe` is a neutral layer and may not ask a subsystem what
    it *decided* — whether a device is schedulable is Carbonite's answer, and exporting it
    here would put a scrape endpoint on the wrong side of the independence contract. The
    conditions below are what the hardware reports; what to do about them is read from
    `bt.accelerators()`, which is the conductor's to assemble.

    Read live on each scrape rather than accumulated, because they are states rather than
    events. Costs a handful of `/sys` reads and NVML calls, which is the right budget for a
    path a monitoring system hits every fifteen seconds and nothing else hits at all.

    Returns:
        Condition name to count, all zero on a host where none of it is readable — a scrape
        config should not have to be conditional on the hardware it is pointed at.
    """
    out = dict.fromkeys(NODE_CONDITION_HELP, 0)
    try:
        from batcher._internal.hardware.amd import ecc_faulted_amd_devices
        from batcher._internal.hardware.fabric import (
            degraded_device_links,
            fabric_error_total,
            nvlink_summary,
            rdma_summary,
        )
        from batcher._internal.hardware.faults import faulted_devices

        out["degraded_links"] = len(degraded_device_links())
        # Both vendors in one gauge. A fleet does not want two alerts for "a device's memory
        # has failed", and NVML reports nothing at all on the AMD half of a mixed fleet.
        out["faulted_devices"] = len(faulted_devices()) + len(ecc_faulted_amd_devices())
        out["nvlink_down_devices"] = int(nvlink_summary()["degraded_devices"])
        out["fabric_errors"] = sum(fabric_error_total().values())
        rdma = rdma_summary()
        out["fabric_ports_down"] = max(0, int(rdma["ports"]) - int(rdma["active_ports"]))
    except Exception as exc:  # pragma: no cover - a scrape must never fail a process
        note_suppressed("observe", "read the node's hardware conditions", exc)
    out.update(_device_state_conditions())
    return out


def _device_state_conditions() -> dict[str, int]:
    """The five conditions read from the deep NVML detail, all zero when it is unreadable.

    Held apart from the block above so one unreadable source cannot zero the other's counts:
    the `/sys` fabric probes and NVML's per-device queries fail independently and for different
    reasons, and a container that mounts one without the other is the normal case rather than
    an odd one.

    Each of these is a *state* rather than an event, and each is invisible to every existing
    series: a clamped device, a saturated host link, a board capped below its default, an
    exhausted mapping aperture, and a clock held below its ceiling all leave every query
    correct and the node a fraction as fast.
    """
    out = {
        "throttled_devices": 0,
        "transfer_bound_devices": 0,
        "power_capped_devices": 0,
        "bar1_pressured_devices": 0,
        "clock_limited_devices": 0,
    }
    try:
        from batcher._internal.hardware.nvml import throttled_devices
        from batcher._internal.hardware.telemetry.clocks import clock_limited_devices
        from batcher._internal.hardware.telemetry.energy import capped_below_default
        from batcher._internal.hardware.telemetry.memory import bar1_pressured_devices
        from batcher._internal.hardware.telemetry.throughput import transfer_bound_devices

        out["throttled_devices"] = len(throttled_devices())
        out["transfer_bound_devices"] = len(transfer_bound_devices())
        out["power_capped_devices"] = len(capped_below_default())
        out["bar1_pressured_devices"] = len(bar1_pressured_devices())
        out["clock_limited_devices"] = len(clock_limited_devices())
    except Exception as exc:  # pragma: no cover - a scrape must never fail a process
        note_suppressed("observe", "read the node's device state conditions", exc)
    try:
        from batcher._internal.hardware.amd import throttled_amd_devices, visible_vram_pressured

        # Both vendors in one gauge, as `faulted_devices` already does. A fleet does not want
        # two alerts for "a device is clamped", and NVML reports nothing at all on the AMD half
        # of a mixed host. Added rather than replaced, because a mixed host has both.
        out["throttled_devices"] += len(throttled_amd_devices())
        out["bar1_pressured_devices"] += len(visible_vram_pressured())
    except Exception as exc:  # pragma: no cover - a scrape must never fail a process
        note_suppressed("observe", "read the node's AMD device state conditions", exc)
    return out
