"""The deep device readings, as Prometheus series a fleet can alert on.

`node_metrics.device_gauges` exports the four figures every GPU dashboard already has: watts,
the cap, degrees, and a utilization ratio. They are the right four to start with, and a fleet
alerting on only those cannot see any of the conditions that actually cost it capacity — a slot
that trained at half width, a device clamped for a third of every stage, a decode pipeline that
never reached the hardware decoder, a board capped 40% below its default limit.

Those are all readable, none of them raises anything, and each leaves every query correct and a
fraction as fast. This module exports them.

**Cardinality is bounded by hardware.** Every series here is labelled by device and by nothing
else, so a node contributes a fixed number of series regardless of how long it runs or how many
queries it serves. That is the property that makes it safe to scrape these every fifteen
seconds forever, and it is why nothing here is labelled by query, stage, or plan.

**Facts, not verdicts.** `observe` is a neutral layer: it may report that a device's host link
is at 90% and that its SMs are at 12%, and it may not report that the device is therefore
transfer-bound, because that is a decision and decisions belong to a subsystem. The verdict is
assembled one module over in `diagnosis`, for a human reading a report, and deliberately does
not leave through a scrape endpoint.

**A scrape must never fail a process.** Every reader here is wrapped, and a source that raises
contributes no series rather than taking the endpoint down with it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from batcher._internal.logging import note_suppressed

__all__ = [
    "accelerator_gauges",
    "link_gauges",
    "utilization_gauges",
]

#: PCIe and NVLink series, as `(metric suffix, help line, how to pull it off one reading)`.
#: Bytes per second rather than a cumulative counter because NVML reports a rate: exporting a
#: sampled rate as a `_total` would make every `rate()` built on it silently wrong.
_LINK_GAUGES: tuple[tuple[str, str, Callable], ...] = (
    (
        "pcie_tx_bytes_per_second",
        "Device-to-host bytes per second",
        lambda r: r.pcie_tx_bytes_per_s,
    ),
    (
        "pcie_rx_bytes_per_second",
        "Host-to-device bytes per second",
        lambda r: r.pcie_rx_bytes_per_s,
    ),
    (
        "pcie_utilization_ratio",
        "Host link use as a fraction of the negotiated link's capacity",
        lambda r: r.pcie_utilization,
    ),
    ("pcie_link_generation", "Negotiated PCIe generation", lambda r: r.pcie_gen),
    ("pcie_link_width", "Negotiated PCIe lane count", lambda r: r.pcie_width),
    (
        "pcie_link_derated",
        "1 when the host link trained below what both ends support",
        lambda r: int(r.link_derated),
    ),
    (
        "nvlink_bytes_per_second",
        "Data-payload bytes per second across the peer fabric",
        lambda r: r.nvlink_bytes_per_s,
    ),
)

#: Clock series. The headroom ratios are exported alongside the raw megahertz because an alert
#: written against megahertz has to encode the part's maximum, which differs across a mixed
#: fleet; the ratio is the same expression on every part.
_CLOCK_GAUGES: tuple[tuple[str, str, Callable], ...] = (
    ("sm_clock_mhz", "Current SM clock", lambda r: r.sm_mhz),
    ("sm_clock_max_mhz", "Highest SM clock this part supports", lambda r: r.sm_max_mhz),
    (
        "sm_clock_headroom_ratio",
        "Fraction of the part's maximum SM clock left unused",
        lambda r: r.sm_headroom,
    ),
    ("memory_clock_mhz", "Current memory clock", lambda r: r.memory_mhz),
    (
        "applications_clock_pinned",
        "1 when an operator has pinned applications below the part's maximum",
        lambda r: int(r.applications_clock_pinned),
    ),
    ("performance_state", "NVML P-state, 0 fastest, -1 unknown", lambda r: r.performance_state),
)

#: Fixed-function engine series. Absent from every device utilization figure, and the whole
#: diagnosis for a media pipeline that is quietly decoding on the SMs.
_ENGINE_GAUGES: tuple[tuple[str, str, Callable], ...] = (
    ("decoder_utilization_ratio", "Fraction of the window NVDEC was busy", lambda r: r.decoder),
    ("encoder_utilization_ratio", "Fraction of the window NVENC was busy", lambda r: r.encoder),
    ("jpeg_utilization_ratio", "Fraction of the window NVJPG was busy", lambda r: r.jpeg),
    ("encoder_sessions", "Live hardware encode sessions", lambda r: r.encoder_sessions),
)

#: Memory-division series. `reserved` and the BAR1 pair are the three a pool's own accounting
#: cannot see, and each of them is a distinct way for an allocation to fail while the device
#: reports plenty free.
_MEMORY_GAUGES: tuple[tuple[str, str, Callable], ...] = (
    (
        "memory_free_bytes",
        "Device memory the driver will currently allocate",
        lambda r: r.free_bytes,
    ),
    (
        "memory_reserved_bytes",
        "Device memory the driver holds for itself and will never hand out",
        lambda r: r.reserved_bytes,
    ),
    ("bar1_used_bytes", "Host-mappable aperture currently mapped", lambda r: r.bar1_used_bytes),
    ("bar1_total_bytes", "Size of the host-mappable aperture", lambda r: r.bar1_total_bytes),
    (
        "memory_temperature_celsius",
        "HBM/GDDR stack temperature, which throttles independently of the core",
        lambda r: r.memory_temperature_c,
    ),
)

#: Hardware performance counters. Present only where DCGM is installed, which is a minority of
#: hosts — and where it is, these are the only series here that answer "how much of the device
#: is this actually using" rather than "was the device busy".
_PROFILE_GAUGES: tuple[tuple[str, str, Callable], ...] = (
    ("sm_active_ratio", "Fraction of SMs with a resident warp", lambda r: r.sm_active),
    ("sm_occupancy_ratio", "Resident warps as a fraction of the maximum", lambda r: r.sm_occupancy),
    (
        "tensor_active_ratio",
        "Fraction of cycles the tensor pipes issued",
        lambda r: r.tensor_active,
    ),
    (
        "dram_active_ratio",
        "Fraction of cycles the memory interface was busy",
        lambda r: r.dram_active,
    ),
)


def _emit(prefix: str, spec: Sequence[tuple[str, str, Callable]], readings: Sequence) -> list[str]:
    """Exposition lines for one metric family across every device, `[]` when nothing answered.

    A reading of exactly zero is emitted rather than skipped, for the same reason
    `node_metrics.device_gauges` does it: zero is a real value for all of these, and a series
    that disappears when it goes quiet breaks every rate and average built on it. A host with
    no readable device emits nothing at all, which is a different thing and reads as such.
    """
    if not readings:
        return []
    lines: list[str] = []
    for suffix, help_text, pull in spec:
        name = f"batcher_{prefix}_{suffix}"
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        for reading in readings:
            lines.append(f'{name}{{device="{reading.index}"}} {pull(reading)}')
    return lines


def link_gauges() -> list[str]:
    """PCIe and NVLink exposition lines for every local device.

    Returns:
        The lines, or `[]` on a host with no readable device.
    """
    try:
        from batcher._internal.hardware.telemetry.throughput import device_throughput

        return _emit("device", _LINK_GAUGES, device_throughput())
    except Exception as exc:  # pragma: no cover - a scrape must never fail a process
        note_suppressed("observe", "read the node's link throughput", exc)
        return []


def utilization_gauges() -> list[str]:
    """Clock, fixed-function engine, and hardware-counter exposition lines.

    The three families that together answer what a single utilization ratio cannot: whether the
    device is being held below its clocks, whether its codec blocks are idle while it does codec
    work, and how much of it a busy kernel is really using.

    Returns:
        The lines, or `[]` on a host where none of the three is readable.
    """
    lines: list[str] = []
    try:
        from batcher._internal.hardware.telemetry.clocks import device_clocks
        from batcher._internal.hardware.telemetry.dcgm import device_profiles
        from batcher._internal.hardware.telemetry.engines import device_engines

        lines.extend(_emit("device", _CLOCK_GAUGES, device_clocks()))
        lines.extend(_emit("device", _ENGINE_GAUGES, device_engines()))
        lines.extend(_emit("device", _PROFILE_GAUGES, device_profiles()))
    except Exception as exc:  # pragma: no cover - a scrape must never fail a process
        note_suppressed("observe", "read the node's device utilization detail", exc)
    return lines


def _memory_gauges() -> list[str]:
    """Memory-division and energy-counter exposition lines."""
    lines: list[str] = []
    try:
        from batcher._internal.hardware.telemetry.energy import device_energy
        from batcher._internal.hardware.telemetry.memory import device_memory

        lines.extend(_emit("device", _MEMORY_GAUGES, device_memory()))
        energy = device_energy()
        if energy:
            # A monotonic joule total, so this one is genuinely a counter and is named as one:
            # a dashboard differences it for watts, and `rate()` on it is correct in a way it
            # would not be on any of the sampled series above.
            lines.append("# HELP batcher_device_energy_joules_total Board energy since driver load")
            lines.append("# TYPE batcher_device_energy_joules_total counter")
            for reading in energy:
                lines.append(
                    f'batcher_device_energy_joules_total{{device="{reading.index}"}} '
                    f"{reading.total_energy_joules}"
                )
            lines.append("# HELP batcher_device_power_derated_ratio Cap below the part's default")
            lines.append("# TYPE batcher_device_power_derated_ratio gauge")
            for reading in energy:
                lines.append(
                    f'batcher_device_power_derated_ratio{{device="{reading.index}"}} '
                    f"{reading.derated_fraction}"
                )
    except Exception as exc:  # pragma: no cover - a scrape must never fail a process
        note_suppressed("observe", "read the node's device memory division", exc)
    return lines


def accelerator_gauges() -> list[str]:
    """Every deep device series, ready to append to the exposition text.

    Costs roughly thirty NVML calls per device plus one DCGM sample, which is the right budget
    for a path a monitoring system hits every fifteen seconds and nothing else hits at all.

    Returns:
        The exposition lines, or `[]` on a host with no readable accelerator.
    """
    return link_gauges() + utilization_gauges() + _memory_gauges()
