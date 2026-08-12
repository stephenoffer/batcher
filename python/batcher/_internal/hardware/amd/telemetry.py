"""The AMD half of the deep device readings, so a mixed fleet has one set of answers.

`devices` reads what an Instinct board *is* and the five figures every tool reads. This reads
the rest — the ones that decide why a stage was slow rather than whether a device is alive —
and it reads the same set `telemetry` reads through NVML, so a report, a scrape endpoint, and a
health check do not each need a vendor branch.

The parity, field for field:

| The question | NVIDIA, through NVML | AMD, through `amdgpu` sysfs |
|---|---|---|
| Memory controller busy | `memory_utilization` | `mem_busy_percent` |
| Host link traffic | PCIe throughput counters | `pcie_bw` packet counters |
| Host-mappable aperture | BAR1 memory info | `mem_info_vis_vram_*` |
| Memory stack temperature | `TEMPERATURE_MEMORY` | hwmon `temp3_input` |
| Settable power range | power limit constraints | hwmon `power1_cap_{min,max}` |
| Current clocks | `nvmlDeviceGetClockInfo` | `pp_dpm_sclk` / `pp_dpm_mclk` |

**`pcie_bw` is a counter, not a rate**, and that is the one place the two vendors differ in
kind rather than in spelling. The kernel publishes packets received, packets sent, and the
maximum payload size; bytes are the product, and a *rate* needs two readings and the time
between them. `pcie_bytes_per_second` is that subtraction, and it refuses to guess from one
reading — the alternative would be reporting a device's entire lifetime of traffic as though it
happened in the last second.

**Reading it costs more than the other files here.** The kernel samples the PCIe counters on
demand and the read blocks for a millisecond or so while it does. That keeps this off any
per-batch path, which is the same budget the NVML side runs on for the same reason.

Every field degrades to `0` when the attribute is absent — an older kernel, a consumer part, a
container that mounted `/sys` read-only without the `amdgpu` tree — and `readable` says which
happened, so an unreadable device is never mistaken for a healthy one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from batcher._internal.hardware.amd.devices import (
    _MICRO,
    _MILLI,
    AMD_PCI_VENDOR,
    AMDGPU_SYSFS_ROOT,
    _hwmon_dir,
    _pci_vendor,
)
from batcher._internal.hardware.sysfs import read_float, read_int, read_text

__all__ = [
    "AmdTelemetry",
    "amd_telemetry",
    "pcie_bytes_per_second",
    "visible_vram_pressured",
]

#: Fraction of the visible (host-mappable) VRAM aperture in use above which host mapping is
#: treated as at risk. The AMD counterpart of the BAR1 threshold, and low for the same reason:
#: a registration either fits in the aperture or fails, and the allocations that consume it
#: arrive in large chunks.
_VISIBLE_PRESSURE = 0.85


@dataclass(frozen=True, slots=True)
class AmdTelemetry:
    """One AMD device's deep readings, shaped to match the NVML side.

    Attributes:
        index: Position in card-number order, matching `AmdDevice.index`.
        card: The DRM node's name, such as `"card0"`.
        memory_busy_percent: Memory controller occupancy, 0-100. The figure that separates a
            memory-bound stage from a compute-bound one, and which `gpu_busy_percent` does not
            include.
        visible_vram_total_bytes: Size of the host-mappable aperture.
        visible_vram_used_bytes: Aperture currently mapped.
        junction_temperature_c: Hotspot temperature, which throttles before the edge sensor
            does and is the one an Instinct clamps on.
        memory_temperature_c: HBM stack temperature, which throttles independently of the core.
        fan_rpm: Fan speed, `0` on a passively cooled datacenter board — which is most of them,
            so zero here is normal rather than a fault.
        power_cap_min_watts: Lowest cap the driver will accept.
        power_cap_max_watts: Highest cap the driver will accept.
        sclk_mhz: Current shader clock.
        mclk_mhz: Current memory clock.
        pcie_packets_received: Lifetime packets received across the host link.
        pcie_packets_sent: Lifetime packets sent.
        pcie_max_payload_bytes: Maximum payload per packet, the multiplier that turns the two
            counters above into bytes.
        readable: Whether any attribute answered.
    """

    index: int
    card: str = ""
    memory_busy_percent: int = 0
    visible_vram_total_bytes: int = 0
    visible_vram_used_bytes: int = 0
    junction_temperature_c: float = 0.0
    memory_temperature_c: float = 0.0
    fan_rpm: int = 0
    power_cap_min_watts: float = 0.0
    power_cap_max_watts: float = 0.0
    sclk_mhz: int = 0
    mclk_mhz: int = 0
    pcie_packets_received: int = 0
    pcie_packets_sent: int = 0
    pcie_max_payload_bytes: int = 0
    readable: bool = False

    @property
    def pcie_bytes_total(self) -> int:
        """Lifetime bytes across the host link, `0` when the counters were unreadable.

        A total since the driver loaded, not a rate. `pcie_bytes_per_second` is what turns two
        of these into a figure comparable with the NVML side.
        """
        packets = self.pcie_packets_received + self.pcie_packets_sent
        return packets * self.pcie_max_payload_bytes

    @property
    def visible_vram_utilization(self) -> float:
        """Fraction of the host-mappable aperture in use, in [0, 1], `0.0` when unknown."""
        if self.visible_vram_total_bytes <= 0:
            return 0.0
        return min(1.0, self.visible_vram_used_bytes / self.visible_vram_total_bytes)

    @property
    def visible_vram_pressured(self) -> bool:
        """Whether host mapping on this device is close to failing.

        The AMD counterpart of BAR1 pressure, and the same failure: a caller that ignores it
        gets a mapping error attributed to whichever library happened to ask last, rather than
        a slow path.
        """
        return (
            self.visible_vram_total_bytes > 0 and self.visible_vram_utilization >= _VISIBLE_PRESSURE
        )

    @property
    def hottest_c(self) -> float:
        """The highest temperature any sensor on the board reported.

        The figure a thermal check should use. An Instinct clamps on its junction sensor, which
        runs well above the edge sensor a single-temperature check reads, so checking the edge
        alone finds the clamp after it has already cost the stage.
        """
        return max(self.junction_temperature_c, self.memory_temperature_c)


def _current_clock_mhz(path: str) -> int:
    """The active level from a `pp_dpm_*` table, in MHz, `0` when unreadable.

    The kernel publishes one line per supported level and marks the active one with a trailing
    `*`, as in `1: 1200Mhz *`. Parsing for the marker rather than taking the last line matters:
    the last line is the *highest* level the part supports, which on an idle device is not the
    one it is running at, and reporting it would show every idle board at its boost clock.
    """
    for line in read_text(path).splitlines():
        if not line.rstrip().endswith("*"):
            continue
        parts = line.split()
        for token in parts:
            lowered = token.lower()
            if lowered.endswith("mhz"):
                try:
                    return int(float(lowered[:-3]))
                except ValueError:
                    return 0
    return 0


def _pcie_bw(device_dir: str) -> tuple[int, int, int]:
    """`(received, sent, max payload)` from `pcie_bw`, `(0, 0, 0)` when unreadable.

    This is the read that costs a millisecond: the kernel samples the counters when asked.
    """
    raw = read_text(os.path.join(device_dir, "pcie_bw")).split()
    if len(raw) < 3:
        return (0, 0, 0)
    try:
        return (int(raw[0]), int(raw[1]), int(raw[2]))
    except ValueError:
        return (0, 0, 0)


def amd_telemetry() -> tuple[AmdTelemetry, ...]:
    """Deep readings for every local AMD device, in card order.

    Not memoized: every field is a live reading. Costs a handful of small `/sys` reads per
    device plus the one blocking `pcie_bw` sample, which puts it on a per-stage cadence.

    Returns:
        One record per AMD device, empty on a host with none. A device whose every attribute
        was refused still reports a record with `readable=False`.
    """
    import glob

    from batcher._internal.hardware.amd.devices import _card_number

    out: list[AmdTelemetry] = []
    nodes = glob.glob(os.path.join(AMDGPU_SYSFS_ROOT, "card*"))
    cards = sorted((c for c in nodes if "-" not in os.path.basename(c)), key=_card_number)
    index = 0
    for card_dir in cards:
        device_dir = os.path.join(card_dir, "device")
        if _pci_vendor(device_dir) != AMD_PCI_VENDOR:
            continue
        hwmon = _hwmon_dir(device_dir)
        received, sent, payload = _pcie_bw(device_dir)
        visible_total = read_int(os.path.join(device_dir, "mem_info_vis_vram_total"))
        busy = read_int(os.path.join(device_dir, "mem_busy_percent"))
        out.append(
            AmdTelemetry(
                index=index,
                card=os.path.basename(card_dir),
                memory_busy_percent=busy,
                visible_vram_total_bytes=visible_total,
                visible_vram_used_bytes=read_int(
                    os.path.join(device_dir, "mem_info_vis_vram_used")
                ),
                # amdgpu's hwmon numbering: 1 is the edge sensor `devices` already reads, 2 is
                # the junction, 3 is the memory stack. Reading them by number rather than by
                # label because the labels are absent on several kernels that publish the
                # values.
                junction_temperature_c=(
                    read_float(os.path.join(hwmon, "temp2_input"), scale=_MILLI) if hwmon else 0.0
                ),
                memory_temperature_c=(
                    read_float(os.path.join(hwmon, "temp3_input"), scale=_MILLI) if hwmon else 0.0
                ),
                fan_rpm=read_int(os.path.join(hwmon, "fan1_input")) if hwmon else 0,
                power_cap_min_watts=(
                    read_float(os.path.join(hwmon, "power1_cap_min"), scale=_MICRO)
                    if hwmon
                    else 0.0
                ),
                power_cap_max_watts=(
                    read_float(os.path.join(hwmon, "power1_cap_max"), scale=_MICRO)
                    if hwmon
                    else 0.0
                ),
                sclk_mhz=_current_clock_mhz(os.path.join(device_dir, "pp_dpm_sclk")),
                mclk_mhz=_current_clock_mhz(os.path.join(device_dir, "pp_dpm_mclk")),
                pcie_packets_received=received,
                pcie_packets_sent=sent,
                pcie_max_payload_bytes=payload,
                readable=bool(visible_total or busy or payload),
            )
        )
        index += 1
    return tuple(out)


def pcie_bytes_per_second(
    before: AmdTelemetry,
    after: AmdTelemetry,
    seconds: float,
) -> float:
    """Host-link bytes per second between two readings of one device.

    The AMD counterpart of NVML's PCIe throughput, assembled rather than read: `amdgpu`
    publishes lifetime packet counters, so a rate needs two of them and the interval between.
    Refusing to derive one from a single reading is the point — the counters cover the life of
    the driver, and treating them as a one-second sample overstates the link by whatever the
    uptime is.

    Args:
        before: Reading at the start of the interval.
        after: Reading at the end.
        seconds: Wall-clock seconds between them.

    Returns:
        Bytes per second, `0.0` when the interval is empty, when either reading was
        unreadable, or when the counters went backwards — which is what a driver reload between
        the readings looks like, and after which the interval simply cannot be measured.
    """
    if seconds <= 0 or not (before.readable and after.readable):
        return 0.0
    delta = after.pcie_bytes_total - before.pcie_bytes_total
    return 0.0 if delta < 0 else delta / seconds


def visible_vram_pressured(
    readings: tuple[AmdTelemetry, ...] | None = None,
) -> tuple[AmdTelemetry, ...]:
    """Devices whose host-mappable aperture is close to exhausted, in card order.

    Args:
        readings: Records to inspect, or `None` to read them live.

    Returns:
        The pressured subset. Empty when none are *or* when the aperture was unreadable, which
        must not be read as headroom.
    """
    records = amd_telemetry() if readings is None else readings
    return tuple(r for r in records if r.visible_vram_pressured)
