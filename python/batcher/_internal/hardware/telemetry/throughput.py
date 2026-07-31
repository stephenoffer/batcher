"""What the wires into and between devices are actually carrying, right now.

`fabric.pcie` and `fabric.nvlink` say how wide a link negotiated and whether it is up. Neither
says how much of it is being *used*, and on a multi-GPU node that is the number that decides
whether a stage is worth accelerating at all. A GPU chain that reads 40 GB across a x16 Gen4
slot spends 1.3 s on the bus before a kernel runs; the same chain reading from device-resident
data spends nothing. Those two are indistinguishable in wall-clock attribution and trivially
distinguishable here.

The three readings and what each one decides:

* **PCIe throughput** — host-to-device and device-to-host bytes per second, sampled by the
  driver over a short window. A device at high PCIe utilization and low SM utilization is
  *transfer-bound*, and the fix is fewer/larger transfers, pinned staging, or keeping the data
  resident — never a bigger batch, which makes it worse.
* **NVLink throughput** — per-device bytes across the peer fabric. A multi-GPU shuffle that
  shows zero here is going over the host, whatever the topology claims, and is paying two PCIe
  crossings for a transfer the fabric could have done directly.
* **Link geometry against its maximum** — a slot that trained to x8 when the part supports x16,
  or to Gen3 on a Gen5 board, halves or quarters every transfer above without failing anything.
  This is the single most common silent capacity loss on rented GPU capacity.

**Throughput readings are rates, not counters.** NVML samples them over its own interval
(roughly 20 ms), so a single reading is a snapshot of whatever happened to be in flight and is
noisy on an idle device. Callers wanting a stage-level figure should take the mean of several,
which is what `telemetry.sampler` exists for.

Every field degrades to `0` when the driver is absent, the query is refused, or the part does
not implement the counter — which is the common case for PCIe throughput on consumer parts and
for NVLink on anything without a peer link.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher._internal.hardware.nvml import _device_count, _nvml, _read

__all__ = [
    "LinkThroughput",
    "device_throughput",
    "nvlink_throughput_bytes",
    "pcie_throughput_bytes",
    "transfer_bound_devices",
]

#: NVML's `nvmlPcieUtilCounter` values: TX (device to host) and RX (host to device). Named
#: constants are read off the binding when present, and these are the documented fallbacks for
#: a binding too old to publish them.
_PCIE_TX, _PCIE_RX = 0, 1

#: `nvmlFieldId` names for the NVLink data-payload throughput counters, by direction. Read by
#: name because the numeric ids differ across NVML releases, and a wrong id does not raise —
#: it returns a plausible figure for a different counter.
_NVLINK_FIELDS = (
    ("NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX", "tx"),
    ("NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX", "rx"),
)

#: `nvmlValueType` -> the union member holding the value. NVML returns a tagged union and the
#: wrong member reads as a garbage integer rather than an error, so the tag is always honored.
_VALUE_MEMBERS = {0: "dVal", 1: "uiVal", 2: "ulVal", 3: "ullVal", 4: "sllVal"}


@dataclass(frozen=True, slots=True)
class LinkThroughput:
    """One device's live link utilization, in bytes per second.

    Attributes:
        index: NVML device index on this host.
        pcie_tx_bytes_per_s: Device-to-host bytes per second across the PCIe link.
        pcie_rx_bytes_per_s: Host-to-device bytes per second across the PCIe link.
        nvlink_tx_bytes_per_s: Data-payload bytes per second this device sent over NVLink,
            summed across its links. Excludes protocol overhead, so it is comparable with the
            PCIe figures rather than with a raw line rate.
        nvlink_rx_bytes_per_s: Data-payload bytes per second this device received over NVLink.
        pcie_gen: Negotiated PCIe generation, `0` when unreported.
        pcie_gen_max: Highest generation this device and slot both support.
        pcie_width: Negotiated PCIe lane count.
        pcie_width_max: Highest lane count this device and slot both support.
        readable: Whether NVML answered any query. False means every field is a default.
    """

    index: int
    pcie_tx_bytes_per_s: float = 0.0
    pcie_rx_bytes_per_s: float = 0.0
    nvlink_tx_bytes_per_s: float = 0.0
    nvlink_rx_bytes_per_s: float = 0.0
    pcie_gen: int = 0
    pcie_gen_max: int = 0
    pcie_width: int = 0
    pcie_width_max: int = 0
    readable: bool = False

    @property
    def pcie_bytes_per_s(self) -> float:
        """Total bytes per second crossing the host link in both directions."""
        return self.pcie_tx_bytes_per_s + self.pcie_rx_bytes_per_s

    @property
    def nvlink_bytes_per_s(self) -> float:
        """Total data-payload bytes per second crossing the peer fabric in both directions."""
        return self.nvlink_tx_bytes_per_s + self.nvlink_rx_bytes_per_s

    @property
    def pcie_capacity_bytes_per_s(self) -> float:
        """Theoretical one-direction capacity of the negotiated link, `0.0` when unknown.

        Per-lane figures are the PCIe base specification's effective payload rates after
        encoding: 0.985 GB/s for Gen3 (8 GT/s, 128b/130b), and doubling per generation. Real
        links do not reach this, so a utilization computed against it is a floor on how loaded
        the bus is, never an overstatement.
        """
        if self.pcie_gen <= 0 or self.pcie_width <= 0:
            return 0.0
        per_lane = 0.985e9 * (2 ** max(0, self.pcie_gen - 3))
        return per_lane * self.pcie_width

    @property
    def pcie_utilization(self) -> float:
        """Fraction of the negotiated link's capacity in use, in [0, 1], `0.0` when unknown.

        Computed against one direction's capacity because PCIe is full duplex: the transfer
        that matters is almost always one-directional, and summing both against a single
        direction's capacity would report a saturated bus as being at 200%.
        """
        capacity = self.pcie_capacity_bytes_per_s
        if capacity <= 0:
            return 0.0
        return min(1.0, max(self.pcie_tx_bytes_per_s, self.pcie_rx_bytes_per_s) / capacity)

    @property
    def link_derated(self) -> bool:
        """Whether the PCIe link trained below what both ends support.

        The failure that costs half the host bandwidth of a node and reports nothing: a reseated
        card, a marginal riser, or a slot the BIOS assigned fewer lanes than the board has.
        """
        if not (self.pcie_gen and self.pcie_gen_max and self.pcie_width and self.pcie_width_max):
            return False
        return self.pcie_gen < self.pcie_gen_max or self.pcie_width < self.pcie_width_max

    @property
    def peer_resident(self) -> bool:
        """Whether this device's traffic is predominantly peer-to-peer rather than host-bound.

        The signal that a multi-GPU stage is actually using the fabric it was placed on. A run
        that reports False while sitting inside an NVLink domain is paying two host crossings
        per exchange for a transfer the fabric would have done once.
        """
        return self.nvlink_bytes_per_s > self.pcie_bytes_per_s


def _counter(nv, name: str, fallback: int) -> int:
    """One NVML enum value, by name, falling back to the documented literal."""
    value = getattr(nv, name, None)
    return fallback if value is None else int(value)


def _field_value(entry) -> float:
    """Decode one `nvmlFieldValue_t`, honoring its type tag; `0.0` on any failure.

    NVML reports per-field success separately from the call's own return code, so an entry that
    the device does not implement arrives inside a successful call with a non-zero
    `nvmlReturn`. Reading its union anyway yields a stale or uninitialized number, which is
    worse than reporting nothing.
    """
    if getattr(entry, "nvmlReturn", 0) != 0:
        return 0.0
    member = _VALUE_MEMBERS.get(int(getattr(entry, "valueType", -1)))
    if member is None:
        return 0.0
    value = getattr(getattr(entry, "value", None), member, None)
    return 0.0 if value is None else float(value)


def _nvlink_throughput(nv, handle) -> tuple[float, float]:
    """`(tx, rx)` NVLink data bytes per second for one device, `(0.0, 0.0)` when unreported.

    NVML publishes these as *field values* rather than as getters, in KiB accumulated since the
    counter last rolled, alongside the sampling interval the driver used. Dividing by that
    interval is what turns them into a rate comparable with the PCIe figures; an entry that
    reports no interval is dropped rather than treated as one second, which would understate a
    busy fabric by orders of magnitude.
    """
    fn = getattr(nv, "nvmlDeviceGetFieldValues", None)
    if fn is None:
        return (0.0, 0.0)
    ids = []
    for name, _direction in _NVLINK_FIELDS:
        field = getattr(nv, name, None)
        if field is None:
            return (0.0, 0.0)
        ids.append(int(field))
    entries = _read(lambda: fn(handle, ids), None)
    if not entries:
        return (0.0, 0.0)
    rates: list[float] = []
    for entry in entries[: len(ids)]:
        kib = _field_value(entry)
        micros = float(getattr(entry, "latencyUsec", 0) or 0)
        rates.append(0.0 if micros <= 0 else kib * 1024.0 / (micros / 1e6))
    while len(rates) < 2:
        rates.append(0.0)
    return (rates[0], rates[1])


def device_throughput() -> tuple[LinkThroughput, ...]:
    """Live link utilization for every device on this host, in NVML index order.

    Not memoized, and deliberately not cheap to call in a loop: the PCIe counters cost a driver
    round trip each and the NVLink fields cost one more. That puts this on a per-stage or
    per-second cadence, the same as `nvml.device_telemetry`.

    Returns:
        One record per device, empty when NVML is unavailable. A device that answered nothing
        still reports a record with `readable=False`, so a caller can tell an idle link from an
        unreadable one — a distinction that matters, because "no PCIe traffic" is the evidence
        for keeping data resident and "no reading" is evidence for nothing at all.
    """
    nv = _nvml()
    if nv is None:
        return ()
    tx_counter = _counter(nv, "NVML_PCIE_UTIL_TX_BYTES", _PCIE_TX)
    rx_counter = _counter(nv, "NVML_PCIE_UTIL_RX_BYTES", _PCIE_RX)
    out: list[LinkThroughput] = []
    for index in range(_device_count(nv)):
        handle = _read(lambda i=index: nv.nvmlDeviceGetHandleByIndex(i), None)
        if handle is None:
            continue
        sentinel = object()
        tx = _read(lambda h=handle: nv.nvmlDeviceGetPcieThroughput(h, tx_counter), sentinel)
        rx = _read(lambda h=handle: nv.nvmlDeviceGetPcieThroughput(h, rx_counter), sentinel)
        nv_tx, nv_rx = _nvlink_throughput(nv, handle)
        out.append(
            LinkThroughput(
                index=index,
                # NVML reports PCIe throughput in KB/s over its own sample interval.
                pcie_tx_bytes_per_s=0.0 if tx is sentinel else float(tx or 0) * 1000.0,
                pcie_rx_bytes_per_s=0.0 if rx is sentinel else float(rx or 0) * 1000.0,
                nvlink_tx_bytes_per_s=nv_tx,
                nvlink_rx_bytes_per_s=nv_rx,
                pcie_gen=int(
                    _read(lambda h=handle: nv.nvmlDeviceGetCurrPcieLinkGeneration(h), 0) or 0
                ),
                pcie_gen_max=int(
                    _read(lambda h=handle: nv.nvmlDeviceGetMaxPcieLinkGeneration(h), 0) or 0
                ),
                pcie_width=int(
                    _read(lambda h=handle: nv.nvmlDeviceGetCurrPcieLinkWidth(h), 0) or 0
                ),
                pcie_width_max=int(
                    _read(lambda h=handle: nv.nvmlDeviceGetMaxPcieLinkWidth(h), 0) or 0
                ),
                readable=tx is not sentinel or rx is not sentinel,
            )
        )
    return tuple(out)


def pcie_throughput_bytes(index: int) -> float:
    """Total bytes per second crossing one device's host link, `0.0` when unreadable.

    Args:
        index: NVML device index.

    Returns:
        Transmit plus receive bytes per second.
    """
    return next((t.pcie_bytes_per_s for t in device_throughput() if t.index == index), 0.0)


def nvlink_throughput_bytes(index: int) -> float:
    """Total data bytes per second crossing one device's peer links, `0.0` when unreadable.

    Args:
        index: NVML device index.

    Returns:
        Transmit plus receive bytes per second, excluding protocol overhead.
    """
    return next((t.nvlink_bytes_per_s for t in device_throughput() if t.index == index), 0.0)


def transfer_bound_devices(
    threshold: float = 0.7,
    readings: tuple[LinkThroughput, ...] | None = None,
) -> tuple[LinkThroughput, ...]:
    """Devices whose host link is the constraint, in index order.

    The diagnosis that changes what a caller should do. A transfer-bound device does not want a
    larger batch — a larger batch moves more bytes across the same saturated link and lengthens
    the stage. It wants the data kept resident, staged from pinned memory, or read straight to
    the device.

    Args:
        threshold: Fraction of the negotiated link's one-direction capacity above which the bus
            is treated as the constraint. The default is deliberately below saturation: PCIe
            efficiency falls off well before 100%, so a link measuring 70% is already the
            limiting factor in practice.
        readings: Records to inspect, or `None` to read them live.

    Returns:
        The transfer-bound subset, empty when none are *or* when nothing was readable.
    """
    records = device_throughput() if readings is None else readings
    return tuple(r for r in records if r.readable and r.pcie_utilization >= threshold)
