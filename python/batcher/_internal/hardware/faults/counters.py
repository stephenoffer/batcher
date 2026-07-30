"""The per-device fault counters NVML publishes — memory rows, retired pages, PCIe replays.

These are the *predictive* half of device health, and they are the half a fleet operator can
act on before a job dies. An HBM device does not fail one byte at a time: it retires a faulty
row and continues, drawing on a fixed pool of spares. Watching that pool is how a bad device is
found while it is still returning correct answers.

What each counter means for a scheduler:

* **Correctable remapped rows** — the device has repaired itself and is fine. Rising counts on
  one device against a fleet baseline mean it is on its way out.
* **Uncorrectable remapped rows** — a row failed hard. The device is working, and the memory
  behind that row is gone.
* **Remapping pending** — a repair is recorded but takes effect on the next reset. The device
  runs normally until then and the row is still faulty, which is why "pending" is a scheduling
  input and not merely a note.
* **Remapping failure** — the spare pool is exhausted. There is no repair left; the device
  needs replacing, and everything scheduled onto it is at risk.
* **Retired pages** — the pre-Ampere spelling of the same idea, still the only signal on older
  parts, so both are read and neither is assumed.
* **PCIe replay counter** — retransmissions on the host link. A climbing count is a marginal
  slot, riser, or cable, and it costs host-to-device bandwidth long before it costs correctness.

Every field degrades to `0` with the driver absent or the query refused, and `readable` says
which happened. A caller must never read all-zero-and-unreadable as a healthy device.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher._internal.hardware.nvml import _decode, _device_count, _nvml, _read

__all__ = [
    "DeviceFaults",
    "device_faults",
    "faulted_devices",
]

#: NVML's `nvmlPageRetirementCause` values: a page retired for a multiple single-bit error
#: history, and one retired for a double-bit error. Both are read because the causes carry
#: different weight — a double-bit retirement means data was already returned wrong.
_RETIREMENT_CAUSES = (0, 1)


@dataclass(frozen=True, slots=True)
class DeviceFaults:
    """One device's accumulated fault counters.

    Attributes:
        index: NVML device index.
        uuid: Stable device UUID, the identifier a fleet's fault history is keyed on.
        pci_address: PCI address, normalized lowercase, `""` when unpublished. What joins this
            record to an Xid event, which the driver reports by address and never by UUID.
        remapped_correctable: Rows remapped after correctable errors.
        remapped_uncorrectable: Rows remapped after uncorrectable errors.
        remap_pending: Whether a remap takes effect only after the next device reset.
        remap_failure: Whether remapping has failed, meaning the spare pool is exhausted.
        retired_pages: Pages retired across both documented causes, for pre-Ampere parts.
        retirement_pending: Whether a page retirement awaits a reset.
        pcie_replay: PCIe replay counter, retransmissions on the host link.
        ecc_volatile_uncorrected: Uncorrectable ECC errors since the driver loaded.
        ecc_aggregate_uncorrected: Uncorrectable ECC errors over the device's lifetime. Held
            apart from the volatile count deliberately: the volatile one is the scheduling
            signal, and the aggregate one is the procurement signal.
        readable: Whether NVML answered at all. False means every field above is a default,
            not a measurement.
    """

    index: int
    uuid: str = ""
    pci_address: str = ""
    remapped_correctable: int = 0
    remapped_uncorrectable: int = 0
    remap_pending: bool = False
    remap_failure: bool = False
    retired_pages: int = 0
    retirement_pending: bool = False
    pcie_replay: int = 0
    ecc_volatile_uncorrected: int = 0
    ecc_aggregate_uncorrected: int = 0
    readable: bool = False

    @property
    def needs_reset(self) -> bool:
        """Whether the device has a repair that only a reset will apply.

        A device in this state is running with a known-faulty row still mapped in. It is
        schedulable, and it should be drained at the next convenient boundary rather than
        during a stage.
        """
        return self.remap_pending or self.retirement_pending

    @property
    def needs_replacement(self) -> bool:
        """Whether the device has exhausted its ability to repair itself.

        The one counter state with no operational remedy: no reset helps, and everything
        scheduled onto the device from here is at risk.
        """
        return self.remap_failure

    @property
    def degraded_memory(self) -> bool:
        """Whether any row or page has been lost to an uncorrectable fault."""
        return self.remapped_uncorrectable > 0 or self.ecc_volatile_uncorrected > 0


def _remapped_rows(nv, handle) -> tuple[int, int, bool, bool]:
    """`(correctable, uncorrectable, pending, failed)`, all zero/False when unsupported.

    NVML returns these as a four-tuple on modern bindings and refuses the call outright on
    pre-Ampere parts, where retired pages are the equivalent signal instead.
    """
    fn = getattr(nv, "nvmlDeviceGetRemappedRows", None)
    if fn is None:
        return (0, 0, False, False)
    value = _read(lambda: fn(handle), None)
    if not isinstance(value, (tuple, list)) or len(value) < 4:
        return (0, 0, False, False)
    return (int(value[0]), int(value[1]), bool(value[2]), bool(value[3]))


def _retired_pages(nv, handle) -> int:
    """Pages retired across both documented causes, `0` when the query is unsupported."""
    fn = getattr(nv, "nvmlDeviceGetRetiredPages", None)
    if fn is None:
        return 0
    total = 0
    for cause in _RETIREMENT_CAUSES:
        pages = _read(lambda c=cause: fn(handle, c), None)
        if isinstance(pages, (tuple, list)):
            total += len(pages)
    return total


def _retirement_pending(nv, handle) -> bool:
    """Whether a page retirement awaits a device reset."""
    fn = getattr(nv, "nvmlDeviceGetRetiredPagesPendingStatus", None)
    if fn is None:
        return False
    return bool(_read(lambda: fn(handle), 0))


def _pci_address(nv, handle) -> str:
    """The device's PCI address, lowercased, or `""` when unpublished."""
    info = _read(lambda: nv.nvmlDeviceGetPciInfo(handle), None)
    if info is None:
        return ""
    return _decode(getattr(info, "busId", "")).lower()


def device_faults() -> tuple[DeviceFaults, ...]:
    """Fault counters for every device on this host, in NVML index order.

    Not memoized: a counter that moved is the entire point. Costs a handful of NVML calls per
    device, which puts it on a health-check cadence rather than a per-batch one.

    Returns:
        One record per device, empty when NVML is unavailable. A device whose queries are all
        refused still reports a record, with `readable=False`, so a caller can tell a healthy
        device from an unreadable one.
    """
    nv = _nvml()
    if nv is None:
        return ()
    out: list[DeviceFaults] = []
    for index in range(_device_count(nv)):
        handle = _read(lambda i=index: nv.nvmlDeviceGetHandleByIndex(i), None)
        if handle is None:
            continue
        correctable, uncorrectable, pending, failed = _remapped_rows(nv, handle)
        sentinel = object()
        replay = _read(lambda h=handle: nv.nvmlDeviceGetPcieReplayCounter(h), sentinel)
        volatile = _read(lambda h=handle: nv.nvmlDeviceGetTotalEccErrors(h, 1, 0), sentinel)
        aggregate = _read(lambda h=handle: nv.nvmlDeviceGetTotalEccErrors(h, 1, 1), sentinel)
        out.append(
            DeviceFaults(
                index=index,
                uuid=_decode(_read(lambda h=handle: nv.nvmlDeviceGetUUID(h), "")),
                pci_address=_pci_address(nv, handle),
                remapped_correctable=correctable,
                remapped_uncorrectable=uncorrectable,
                remap_pending=pending,
                remap_failure=failed,
                retired_pages=_retired_pages(nv, handle),
                retirement_pending=_retirement_pending(nv, handle),
                pcie_replay=0 if replay is sentinel else int(replay or 0),
                ecc_volatile_uncorrected=0 if volatile is sentinel else int(volatile or 0),
                ecc_aggregate_uncorrected=0 if aggregate is sentinel else int(aggregate or 0),
                readable=replay is not sentinel or volatile is not sentinel,
            )
        )
    return tuple(out)


def faulted_devices(faults: tuple[DeviceFaults, ...] | None = None) -> tuple[DeviceFaults, ...]:
    """Devices with a fault a scheduler should act on, in index order.

    Args:
        faults: Records to inspect, or `None` to read them live.

    Returns:
        The devices needing a reset, needing replacement, or holding uncorrectable memory
        damage. Empty when the fleet is clean *or* unreadable; `DeviceFaults.readable` is what
        tells those apart, and a fleet where it is False must not be quarantined for it.
    """
    records = device_faults() if faults is None else faults
    return tuple(
        f
        for f in records
        if f.readable and (f.needs_reset or f.needs_replacement or f.degraded_memory)
    )
