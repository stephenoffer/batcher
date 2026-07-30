"""NVLink, per link and per device — whether the fast path between devices is actually up.

A node's NVLink fabric is the difference between a multi-device stage that exchanges at
hundreds of gigabytes per second and one that falls back to PCIe at a tenth of that. The
fallback is silent: CUDA peer-to-peer copies still work when a link is down, they simply route
through the host, and the stage completes with the right answer and a runtime nobody can
account for. `hardware.fabric.topology` reasons about the domain a device *model* has; this
module reads what the fabric on this particular board is doing right now.

Two signals, both of which a GPU fleet operator acts on:

* **Inactive links.** A device whose links are down has been isolated from its peers. It is
  still a working GPU for single-device work, and it is the wrong place to schedule a
  collective. Degrading a device out of collective placement is cheaper than discovering it
  through a stage that runs at a fifth of the expected rate.
* **Replay, recovery, and CRC counters.** These climb on a marginal link before it drops.
  A device accumulating recovery events is on its way to an outage, and the counters are the
  only warning that arrives before the job fails.

Reads through NVML, and reports nothing rather than raising anywhere NVML declines: consumer
parts have no NVLink at all, MIG instances refuse the queries, and a container without the
driver mounted refuses everything.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher._internal.hardware.nvml import _decode, _device_count, _nvml, _read

__all__ = [
    "NVLINK_ERROR_COUNTERS",
    "NvLinkStatus",
    "nvlink_degraded_devices",
    "nvlink_status",
    "nvlink_summary",
    "p2p_pairs",
]

#: NVML's per-link error counters, in enum order, with the name each is reported under.
#:
#: * `replay` — a flit was retransmitted; individually harmless, and a rising rate is not.
#: * `recovery` — the link retrained itself, which stalls traffic across it for milliseconds.
#: * `crc_flit` / `crc_data` — corruption caught by the link's own check.
NVLINK_ERROR_COUNTERS = ("replay", "recovery", "crc_flit", "crc_data")

#: Links to probe per device. NVML has no "how many links does this device have" call that is
#: available on every driver, so the probe walks up to this many and stops counting at the
#: first index the driver rejects. Eighteen covers every shipping part through the B200.
_MAX_LINKS = 18


@dataclass(frozen=True, slots=True)
class NvLinkStatus:
    """One device's NVLink fabric state.

    Attributes:
        index: NVML device index.
        uuid: Stable device UUID, the identifier a health history should be keyed on.
        links: Links the driver reported a state for at all.
        active_links: Links currently up.
        errors: Counter name to summed count across the device's links, from
            `NVLINK_ERROR_COUNTERS`. A counter NVML declined to report is absent rather than
            zero, so "no errors" and "no visibility" stay distinguishable.
        peers: PCI addresses at the far end of each active link, deduplicated. Empty when the
            driver does not publish remote endpoints, which is common inside a container.
    """

    index: int
    uuid: str = ""
    links: int = 0
    active_links: int = 0
    errors: dict[str, int] | None = None
    peers: tuple[str, ...] = ()

    @property
    def degraded(self) -> bool:
        """Whether the device has links the driver reported but did not bring up.

        False for a device with no NVLink at all: absence of a fabric is not a fault, and
        flagging every PCIe-attached device as degraded would bury the ones that are.
        """
        return self.links > 0 and self.active_links < self.links

    @property
    def total_errors(self) -> int:
        """Summed link errors across every reported counter, `0` when none were readable."""
        return sum((self.errors or {}).values())


def _link_state(nv, handle, link: int) -> int | None:
    """`1` when a link is up, `0` when down, `None` when the driver refuses the index.

    `None` is what terminates the walk: a device with twelve links rejects the thirteenth, and
    that rejection is the only portable way to learn the link count.
    """
    fn = getattr(nv, "nvmlDeviceGetNvLinkState", None)
    if fn is None:
        return None
    sentinel = object()
    value = _read(lambda: fn(handle, link), sentinel)
    if value is sentinel:
        return None
    return 1 if int(value) else 0


def _link_errors(nv, handle, link: int) -> dict[str, int]:
    """Error counters for one link, omitting any the driver declines to report."""
    fn = getattr(nv, "nvmlDeviceGetNvLinkErrorCounter", None)
    if fn is None:
        return {}
    out: dict[str, int] = {}
    for counter, name in enumerate(NVLINK_ERROR_COUNTERS):
        sentinel = object()
        value = _read(lambda c=counter: fn(handle, link, c), sentinel)
        if value is not sentinel:
            out[name] = int(value)
    return out


def _remote_pci(nv, handle, link: int) -> str:
    """The PCI address at the far end of a link, or `""` when unpublished."""
    fn = getattr(nv, "nvmlDeviceGetNvLinkRemotePciInfo", None)
    if fn is None:
        return ""
    info = _read(lambda: fn(handle, link), None)
    if info is None:
        return ""
    return _decode(getattr(info, "busId", "")).lower()


def nvlink_status() -> tuple[NvLinkStatus, ...]:
    """NVLink state for every device on this host, in NVML index order.

    Not memoized: link state and error counters are live readings, and a cached "the fabric is
    healthy" is exactly the answer that outlives the fabric being healthy. Costs a handful of
    NVML calls per link, so it belongs on a per-stage cadence rather than a per-batch one.

    Returns:
        One record per device, empty when NVML is unavailable. A device with no NVLink reports
        a record with zero links rather than being omitted, so a caller can tell "no fabric"
        from "no visibility".
    """
    nv = _nvml()
    if nv is None:
        return ()
    out: list[NvLinkStatus] = []
    for index in range(_device_count(nv)):
        handle = _read(lambda i=index: nv.nvmlDeviceGetHandleByIndex(i), None)
        if handle is None:
            continue
        links = active = 0
        errors: dict[str, int] = {}
        peers: list[str] = []
        for link in range(_MAX_LINKS):
            state = _link_state(nv, handle, link)
            if state is None:
                break
            links += 1
            active += state
            for name, count in _link_errors(nv, handle, link).items():
                errors[name] = errors.get(name, 0) + count
            if state:
                peer = _remote_pci(nv, handle, link)
                if peer and peer not in peers:
                    peers.append(peer)
        out.append(
            NvLinkStatus(
                index=index,
                uuid=_decode(_read(lambda h=handle: nv.nvmlDeviceGetUUID(h), "")),
                links=links,
                active_links=active,
                errors=errors or None,
                peers=tuple(peers),
            )
        )
    return tuple(out)


def nvlink_degraded_devices(
    status: tuple[NvLinkStatus, ...] | None = None,
    *,
    error_threshold: int = 0,
) -> tuple[NvLinkStatus, ...]:
    """Devices whose fabric is down, partially down, or accumulating errors.

    Args:
        status: Readings to inspect, or `None` to take them live.
        error_threshold: Summed link errors above which a device counts as degraded even with
            every link up. The default of `0` reports only structurally degraded devices,
            because replay counters are non-zero on healthy hardware and a threshold that
            flags them would quarantine a whole fleet.

    Returns:
        The degraded subset, in index order. Empty when the fabric is healthy or unreadable.
    """
    records = nvlink_status() if status is None else status
    return tuple(
        s
        for s in records
        if s.degraded or (error_threshold > 0 and s.total_errors > error_threshold)
    )


def p2p_pairs(status: tuple[NvLinkStatus, ...] | None = None) -> tuple[tuple[int, int], ...]:
    """Device index pairs that share at least one active NVLink, each pair once.

    The set a collective should be placed within: a pair on this list exchanges on the fabric,
    and a pair off it exchanges through the host at PCIe rate.

    Args:
        status: Readings to inspect, or `None` to take them live.

    Returns:
        Ascending `(low, high)` index pairs. Empty when no remote endpoints are published,
        which is the honest answer inside a container that hides the PCI tree — a caller then
        falls back to the model's domain width from `device_specs`.
    """
    records = nvlink_status() if status is None else status
    nv = _nvml()
    by_pci: dict[str, int] = {}
    for record in records if nv is not None else ():
        handle = _read(lambda i=record.index: nv.nvmlDeviceGetHandleByIndex(i), None)
        info = _read(lambda h=handle: nv.nvmlDeviceGetPciInfo(h), None) if handle else None
        bus = _decode(getattr(info, "busId", "")).lower() if info is not None else ""
        if bus:
            by_pci[bus] = record.index
    pairs: set[tuple[int, int]] = set()
    for record in records:
        for peer in record.peers:
            other = by_pci.get(peer)
            if other is not None and other != record.index:
                pairs.add((min(record.index, other), max(record.index, other)))
    return tuple(sorted(pairs))


def nvlink_summary(status: tuple[NvLinkStatus, ...] | None = None) -> dict:
    """A flat description of the node's device fabric, for the decision log and dashboard.

    Args:
        status: Readings to inspect, or `None` to take them live.

    Returns:
        Device count, total and active link counts, how many devices are degraded, how many
        device pairs share an active link, and the summed error counters. All zero on a node
        with no NVLink and on one where NVML is unavailable, which a caller distinguishes with
        `nvml_available()`.
    """
    records = nvlink_status() if status is None else status
    errors: dict[str, int] = {}
    for record in records:
        for name, count in (record.errors or {}).items():
            errors[name] = errors.get(name, 0) + count
    return {
        "devices": len(records),
        "links": sum(r.links for r in records),
        "active_links": sum(r.active_links for r in records),
        "degraded_devices": len(nvlink_degraded_devices(records)),
        # Pairs that can actually exchange on the fabric. A link count says how much fabric
        # exists; this says how much of it connects the devices a collective would be placed
        # across, which is the figure that changes when one board drops off the mesh.
        "peer_pairs": len(p2p_pairs(records)),
        "errors": errors,
    }
