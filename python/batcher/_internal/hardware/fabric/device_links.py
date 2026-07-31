"""The host link each accelerator is actually on, as opposed to the one its datasheet has.

Every cost model that decides whether a stage is worth moving to a device charges the host
copy, and every one of them charges it at the link the device *model* ships with. A real board
frequently negotiates lower. A card reseated during maintenance, a riser in a dense chassis, a
marginal cable, or a power event drops the link a generation or halves its width, and nothing
reports it: the device enumerates, runs, and returns correct results at a quarter of its
host-to-device bandwidth.

That single number is the term a device decision is most sensitive to. A projection over wide
rows is transfer-bound, so a link at a quarter of nameplate moves the crossover point by a
factor of four — and the decision it corrupts is the one that says "this stage is worth a GPU",
which is then wrong on exactly the nodes that have the problem.

This module is the join between the two halves: NVML says which PCI address each device is at,
`fabric.pcie` says what that address's link negotiated. Both degrade to nothing, and the
efficiency then reads as `1.0` — the nameplate assumption every caller already made.
"""

from __future__ import annotations

import contextlib
import os

from batcher._internal.hardware.fabric.pcie import (
    PcieLink,
    degraded_pcie_links,
    pcie_class,
    pcie_link,
)
from batcher._internal.hardware.nvml import _decode, _device_count, _nvml, _read

__all__ = [
    "degraded_device_links",
    "device_cpu_affinity",
    "device_link_efficiency",
    "device_numa_nodes",
    "device_pcie_links",
    "gpu_pci_addresses",
    "nearest_rdma_device",
    "visible_device_indices",
]


def gpu_pci_addresses() -> tuple[str, ...]:
    """PCI addresses of the local accelerators, one per device in NVML index order.

    **Position is the NVML index**, which is why a device whose address cannot be read holds
    an empty string rather than being dropped. Compacting the list would renumber every device
    after the unreadable one, and each caller here indexes it by device — so a single refused
    query would silently attribute one board's NUMA node, host link, and degradation ratio to
    a different board. On a node of identical devices that answer even looks right.

    On a host with no NVML the AMD devices are used instead, in the order the `amdgpu` driver
    enumerates them, which is the order every index here already means on that host. Without
    it an Instinct node had no addresses, so its NUMA homes, its nearest NICs, and every
    renegotiated host link were invisible — the failure this whole module exists to catch,
    skipped for a whole vendor.

    Returns:
        Lowercased addresses with `""` where the driver did not publish one, empty when
        neither vendor is readable.
    """
    nv = _nvml()
    if nv is None:
        from batcher._internal.hardware.amd import amd_devices

        return tuple(device.address.lower() for device in amd_devices())
    out: list[str] = []
    for index in range(_device_count(nv)):
        handle = _read(lambda i=index: nv.nvmlDeviceGetHandleByIndex(i), None)
        if handle is None:
            out.append("")
            continue
        info = _read(lambda h=handle: nv.nvmlDeviceGetPciInfo(h), None)
        out.append(_decode(getattr(info, "busId", "")).lower() if info is not None else "")
    return tuple(out)


def device_pcie_links() -> tuple[PcieLink, ...]:
    """The negotiated host link of every local accelerator.

    Returns:
        One record per device, positionally aligned with the NVML index, empty without NVML.
        A device whose address did not resolve carries an all-zero record rather than being
        omitted, which keeps the alignment every caller indexes by. Each record carries both
        the negotiated and the capable link, so a caller can tell a slow device from a slow
        *link*.
    """
    return tuple(pcie_link(address) for address in gpu_pci_addresses())


def degraded_device_links() -> tuple[PcieLink, ...]:
    """The accelerators whose host link negotiated below what it is capable of.

    The check worth running once per node at startup. These devices pass every health check
    and every correctness test, and they feed at a fraction of the rate for the life of the
    node.

    Returns:
        The degraded links, in device order. Empty when every link is at full capability or
        when nothing could be read.
    """
    return degraded_pcie_links(gpu_pci_addresses())


def device_link_efficiency() -> float:
    """The fraction of nameplate host bandwidth this node's accelerators actually have.

    The *worst* device's ratio, not the mean. A stage is placed across the devices it is
    given, and its rate is set by the slowest link in that set, so averaging a degraded device
    away produces a transfer estimate no device on the node will meet.

    Returns:
        A ratio in `(0, 1]`. `1.0` when nothing is degraded and — deliberately — also when the
        links cannot be read, which keeps the nameplate assumption every caller already makes
        rather than inventing a penalty for a node it could not see.
    """
    links = device_pcie_links()
    # An unreadable link reports a ratio of 1.0, so it neither penalizes the node nor hides a
    # degraded sibling: the minimum is still taken over every device that *did* answer.
    ratios = [link.degradation_ratio for link in links if link.degradation_ratio > 0.0]
    return min(ratios) if ratios else 1.0


def device_numa_nodes() -> tuple[int, ...]:
    """The NUMA node each local accelerator hangs off, in device order.

    Returns:
        One entry per device whose address resolved, `-1` where the kernel does not publish a
        node. Empty when NVML or the PCI tree is unavailable.
    """
    return tuple(link.numa_node for link in device_pcie_links())


def device_cpu_affinity(address: str) -> tuple[int, ...]:
    """The CPUs sitting on the same NUMA node as one device.

    The set a device's host-side feeder threads should run on. A decode thread on the far
    socket reads its input across the inter-socket link, writes the staging buffer there, and
    then the DMA engine reads it back across the same link — three crossings for work that
    should have had none. On a dense GPU node with two sockets and eight devices, half the
    devices get that treatment by default, and the symptom is a device at 50% utilization with
    no visible cause.

    Args:
        address: The device's PCI address.

    Returns:
        CPU ids on the device's node, ascending, restricted to this process's affinity mask so
        a container pinned to a subset never gets told to use a core it cannot. Empty when the
        kernel does not publish the mapping, when the intersection is empty (a container pinned
        entirely to the far socket — where binding would be a hang, not an optimization), or on
        a machine with no NUMA information.
    """
    raw = _read_local_cpulist(address)
    if not raw:
        return ()
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        with contextlib.suppress(OSError):
            raw &= set(getaffinity(0))
    return tuple(sorted(raw))


def _read_local_cpulist(address: str) -> set[int]:
    """The CPU set the kernel says is local to a PCI device, empty when unpublished."""
    from batcher._internal.hardware.fabric.pcie import PCIE_SYSFS_ROOT
    from batcher._internal.hardware.topology import _parse_cpu_list

    try:
        with open(os.path.join(PCIE_SYSFS_ROOT, address, "local_cpulist")) as f:
            return _parse_cpu_list(f.read())
    except OSError:
        return set()


def visible_device_indices() -> tuple[int, ...]:
    """NVML indices of the devices this process may actually use, in the order CUDA sees them.

    The translation nothing else does, and getting it wrong is silent. NVML enumerates every
    device on the *host*; CUDA enumerates only those `CUDA_VISIBLE_DEVICES` names, renumbered
    from zero. A worker handed the host's device 5 therefore calls it device 0, and any code
    that asks NVML about "device 0" gets the wrong board's PCI address, NUMA node, and link —
    on a node where every device is identical, the answer even looks plausible.

    Returns:
        NVML indices in visible order. Every device when the variable is unset or empty (the
        CUDA default). An entry naming a UUID is resolved against NVML; one naming neither an
        index nor a known UUID is dropped, because a device this process cannot identify is one
        it must not make a placement decision about.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    nv = _nvml()
    count = _device_count(nv) if nv is not None else 0
    if raw is None or not raw.strip():
        return tuple(range(count))
    by_uuid = _uuid_indices(nv, count)
    out: list[int] = []
    for token in raw.split(","):
        entry = token.strip()
        if not entry:
            continue
        if entry.isdigit():
            index = int(entry)
            # CUDA stops at the first invalid entry rather than skipping it, and so does this:
            # the devices after an unresolvable one are not visible either.
            if index >= count:
                break
            out.append(index)
        elif entry in by_uuid:
            out.append(by_uuid[entry])
        else:
            break
    return tuple(out)


def _uuid_indices(nv, count: int) -> dict[str, int]:
    """Device UUID to NVML index, empty without a driver."""
    if nv is None:
        return {}
    out: dict[str, int] = {}
    for index in range(count):
        handle = _read(lambda i=index: nv.nvmlDeviceGetHandleByIndex(i), None)
        if handle is None:
            continue
        uuid = _decode(_read(lambda h=handle: nv.nvmlDeviceGetUUID(h), ""))
        if uuid:
            out[uuid] = index
    return out


def nearest_rdma_device(ordinal: int = 0) -> str:
    """The RDMA NIC closest to one accelerator on the PCI bus, or `""`.

    The pairing a GPU-to-fabric transfer depends on and nothing else establishes. A dense node
    carries several NICs, and traffic between a device and a NIC on a different root complex
    crosses the inter-socket link in both directions — for a transfer whose entire purpose is
    to leave the node. Choosing the first NIC in the list is right by luck on half a
    two-socket box.

    Ranked by `pcie_class`, so a NIC under the same switch beats one merely on the same root
    complex, which beats one across the socket. Ties break on device name, so the choice is
    stable across processes on the same node — two workers that disagree about which NIC is
    nearest would each be right and would still contend.

    Args:
        ordinal: The device's index as this process sees it (CUDA's numbering).

    Returns:
        The RDMA device name (`"mlx5_3"`), or `""` when there is no active fabric, when the
        device's address is unreadable, or when no NIC publishes one.
    """
    from batcher._internal.hardware.fabric.pcie import PCIE_CLASSES
    from batcher._internal.hardware.fabric.rdma import active_rdma_devices

    visible = visible_device_indices()
    addresses = gpu_pci_addresses()
    if ordinal < 0 or ordinal >= len(visible):
        return ""
    index = visible[ordinal]
    if index >= len(addresses) or not addresses[index]:
        return ""
    device_address = addresses[index]
    candidates = [n for n in active_rdma_devices() if n.pci_address]
    if not candidates:
        return ""
    ranked = min(
        candidates,
        key=lambda nic: (PCIE_CLASSES.index(pcie_class(device_address, nic.pci_address)), nic.name),
    )
    return ranked.name
