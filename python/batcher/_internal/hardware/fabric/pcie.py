"""A device's PCIe link, its NUMA home, and how far two devices sit apart on the bus.

Everything that is not on NVLink reaches a GPU over PCIe, and the link is not the width the
datasheet promises. Renegotiation to a lower generation or a narrower width is a routine
consequence of a reseated card, a riser, a marginal cable, or a power event, and the machine
keeps working afterwards — at a quarter of the host-to-device bandwidth, with nothing in any
log to say so. On a fleet where a staging pipeline is already the thing feeding the device,
that is the difference between a saturated GPU and one running at 25%.

Distance on the bus matters for a second reason. Two devices under one PCIe switch exchange
peer-to-peer without touching the host; two under different root complexes exchange across the
inter-socket link, which is both slower and contended with every other socket-crossing access.
The same distance decides which NIC a device should use for GPUDirect RDMA: the one on its own
root complex, not the one that happens to be first in the list.

Both facts come off `/sys/bus/pci`. Off Linux, or in a container without the PCI tree mounted,
every entry point reports the neutral answer — an unknown link and the coarsest distance class
— and callers keep the behavior they had before this existed.
"""

from __future__ import annotations

import functools
import os
import re
from dataclasses import dataclass

from batcher._internal.hardware.sysfs import read_text

__all__ = [
    "PCIE_CLASSES",
    "PCIE_SYSFS_ROOT",
    "PcieLink",
    "degraded_pcie_links",
    "device_numa_node",
    "pcie_bandwidth_gbps",
    "pcie_class",
    "pcie_link",
]

#: Where the kernel publishes PCI devices. A constant so a test can point it at a fake tree.
PCIE_SYSFS_ROOT = "/sys/bus/pci/devices"

#: Distance classes between two PCI devices, closest first. The names are NVIDIA's, because
#: `nvidia-smi topo -m` is what every operator on a GPU fleet already reads, and inventing a
#: second vocabulary for the same fact would mean translating between them in every report.
#:
#: * `pix` — one PCIe switch between them; peer-to-peer stays on the switch.
#: * `pxb` — several switches, still below one host bridge.
#: * `phb` — the same host bridge (root complex), so traffic reaches the CPU and turns around.
#: * `node` — different host bridges on one NUMA node.
#: * `sys` — different NUMA nodes; traffic crosses the inter-socket link.
PCIE_CLASSES = ("pix", "pxb", "phb", "node", "sys")

#: Effective per-lane throughput in Gb/s, by PCIe generation, after line encoding (8b/10b for
#: gen 1-2, 128b/130b from gen 3). These are the figures the specification's own throughput
#: tables give, not the raw transfer rates, because what a caller sizes a staging buffer
#: against is payload bandwidth.
_LANE_GBPS: dict[int, float] = {
    1: 2.0,
    2: 4.0,
    3: 7.877,
    4: 15.754,
    5: 31.508,
    6: 63.015,
}

#: Raw transfer rate to generation. The kernel publishes `"16.0 GT/s PCIe"`; the generation is
#: what every other source names, so it is what this module reports.
_GT_TO_GEN: dict[str, int] = {
    "2.5": 1,
    "5.0": 2,
    "8.0": 3,
    "16.0": 4,
    "32.0": 5,
    "64.0": 6,
}

_ADDRESS_RE = re.compile(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]")


def _parse_gen(raw: str) -> int:
    """PCIe generation from a kernel link-speed string, or `0` when unparseable."""
    token = raw.split()[0] if raw else ""
    return _GT_TO_GEN.get(token, 0)


def _parse_width(raw: str) -> int:
    """Lane count from a kernel link-width string (`"x16"` or `"16"`), or `0`."""
    try:
        return int(raw.lstrip("xX"))
    except ValueError:
        return 0


@dataclass(frozen=True, slots=True)
class PcieLink:
    """One device's PCIe link, current against capable.

    Attributes:
        address: PCI address (`"0000:0c:00.0"`).
        gen: Negotiated generation, `0` when unpublished.
        width: Negotiated lane count, `0` when unpublished.
        max_gen: Generation both ends are capable of, `0` when unpublished.
        max_width: Lane count both ends are capable of, `0` when unpublished.
        numa_node: NUMA node the device hangs off, `-1` when unpublished.
    """

    address: str
    gen: int = 0
    width: int = 0
    max_gen: int = 0
    max_width: int = 0
    numa_node: int = -1

    @property
    def bandwidth_gbps(self) -> float:
        """Payload bandwidth of the negotiated link, `0.0` when either half is unknown."""
        return pcie_bandwidth_gbps(self.gen, self.width)

    @property
    def max_bandwidth_gbps(self) -> float:
        """Payload bandwidth the link is capable of, `0.0` when either half is unknown."""
        return pcie_bandwidth_gbps(self.max_gen, self.max_width)

    @property
    def degraded(self) -> bool:
        """Whether the link negotiated below what both ends can do.

        False when either side of the comparison is unpublished: an unreadable link is not
        evidence of a degraded one, and reporting it as degraded would send an operator to
        inspect healthy hardware.
        """
        if not (self.gen and self.width and self.max_gen and self.max_width):
            return False
        return self.gen < self.max_gen or self.width < self.max_width

    @property
    def degradation_ratio(self) -> float:
        """Negotiated bandwidth as a fraction of capable, `1.0` when not comparable.

        The figure that says how much a degraded link is actually costing: a x16 gen-5 device
        running at x8 gen-4 reports `0.25`, which is the factor its host-to-device transfers
        are multiplied by.
        """
        capable = self.max_bandwidth_gbps
        if capable <= 0.0 or self.bandwidth_gbps <= 0.0:
            return 1.0
        return min(1.0, self.bandwidth_gbps / capable)


def pcie_bandwidth_gbps(gen: int, width: int) -> float:
    """Payload bandwidth of a PCIe link, in gigabits per second.

    Args:
        gen: PCIe generation, 1 through 6.
        width: Lane count.

    Returns:
        Effective payload bandwidth, or `0.0` when the generation is outside the table or the
        width is not positive. Zero reads as "unknown" downstream, never as "a slow link".
    """
    if width <= 0:
        return 0.0
    return _LANE_GBPS.get(gen, 0.0) * width


@functools.lru_cache(maxsize=64)
def pcie_link(address: str) -> PcieLink:
    """The link state of one PCI device.

    Memoized per address: a link renegotiates on a hardware event, not under a running query,
    and the callers ask per placement decision. `reset_fabric_probes()` clears it.

    Args:
        address: PCI address, as `"0000:0c:00.0"`.

    Returns:
        The link, with `0` in every field the kernel does not publish. An address that does
        not exist reports an all-zero record rather than raising, because a caller resolving a
        device that has since been unbound should degrade, not fail.
    """
    if not _ADDRESS_RE.fullmatch(address or ""):
        return PcieLink(address=address or "")
    base = os.path.join(PCIE_SYSFS_ROOT, address)
    return PcieLink(
        address=address,
        gen=_parse_gen(read_text(os.path.join(base, "current_link_speed"))),
        width=_parse_width(read_text(os.path.join(base, "current_link_width"))),
        max_gen=_parse_gen(read_text(os.path.join(base, "max_link_speed"))),
        max_width=_parse_width(read_text(os.path.join(base, "max_link_width"))),
        numa_node=device_numa_node(address),
    )


@functools.lru_cache(maxsize=64)
def device_numa_node(address: str) -> int:
    """The NUMA node a PCI device hangs off, or `-1` when unpublished.

    What decides where a staging buffer for that device should be allocated: a host buffer on
    the wrong node crosses the inter-socket link on every transfer, in addition to the PCIe
    hop it was always going to make.

    Args:
        address: PCI address.

    Returns:
        The NUMA node id, or `-1` on a single-node machine, off Linux, or when the kernel does
        not publish it. Callers read `-1` as "no preference".
    """
    raw = read_text(os.path.join(PCIE_SYSFS_ROOT, address, "numa_node"))
    try:
        node = int(raw)
    except ValueError:
        return -1
    return node if node >= 0 else -1


def _pci_path(address: str) -> tuple[str, ...]:
    """The PCI bridge chain above a device, root first, empty when unresolvable.

    `/sys/bus/pci/devices/<addr>` symlinks to its position in the device tree, so the resolved
    path spells out every bridge between the root complex and the device. Comparing two of
    those chains is how bus distance is established without a vendor tool.
    """
    try:
        target = os.path.realpath(os.path.join(PCIE_SYSFS_ROOT, address))
    except OSError:
        return ()
    parts = [p for p in target.split(os.sep) if p]
    try:
        start = next(i for i, p in enumerate(parts) if p.startswith("pci"))
    except StopIteration:
        return ()
    return tuple(parts[start:])


def pcie_class(a: str, b: str) -> str:
    """How far apart two PCI devices are, as a class from `PCIE_CLASSES`.

    Args:
        a: One device's PCI address.
        b: The other's.

    Returns:
        The closest class that applies. Two unresolvable addresses report `"sys"`, the
        coarsest class, because assuming distance costs a caller a suboptimal placement while
        assuming proximity costs it a peer-to-peer transfer that silently routes through the
        host.
    """
    if a and a == b:
        return "pix"
    path_a, path_b = _pci_path(a), _pci_path(b)
    if not path_a or not path_b:
        return "sys"
    if path_a[0] != path_b[0]:
        node_a, node_b = device_numa_node(a), device_numa_node(b)
        if node_a >= 0 and node_a == node_b:
            return "node"
        return "sys"
    # Shared root complex: the number of common bridges below it says how many switches the
    # traffic stays inside. Both devices' own entries are excluded from the comparison.
    common = 0
    for x, y in zip(path_a[1:-1], path_b[1:-1], strict=False):
        if x != y:
            break
        common += 1
    if common >= 2:
        return "pix"
    if common == 1:
        return "pxb"
    return "phb"


def degraded_pcie_links(addresses: tuple[str, ...]) -> tuple[PcieLink, ...]:
    """The links among `addresses` that negotiated below what they are capable of.

    The check worth running once at startup on every accelerator node. A device on a
    half-width link is not broken and will not fail a health check; it will simply feed at
    half rate for the life of the node, which is invisible from a task's own timings.

    Args:
        addresses: PCI addresses to inspect.

    Returns:
        The degraded links, in the order given. Empty when every link is at full negotiated
        capability or when the kernel publishes nothing to compare.
    """
    return tuple(link for a in addresses if (link := pcie_link(a)).degraded)
