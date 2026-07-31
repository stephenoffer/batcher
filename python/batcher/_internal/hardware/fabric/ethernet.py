"""The node's Ethernet links, for the large share of GPU capacity that has no RDMA.

`rdma` reads InfiniBand and RoCE, which is what the top tier of GPU capacity is wired with.
Most of it is not. A rented eight-GPU node on the commodity tier reaches the network through
ordinary Ethernet at 25, 100, or 200 Gb/s, and on that node `fabric_bandwidth_gbps()` returned
zero — not "slow", but "unknown", which sent the cost model back to its built-in default. The
default is a guess; `/sys/class/net/<iface>/speed` is a measurement, and it is available on
every Linux host without a driver, a library, or a privilege.

Three things make the reading harder than opening one file, and each has a wrong answer that
looks plausible:

* **A container sees interfaces it cannot use.** `lo`, `docker0`, a veth pair, and a CNI bridge
  are all in `/sys/class/net` and none of them carries a byte off the node. A physical
  interface is one with a `device` symlink into the PCI or platform bus, which is the check
  used here; the virtual ones have no such link.
* **A bonded pair is one link, not two.** Summing a bond's members double-counts, because the
  bond and its slaves both appear. Slaves are identified by their `master` link and excluded,
  so the bond's own published speed — which is the aggregate — is the figure that counts.
* **Speed is often unreadable, and `-1` when it is.** The kernel returns `-1` from a virtual
  interface, from one that is down, and from some drivers that do not implement it. That is
  unknown, not zero, and it must not be summed as though the link were absent.

The convention this file keeps, like the rest of the package: zero means unknown, and a caller
that needs to distinguish "no network" from "nobody looked" asks whether any interface was
found at all rather than reading the figure.

A neutral utility: any layer may import `_internal`.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

__all__ = [
    "ETHERNET_SYSFS_ROOT",
    "EthernetLink",
    "ethernet_bandwidth_gbps",
    "ethernet_links",
    "ethernet_summary",
]

#: Where the kernel publishes network interfaces. A constant so a test can fake the tree.
ETHERNET_SYSFS_ROOT = "/sys/class/net"


@dataclass(frozen=True)
class EthernetLink:
    """One physical network interface.

    Attributes:
        name: Interface name, such as `"ens5"` or `"bond0"`.
        speed_gbps: Negotiated line rate in gigabits per second, `0.0` when the kernel does
            not publish one. Not throughput: a 100 Gb/s link delivers well under that to a
            single stream, and this is the ceiling rather than the expectation.
        up: Whether the interface is operationally up and has carrier.
        bonded: Whether this interface is a bond aggregating others.
        address: PCI address of the underlying device, or `""` for a bond or a platform
            device. Lets a caller pair an interface with the NUMA node it hangs off.
    """

    name: str
    speed_gbps: float = 0.0
    up: bool = False
    bonded: bool = False
    address: str = ""


def _read(path: str) -> str:
    """One sysfs attribute, stripped, or `""` when it cannot be read."""
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def _speed_gbps(iface_dir: str) -> float:
    """Negotiated line rate in Gb/s, or `0.0`.

    The kernel publishes megabits per second, and returns `-1` for an interface that is down,
    is virtual, or whose driver does not implement the query. All three are unknown.
    """
    raw = _read(os.path.join(iface_dir, "speed"))
    try:
        mbps = int(raw)
    except ValueError:
        return 0.0
    return mbps / 1000.0 if mbps > 0 else 0.0


def _is_physical(iface_dir: str) -> bool:
    """Whether an interface is backed by real hardware rather than by the kernel.

    A physical interface has a `device` link into a bus. Loopback, bridges, veth pairs, tun
    devices, and the interfaces a container runtime creates do not, which is what keeps a
    node's fabric estimate from counting a docker bridge as network capacity.
    """
    return os.path.exists(os.path.join(iface_dir, "device"))


def _is_slave(iface_dir: str) -> bool:
    """Whether an interface is enslaved to a bond or bridge, and so already counted by it."""
    return os.path.exists(os.path.join(iface_dir, "master"))


def _is_bond(iface_dir: str) -> bool:
    """Whether an interface is a bond. `bonding/` is present only on the aggregate."""
    return os.path.isdir(os.path.join(iface_dir, "bonding"))


def _pci_address(iface_dir: str) -> str:
    """PCI address behind an interface, or `""` for a bond or a platform device."""
    device = os.path.join(iface_dir, "device")
    if not os.path.exists(device):
        return ""
    base = os.path.basename(os.path.realpath(device))
    return base if base.count(":") == 2 else ""


def ethernet_links() -> tuple[EthernetLink, ...]:
    """Every network interface that can carry traffic off this node, in name order.

    Bonds are reported and their members are not, so a bonded pair counts once. Virtual
    interfaces are excluded entirely.

    Returns:
        One entry per usable interface, empty off Linux and in a container without
        `/sys/class/net`.
    """
    links: list[EthernetLink] = []
    for iface_dir in sorted(glob.glob(os.path.join(ETHERNET_SYSFS_ROOT, "*"))):
        name = os.path.basename(iface_dir)
        bond = _is_bond(iface_dir)
        if not bond and (not _is_physical(iface_dir) or _is_slave(iface_dir)):
            continue
        links.append(
            EthernetLink(
                name=name,
                speed_gbps=_speed_gbps(iface_dir),
                up=_read(os.path.join(iface_dir, "operstate")) == "up"
                and _read(os.path.join(iface_dir, "carrier")) == "1",
                bonded=bond,
                address=_pci_address(iface_dir),
            )
        )
    return tuple(links)


def ethernet_bandwidth_gbps() -> float:
    """Summed line rate of this node's up Ethernet links, in Gb/s.

    The figure a cost model should price a shuffled byte against when there is no RDMA fabric.
    It is an upper bound on what the node can move and a poor estimate of what one stream will
    see, which is why the caller applies its own efficiency factor rather than this returning
    a discounted number: a discount applied here would be invisible to a caller that already
    applied one.

    Returns:
        Summed rate over up interfaces, `0.0` when none is up or none publishes a rate.
    """
    return sum(link.speed_gbps for link in ethernet_links() if link.up)


def ethernet_summary() -> dict:
    """A flat description of the node's Ethernet, for the decision log and the dashboard.

    Returns:
        `interfaces`, `up`, and `total_gbps`, plus `fastest` — the name of the highest-rate up
        interface, which is the one a transfer will actually take and the one worth naming in
        a report.
    """
    links = ethernet_links()
    up = [link for link in links if link.up]
    fastest = max(up, key=lambda link: link.speed_gbps, default=None)
    return {
        "interfaces": len(links),
        "up": len(up),
        "total_gbps": round(sum(link.speed_gbps for link in up), 1),
        "fastest": fastest.name if fastest is not None else "",
    }
