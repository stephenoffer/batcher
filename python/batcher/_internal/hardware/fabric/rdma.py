"""InfiniBand and RoCE NICs — the wire a cross-node shuffle actually runs on.

A GPU datacenter's network is not one number. A node carries several RDMA NICs, each with its
own rate, its own state, and its own fabric partition, and the difference between them decides
what a shuffle costs: eight 400 Gb/s InfiniBand ports sustain a different stage than one
25 Gb/s Ethernet port, and a run that assumes the former on a node provisioned with the latter
plans a shuffle it cannot afford. Nothing in a cluster manager reports this. Ray reports cores,
memory, and a GPU count; the NIC is invisible to it.

Three facts here that a scheduler cannot get anywhere else:

* **Aggregate rate.** The sum of the node's active port rates is the ceiling on everything the
  shuffle can move, and it is the denominator in any honest transfer-time estimate.
* **State.** A port that is `DOWN` or `INIT` carries nothing. Neoclouds ship nodes with more
  ports cabled than active, so counting ports rather than *active* ports over-estimates the
  fabric by an integer factor, which is exactly the direction that turns a plan into a stall.
* **Partition.** Two nodes on different InfiniBand partitions cannot reach each other over the
  fast path however close they are physically. The partition key is what `topology.FABRIC_LABEL`
  names, and reading it here means an operator does not have to label it by hand.

**Nothing here is fabricated.** A rate string the kernel does not publish reads as `0.0`, an
absent `/sys/class/infiniband` reads as no devices, and a caller that needs a number for an
unreadable fabric takes it from configuration. The alternative — assuming a plausible datacenter
NIC — produces a confident transfer estimate that is wrong by an order of magnitude on any node
that is not the one the constant was written for.
"""

from __future__ import annotations

import functools
import glob
import os
import re
from dataclasses import dataclass

__all__ = [
    "RDMA_SYSFS_ROOT",
    "RdmaDevice",
    "active_rdma_devices",
    "fabric_bandwidth_gbps",
    "fabric_interface_address",
    "fabric_partition",
    "rdma_available",
    "rdma_devices",
    "rdma_link_layers",
    "rdma_net_interfaces",
    "rdma_summary",
    "reset_fabric_probes",
]

#: Where the kernel publishes RDMA devices. A module-level constant rather than a literal
#: because tests point it at a fake tree, and because a container may bind-mount it elsewhere.
RDMA_SYSFS_ROOT = "/sys/class/infiniband"

#: `rate` reads as `"200 Gb/sec (4X HDR)"`. Only the leading figure is a fact about throughput;
#: the parenthesised part names the signalling, which varies in spelling across driver versions.
_RATE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*Gb/sec")

#: Port states the kernel reports, as `"4: ACTIVE"`. Only `ACTIVE` carries traffic; `INIT` and
#: `ARMED` are stages of bring-up that a subnet manager may never finish on a miscabled port.
_ACTIVE_STATE = "ACTIVE"

#: A PCI address as the kernel spells it in the device tree. A software RDMA device (`rxe`,
#: `siw`) resolves to something that is not one, and reads as "no address" rather than as a
#: fabricated one.
_ADDRESS_RE = re.compile(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]")


def _read_text(path: str) -> str:
    """The stripped contents of a `/sys` file, or `""` when it cannot be read."""
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def _parse_rate_gbps(raw: str) -> float:
    """Gigabits per second from a kernel `rate` string, or `0.0` when unparseable.

    Args:
        raw: The `rate` attribute, such as `"200 Gb/sec (4X HDR)"`.

    Returns:
        The rate in Gb/s, or `0.0` when the string is absent or in a shape this does not
        recognize. Zero reads downstream as "unknown", never as "a slow link".
    """
    match = _RATE_RE.match(raw)
    if match is None:
        return 0.0
    try:
        return float(match.group(1))
    except ValueError:
        return 0.0


def _parse_state(raw: str) -> str:
    """The state name from a kernel `state` string like `"4: ACTIVE"`."""
    if ":" in raw:
        return raw.split(":", 1)[1].strip().upper()
    return raw.strip().upper()


@dataclass(frozen=True, slots=True)
class RdmaDevice:
    """One RDMA port, as the kernel describes it.

    Attributes:
        name: Device name (`"mlx5_0"`), the handle a verbs consumer opens.
        port: Port number on that device, 1-based as the kernel numbers them.
        link_layer: `"InfiniBand"` or `"Ethernet"` (RoCE), `""` when not published. The two
            need different fabric configuration, so a caller that must choose cannot treat
            them as one class.
        rate_gbps: Negotiated port rate, `0.0` when unpublished.
        state: Port state (`"ACTIVE"`, `"DOWN"`, `"INIT"`, `"ARMED"`), `""` when unpublished.
        partition_key: Default pkey (`"0xffff"` on an unpartitioned fabric), `""` when absent.
            Two nodes with different keys are on different fabrics regardless of proximity.
        node_guid: The device's stable GUID, `""` when unpublished.
        pci_address: The NIC's PCI address (`"0000:0c:00.0"`), `""` when not resolvable. This
            is what pairs a NIC with the device that should use it for GPUDirect RDMA.
        numa_node: NUMA node the NIC is attached to, `-1` when unpublished. Staging a transfer
            on the wrong node crosses the interconnect twice.
    """

    name: str
    port: int
    link_layer: str = ""
    rate_gbps: float = 0.0
    state: str = ""
    partition_key: str = ""
    node_guid: str = ""
    pci_address: str = ""
    numa_node: int = -1

    @property
    def active(self) -> bool:
        """Whether this port is carrying traffic."""
        return self.state == _ACTIVE_STATE

    @property
    def roce(self) -> bool:
        """Whether this is RDMA over Converged Ethernet rather than native InfiniBand."""
        return self.link_layer.lower() == "ethernet"


def _pci_address(device_dir: str) -> str:
    """The PCI address behind an RDMA device directory, or `""`.

    `/sys/class/infiniband/<dev>/device` symlinks into the PCI tree, so the basename of the
    resolved target is the address. A virtual or software device (`rxe`, `siw`) resolves to
    something that is not an address, and reads as `""` rather than as a fabricated one.
    """
    try:
        target = os.path.realpath(os.path.join(device_dir, "device"))
    except OSError:
        return ""
    base = os.path.basename(target)
    return base if _ADDRESS_RE.fullmatch(base) else ""


def _ports(device_dir: str) -> list[int]:
    """Port numbers published under an RDMA device directory, ascending."""
    ports: list[int] = []
    for entry in glob.glob(os.path.join(device_dir, "ports", "*")):
        try:
            ports.append(int(os.path.basename(entry)))
        except ValueError:
            continue
    return sorted(ports)


@functools.lru_cache(maxsize=1)
def rdma_devices() -> tuple[RdmaDevice, ...]:
    """Every RDMA port on this node, in device then port order.

    Memoized: cabling does not change under a running process, and the callers ask on every
    placement decision. `reset_fabric_probes()` clears it for a test faking the `/sys` tree.

    Returns:
        One record per port, empty when the node has no RDMA hardware, when the driver is not
        loaded, or when the container did not mount `/sys/class/infiniband`.
    """
    out: list[RdmaDevice] = []
    for device_dir in sorted(glob.glob(os.path.join(RDMA_SYSFS_ROOT, "*"))):
        name = os.path.basename(device_dir)
        node_guid = _read_text(os.path.join(device_dir, "node_guid"))
        pci = _pci_address(device_dir)
        numa = _read_text(os.path.join(device_dir, "device", "numa_node"))
        try:
            numa_node = int(numa)
        except ValueError:
            numa_node = -1
        for port in _ports(device_dir):
            port_dir = os.path.join(device_dir, "ports", str(port))
            out.append(
                RdmaDevice(
                    name=name,
                    port=port,
                    link_layer=_read_text(os.path.join(port_dir, "link_layer")),
                    rate_gbps=_parse_rate_gbps(_read_text(os.path.join(port_dir, "rate"))),
                    state=_parse_state(_read_text(os.path.join(port_dir, "state"))),
                    partition_key=_read_text(os.path.join(port_dir, "pkeys", "0")),
                    node_guid=node_guid,
                    pci_address=pci,
                    numa_node=numa_node,
                )
            )
    return tuple(out)


def active_rdma_devices() -> tuple[RdmaDevice, ...]:
    """Only the ports that are up and carrying traffic.

    The set every bandwidth and placement decision is made against. A cabled-but-inactive port
    is indistinguishable from an absent one for anything the engine plans.

    Returns:
        The active subset of `rdma_devices()`, empty when nothing is up.
    """
    return tuple(d for d in rdma_devices() if d.active)


def rdma_available() -> bool:
    """Whether this node has at least one active RDMA port.

    Returns:
        True when an RDMA fast path exists. False on a node with no RDMA hardware, with the
        driver absent, or with every port down — all three of which mean the same thing to a
        caller choosing a transport.
    """
    return bool(active_rdma_devices())


def fabric_bandwidth_gbps() -> float:
    """The node's total active RDMA port rate, in gigabits per second.

    The ceiling on everything a cross-node stage can move, and the denominator of any transfer
    estimate worth making. Ports whose rate the kernel does not publish contribute nothing,
    so the figure is a lower bound rather than an optimistic one.

    Returns:
        Summed active port rate, `0.0` when no port is active or no rate is published.
    """
    return sum(d.rate_gbps for d in active_rdma_devices())


def rdma_link_layers() -> dict[str, int]:
    """How many active ports of each link layer the node has.

    Returns:
        Link-layer name to active port count (`{"InfiniBand": 8}`), empty when nothing is
        active. Ports whose link layer is unpublished are counted under `"unknown"` rather
        than being dropped, because a port that carries traffic still bounds the fabric.
    """
    counts: dict[str, int] = {}
    for device in active_rdma_devices():
        key = device.link_layer or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def fabric_partition() -> str:
    """The fabric partition this node is on, as a label value.

    The partition key of its active ports, which is what decides reachability over the fast
    path. A node whose ports disagree reports `""` rather than picking one, because a
    multi-partition node is not on *a* fabric and a caller must not treat it as though it is.

    Returns:
        The shared partition key, or `""` when unreadable, absent, or not unique.
    """
    keys = {d.partition_key for d in active_rdma_devices() if d.partition_key}
    return keys.pop() if len(keys) == 1 else ""


def rdma_summary() -> dict:
    """A flat description of the node's fabric, for the decision log and the dashboard.

    Returns:
        Port counts, aggregate rate, link layers, partition, and the distinct NIC models by
        name. Every field is zero or empty on a node with no readable fabric, which is the
        answer that keeps a caller on its configured default.
    """
    active = active_rdma_devices()
    return {
        "rdma_available": bool(active),
        "ports": len(rdma_devices()),
        "active_ports": len(active),
        "bandwidth_gbps": fabric_bandwidth_gbps(),
        "link_layers": rdma_link_layers(),
        "partition": fabric_partition(),
        "devices": sorted({d.name for d in active}),
        "numa_nodes": sorted({d.numa_node for d in active if d.numa_node >= 0}),
    }


def reset_fabric_probes() -> None:
    """Forget the memoized fabric readings, so the next call re-reads `/sys`.

    Registered with `reset_hardware_probes()`; called directly only by a test that fakes the
    sysfs tree, in the same shape every other probe in this package offers.
    """
    from batcher._internal.hardware.fabric import pcie

    rdma_devices.cache_clear()
    pcie.pcie_link.cache_clear()
    pcie.device_numa_node.cache_clear()


def rdma_net_interfaces() -> tuple[str, ...]:
    """Network interface names backed by an active RDMA port, in device order.

    The join nothing else makes: an RDMA device (`mlx5_0`) and the interface an IP socket
    would use to reach the same wire (`ib0`, or an Ethernet name under RoCE) are different
    identifiers for the same hardware, and only the kernel relates them. Without it a node's
    fast fabric is invisible to anything that dials an address — which is every TCP and gRPC
    connection the engine makes.

    Returns:
        Interface names, deduplicated, empty when no port is active or the kernel does not
        publish the mapping.
    """
    names: list[str] = []
    for device in active_rdma_devices():
        net_dir = os.path.join(RDMA_SYSFS_ROOT, device.name, "device", "net")
        try:
            entries = sorted(os.listdir(net_dir))
        except OSError:
            continue
        for name in entries:
            if name not in names:
                names.append(name)
    return tuple(names)


def fabric_interface_address() -> str:
    """The IPv4 address of this node's fastest active fabric interface, or `""`.

    What a peer should dial to reach this node over the fabric rather than over whatever
    interface the default route happens to name. On a GPU node those are usually different
    wires by two orders of magnitude: the management NIC carries the node's advertised IP,
    while the InfiniBand ports carry nothing unless something addresses them.

    Returns:
        A dotted-quad address, or `""` when no active fabric interface has one — an
        InfiniBand port with no IPoIB address configured is common and is not an error, it
        simply means the fabric is not IP-addressable and a caller must keep its existing
        address.
    """
    interfaces = rdma_net_interfaces()
    if not interfaces:
        return ""
    try:
        import socket

        import psutil

        addrs = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
    except Exception:
        return ""
    for name in interfaces:
        # A configured-but-down interface has an address that nothing answers on, which is
        # worse than having none: a peer dialing it waits for a timeout instead of failing.
        if not getattr(stats.get(name), "isup", False):
            continue
        for addr in addrs.get(name, ()):
            if addr.family == socket.AF_INET and addr.address:
                return str(addr.address)
    return ""
