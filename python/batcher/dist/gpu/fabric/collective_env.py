"""Telling the collective library which wires this node has, instead of letting it guess.

A multi-GPU stage that runs its own collective — a tensor-parallel model, an all-reduce inside
a UDF, any NCCL or RCCL group — discovers the node's fabric by probing at initialization. The
probe is good and it is not omniscient, and on rented GPU capacity it goes wrong in ways that
never surface as an error:

* It picks a NIC per device by its own ordering, which on a multi-rail node is a coin toss:
  half the devices end up on a NIC across the socket from them, and every byte they send
  crosses the inter-socket link twice.
* It cannot tell an RDMA NIC that carries the fabric from the management interface next to it,
  so a socket-based fallback can land on the 1 Gb/s wire while a 400 Gb/s port sits idle.
* It enables peer-to-peer on pairs the bus cannot actually serve, then discovers the copies are
  staged through host memory after paying to find out.

Batcher has already measured all three (`fabric.rails`, `fabric.rdma`, `fabric.p2p`), so it can
hand the library the answers rather than let it re-derive them. This module turns the measured
topology into the standard environment variables and puts them in the GPU task's `runtime_env`.

Two rules make this safe. **Nothing is invented**: a variable is set only when a probe answered,
so a node whose fabric cannot be read gets the empty environment and the library's own probe,
exactly as before. And **the operator always wins**: a variable already set in the environment
or in the caller's `runtime_env` is never overwritten, because a deployment that pins
`NCCL_IB_HCA` has a reason that no probe can see.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

__all__ = [
    "COLLECTIVE_VARS",
    "collective_env",
    "gdr_level",
    "ib_hca_list",
    "merge_env",
    "node_collective_env",
    "p2p_disabled",
    "socket_ifnames",
]

#: Every variable this module may set. Named as a set so a caller — or a test — can assert that
#: nothing outside it is touched, and so `merge_env` has one list to check the operator's own
#: settings against rather than a literal repeated per variable.
COLLECTIVE_VARS = (
    "NCCL_IB_HCA",
    "NCCL_SOCKET_IFNAME",
    "NCCL_NET_GDR_LEVEL",
    "NCCL_P2P_DISABLE",
    "NCCL_CROSS_NIC",
)

#: How close a NIC must be to a device before GPUDirect RDMA is worth using, expressed in the
#: library's own vocabulary. `PIX` means "same PCIe switch"; at greater distances the DMA path
#: crosses a host bridge, where it is no longer faster than staging through host memory and is
#: measurably less reliable on some chipsets. The library's own default is more permissive,
#: which is the conservative direction for *it* and the wrong one here: the node's real distance
#: is known.
_GDR_BY_CLASS = {"pix": "PIX", "pxb": "PXB", "phb": "PHB"}


def ib_hca_list(assignment: Mapping[int, str] | None = None) -> str:
    """The rail-aligned NIC list, as `NCCL_IB_HCA` spells it.

    One NIC per device in device order, which is how the variable is read when it names several:
    device `i` of the local group uses entry `i`. That is exactly the rail map, so the variable
    is the map written out.

    Args:
        assignment: A device-to-NIC map from `fabric.rails.rail_map`, or `None` to read one
            live.

    Returns:
        A comma-separated list (`"mlx5_0,mlx5_1"`), or `""` when no rail map could be read —
        which leaves the library to choose, as it did before.
    """
    from batcher._internal.hardware.fabric.rails import rail_map

    placed = rail_map() if assignment is None else assignment
    if not placed:
        return ""
    return ",".join(placed[ordinal] for ordinal in sorted(placed))


def socket_ifnames(interfaces: Sequence[str] | None = None) -> str:
    """The interfaces a socket-based collective should dial, as `NCCL_SOCKET_IFNAME` spells it.

    The fabric's own interfaces, not the node's first one. Bootstrap traffic is small, so the
    wire it uses rarely shows up in a profile — but a bootstrap that lands on a management
    interface the fabric subnet cannot route to does not run slowly, it hangs at
    initialization, which is the hardest failure in this area to attribute.

    Args:
        interfaces: Interface names, or `None` to take the RDMA-backed ones live.

    Returns:
        A comma-separated list, or `""` when no fabric interface could be read.
    """
    from batcher._internal.hardware.fabric.rdma import rdma_net_interfaces

    names = rdma_net_interfaces() if interfaces is None else tuple(interfaces)
    return ",".join(names)


def gdr_level(device_class: str = "") -> str:
    """How close a NIC must be to a device for GPUDirect RDMA, as `NCCL_NET_GDR_LEVEL` spells it.

    Args:
        device_class: The class between the device and its rail's NIC, from `fabric.pcie` or
            `fabric.p2p`. Empty means unmeasured.

    Returns:
        A level name, or `""` when the distance was not measured or is beyond the point where
        the DMA path pays — in which case the library keeps its own default rather than being
        told to use a path this node's bus does not serve well.
    """
    return _GDR_BY_CLASS.get(device_class, "")


def p2p_disabled(staged_pairs: int = 0, devices: int = 0) -> str:
    """Whether to turn peer-to-peer off, as `NCCL_P2P_DISABLE` spells it.

    Turned off only when *every* pair on the node has to stage through host memory. A node
    with some direct pairs keeps peer-to-peer on, because the library uses it per pair and
    disabling it globally would give up the pairs that work to avoid the ones that do not.

    Args:
        staged_pairs: Pairs that must bounce through host memory, from `p2p.host_staged_pairs`.
        devices: How many devices the node has.

    Returns:
        `"1"` when peer-to-peer cannot help at all, `""` (no opinion) otherwise — including on
        a node whose topology was not read, where the library's probe is better than a guess.
    """
    if devices < 2 or staged_pairs <= 0:
        return ""
    return "1" if staged_pairs == devices * (devices - 1) // 2 else ""


def collective_env(
    *,
    assignment: Mapping[int, str] | None = None,
    interfaces: Sequence[str] | None = None,
    device_class: str = "",
    staged_pairs: int = 0,
    devices: int = 0,
) -> dict[str, str]:
    """The whole collective environment for this node, from what its wires actually are.

    Every argument defaults to reading the topology live; passing them makes the function pure,
    which is how a node shape is tested on a host that does not have it.

    Args:
        assignment: Device-to-NIC map, or `None` to read one live.
        interfaces: Fabric interface names, or `None` to read them live.
        device_class: Class between a device and its NIC, `""` when unmeasured.
        staged_pairs: Pairs that must stage through the host.
        devices: Device count on the node.

    Returns:
        Variable to value, containing only the variables a probe could answer. Empty on a node
        whose fabric is unreadable, which is the whole of the degradation path: the collective
        library then probes for itself exactly as it did before this existed.
    """
    env: dict[str, str] = {}
    hca = ib_hca_list(assignment)
    if hca:
        env["NCCL_IB_HCA"] = hca
        # One NIC per device is a rail-aligned layout, and a ring that hops between NICs
        # inside it undoes the alignment. Told only when the alignment is known to exist.
        env["NCCL_CROSS_NIC"] = "0"
    ifnames = socket_ifnames(interfaces)
    if ifnames:
        env["NCCL_SOCKET_IFNAME"] = ifnames
    level = gdr_level(device_class)
    if level:
        env["NCCL_NET_GDR_LEVEL"] = level
    disable = p2p_disabled(staged_pairs, devices)
    if disable:
        env["NCCL_P2P_DISABLE"] = disable
    return env


def merge_env(
    base: Mapping[str, str] | None,
    derived: Mapping[str, str],
    *,
    process_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Fold the derived variables into an existing environment without overriding a decision.

    A variable already set — in the caller's environment block or in this process, which is
    what a worker inherits — is left exactly as it is. A deployment that pins `NCCL_IB_HCA`
    has a reason no probe can see: a partitioned fabric, a NIC reserved for storage, a vendor
    workaround. Silently replacing it with a measurement would be correct on paper and wrong
    in the one deployment that had to write it down.

    Args:
        base: The caller's existing environment block, or `None`.
        derived: What `collective_env` produced.
        process_env: The environment to check for operator settings, or `None` for this
            process's own.

    Returns:
        A new dict; the inputs are not modified.
    """
    ambient = os.environ if process_env is None else process_env
    out = dict(base or {})
    for key, value in derived.items():
        if key in out or ambient.get(key):
            continue
        out[key] = value
    return out


def node_collective_env() -> dict[str, str]:
    """The collective environment for the node this call runs on, read live.

    The one entry point that touches the machine; everything else is pure. Used by the GPU
    task's `runtime_env` so a worker inherits the node's measured fabric rather than probing
    for it.

    Returns:
        Variable to value, empty when nothing about the node's fabric could be read.
    """
    from batcher._internal.hardware.fabric.device_links import gpu_pci_addresses
    from batcher._internal.hardware.fabric.p2p import host_staged_pairs, peer_matrix
    from batcher._internal.hardware.fabric.pcie import PCIE_CLASSES, pcie_class
    from batcher._internal.hardware.fabric.rails import rail_map
    from batcher._internal.hardware.fabric.rdma import active_rdma_devices

    placed = rail_map()
    matrix = peer_matrix()
    addresses = gpu_pci_addresses()
    nic_addresses = {n.name: n.pci_address for n in active_rdma_devices()}
    # The distance a GPUDirect path would cross, taken from the *worst* device-to-NIC pair in
    # the rail map: the level is one setting for the node, and sizing it on the best pair
    # would enable the DMA path for a device whose NIC is across the socket.
    classes = [
        pcie_class(addresses[ordinal], nic_addresses.get(nic, ""))
        for ordinal, nic in placed.items()
        if 0 <= ordinal < len(addresses) and addresses[ordinal] and nic_addresses.get(nic)
    ]
    worst = max(classes, key=PCIE_CLASSES.index) if classes else ""
    return collective_env(
        assignment=placed,
        device_class=worst,
        staged_pairs=len(host_staged_pairs(matrix)),
        devices=len(matrix),
    )
