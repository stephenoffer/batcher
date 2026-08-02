"""Which NIC each accelerator leaves the node through, and whether the rails are balanced.

A dense GPU node has more than one NIC, and the pairing is not decorative. A transfer between
a device and a NIC under a different root complex crosses the inter-socket link in both
directions, for a transfer whose whole purpose is to leave the node; on an eight-device,
eight-NIC box the wrong pairing is a 2-4x loss on every cross-node byte, and nothing reports
it. `device_links.nearest_rdma_device` answers this for *one* device, greedily. That is the
right answer per device and the wrong one per node: eight devices asked independently can all
name the same NIC, and then one rail carries the whole shuffle while seven sit idle.

This module is the node-wide version. `assign_rails` is the pure decision — devices and NICs
in, a device-to-NIC map out — and it balances: a device takes its closest NIC unless that NIC
already holds its share and an equally close one is free. The live wrappers (`rail_map`,
`rails`, `rail_summary`) read the topology and apply it, degrading to an empty map on a host
with no RDMA fabric, no PCI tree, or no accelerators, where the caller keeps whatever NIC
selection it had.

`rail_imbalance` is the figure worth alerting on: rails balanced 1:1 with devices carry the
node's full port rate, and a node whose devices all landed on one rail carries a fraction of
it while every counter says the fabric is healthy.

A neutral utility: any layer may import `_internal`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

__all__ = [
    "Rail",
    "assign_rails",
    "device_rail_bandwidth_gbps",
    "rail_aligned",
    "rail_for_device",
    "rail_imbalance",
    "rail_map",
    "rail_summary",
    "rails",
]


@dataclass(frozen=True)
class Rail:
    """One NIC and the accelerators that leave the node through it.

    Attributes:
        nic: RDMA device name (`"mlx5_3"`), the handle a verbs consumer opens.
        rate_gbps: The port's negotiated rate, `0.0` when the kernel does not publish one.
        devices: Accelerator ordinals assigned to this rail, ascending. Empty for a NIC no
            device was placed on, which is itself the finding on an unbalanced node.
        numa_node: The NIC's NUMA home, `-1` when unpublished.
    """

    nic: str
    rate_gbps: float = 0.0
    devices: tuple[int, ...] = field(default_factory=tuple)
    numa_node: int = -1

    @property
    def loaded(self) -> bool:
        """Whether any device leaves through this rail."""
        return bool(self.devices)

    @property
    def share_gbps(self) -> float:
        """Port rate divided by the devices contending for it, `0.0` when nothing is on it.

        The figure a staging plan should size against: a device on a 400 Gb/s rail it shares
        with three others has 100 Gb/s, and sizing against the port rate over-commits by 4x.
        """
        return self.rate_gbps / len(self.devices) if self.devices else 0.0


def assign_rails(
    device_addresses: Sequence[str],
    nics: Sequence[tuple[str, str]],
    distance: Callable[[str, str], int],
) -> dict[int, str]:
    """Assign each accelerator to a NIC, closest first, balanced across equals.

    The node-wide answer to the question `nearest_rdma_device` answers per device. Devices are
    placed in order of how *decided* their choice is: a device with one clearly closest NIC is
    placed before one whose NICs are all equidistant, so the constrained placements take their
    rail before the free ones consume it. Within a distance tier the least-loaded NIC wins, so
    a node whose devices all sit under one switch still spreads across that switch's NICs.

    Balance never overrides distance. A device is moved to a second NIC only when that NIC is
    *equally* close: crossing a socket to balance a rail costs more than the imbalance saves.

    Args:
        device_addresses: PCI address per accelerator ordinal. An empty entry is a device whose
            address could not be read; it is left unassigned rather than placed on a guess.
        nics: `(name, pci_address)` per candidate NIC, already filtered to active ports.
        distance: Callable `(a, b) -> int` ranking two PCI addresses, lower being closer.

    Returns:
        Accelerator ordinal to NIC name. Devices with no readable address, and every device
        when there are no NICs, are absent — an absent entry means "no opinion", which leaves
        the caller's existing selection alone.
    """
    if not nics:
        return {}

    ordered: list[tuple[int, list[tuple[int, str]]]] = []
    for ordinal, address in enumerate(device_addresses):
        if not address:
            continue
        ranked = sorted((int(distance(address, nic_addr)), name) for name, nic_addr in nics)
        if ranked:
            ordered.append((ordinal, ranked))
    # Placed in order of how much each device stands to lose by not getting its first choice
    # (the spread between its closest and furthest NIC). The constrained devices take their
    # rail before the ones that are equidistant from everything consume it, so balancing never
    # displaces a device that had only one good answer.
    ordered.sort(key=lambda item: (-(item[1][-1][0] - item[1][0][0]), item[0]))

    load: dict[str, int] = {name: 0 for name, _ in nics}
    out: dict[int, str] = {}
    for ordinal, ranked in ordered:
        best_class = ranked[0][0]
        equals = [name for cls, name in ranked if cls == best_class]
        chosen = min(equals, key=lambda name: (load[name], name))
        out[ordinal] = chosen
        load[chosen] += 1
    return dict(sorted(out.items()))


def rail_map() -> dict[int, str]:
    """The live device-to-NIC assignment for this node.

    Reads the accelerators' PCI addresses and the active RDMA ports, then applies
    `assign_rails`. Uses the *visible* device ordering, so a worker handed a subset through
    `CUDA_VISIBLE_DEVICES` gets ordinals it can pass to CUDA rather than host indices.

    Returns:
        Accelerator ordinal to NIC name, empty on a node with no RDMA fabric, no readable PCI
        tree, or no accelerators.
    """
    from batcher._internal.hardware.fabric.device_links import (
        gpu_pci_addresses,
        visible_device_indices,
    )
    from batcher._internal.hardware.fabric.pcie import PCIE_CLASSES, pcie_class
    from batcher._internal.hardware.fabric.rdma import active_rdma_devices

    addresses = gpu_pci_addresses()
    if not addresses:
        return {}
    visible = [addresses[i] if i < len(addresses) else "" for i in visible_device_indices()]
    candidates = [(n.name, n.pci_address) for n in active_rdma_devices() if n.pci_address]
    return assign_rails(visible, candidates, lambda a, b: PCIE_CLASSES.index(pcie_class(a, b)))


def rails(assignment: Mapping[int, str] | None = None) -> tuple[Rail, ...]:
    """Every active NIC on the node with the devices assigned to it.

    Args:
        assignment: A device-to-NIC map, or `None` to take one live.

    Returns:
        One `Rail` per active NIC, ordered by name. NICs with no device are included: a rail
        nothing was placed on is the evidence of an imbalance, and dropping it would hide the
        thing this module exists to show.
    """
    from batcher._internal.hardware.fabric.rdma import active_rdma_devices

    placed = rail_map() if assignment is None else dict(assignment)
    by_nic: dict[str, list[int]] = defaultdict(list)
    for ordinal, nic in placed.items():
        by_nic[nic].append(ordinal)
    out = [
        Rail(
            nic=nic.name,
            rate_gbps=nic.rate_gbps,
            devices=tuple(sorted(by_nic.get(nic.name, ()))),
            numa_node=nic.numa_node,
        )
        for nic in active_rdma_devices()
    ]
    return tuple(sorted(out, key=lambda r: r.nic))


def rail_for_device(ordinal: int, assignment: Mapping[int, str] | None = None) -> str:
    """The NIC accelerator `ordinal` should leave the node through, or `""`.

    Args:
        ordinal: The device's index as this process sees it (CUDA's numbering).
        assignment: A device-to-NIC map, or `None` to take one live.

    Returns:
        The NIC name, or `""` when the device has no rail — no fabric, no readable address, or
        an ordinal this node does not have.
    """
    placed = rail_map() if assignment is None else assignment
    return placed.get(ordinal, "")


def rail_aligned(ordinal: int, nic: str, assignment: Mapping[int, str] | None = None) -> bool:
    """Whether a transfer for device `ordinal` on NIC `nic` stays on its own rail.

    False when the pairing crosses to another rail *and* the device has one of its own. A
    device with no assignment is not misaligned — there is no rail to be off — so the answer
    is True, which keeps a caller from reporting a fault on a node that simply has no fabric.

    Args:
        ordinal: The device's index as this process sees it.
        nic: The NIC the transfer would actually use.
        assignment: A device-to-NIC map, or `None` to take one live.

    Returns:
        True when aligned or when there is no opinion, False when the transfer leaves through
        a NIC that is not this device's.
    """
    own = rail_for_device(ordinal, assignment)
    return not own or not nic or own == nic


def rail_imbalance(rail_records: Sequence[Rail] | None = None) -> float:
    """How unevenly the node's devices are spread over its rails, `0.0` being perfect.

    Defined as `1 - (mean devices per rail / max devices on any rail)` over **every** active
    rail, and `0.0` for a node with one rail or none — a single rail cannot be unbalanced
    against itself.

    **Every rail, including the empty ones.** Measuring only the loaded rails made the metric
    blind to the exact fault it is documented to catch: a node whose eight devices all chose
    one NIC has a single loaded rail, so the load list has one entry, the `<= 1` guard fires,
    and the answer is `0.0` — perfectly balanced. That is the worst wiring a dense node can
    have, and it is also the one it reported as healthy, alongside `ACTIVE` on every port,
    zero errors, and the full summed port rate while carrying an eighth of it. `rails()`
    deliberately keeps a record for every active NIC so this evidence exists; filtering it out
    here threw it away one line later.

    Args:
        rail_records: Rails to measure, or `None` to take them live.

    Returns:
        `0.0` (balanced or undecidable) through just under `1.0` (everything on one rail of
        many). `0.0` when no device is placed at all, which is an absent topology rather than
        an imbalanced one.
    """
    records = rails() if rail_records is None else rail_records
    loads = [len(r.devices) for r in records]
    if len(loads) <= 1:
        return 0.0
    peak = max(loads)
    return 0.0 if peak <= 0 else 1.0 - (sum(loads) / len(loads)) / peak


def device_rail_bandwidth_gbps(ordinal: int, rail_records: Sequence[Rail] | None = None) -> float:
    """The port rate accelerator `ordinal` actually has for cross-node transfers.

    The node's *total* fabric bandwidth is the wrong denominator for a single device's
    transfer: it can only use the rail it is on, shared with whatever else landed there. This
    is that share.

    Args:
        ordinal: The device's index as this process sees it.
        rail_records: Rails to read, or `None` to take them live.

    Returns:
        Gigabits per second, `0.0` when the device has no rail or its rate is unpublished.
    """
    for rail in rails() if rail_records is None else rail_records:
        if ordinal in rail.devices:
            return rail.share_gbps
    return 0.0


def rail_summary(rail_records: Sequence[Rail] | None = None) -> dict:
    """The node's rail layout as one record, for a report or a health gauge.

    Args:
        rail_records: Rails to summarize, or `None` to take them live.

    Returns:
        `rails` (count), `loaded_rails`, `devices` (placed count), `imbalance`,
        `total_gbps` over loaded rails, and `assignment` (NIC to device ordinals). Empty
        counts and a `0.0` imbalance on a node with no fabric.
    """
    records = rails() if rail_records is None else rail_records
    loaded = [r for r in records if r.loaded]
    return {
        "rails": len(records),
        "loaded_rails": len(loaded),
        "devices": sum(len(r.devices) for r in records),
        "imbalance": rail_imbalance(records),
        "total_gbps": sum(r.rate_gbps for r in loaded),
        "assignment": {r.nic: list(r.devices) for r in loaded},
    }
