"""What a byte costs when the data is on a device, and how wide a stage may fan out before it.

`kyber.cost.fabric` prices a shuffled byte against the node's *summed* active port rate. For
host-resident data that is the right denominator. For device-resident data it is wrong twice
over, and both errors point the same way — optimistic:

* **A device cannot use the whole fabric.** It uses the rail it is on, shared with whatever
  else landed there (`fabric.rails`). Eight devices on one NIC have an eighth of the node's
  fabric each, and the summed figure says they each have all of it.
* **A device's bytes cross the host link first.** Whatever the NIC sustains, the payload is
  bounded by the PCIe link between the device and host memory. Summing eight 400 Gb/s ports to
  400 GB/s while the board negotiates 64 GB/s over PCIe overstates a device shuffle six-fold.

So the effective rate for a device's off-node byte is the *minimum* of its rail share and its
host link, and this module derives the cost weight from that. The same reasoning bounds fan-out
width: a stage whose devices exchange with each other should stay inside one fabric island,
because the device the island does not hold turns every round of the schedule into its slowest
pair (`fabric.p2p`).

Kyber decides; nothing here executes or measures. Every entry point falls back to the behavior
that existed before it — an unreadable rail map, an unknown device model, or a node with no
fabric all yield "no opinion", and the caller keeps the figure it had.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher._internal.hardware.fabric.p2p import fabric_fraction, peer_islands
from batcher.kyber.cost.fabric import REFERENCE_LOCAL_GBPS

__all__ = [
    "DeviceFabric",
    "device_exchange_gbps",
    "device_fabric",
    "device_net_gbps",
    "device_net_weight",
    "fabric_bounded_width",
    "widest_fabric_island",
]

#: A device byte is never cheaper than a local one, and never priced past the point where every
#: plan that moves it has already lost. The same band `cost.fabric` clamps the host weight to,
#: because the two weights are compared against each other in a plan's total and a wider band
#: here would let a device stage outrank a host stage on arithmetic rather than on physics.
_MIN_WEIGHT = 1.0
_MAX_WEIGHT = 32.0


@dataclass(frozen=True, slots=True)
class DeviceFabric:
    """What one device can actually move, and how much of its group is on the coherent fabric.

    Attributes:
        rail_gbps: The device's share of its NIC's port rate, in gigabits per second. `0.0`
            when there is no rail map — no RDMA fabric, or an unreadable PCI tree.
        host_link_gbps: The negotiated device-to-host link, in gigabytes per second. `0.0`
            when the board's link cannot be read.
        island: How many devices are in the largest coherent group this node has. `0` when the
            topology is unreadable; `1` on a board with no fabric at all.
        fabric_share: Fraction of the group's pairs that exchange on the fabric, `0.0` through
            `1.0`.
    """

    rail_gbps: float = 0.0
    host_link_gbps: float = 0.0
    island: int = 0
    fabric_share: float = 0.0

    @property
    def readable(self) -> bool:
        """Whether anything was measured. False means every derived figure is "no opinion"."""
        return self.rail_gbps > 0.0 or self.host_link_gbps > 0.0 or self.island > 0

    def summary(self) -> dict:
        """The record as one flat dict, for a report or a decision log.

        Returns:
            `rail_gbps`, `host_link_gbps`, `island`, `fabric_share`, and the derived
            `net_gbps`.
        """
        return {
            "rail_gbps": self.rail_gbps,
            "host_link_gbps": self.host_link_gbps,
            "island": self.island,
            "fabric_share": self.fabric_share,
            "net_gbps": device_net_gbps(self),
        }


def device_fabric(ordinal: int = 0) -> DeviceFabric:
    """Read what one device's wires can carry, and how coherent its neighbours are.

    Args:
        ordinal: The device's index as this process sees it (CUDA's numbering).

    Returns:
        A `DeviceFabric`. Every field is zero on a host with no accelerators, no fabric, or no
        readable PCI tree, which `readable` reports and every consumer here treats as "keep
        what you had".
    """
    from batcher._internal.hardware.fabric.device_links import device_pcie_links
    from batcher._internal.hardware.fabric.rails import device_rail_bandwidth_gbps

    rail = device_rail_bandwidth_gbps(ordinal)
    islands = peer_islands()
    widest = max((len(i) for i in islands), default=0)
    # The *negotiated* link, not the model's nameplate: a card that renegotiated to half width
    # still enumerates and still returns correct results, and this is the term a device
    # decision is most sensitive to. Published in gigabits, consumed here in gigabytes.
    links = device_pcie_links()
    host_link = links[ordinal].bandwidth_gbps / 8.0 if 0 <= ordinal < len(links) else 0.0
    group = max((i for i in islands if ordinal in i), key=len, default=())
    return DeviceFabric(
        rail_gbps=rail,
        host_link_gbps=host_link,
        island=widest,
        fabric_share=fabric_fraction(group) if len(group) > 1 else 0.0,
    )


def device_net_gbps(fabric: DeviceFabric) -> float:
    """What one device sustains for an off-node byte, in gigabytes per second.

    The minimum of the two wires the byte crosses: the rail it leaves on, and the host link it
    reaches the rail through. Taking the minimum rather than either alone is the whole point —
    each of the two figures alone over-states the path by whatever the other one is.

    Args:
        fabric: The device's wires, from `device_fabric`.

    Returns:
        Gigabytes per second, `0.0` when neither wire could be read. When only one is readable
        that one is used: a partially readable node is better priced against the half it knows
        than against a constant.
    """
    rail_bytes = fabric.rail_gbps / 8.0  # port rates are in gigabits, cost models in gigabytes
    known = [r for r in (rail_bytes, fabric.host_link_gbps) if r > 0.0]
    return min(known) if known else 0.0


def device_net_weight(
    fabric: DeviceFabric | None = None, local_gbps: float = REFERENCE_LOCAL_GBPS
) -> float | None:
    """What one byte shuffled *off a device* costs relative to one local byte.

    The device-resident counterpart of `cost.fabric.fabric_net_weight`, and it is the figure
    that should price a stage whose data is already on a board. Kept as a separate weight
    rather than replacing the host one: a plan mixes host and device stages, and pricing a
    host shuffle at a device's link would push the enumerator away from shuffles the host can
    perform perfectly well.

    Args:
        fabric: The device's wires, or `None` to read device 0 live.
        local_gbps: Effective host memory bandwidth, the denominator a local byte is priced at.

    Returns:
        The weight, clamped to `[1.0, 32.0]`, or `None` when nothing about the device's wires
        could be read. `None` means keep the configured weight: an unreadable link is not
        evidence of a slow one.
    """
    record = device_fabric() if fabric is None else fabric
    rate = device_net_gbps(record)
    if rate <= 0.0 or local_gbps <= 0.0:
        return None
    return min(_MAX_WEIGHT, max(_MIN_WEIGHT, local_gbps / rate))


def device_exchange_gbps(devices: int, fabric: DeviceFabric, nvlink_gbps: float = 0.0) -> float:
    """What a redistribution *between* devices sustains, in gigabytes per second.

    Inside one island the exchange runs on the fabric. Past it, the bytes leave the island
    through the host link, which is the rate the whole exchange then runs at however fast the
    fabric inside each island is — a group of ten on a node of eight is bounded by the two
    that are not on it.

    Args:
        devices: How many devices take part.
        fabric: The device's wires, from `device_fabric`.
        nvlink_gbps: The device model's NVLink rate, `0.0` when the model is unknown.

    Returns:
        Gigabytes per second, `0.0` when neither rate is known or fewer than two devices take
        part.
    """
    if devices < 2:
        return 0.0
    if fabric.island >= devices and nvlink_gbps > 0.0:
        return nvlink_gbps
    return fabric.host_link_gbps


def widest_fabric_island(islands: tuple[tuple[int, ...], ...] | None = None) -> int:
    """How many devices the node's largest coherent group holds.

    Args:
        islands: `p2p.peer_islands` output, or `None` to read it live.

    Returns:
        The size, `0` when the topology is unreadable. Zero is "no opinion" and must not be
        read as "no fabric": a board with no fabric reports `1`, one island per device.
    """
    groups = peer_islands() if islands is None else islands
    return max((len(i) for i in groups), default=0)


def fabric_bounded_width(desired: int, island: int, *, exchanges: bool = True) -> int:
    """Cap a stage's device fan-out at the fabric island, when the stage exchanges.

    A fan-out of independent shards — the sharded aggregate, where each device reads its own
    slice and nothing crosses — should be as wide as the fleet allows, and this leaves it
    alone. A stage whose devices talk to each other should not: the device outside the island
    turns every round of the schedule into its slowest pair, so the ninth device on an
    eight-wide fabric makes the collective slower than eight would have been.

    Args:
        desired: The width the stage asked for.
        island: The widest coherent group, from `widest_fabric_island`.
        exchanges: Whether the stage's devices exchange with each other. False leaves `desired`
            untouched.

    Returns:
        The width to use. `desired` is returned unchanged when the stage does not exchange,
        when the topology is unreadable (`island <= 0`), or when it already fits — so every
        path that cannot be improved keeps exactly the behavior it had.
    """
    if not exchanges or island <= 0 or desired <= island:
        return desired
    return max(1, island)
