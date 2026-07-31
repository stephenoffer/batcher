"""The wires between the devices, in the report the operator already reads.

`report._add_fabric` says what the node's NICs and NVLink are *capable* of. Two things it does
not say decide what a multi-GPU stage actually gets, and both are invisible to every counter:

* **Which NIC each device leaves through.** A node whose devices all landed on one rail reports
  `ACTIVE` on every port, zero errors, and the full summed port rate, while carrying a fraction
  of it. The imbalance is the finding, and nothing else in the report can express it.
* **Which device pairs can copy directly.** A pair the bus keeps apart stages through host
  memory whatever the copy is called, so a redistribution across those pairs runs at the host
  link rather than at the fabric — again with nothing wrong anywhere.

Both are added to the existing `fabric` section rather than given one of their own, because a
reader looking for "why is my exchange slow" is already there. Both are omitted entirely when
the topology cannot be read, so the report stays the same size on a host without one.
"""

from __future__ import annotations

__all__ = ["add_wires", "device_cost_section", "peer_section", "rail_section", "wire_problems"]

#: Imbalance above which the rails are worth a line in the problem list. A little unevenness is
#: unavoidable on a node whose device count is not a multiple of its NIC count; a third of the
#: fleet's rail capacity going unused is a wiring or a driver-ordering fault.
_IMBALANCE_ALERT = 0.34


def rail_section() -> dict:
    """How the node's devices are spread over its NICs.

    Returns:
        `rails` (count), `loaded_rails`, `devices` placed, `imbalance` (`0.0` balanced),
        `total_gbps` over the loaded rails, and `assignment` (NIC to device ordinals). Empty
        on a node with no RDMA fabric, no readable PCI tree, or no accelerators.
    """
    from batcher._internal.hardware.fabric.rails import rail_summary

    summary = rail_summary()
    return summary if summary.get("loaded_rails") else {}


def peer_section() -> dict:
    """Which devices on this node can exchange without touching host memory.

    Returns:
        `devices`, `islands` (coherent groups), `largest_island`, `fabric_pairs`,
        `staged_pairs` (pairs that must bounce through the host), and `class` (the node's
        worst pair). Empty when the device topology cannot be read.
    """
    from batcher._internal.hardware.fabric.p2p import peer_summary

    summary = peer_summary()
    return summary if summary.get("devices") else {}


def device_cost_section() -> dict:
    """What a byte shuffled off a *device* costs, against what a host byte costs.

    The report already carries the host figure, which prices a shuffled byte against the
    node's summed port rate. That denominator is wrong for data sitting on a device in two
    directions at once: a device uses the one rail it is on, and its bytes cross the host link
    before they reach any rail. Both errors are optimistic, so a stage planned against the host
    figure expects bandwidth the device does not have.

    The host-side transfer shape is reported with it, because "the link is the ceiling" and
    "the link is idle between copies" are different problems with the same symptom, and the
    second is answered by a chunk size and a ring depth rather than by more devices.

    Returns:
        `rail_gbps`, `host_link_gbps`, `net_gbps` (the minimum of the two, which is what the
        device actually sustains), `weight` (that rate as a multiple of a local byte), and
        `staging` (the chunk, depth, and pinning this link wants). Empty when nothing about
        the device's wires could be read.
    """
    from batcher.carbonite.transfer.staging import plan_staging
    from batcher.kyber.gpu.exchange import device_fabric, device_net_weight

    record = device_fabric()
    if not record.readable:
        return {}
    summary = record.summary()
    weight = device_net_weight(record)
    if weight is not None:
        summary["weight"] = round(weight, 2)
    if record.host_link_gbps > 0.0:
        # Sized for a transfer large enough to amortize pinning, which is the regime a feed
        # runs in; a per-batch figure would report "pageable" for a ring that is reused all
        # stage long.
        summary["staging"] = plan_staging(1 << 30, record.host_link_gbps).summary()
    return summary


def add_wires(fabric: dict) -> None:
    """Add the rail, peer, and device-cost sections to a report's `fabric` block, in place.

    Args:
        fabric: The report's fabric mapping, which is modified.
    """
    rails = rail_section()
    if rails:
        fabric["rails"] = rails
    peers = peer_section()
    if peers:
        fabric["peers"] = peers
    device_cost = device_cost_section()
    if device_cost:
        fabric["device_cost"] = device_cost


def wire_problems(fabric: dict) -> list[str]:
    """The wiring conditions worth an alert, as complete sentences.

    Two, and both are the shape this report exists for: correct, healthy, and slower than the
    hardware. An unreadable topology yields nothing rather than a complaint, so a deployment
    check does not fail a fleet whose base image stopped publishing a PCI tree.

    Args:
        fabric: The report's fabric mapping.

    Returns:
        The findings, empty when the rails are even and every pair can copy directly.
    """
    out: list[str] = []
    rails = fabric.get("rails") or {}
    imbalance = float(rails.get("imbalance", 0.0))
    if imbalance >= _IMBALANCE_ALERT and rails.get("loaded_rails", 0) > 1:
        out.append(
            f"fabric: devices are unevenly spread over {rails['loaded_rails']} rails "
            f"({imbalance:.0%} imbalance), so a cross-node stage uses part of the port rate"
        )
    peers = fabric.get("peers") or {}
    devices = int(peers.get("devices", 0))
    staged = int(peers.get("staged_pairs", 0))
    if devices > 1 and staged == devices * (devices - 1) // 2:
        out.append(
            f"fabric: no pair of the {devices} devices can copy directly, so every "
            "device-to-device exchange stages through host memory"
        )
    return out
