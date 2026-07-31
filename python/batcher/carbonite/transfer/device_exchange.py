"""Redistributing between the devices of one node without serializing them behind each other.

A sharded GPU stage that has to redistribute — a hash exchange before a device-side group-by,
a build side spread across four boards — moves bytes between devices that are already holding
them. Done naively that is a sequence of copies through host memory: device 0 to host, host to
device 1, one pair at a time, and the node's fabric sits idle while its slowest wire carries
everything.

Two things fix it, and both are scheduling rather than semantics. **Pair the transfers** so no
device is the source or the destination of two copies at once: an all-to-all over `n` devices
decomposes into `n-1` rounds of `n/2` disjoint pairs, and every round then runs at link rate
instead of contending. And **order the ring** by the fabric rather than by device index, so a
reduction walks NVLink where NVLink exists — an index-ordered ring on a two-board node crosses
the bus twice per revolution for no reason but the numbering.

This module is the pure decision: a topology in (`_internal.hardware.fabric.p2p`), a schedule
and its predicted duration out. Nothing here copies a byte or imports a device runtime, which
is what keeps it testable against a described node. `worth_device_exchange` is the gate the
caller actually asks: a plan that predicts no gain over the host path is not worth the second
code path, and on a node with no fabric it reliably predicts none.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from batcher._internal.hardware.fabric.p2p import p2p_capable, peer_bandwidth_gbps, peer_class

__all__ = [
    "ExchangePlan",
    "ExchangeStep",
    "all_reduce_seconds",
    "exchange_seconds",
    "pairwise_rounds",
    "plan_exchange",
    "ring_bandwidth_gbps",
    "ring_order",
    "staged_pairs_in",
    "worth_device_exchange",
]

#: How much faster a device-to-device plan must be before it is worth running instead of the
#: host path. The device path is the second implementation of a movement the host path already
#: performs correctly, so a plan that ties loses: the margin buys the risk of the newer path.
_WORTH_MARGIN = 1.25


@dataclass(frozen=True)
class ExchangeStep:
    """One round of an exchange: pairs that can copy simultaneously without contending.

    Attributes:
        index: The round's position in the schedule, from zero.
        pairs: `(a, b)` device pairs exchanging in this round. Every device appears in at most
            one pair, which is what makes the round congestion-free.
    """

    index: int
    pairs: tuple[tuple[int, int], ...] = field(default_factory=tuple)

    @property
    def width(self) -> int:
        """How many pairs copy simultaneously in this round."""
        return len(self.pairs)


@dataclass(frozen=True)
class ExchangePlan:
    """A whole device-to-device redistribution: its schedule, its cost, and what it stages.

    Attributes:
        steps: The rounds, in order.
        ring: Device order for a ring reduction, fastest-link-first. Empty when fewer than two
            devices are involved.
        staged: Pairs whose copy has to bounce through host memory because the bus puts a host
            bridge between them. Empty on a fully direct node.
        seconds: Predicted duration for the modelled byte volume, `0.0` when no rate is known.
        host_seconds: What the same movement costs through host memory, for comparison.
    """

    steps: tuple[ExchangeStep, ...] = field(default_factory=tuple)
    ring: tuple[int, ...] = field(default_factory=tuple)
    staged: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    seconds: float = 0.0
    host_seconds: float = 0.0

    @property
    def rounds(self) -> int:
        """How many congestion-free rounds the schedule takes."""
        return len(self.steps)

    @property
    def fully_direct(self) -> bool:
        """Whether every pair copies device-to-device with no host bounce."""
        return not self.staged

    @property
    def speedup(self) -> float:
        """How many times faster than the host path, `0.0` when either cost is unknown.

        The figure `worth_device_exchange` gates on, exposed because a caller that declines
        the device path should be able to say by how much it lost rather than only that it did.
        """
        if self.seconds <= 0.0 or self.host_seconds <= 0.0:
            return 0.0
        return self.host_seconds / self.seconds

    def summary(self) -> dict:
        """The plan as one flat record, for a report or a debug note.

        Returns:
            `rounds`, `pairs` (total), `staged` (count), `ring`, `seconds`, `host_seconds`,
            and `speedup`.
        """
        return {
            "rounds": self.rounds,
            "pairs": sum(s.width for s in self.steps),
            "staged": len(self.staged),
            "ring": list(self.ring),
            "seconds": self.seconds,
            "host_seconds": self.host_seconds,
            "speedup": self.speedup,
        }


def pairwise_rounds(devices: Sequence[int]) -> tuple[ExchangeStep, ...]:
    """A congestion-free schedule in which every pair of devices exchanges exactly once.

    The circle method: fix one device and rotate the rest, which yields `n-1` rounds of `n/2`
    disjoint pairs for an even count. An odd count is scheduled as if one more device were
    present, and the pair holding that phantom sits the round out — `n` rounds, one device
    idle in each, which is the best any schedule can do with an odd number of endpoints.

    Disjointness is the point. A device that is the source of one copy and the destination of
    another in the same round has both transfers running through one link, which halves each
    of them and makes the round's duration the sum rather than the maximum.

    Args:
        devices: Device indices taking part. Duplicates are collapsed and order is normalized,
            so two callers describing the same node get the same schedule.

    Returns:
        The rounds in order. Empty for fewer than two devices, where there is nothing to
        exchange.
    """
    members = sorted(set(devices))
    if len(members) < 2:
        return ()
    phantom = -1
    ring = [*members, phantom] if len(members) % 2 else list(members)
    n = len(ring)
    fixed, rotating = ring[0], ring[1:]
    steps: list[ExchangeStep] = []
    for round_index in range(n - 1):
        order = [fixed, *rotating]
        pairs = []
        for i in range(n // 2):
            a, b = order[i], order[n - 1 - i]
            if a != phantom and b != phantom:
                pairs.append((min(a, b), max(a, b)))
        steps.append(ExchangeStep(round_index, tuple(sorted(pairs))))
        rotating = [rotating[-1], *rotating[:-1]]
    return tuple(steps)


def ring_order(
    devices: Sequence[int], matrix: Sequence[Sequence[str]] | None = None
) -> tuple[int, ...]:
    """Device order for a ring reduction, following the fastest links available.

    A ring reduction's rate is its *worst* hop, so the order is chosen to keep every hop on
    the best class the node has: greedy nearest-neighbour from each possible start, keeping
    whichever tour has the best worst hop. On a node with no fabric this returns index order,
    which is what an unaware caller already used.

    Args:
        devices: Device indices taking part.
        matrix: A `p2p.peer_matrix`, or `None` to take one live.

    Returns:
        The devices as a tour, starting at the lowest index of the best tour found. Empty for
        fewer than two devices.
    """
    from batcher._internal.hardware.fabric.p2p import P2P_CLASSES, peer_matrix

    members = sorted(set(devices))
    if len(members) < 2:
        return ()
    m = peer_matrix() if matrix is None else matrix
    rank = {name: i for i, name in enumerate(P2P_CLASSES)}
    fallback = len(P2P_CLASSES)

    def hop(a: int, b: int) -> int:
        return rank.get(peer_class(a, b, m), fallback)

    best: tuple[int, tuple[int, ...]] | None = None
    for start in members:
        tour = [start]
        remaining = set(members) - {start}
        while remaining:
            here = tour[-1]
            nxt = min(remaining, key=lambda j: (hop(here, j), j))
            tour.append(nxt)
            remaining.discard(nxt)
        # The tour closes, so the hop back to the start counts as much as any other.
        hops = [hop(tour[i], tour[i + 1]) for i in range(len(tour) - 1)]
        worst = max([*hops, hop(tour[-1], tour[0])])
        if best is None or (worst, tour[0]) < (best[0], best[1][0]):
            best = (worst, tuple(tour))
    return best[1] if best is not None else ()


def ring_bandwidth_gbps(
    order: Sequence[int],
    matrix: Sequence[Sequence[str]] | None = None,
    *,
    nvlink_gbps: float = 0.0,
    pcie_gbps: float = 0.0,
) -> float:
    """The rate a ring runs at: its slowest hop, including the hop that closes it.

    Args:
        order: The ring, as `ring_order` returns it.
        matrix: A `p2p.peer_matrix`, or `None` to take one live.
        nvlink_gbps: The device model's NVLink rate.
        pcie_gbps: The negotiated host link rate.

    Returns:
        Gigabytes per second, `0.0` for a ring of fewer than two devices or when no rate is
        known.
    """
    ring = list(order)
    if len(ring) < 2:
        return 0.0
    hops = [(ring[i], ring[i + 1]) for i in range(len(ring) - 1)] + [(ring[-1], ring[0])]
    return min(
        peer_bandwidth_gbps(peer_class(a, b, matrix), nvlink_gbps=nvlink_gbps, pcie_gbps=pcie_gbps)
        for a, b in hops
    )


def staged_pairs_in(
    devices: Sequence[int], matrix: Sequence[Sequence[str]] | None = None
) -> tuple[tuple[int, int], ...]:
    """Pairs among `devices` whose exchange has to bounce through host memory.

    Args:
        devices: Device indices taking part.
        matrix: A `p2p.peer_matrix`, or `None` to take one live.

    Returns:
        Ascending `(low, high)` pairs, each once. Empty when every pair copies directly.
    """
    members = sorted(set(devices))
    return tuple(
        (a, b)
        for i, a in enumerate(members)
        for b in members[i + 1 :]
        if not p2p_capable(a, b, matrix)
    )


def exchange_seconds(
    bytes_moved: int,
    steps: Sequence[ExchangeStep],
    matrix: Sequence[Sequence[str]] | None = None,
    *,
    nvlink_gbps: float = 0.0,
    pcie_gbps: float = 0.0,
) -> float:
    """How long a scheduled all-to-all takes, in seconds.

    Bytes are divided evenly over the pairs — an unskewed exchange, which is what a hash
    partitioner produces — and each round costs its *slowest* pair, because the round ends when
    its last copy does. Summing the rounds is then the schedule's duration.

    Args:
        bytes_moved: Total bytes crossing between devices, over the whole exchange.
        steps: The schedule, from `pairwise_rounds`.
        matrix: A `p2p.peer_matrix`, or `None` to take one live.
        nvlink_gbps: The device model's NVLink rate.
        pcie_gbps: The negotiated host link rate.

    Returns:
        Seconds, `0.0` when there is nothing to move. A pair whose rate is unknown contributes
        nothing rather than an infinity: an unpriced link makes the estimate optimistic, and
        `worth_device_exchange` refuses on an unpriced node for exactly that reason.
    """
    pairs = sum(s.width for s in steps)
    if bytes_moved <= 0 or pairs == 0:
        return 0.0
    per_pair = bytes_moved / pairs
    total = 0.0
    for step in steps:
        rates = [
            peer_bandwidth_gbps(
                peer_class(a, b, matrix), nvlink_gbps=nvlink_gbps, pcie_gbps=pcie_gbps
            )
            for a, b in step.pairs
        ]
        usable = [r for r in rates if r > 0.0]
        if usable:
            total += per_pair / (min(usable) * 1e9)
    return total


def all_reduce_seconds(bytes_per_device: int, devices: int, ring_gbps: float) -> float:
    """How long a ring all-reduce of `bytes_per_device` takes across `devices`.

    The standard bound: a ring all-reduce is a reduce-scatter followed by an all-gather, and
    each moves `(n-1)/n` of a device's buffer over the ring, so the total is `2(n-1)/n` times
    the buffer at the ring's rate. Independent of `n` to first order, which is the property
    that makes a ring the right shape for a wide node.

    Args:
        bytes_per_device: The buffer each device contributes.
        devices: How many devices are in the ring.
        ring_gbps: The ring's rate, from `ring_bandwidth_gbps`.

    Returns:
        Seconds, `0.0` when there is nothing to reduce, fewer than two devices, or no known
        rate.
    """
    if bytes_per_device <= 0 or devices < 2 or ring_gbps <= 0.0:
        return 0.0
    return 2.0 * (devices - 1) / devices * bytes_per_device / (ring_gbps * 1e9)


def plan_exchange(
    devices: Sequence[int],
    bytes_moved: int,
    matrix: Sequence[Sequence[str]] | None = None,
    *,
    nvlink_gbps: float = 0.0,
    pcie_gbps: float = 0.0,
    host_gbps: float = 0.0,
) -> ExchangePlan:
    """The whole redistribution: schedule, ring, staged pairs, and what each path costs.

    Args:
        devices: Device indices taking part.
        bytes_moved: Total bytes crossing between devices.
        matrix: A `p2p.peer_matrix`, or `None` to take one live.
        nvlink_gbps: The device model's NVLink rate.
        pcie_gbps: The negotiated host link rate.
        host_gbps: The rate the host path achieves, which is one link's rate for the whole
            movement because every copy shares it. `0.0` leaves `host_seconds` unknown.

    Returns:
        An `ExchangePlan`. A single device or none yields an empty plan whose costs are zero,
        which `worth_device_exchange` reads as "not worth it" rather than as free.
    """
    members = sorted(set(devices))
    steps = pairwise_rounds(members)
    seconds = exchange_seconds(
        bytes_moved, steps, matrix, nvlink_gbps=nvlink_gbps, pcie_gbps=pcie_gbps
    )
    # The host path serializes: every byte crosses the one host link twice, down from the
    # source device and back up to the destination, with no second wire to overlap onto.
    host_seconds = 2.0 * bytes_moved / (host_gbps * 1e9) if host_gbps > 0.0 and members else 0.0
    return ExchangePlan(
        steps=steps,
        ring=ring_order(members, matrix),
        staged=staged_pairs_in(members, matrix),
        seconds=seconds,
        host_seconds=host_seconds,
    )


def worth_device_exchange(plan: ExchangePlan, margin: float = _WORTH_MARGIN) -> bool:
    """Whether a device-to-device exchange is worth running instead of the host path.

    Three refusals, each for a reason the cost figures alone would not carry. A plan with no
    rounds has nothing to schedule. A plan whose cost or whose host cost is unknown is refused
    rather than assumed favorable, because an unpriced link makes `seconds` optimistic and the
    optimism would be spent on the newer of two code paths. And a plan that merely ties loses:
    the host path already moves these bytes correctly.

    Args:
        plan: The plan from `plan_exchange`.
        margin: How many times faster the device path must be. Below `1.0` it is clamped to
            `1.0`, since a plan that is slower is never worth running.

    Returns:
        True when the device path should be used.
    """
    if not plan.steps or plan.seconds <= 0.0 or plan.host_seconds <= 0.0:
        return False
    return plan.speedup >= max(1.0, margin)
