"""How workers land on the fleet's nodes, and how much of an exchange each tier carries.

Split from `cluster` along the seam between *what the fleet is* — one record per node, which
`ClusterShape` owns — and *what falls out of it* when a plan is spread over it. The arithmetic
here is the load-bearing part: every tiered cost decision multiplies through these shares, so
a share set that does not partition the exchange silently rescales every `net` cost that reads
it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from batcher.plan.resource.cluster import NodeShape

__all__ = ["LocalityShares"]


@dataclass(frozen=True, slots=True)
class LocalityShares:
    """How a hash exchange across `W` workers splits over the interconnect tiers.

    Five fractions that sum to `1.0`, each the share of the exchange whose *fastest common
    tier* is that one. Shares of the data, not of the worker pairs: a producer sends an equal
    slice to each of the `W` buckets, so the share on a tier is the share of buckets it reaches.

    Attributes:
        local: Stays in the producer's own worker. Never crosses anything.
        intra_domain: Reaches another device inside the same coherent fabric — NVLink, at a
            bandwidth within a small factor of host memory.
        intra_node: Reaches another worker on the same host but outside its fabric domain, so
            it crosses PCIe and host memory rather than the network.
        intra_rack: Leaves the host but stays inside the rack, on the fabric's fastest tier.
        cross_rack: Crosses the general network. The tier every existing cost figure assumed
            for the whole exchange.
    """

    local: float = 0.0
    intra_domain: float = 0.0
    intra_node: float = 0.0
    intra_rack: float = 0.0
    cross_rack: float = 1.0

    @property
    def off_node(self) -> float:
        """Share that leaves the host: everything above the node tier."""
        return self.intra_rack + self.cross_rack

    @property
    def on_node(self) -> float:
        """Share that never leaves the host, including what never leaves the worker."""
        return self.local + self.intra_domain + self.intra_node

    def weighted(
        self,
        *,
        local: float = 0.0,
        intra_domain: float,
        intra_node: float,
        intra_rack: float,
        cross_rack: float,
    ) -> float:
        """Collapse the shares to one figure with a per-tier price.

        Args:
            local: Price of the share that never leaves its worker.
            intra_domain: Price of a byte on the coherent device fabric.
            intra_node: Price of a byte crossing the host but not the network.
            intra_rack: Price of a byte on the rack's fabric.
            cross_rack: Price of a byte on the general network.

        Returns:
            The share-weighted price, in whatever unit the prices were given in.
        """
        return (
            self.local * local
            + self.intra_domain * intra_domain
            + self.intra_node * intra_node
            + self.intra_rack * intra_rack
            + self.cross_rack * cross_rack
        )


def _spread(capacities: Sequence[int], workers: int) -> list[int]:
    """Place `workers` as evenly as `capacities` allow — the fleet's default SPREAD strategy.

    Even placement bounded by capacity, not placement proportional to it. That matches what the
    engine actually asks for (`SchedulingEnvelope.placement_strategy` defaults to `SPREAD`), and
    where it is wrong it is wrong in the safe direction: spreading is the *least* local
    arrangement, so a share derived from it never over-states how much of an exchange stays
    home. Over-stating locality under-charges a shuffle, which is how a plan comes to move data
    nobody budgeted for; under-stating it only costs a pessimistic ranking.

    Nodes fill by descending capacity, so a node whose capacity runs out drops out while the
    rest keep taking. Workers beyond the fleet's whole capacity — an over-subscribed grant,
    which the scheduler does allow — are dealt evenly rather than piled onto the largest.
    """
    usable = [max(0, int(c)) for c in capacities]
    occupied = sorted((i for i, c in enumerate(usable) if c > 0), key=lambda i: (-usable[i], i))
    out = [0] * len(usable)
    if workers <= 0 or not occupied:
        return out

    remaining = workers
    active = list(occupied)
    while active and remaining > 0:
        base, extra = divmod(remaining, len(active))
        if base == 0:  # fewer workers than nodes: one each to the largest
            for index in active[:remaining]:
                out[index] += 1
            return out
        filled = [i for i in active if usable[i] - out[i] <= base]
        if not filled:  # nobody is capacity-bound: deal the whole remainder out
            for rank, index in enumerate(active):
                out[index] += base + (1 if rank < extra else 0)
            return out
        for index in filled:
            remaining -= usable[index] - out[index]
            out[index] = usable[index]
        active = [i for i in active if out[i] < usable[i]]

    if remaining > 0:  # over-subscribed: more workers than the fleet has capacity units
        base, extra = divmod(remaining, len(occupied))
        for rank, index in enumerate(occupied):
            out[index] += base + (1 if rank < extra else 0)
    return out


def _group_share(sizes: Sequence[int], workers: int) -> float:
    """Share of a uniform hash exchange whose destination is inside the producer's own group.

    A producer sends `1/W` of its data to each bucket, so the share reaching its own group is
    that group's worker count over `W`. Weighting each group by the data it *produces* — also
    its worker count over `W` — gives `sum(g^2) / W^2`, the standard collision form.

    Args:
        sizes: Workers in each group, one entry per group. Groups must partition the fleet,
            since the shares are read as a containment hierarchy.
        workers: Total workers in the exchange.

    Returns:
        The share in `[0, 1]`, `0.0` for an empty exchange.
    """
    if workers <= 0:
        return 0.0
    total = float(workers) ** 2
    return sum(float(size) ** 2 for size in sizes) / total if total > 0 else 0.0


def _domain_split(node: NodeShape, placed: int, unit: str) -> list[int]:
    """`placed` workers on `node`, split over the coherent fabric groups they land in.

    Only a device fleet has a coherent fabric at all. A CPU worker exchanging with another CPU
    worker on the same host copies through host memory whatever wires the accelerators
    together, so each one is its own group and the domain tier collapses to the *local* share —
    which leaves everything on-host in the node tier, priced at host bandwidth. Folding CPU
    workers into a device domain instead would charge a relational shuffle the NVLink rate for
    traffic that never touches NVLink, discounting it by more than an order of magnitude.
    """
    if placed <= 0:
        return []
    if unit != "gpu":
        return [1] * placed
    domains = node.domains
    if domains <= 1:
        return [placed]
    return [count for count in _spread([node.local_domain] * domains, placed) if count > 0]
