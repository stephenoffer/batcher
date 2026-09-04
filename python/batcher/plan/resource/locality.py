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


def _apportion(sizes: Sequence[int], total: int) -> list[int]:
    """Share `total` out across groups of `sizes` nodes, by largest remainder.

    Integer arithmetic throughout, so the result is exact and never depends on how a float
    quotient rounded. Ties go to the earlier group, which is what makes this reduce to "the
    first `total` groups" when every group holds one node.

    Args:
        sizes: Nodes in each group.
        total: Workers to share out. Never more than `sum(sizes)` at any call site here, so
            no group is ever given more workers than it has nodes.

    Returns:
        The share for each group, in the input order, summing to `total`.
    """
    population = sum(sizes)
    if population <= 0 or total <= 0:
        return [0] * len(sizes)
    shares = [total * size // population for size in sizes]
    remainders = [total * size % population for size in sizes]
    order = sorted(range(len(sizes)), key=lambda i: (-remainders[i], i))
    for index in order[: total - sum(shares)]:
        shares[index] += 1
    return shares


def _spread_census(classes: Sequence[tuple[int, int]], workers: int) -> list[int]:
    """`_spread` over a census: `(capacity, how many nodes have it)` in, total placed per class.

    The same placement `_spread` computes, at O(classes) instead of O(nodes). A hundred-thousand
    -node fleet is a handful of instance types across a handful of zones, so the class count is
    two or three orders of magnitude smaller — and this is the arithmetic behind
    `ClusterShape.locality_shares`, which the cost model evaluates once per pipeline breaker.
    Measured on a synthetic 100,000-node fleet, one `locality_shares` call took 106 ms.

    **Identical to `_spread` on the expanded fleet whenever each class's nodes are contiguous
    in that expansion**, which is how `ClusterShape` expands one. Contiguity is load-bearing
    rather than incidental: `_spread` hands its remainder to the first `extra` nodes in
    `(-capacity, index)` order, so two interleaved classes of equal capacity would split that
    remainder differently from two adjacent ones. `tests/unit/test_locality_census.py` holds
    this to `_spread` itself over randomized fleets.

    Within one class the per-node placements differ by at most one — `_spread` deals evenly
    among equal capacities — so the class total is all a caller needs: `divmod(total, count)`
    recovers the individual placements exactly.

    Args:
        classes: `(capacity, node count)` per class, in the fleet's own order.
        workers: Workers to place.

    Returns:
        Workers placed on each class as a whole, one entry per class, in the input order.
    """
    capacities = [max(0, int(capacity)) for capacity, _ in classes]
    counts = [max(0, int(count)) for _, count in classes]
    per_node = [0] * len(classes)
    occupied = sorted(
        (j for j in range(len(classes)) if capacities[j] > 0 and counts[j] > 0),
        key=lambda j: (-capacities[j], j),
    )
    totals = [0] * len(classes)
    if workers <= 0 or not occupied:
        return totals

    def _deal(order: Sequence[int], base: int, extra: int) -> list[int]:
        """`base` to every node in `order`, then `extra` more shared out across the classes.

        The `extra` are apportioned by node count rather than handed to the first classes in
        order, and that is the one place a census cannot simply replay what `_spread` did.
        `_spread` gives its remainder to the first `extra` *nodes* in `(-capacity, index)`
        order, where `index` is the fleet's own node ordering — which was the node id, and Ray
        node ids are random. So on a homogeneous fleet the per-node form drew an arbitrary
        sample of nodes, and a class-contiguous replay of it draws the most clustered sample
        instead: every worker lands in the first class, which is one rack.

        That is not a neutral difference. It over-states rack locality, and over-stating
        locality under-charges a shuffle — the direction this module's placement is
        deliberately biased *against*. Apportioning by node count is the expectation of the
        arbitrary draw it replaces, so it is unbiased rather than merely different, and on a
        fleet of single-node classes — every `ClusterShape` built by hand, and any fleet whose
        machines are all distinguishable — it reduces to exactly "the first `extra`", which is
        what `_spread` does.
        """
        out = [per_node[j] * counts[j] for j in range(len(classes))]
        for j, share in zip(order, _apportion([counts[j] for j in order], extra), strict=True):
            out[j] += base * counts[j] + share
        return out

    remaining = workers
    active = list(occupied)
    while active and remaining > 0:
        base, extra = divmod(remaining, sum(counts[j] for j in active))
        if base == 0:  # fewer workers than nodes: one each to the largest
            return _deal(active, 0, remaining)
        filled = [j for j in active if capacities[j] - per_node[j] <= base]
        if not filled:  # nobody is capacity-bound: deal the whole remainder out
            return _deal(active, base, extra)
        for j in filled:
            remaining -= (capacities[j] - per_node[j]) * counts[j]
            per_node[j] = capacities[j]
        active = [j for j in active if per_node[j] < capacities[j]]

    if remaining > 0:  # over-subscribed: more workers than the fleet has capacity units
        base, extra = divmod(remaining, sum(counts[j] for j in occupied))
        return _deal(occupied, base, extra)
    return [per_node[j] * counts[j] for j in range(len(classes))]


def _group_share_census(groups: Sequence[tuple[int, int]], workers: int) -> float:
    """`_group_share` over `(group_size, how many groups are that size)` pairs.

    The census form of the same sum. A fleet describes itself as classes of identical nodes,
    so the groups an exchange falls into arrive already counted; expanding them back to one
    entry per group to sum their squares would put the O(nodes) term back into a figure the
    cost model reads once per pipeline breaker.

    Args:
        groups: `(size, multiplicity)` pairs. Groups must partition the fleet.
        workers: Total workers in the exchange.

    Returns:
        The share in `[0, 1]`, `0.0` for an empty exchange.
    """
    if workers <= 0:
        return 0.0
    total = float(workers) ** 2
    if total <= 0:
        return 0.0
    return sum(float(size) ** 2 * multiplicity for size, multiplicity in groups) / total


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
    return _group_share_census([(size, 1) for size in sizes], workers)


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
