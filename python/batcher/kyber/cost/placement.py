"""Should a breaker's workers be packed onto few nodes, or spread across many?

Kyber annotates every breaker with `ResourceBounds.prefers_locality`, which Carbonite turns
into a `PACK` or `SPREAD` preference and `dist` resolves against the live cluster. The rule
deciding it has been one absolute byte threshold: a shuffle under `locality_max_bytes` (4 MiB)
prefers PACK, anything larger prefers SPREAD.

That threshold answers the question it was written for — a *tiny* exchange between a handful of
workers should not be scattered across a network for nothing — and it is blind to the property
that actually decides the trade on a dense fleet. Packing does not merely avoid a small
shuffle; it **moves an exchange down a tier**. Sixteen workers packed onto two eight-device
nodes keep roughly half their all-to-all traffic inside a host, where it moves at host-memory
or NVLink rate instead of at the NIC's. On a 25 Gb/s fleet that is a factor of six on half the
bytes, and it grows with the shuffle rather than shrinking — so the existing rule refuses PACK
in exactly the regime where packing is worth the most.

The other half of the trade is real too, which is why this does not simply invert the rule.
Packing concentrates the fleet: those workers share one node's memory bandwidth, one node's
page cache, and one node's NIC for reading their inputs, and a gang on two nodes has two nodes'
worth of failure domain. That cost is **not priced here**, and saying so plainly matters more
than modelling it badly — a per-node read bandwidth is a figure this process cannot measure,
and inventing one would put a fabricated number under a placement decision.

It is bounded instead. Packing is preferred only when the gang fits on a *handful* of nodes
(`_MAX_PACKED_NODES`) and only when the tier saving is a material fraction of the exchange.
Past that bound the concentration argument is the one that dominates and the answer stays
SPREAD, which is what it was before. So this rule can only move a plan that both fits tightly
and gains a lot, and everything else keeps the byte threshold exactly.

Kyber decides; nothing here executes. `dist` still makes the final call against the live
cluster and may downgrade either way.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher.kyber.cost.locality import locality_factor

__all__ = ["PlacementAdvice", "prefers_locality"]

#: Minimum tier saving worth concentrating a fleet for, as a fraction of the shuffle's cost.
#: Below this the packed and spread plans move nearly the same bytes at nearly the same price,
#: and the packing has bought a smaller failure domain and a narrower read path for nothing.
_MIN_SAVING = 0.15

#: Nodes a packed gang may occupy before it stops being "packed" in any useful sense. Past a
#: handful the exchange is crossing the network anyway, the tier saving has already been
#: counted by the locality factor, and the concentration argument no longer applies. Stated as
#: a bound rather than derived, because what it really encodes is the failure-domain limit an
#: operator is willing to accept, and that is a policy question with no measurable answer.
_MAX_PACKED_NODES = 4


@dataclass(frozen=True, slots=True)
class PlacementAdvice:
    """Whether to prefer co-locating a breaker's workers, and what decided it.

    Attributes:
        pack: True to prefer `PACK`. Consumed as `ResourceBounds.prefers_locality`.
        saving: Fraction of the exchange's cost that packing removes, `0.0` when unknown.
        nodes: Nodes the packed gang would occupy, `0` when the fleet shape is unknown.
        reason: One line for the decision log.
    """

    pack: bool
    saving: float = 0.0
    nodes: int = 0
    reason: str = ""


def _packed_nodes(hardware, workers: int) -> int:
    """Nodes a `workers`-wide gang would occupy if packed as tightly as capacity allows.

    `0` when the fleet's shape is unknown, which is the signal to fall back to the byte
    threshold rather than to reason about a topology nobody reported.
    """
    cluster = getattr(hardware, "cluster", None)
    if cluster is None or not cluster.known:
        return 0
    # Densest-first, because a gang packs onto the biggest hosts it can reach. Using the mean
    # would report a fleet of one fat node and many thin ones as unable to pack, which is
    # precisely the fleet where packing is most available.
    capacities = sorted((n.cpu_cores for n in cluster.nodes if n.cpu_cores > 0), reverse=True)
    if not capacities:
        return 0
    used = 0
    for capacity in capacities:
        if workers <= 0:
            break
        workers -= capacity
        used += 1
    return used if workers <= 0 else 0


def prefers_locality(
    hardware,
    workers: int,
    shuffle_bytes: float,
    threshold_bytes: int,
) -> PlacementAdvice:
    """Whether this breaker's workers should be packed onto few nodes.

    Args:
        hardware: The `HardwareProfile` being planned against.
        workers: The breaker's desired task fan-out.
        shuffle_bytes: Bytes the breaker exchanges (its input volume in bytes).
        threshold_bytes: `optimizer.locality_max_bytes` — the existing absolute rule, which
            stands wherever the fleet cannot be reasoned about.

    Returns:
        The advice. Identical to the pre-existing threshold rule on an unreadable fleet, on a
        single node, and wherever packing would buy less than a material fraction of the
        exchange — so the only plans this moves are the ones where concentrating the gang
        provably drops a large exchange onto a faster tier.
    """
    small = shuffle_bytes <= max(0, threshold_bytes)
    if small:
        # The original rule, and it stands on its own terms: an exchange this small should not
        # be scattered whatever the fleet looks like.
        return PlacementAdvice(True, reason=f"~{shuffle_bytes:.0f}B under the locality threshold")

    nodes = _packed_nodes(hardware, workers)
    if nodes == 0 or nodes > _MAX_PACKED_NODES or workers <= 1:
        return PlacementAdvice(
            False,
            nodes=nodes,
            reason=(
                "fleet shape unknown; keeping the byte threshold"
                if nodes == 0
                else f"a packed gang would still span {nodes} nodes"
            ),
        )

    spread = locality_factor(hardware, workers)
    packed = locality_factor(_packed_view(hardware, nodes), workers)
    if spread <= 0.0:
        return PlacementAdvice(False, nodes=nodes, reason="exchange not priceable")
    saving = max(0.0, 1.0 - packed / spread)
    if saving < _MIN_SAVING:
        return PlacementAdvice(
            False,
            saving=saving,
            nodes=nodes,
            reason=f"packing onto {nodes} nodes saves only {saving:.0%} of the exchange",
        )
    return PlacementAdvice(
        True,
        saving=saving,
        nodes=nodes,
        reason=(
            f"packing onto {nodes} nodes drops {saving:.0%} of a "
            f"~{shuffle_bytes / 1e9:.1f}GB exchange onto a faster tier"
        ),
    )


def _packed_view(hardware, nodes: int):
    """`hardware` as it would look with the gang confined to its `nodes` densest hosts.

    A view rather than a mutation: the profile is frozen and shared, and the packed shape is
    only ever used to *price* an alternative that has not been chosen yet.

    Falls back to the input profile whenever the narrowed fleet cannot be built, so a failure
    here makes the saving zero and the advice "spread", which is the pre-existing answer.
    """
    from dataclasses import replace

    from batcher.plan.resource import ClusterShape

    cluster = getattr(hardware, "cluster", None)
    if cluster is None or not cluster.known:
        return hardware
    densest = sorted(cluster.nodes, key=lambda n: (-n.cpu_cores, n.node_id))[:nodes]
    if not densest:
        return hardware
    return replace(hardware, cluster=ClusterShape(nodes=tuple(densest)))
