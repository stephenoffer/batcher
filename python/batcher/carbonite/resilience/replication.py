"""Where each mapper's shuffle output is copied, so a lost worker costs a fetch not a recompute.

A mapper publishes its buckets into its own worker's memory. With one copy, losing that
worker destroys them and the only way back is *lineage recompute*: re-read the source
partition from object storage and re-run the map — the longest phase of most queries.
Replicating the output onto other workers turns that into a re-fetch from a survivor.

The trade is only affordable because of the mergeable algebra: what a mapper publishes is
`partial` state (already aggregated), typically far smaller than the source that produced
it, so a copy costs a fraction of a recompute. This module owns the *decision* — which
workers hold which copy — as a pure function; `dist.replication` performs it.

Carbonite owns it because it is a protect-against-loss resource decision (which node
carries which copy, and at what memory cost), the same way `transfer.placement` owns
locality-aware reducer placement. It never moves a byte.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

__all__ = ["assign_replica_hosts"]


def assign_replica_hosts(
    primaries: Mapping[int, int],
    nodes: Sequence[str],
    factor: int,
    dead: frozenset[int] | set[int] = frozenset(),
) -> dict[int, list[int]]:
    """Pick the workers that hold a replica of each source's shuffle output.

    A replica is only useful if it dies *independently* of the primary, so a worker on a
    different node always outranks one on the primary's node — a node loss (the unit a
    spot reclamation actually takes) must not remove every copy. Among equally-good
    candidates the least-loaded worker wins, so replicas spread evenly instead of piling
    onto one host. Returns at most `factor - 1` replicas per source, and fewer (or none)
    when the cluster is too small to place them — replication is an optimization, so a
    cluster that cannot host a copy degrades to the recompute path rather than failing.

    Args:
        primaries: Source id → the worker index whose memory holds that source's output.
        nodes: Node id per worker index, so a replica can be placed off the primary's node.
        factor: Total copies wanted per source (1 = no replica, the default).
        dead: Workers known to be gone; never assigned a copy.

    Returns:
        Source id → the worker indices holding a replica, excluding the primary.

    Examples:
        .. doctest::

            >>> from batcher.carbonite.resilience.replication import assign_replica_hosts
            >>> # Four workers on two nodes; each source's replica lands off its own node.
            >>> nodes = ["n1", "n1", "n2", "n2"]
            >>> assign_replica_hosts({0: 0, 1: 1, 2: 2, 3: 3}, nodes, factor=2)
            {0: [2], 1: [3], 2: [0], 3: [1]}

            >>> # factor=1 keeps the single copy (no replication).
            >>> assign_replica_hosts({0: 0, 1: 1}, ["n1", "n2"], factor=1)
            {0: [], 1: []}
    """
    wanted = max(0, factor - 1)
    out: dict[int, list[int]] = {src: [] for src in primaries}
    if wanted == 0 or not nodes:
        return out

    live = [w for w in range(len(nodes)) if w not in dead]
    load = dict.fromkeys(live, 0)
    for src in sorted(primaries):
        primary = primaries[src]
        primary_node = nodes[primary] if primary < len(nodes) else None
        # Off-node first (an independent failure domain), then least-loaded, then stable
        # by index so the assignment is deterministic — a replay picks the same hosts.
        candidates = sorted(
            (w for w in live if w != primary),
            key=lambda w: (nodes[w] == primary_node, load[w], w),
        )
        chosen = candidates[:wanted]
        out[src] = chosen
        for w in chosen:
            load[w] += 1
    return out
