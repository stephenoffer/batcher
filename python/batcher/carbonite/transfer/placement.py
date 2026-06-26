"""Locality-aware reducer placement — put a reducer where its data already is.

A hash-shuffle reducer fetches its bucket from every mapper. When the bucket's bytes
are spread evenly across the cluster (the unskewed case) no placement helps — every
reducer fetches the same fraction from each node. But when a bucket is *concentrated*
on one node (a skewed key, or a co-partitioned upstream), hosting that reducer on the
same node turns the bulk of its fetches into same-node `SHARED_MEMORY`/`DIRECT_MEMORY`
hits instead of network transfers.

This module is the pure decision: given how many bytes of each bucket sit on each node,
`reducer_affinity` names the node a bucket is concentrated on (or leaves it out when no
node dominates), and `assign_reducer_hosts` turns that into a host-actor index per
reducer — load-balanced across a node's actors, falling back to the default round-robin
for unconcentrated buckets. Placement never changes the *result*, only where the bytes
travel, so it is always safe; the orchestrator (`dist`) measures the bytes and applies
the assignment, keeping this layer free of Ray.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

__all__ = ["assign_reducer_hosts", "reducer_affinity"]

# A bucket is "concentrated" — worth pulling its reducer onto a node — only when that
# node holds at least this fraction of the bucket's bytes. At/below it the bucket is
# effectively uniform and placement would just unbalance the fleet for no locality gain.
_CONCENTRATION = 0.5


def reducer_affinity(
    bucket_node_bytes: Mapping[int, Mapping[str, int]], concentration: float = _CONCENTRATION
) -> dict[int, str]:
    """The node each bucket is concentrated on, for buckets that have one.

    `bucket_node_bytes[r][node]` is how many bytes of reducer-bucket `r` live on `node`.
    A bucket is included only when its top node holds `>= concentration` of its bytes
    (a skewed/co-located bucket); a uniformly-spread bucket is omitted, so the caller
    keeps its default placement for it. Deterministic: ties break on the node id.
    """
    out: dict[int, str] = {}
    for bucket, node_bytes in bucket_node_bytes.items():
        total = sum(node_bytes.values())
        if total <= 0:
            continue
        node, nbytes = max(node_bytes.items(), key=lambda kv: (kv[1], kv[0]))
        share = nbytes / total
        n_with_data = sum(1 for b in node_bytes.values() if b > 0)
        # Concentrated = a clear majority (>= `concentration`) that also beats the
        # uniform share (`1/n_with_data`), so an evenly-split bucket — e.g. 50/50 across
        # two nodes, where one node holds exactly the uniform 50% — is NOT flagged.
        if share >= concentration and (n_with_data <= 1 or share > 1.0 / n_with_data):
            out[bucket] = node
    return out


def assign_reducer_hosts(
    n_reducers: int, actor_nodes: Sequence[str], affinity: Mapping[int, str]
) -> list[int]:
    """Host-actor index for each of `n_reducers` reducers.

    A bucket with an `affinity` node is hosted on an actor *on that node* (round-robin
    across the node's actors, so one hot node's reducers still spread over its actors);
    every other bucket keeps the default `reducer r → actor r` round-robin, so an
    unskewed shuffle's placement — and behavior — is exactly as before. `actor_nodes[i]`
    is the node id actor `i` runs on.
    """
    nodes_to_actors: dict[str, list[int]] = defaultdict(list)
    for i, node in enumerate(actor_nodes):
        nodes_to_actors[node].append(i)

    n_actors = len(actor_nodes)
    hosts = [r % n_actors if n_actors else 0 for r in range(n_reducers)]
    cursor: dict[str, int] = defaultdict(int)
    for r in range(n_reducers):
        node = affinity.get(r)
        actors_on_node = nodes_to_actors.get(node) if node is not None else None
        if actors_on_node:
            hosts[r] = actors_on_node[cursor[node] % len(actors_on_node)]
            cursor[node] += 1
    return hosts
