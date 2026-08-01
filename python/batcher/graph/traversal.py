"""Distances: how far each node is from a starting set, in hops or in weight.

Both functions here expand a frontier one round at a time, which makes their cost
proportional to the *reached* part of the graph rather than to the whole of it. Starting
from one node in a graph of a billion edges is cheap if that node's neighbourhood is
small, and that is the usual case.

The distances are single-source (or multi-source), never all-pairs. An all-pairs distance
matrix on any graph large enough to need this engine is quadratic in node count and does
not fit anywhere, so the honest interface is "from these nodes", and the summary
statistics that normally come from all-pairs (`harmonic_centrality`, eccentricity) are
computed from a sample of sources and labelled as estimates.
"""

from __future__ import annotations

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.api.dataset import Dataset
from batcher.graph._graph import DST, NODE, SRC, WEIGHT, Graph
from batcher.graph._iterate import checkpoint

__all__ = [
    "bfs",
    "diameter_estimate",
    "harmonic_centrality",
    "k_hop_neighbors",
    "reachable_from",
    "shortest_path_lengths",
]


def _seed_frontier(g: Graph, sources: Dataset, node: str) -> Dataset:
    seeds = sources.select(**{NODE: bt.col(node)}).distinct()
    present = seeds.join(g.nodes(), on=NODE, how="semi")
    if present.count() == 0:
        raise PlanError(
            "none of the source nodes appear in the graph, so nothing can be reached from them"
        )
    return present


def bfs(g: Graph, sources: Dataset, *, node: str = "node", max_depth: int = 10) -> Dataset:
    """Hop distance from a set of source nodes, by frontier expansion.

    Each round expands the frontier by one hop and keeps only the nodes not already
    reached, so a node's recorded depth is its shortest hop distance and every node is
    visited once.

    Args:
        g: The graph. Edge direction is respected; symmetrize first with
            `Graph.to_undirected` for an undirected search.
        sources: The nodes to start from. Multiple sources give the distance to the
            *nearest* one, which is what a "how far is each node from any warehouse"
            question wants.
        node: The column in `sources` holding the node id.
        max_depth: Stop after this many hops. Nodes further away are simply absent from
            the result, which is what makes a bounded search cheap.

    Returns:
        A dataset of `node` and `depth`, holding only the nodes actually reached.

    Raises:
        PlanError: If `max_depth` is negative, or no source is in the graph.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, bfs
            >>> e = bt.from_pydict({"src": [1, 2, 3], "dst": [2, 3, 4]})
            >>> out = bfs(Graph.from_edges(e), bt.from_pydict({"node": [1]}))
            >>> out.sort("node").to_pydict()
            {'node': [1, 2, 3, 4], 'depth': [0, 1, 2, 3]}
    """
    if max_depth < 0:
        raise PlanError(f"max_depth must be non-negative, got {max_depth}")
    frontier = _seed_frontier(g, sources, node)
    visited = checkpoint(frontier.select(**{NODE: bt.col(NODE), "depth": bt.lit(0)}))
    current = checkpoint(frontier)
    edges = g.edges.cache()
    for depth in range(1, max_depth + 1):
        if current.count() == 0:
            break
        # One hop out, minus everything already reached: the anti-join is what keeps a
        # node's first-recorded depth its shortest one.
        nxt = (
            edges.join(current.select(**{SRC: bt.col(NODE)}), on=SRC, how="semi")
            .select(**{NODE: bt.col(DST)})
            .distinct()
            .join(visited.select(NODE), on=NODE, how="anti")
        )
        nxt = checkpoint(nxt)
        if nxt.count() == 0:
            break
        visited = checkpoint(
            visited.union(nxt.select(**{NODE: bt.col(NODE), "depth": bt.lit(depth)}))
        )
        current = nxt
    return visited


def shortest_path_lengths(
    g: Graph, sources: Dataset, *, node: str = "node", max_iterations: int = 20
) -> Dataset:
    """Weighted shortest-path distance from a set of sources, by edge relaxation.

    Bellman-Ford's relaxation, one round per iteration: every round each node takes the
    smallest of its current distance and the distance through any incoming edge. Converges
    once nothing improves.

    **Negative edge weights are rejected.** With one, "shortest" has no meaning on a graph
    containing a negative cycle, and the loop would drive distances to minus infinity
    while looking like it was working.

    Args:
        g: The graph.
        sources: The nodes to start from, at distance 0.
        node: The column in `sources` holding the node id.
        max_iterations: The cap on relaxation rounds.

    Returns:
        A dataset of `node` and `distance`, holding only the nodes actually reached.

    Raises:
        PlanError: If any edge weight is negative, or no source is in the graph.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, shortest_path_lengths
            >>> e = bt.from_pydict(
            ...     {"src": [1, 1, 2], "dst": [2, 3, 3], "w": [1.0, 9.0, 1.0]}
            ... )
            >>> g = Graph.from_edges(e, weight="w")
            >>> out = shortest_path_lengths(g, bt.from_pydict({"node": [1]}))
            >>> out.sort("node").to_pydict()
            {'node': [1, 2, 3], 'distance': [0.0, 1.0, 2.0]}
    """
    if max_iterations < 1:
        raise PlanError(f"max_iterations must be at least 1, got {max_iterations}")
    negatives = g.edges.filter(bt.col(WEIGHT) < 0.0).count()
    if negatives:
        raise PlanError(
            f"shortest_path_lengths(): {negatives} edge(s) have a negative weight. "
            "Shortest paths are undefined on a graph with a negative cycle, and the "
            "relaxation would diverge rather than fail, so this is refused up front."
        )
    frontier = _seed_frontier(g, sources, node)
    dist = checkpoint(frontier.select(**{NODE: bt.col(NODE), "distance": bt.lit(0.0)}))
    edges = g.edges.cache()
    for _ in range(max_iterations):
        relaxed = (
            edges.join(
                dist.select(**{SRC: bt.col(NODE), "_d": bt.col("distance")}),
                on=SRC,
                how="inner",
            )
            .select(**{NODE: bt.col(DST), "_c": bt.col("_d") + bt.col(WEIGHT)})
            .group_by(NODE)
            .agg(_c=bt.min("_c"))
        )
        merged = checkpoint(
            dist.join(relaxed, on=NODE, how="outer").select(
                **{
                    NODE: bt.col(NODE),
                    "distance": bt.min_horizontal(
                        bt.coalesce(bt.col("distance"), bt.col("_c")),
                        bt.coalesce(bt.col("_c"), bt.col("distance")),
                    ),
                }
            )
        )
        if merged.count() == dist.count():
            improved = (
                merged.select(**{NODE: bt.col(NODE), "_new": bt.col("distance")})
                .join(dist.select(**{NODE: bt.col(NODE), "_old": bt.col("distance")}), on=NODE)
                .filter(bt.col("_new") < bt.col("_old"))
                .count()
            )
            if improved == 0:
                return merged
        dist = merged
    return dist


def k_hop_neighbors(g: Graph, sources: Dataset, k: int, *, node: str = "node") -> Dataset:
    """Every node within `k` hops of a source, excluding the sources themselves.

    The neighbourhood extraction step of every graph-ML pipeline: a GNN layer sees one
    hop, two layers see two, and this is how you cut the subgraph a batch needs out of a
    graph too big to hold.

    Args:
        g: The graph.
        sources: The nodes to expand from.
        k: How many hops.
        node: The column in `sources` holding the node id.

    Returns:
        A dataset of `node` and `depth`, with `depth` from 1 to `k`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, k_hop_neighbors
            >>> e = bt.from_pydict({"src": [1, 2, 3], "dst": [2, 3, 4]})
            >>> out = k_hop_neighbors(Graph.from_edges(e), bt.from_pydict({"node": [1]}), 2)
            >>> out.sort("node").to_pydict()
            {'node': [2, 3], 'depth': [1, 2]}
    """
    return bfs(g, sources, node=node, max_depth=k).filter(bt.col("depth") > 0)


def reachable_from(
    g: Graph, sources: Dataset, *, node: str = "node", max_depth: int = 100
) -> Dataset:
    """Every node reachable from a set of sources, at any distance.

    Args:
        g: The graph.
        sources: The nodes to start from.
        node: The column in `sources` holding the node id.
        max_depth: The safety cap on hops.

    Returns:
        A one-column dataset of `node`, including the sources.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, reachable_from
            >>> e = bt.from_pydict({"src": [1, 2, 8], "dst": [2, 3, 9]})
            >>> out = reachable_from(Graph.from_edges(e), bt.from_pydict({"node": [1]}))
            >>> sorted(out.to_pydict()["node"])
            [1, 2, 3]
    """
    return bfs(g, sources, node=node, max_depth=max_depth).select(NODE)


def harmonic_centrality(
    g: Graph, sources: Dataset, *, node: str = "node", max_depth: int = 10
) -> Dataset:
    """How close each node is to a sample of sources, as a sum of reciprocal distances.

    Harmonic rather than plain closeness, because a node that some sources cannot reach
    has an infinite distance to them; harmonic centrality scores that as zero and keeps
    going, while closeness would be undefined. That makes it the measure that works on a
    disconnected graph, which is most of them.

    **This is an estimate over the sources you provide**, not the true centrality over all
    node pairs, which is quadratic and does not fit. Sample the sources (`Dataset.sample`)
    and the ranking converges quickly even when the values do not.

    Args:
        g: The graph.
        sources: The nodes to measure distance from.
        node: The column in `sources` holding the node id.
        max_depth: The hop cap; nodes further away contribute nothing.

    Returns:
        A dataset of `node` and `harmonic_centrality`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, harmonic_centrality
            >>> e = bt.from_pydict({"src": [1, 2, 3], "dst": [2, 3, 4]})
            >>> g = Graph.from_edges(e).to_undirected()
            >>> out = harmonic_centrality(g, bt.from_pydict({"node": [1, 4]}))
            >>> out.sort("node").to_pydict()["harmonic_centrality"]
            [0.3333333333333333, 1.5, 1.5, 0.3333333333333333]
    """
    seeds = _seed_frontier(g, sources, node).to_pydict()[NODE]
    reached = None
    for seed in seeds:
        one = bfs(g, bt.from_pydict({NODE: [seed]}), node=NODE, max_depth=max_depth).filter(
            bt.col("depth") > 0
        )
        scored = one.select(
            **{NODE: bt.col(NODE), "_s": bt.lit(1.0) / bt.col("depth").cast("float64")}
        )
        reached = scored if reached is None else reached.union(scored)
    if reached is None:
        return g.nodes().select(**{NODE: bt.col(NODE), "harmonic_centrality": bt.lit(0.0)})
    summed = reached.group_by(NODE).agg(_s=bt.sum("_s"))
    return (
        g.nodes()
        .join(summed, on=NODE, how="left")
        .select(
            **{
                NODE: bt.col(NODE),
                "harmonic_centrality": bt.coalesce(bt.col("_s"), bt.lit(0.0)),
            }
        )
    )


def diameter_estimate(
    g: Graph, sources: Dataset, *, node: str = "node", max_depth: int = 50
) -> int:
    """The longest shortest path found from a sample of sources.

    A **lower bound** on the true diameter, and named for it: the real diameter is the
    maximum over all pairs, and this searches from the sources you give it. Sampling a few
    dozen high-degree nodes gets within a hop or two of the truth on most real graphs,
    which is enough to size a `max_iterations` for the iterative algorithms.

    Args:
        g: The graph.
        sources: The nodes to search from.
        node: The column in `sources` holding the node id.
        max_depth: The hop cap. A result equal to this means the search was truncated and
            the true diameter is larger.

    Returns:
        The largest depth reached, or 0 when nothing is reachable.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, diameter_estimate
            >>> e = bt.from_pydict({"src": [1, 2, 3], "dst": [2, 3, 4]})
            >>> diameter_estimate(Graph.from_edges(e), bt.from_pydict({"node": [1]}))
            3
    """
    got = bfs(g, sources, node=node, max_depth=max_depth).agg(d=bt.max("depth")).to_pydict()["d"]
    return int(got[0]) if got and got[0] is not None else 0
