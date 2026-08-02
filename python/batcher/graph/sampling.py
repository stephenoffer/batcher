"""Sampling a graph: random walks, bounded neighbourhoods, and induced subgraphs.

This is the module graph ML actually runs on. A GNN cannot see a billion-edge graph at
once, and the two standard ways around that are both here: sample a fixed number of
neighbours per node so a layer's cost is bounded regardless of degree (`neighbor_sample`),
or walk the graph and treat the walks as sentences (`random_walks`), which is how
DeepWalk and node2vec turn a graph into something an embedding model can read.

Every function takes an explicit `seed` and is deterministic given one. That is not
politeness: an embedding trained on walks you cannot regenerate is an embedding you cannot
debug, and a neighbour sample that changes between the train and inference passes is a
silent accuracy loss that looks like drift.
"""

from __future__ import annotations

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.api.dataset import Dataset
from batcher.graph._graph import DST, NODE, SRC, WEIGHT, Graph
from batcher.graph._iterate import checkpoint

__all__ = [
    "edge_sample",
    "ego_network",
    "neighbor_sample",
    "node_sample",
    "random_walks",
]


def _check_seed(seed: int) -> None:
    if seed < 0:
        raise PlanError(f"seed must be non-negative, got {seed}")


def neighbor_sample(g: Graph, k: int, *, seed: int = 0) -> Dataset:
    """At most `k` outgoing edges per node, chosen deterministically.

    The bound that makes a GNN layer affordable: without it, one celebrity node with a
    million neighbours dominates a batch and the layer's cost is set by the worst node
    rather than the average. GraphSAGE is exactly this, applied once per layer.

    Selection is by a hash of the edge rather than by a shuffle, so it is stable across
    runs, across partitionings, and between a training and an inference pass. Two nodes
    with the same neighbours get the same sample, which is what makes a cached embedding
    still valid.

    Args:
        g: The graph.
        k: The maximum number of edges to keep per source node.
        seed: Varies the selection. The same seed always gives the same sample.

    Returns:
        An edge table with at most `k` rows per `src`.

    Raises:
        PlanError: If `k` is not positive or `seed` is negative.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, neighbor_sample
            >>> e = bt.from_pydict({"src": [1, 1, 1, 2], "dst": [2, 3, 4, 3]})
            >>> out = neighbor_sample(Graph.from_edges(e), 2)
            >>> sorted(out.group_by("src").agg(n=bt.count()).to_pydict()["n"])
            [1, 2]
    """
    if k < 1:
        raise PlanError(f"k must be positive, got {k}")
    _check_seed(seed)
    ranked = g.edges.with_columns(
        _h=bt.hash_rows(bt.col(SRC), bt.col(DST), bt.lit(seed))
    ).with_columns(_rank=bt.col("_h").rank().over(partition_by=SRC, order_by="_h"))
    return ranked.filter(bt.col("_rank") <= bt.lit(k)).select(SRC, DST, WEIGHT)


def random_walks(
    g: Graph,
    starts: Dataset,
    length: int,
    *,
    node: str = "node",
    seed: int = 0,
) -> Dataset:
    """Fixed-length random walks from a set of starting nodes.

    The input to DeepWalk and node2vec: each walk is a sequence of node ids, and feeding
    those to a word-embedding model produces node embeddings whose geometry reflects the
    graph's structure. Walks are the standard way to do that because they turn an
    irregular structure into the regular one an embedding model expects.

    A walk that reaches a node with no outgoing edges stops there, so walks can be shorter
    than `length`. That is reported rather than padded, because a padded walk would teach
    the model that the dead-end node is followed by whatever the padding is.

    Args:
        g: The graph. Edge weights bias the step: a neighbour reached by a heavier edge is
            proportionally more likely.
        starts: The nodes to walk from, one walk per row.
        length: How many steps to take. A walk holds up to `length + 1` nodes.
        node: The column in `starts` holding the node id.
        seed: Varies the walks. The same seed always gives the same walks.

    Returns:
        A dataset of `walk` (an id per walk), `step` (0-based position) and `node`.

    Raises:
        PlanError: If `length` is not positive or `seed` is negative.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, random_walks
            >>> e = bt.from_pydict({"src": [1, 2, 3], "dst": [2, 3, 1]})
            >>> walks = random_walks(Graph.from_edges(e), bt.from_pydict({"node": [1]}), 3)
            >>> walks.sort("step").to_pydict()["node"]
            [1, 2, 3, 1]
    """
    if length < 1:
        raise PlanError(f"length must be positive, got {length}")
    _check_seed(seed)
    current = checkpoint(
        starts.select(**{NODE: bt.col(node)})
        .with_row_index("walk")
        # `with_row_index` produces a non-nullable column, while the grouped result each
        # round produces a nullable one, and a union of the two is an Arrow schema
        # mismatch rather than a widening. The coalesce is what makes the two sides the
        # same type; it changes no value.
        .select(walk=bt.coalesce(bt.col("walk"), bt.lit(0)), **{NODE: bt.col(NODE)})
    )
    trail = checkpoint(current.select("walk", step=bt.lit(0), **{NODE: bt.col(NODE)}))
    edges = g.edges.cache()
    for step in range(1, length + 1):
        if current.count() == 0:
            break
        # Pick one outgoing edge per walk: score every candidate by a hash of
        # (walk, step, destination) scaled by the edge weight, and take the best. Hashing
        # rather than sampling keeps the walk reproducible and keeps the choice inside the
        # engine instead of pulling candidates into Python.
        candidates = current.join(
            edges.select(**{NODE: bt.col(SRC), "_to": bt.col(DST), "_w": bt.col(WEIGHT)}),
            on=NODE,
            how="inner",
        ).with_columns(
            _score=(
                bt.hash_rows(bt.col("walk"), bt.lit(step * 2_654_435_761 + seed), bt.col("_to"))
                .abs()
                .cast("float64")
                % bt.lit(1_000_003.0)
            )
            * bt.col("_w")
        )
        best = candidates.group_by("walk").agg(_score=bt.max("_score"))
        chosen = checkpoint(
            candidates.join(best, on=["walk", "_score"], how="inner")
            .group_by("walk")
            .agg(**{NODE: bt.min("_to")})
        )
        if chosen.count() == 0:
            break
        trail = checkpoint(
            trail.union(chosen.select("walk", step=bt.lit(step), **{NODE: bt.col(NODE)}))
        )
        current = chosen
    return trail


def node_sample(g: Graph, fraction: float, *, seed: int = 0) -> Dataset:
    """A deterministic fraction of the nodes, by hash.

    Hashing rather than shuffling means a node's membership does not depend on how many
    other nodes there are, so a sample taken today still contains the same nodes after the
    graph grows. That is what makes an incrementally-updated graph's sample stable.

    Args:
        g: The graph.
        fraction: The share of nodes to keep, in `(0, 1]`.
        seed: Varies which nodes are kept.

    Returns:
        A one-column dataset of `node`.

    Raises:
        PlanError: If `fraction` is outside `(0, 1]` or `seed` is negative.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, node_sample
            >>> e = bt.from_pydict({"src": list(range(50)), "dst": list(range(1, 51))})
            >>> kept = node_sample(Graph.from_edges(e), 0.5).count()
            >>> 10 < kept < 42
            True
    """
    if not 0.0 < fraction <= 1.0:
        raise PlanError(f"fraction must be in (0, 1], got {fraction}")
    _check_seed(seed)
    cutoff = int(fraction * 1_000_003)
    return (
        g.nodes()
        .filter(bt.hash_rows(bt.col(NODE), bt.lit(seed)).abs() % bt.lit(1_000_003) < bt.lit(cutoff))
        .select(NODE)
    )


def edge_sample(g: Graph, fraction: float, *, seed: int = 0) -> Graph:
    """A deterministic fraction of the edges, by hash.

    The right way to build a link-prediction train/test split: hold out a fraction of the
    edges, train on the rest, and score whether the held-out ones come back. Sampling
    *nodes* for that would remove whole neighbourhoods and make the task a different one.

    Args:
        g: The graph.
        fraction: The share of edges to keep, in `(0, 1]`.
        seed: Varies which edges are kept.

    Returns:
        A graph over the sampled edges.

    Raises:
        PlanError: If `fraction` is outside `(0, 1]` or `seed` is negative.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, edge_sample
            >>> e = bt.from_pydict({"src": list(range(100)), "dst": list(range(1, 101))})
            >>> kept = edge_sample(Graph.from_edges(e), 0.5).num_edges()
            >>> 25 < kept < 75
            True
    """
    if not 0.0 < fraction <= 1.0:
        raise PlanError(f"fraction must be in (0, 1], got {fraction}")
    _check_seed(seed)
    cutoff = int(fraction * 1_000_003)
    kept = g.edges.filter(
        bt.hash_rows(bt.col(SRC), bt.col(DST), bt.lit(seed)).abs() % bt.lit(1_000_003)
        < bt.lit(cutoff)
    )
    from dataclasses import replace

    return replace(g, edges=kept)


def ego_network(g: Graph, center: object, radius: int = 1) -> Graph:
    """The subgraph induced on everything within `radius` hops of one node.

    The unit a graph-ML batch is usually built from, and the way to look at one node's
    situation without materializing anything else. The centre is included.

    Args:
        g: The graph.
        center: The node id at the centre.
        radius: How many hops out to include.

    Returns:
        The induced subgraph.

    Raises:
        PlanError: If `radius` is negative.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, ego_network
            >>> e = bt.from_pydict({"src": [1, 2, 3, 9], "dst": [2, 3, 4, 9]})
            >>> sorted(ego_network(Graph.from_edges(e), 1, 2).nodes().to_pydict()["node"])
            [1, 2, 3]
    """
    if radius < 0:
        raise PlanError(f"radius must be non-negative, got {radius}")
    from batcher.graph.traversal import bfs

    seeds = bt.from_pydict({NODE: [center]})
    reached = bfs(g.to_undirected(), seeds, node=NODE, max_depth=radius).select(NODE)
    return g.subgraph(checkpoint(reached))
