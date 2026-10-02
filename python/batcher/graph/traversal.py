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

from collections.abc import Iterator

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.api.dataset import Dataset
from batcher.graph._graph import DST, NODE, SRC, WEIGHT, Graph, walked
from batcher.graph._iterate import checkpoint, fixpoint_rounds, require_fixpoint

__all__ = [
    "bfs",
    "diameter_estimate",
    "harmonic_centrality",
    "is_dag",
    "k_hop_neighbors",
    "nodes_in_cycles",
    "reachable_from",
    "shortest_path_lengths",
    "topological_order",
]


def _seed_frontier(g: Graph, sources: Dataset, node: str) -> Dataset:
    seeds = sources.select(**{NODE: bt.col(node)}).distinct()
    present = seeds.join(g.nodes(), on=NODE, how="semi")
    if present.count() == 0:
        raise PlanError(
            "none of the source nodes appear in the graph, so nothing can be reached from them"
        )
    return present


def bfs(g: Graph, sources: Dataset, *, node: str = "node", max_depth: int | None = None) -> Dataset:
    """Hop distance from a set of source nodes, by frontier expansion.

    Each round expands the frontier by one hop and keeps only the nodes not already
    reached, so a node's recorded depth is its shortest hop distance and every node is
    visited once.

    Args:
        g: The graph. Edge direction is respected on a directed graph; one built with
            `directed=False` is searched both ways.
        sources: The nodes to start from. Multiple sources give the distance to the
            *nearest* one, which is what a "how far is each node from any warehouse"
            question wants.
        node: The column in `sources` holding the node id.
        max_depth: Stop after this many hops, or `None` (the default) to search until
            nothing new is reached. Nodes beyond a cap are absent from the result, which
            is what makes a bounded search cheap; it is a filter you choose, not a
            convergence limit, so reaching it is not an error.

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
    if max_depth is not None and max_depth < 0:
        raise PlanError(f"max_depth must be non-negative, got {max_depth}")
    frontier = _seed_frontier(g, sources, node)
    visited = checkpoint(frontier.select(**{NODE: bt.col(NODE), "depth": bt.lit(0)}))
    current = checkpoint(frontier)
    edges = walked(g).edges.cache()
    depth = 0
    while max_depth is None or depth < max_depth:
        depth += 1
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
    g: Graph, sources: Dataset, *, node: str = "node", max_iterations: int | None = None
) -> Dataset:
    """Weighted shortest-path distance from a set of sources, by edge relaxation.

    Bellman-Ford's relaxation, one round per iteration: every round each node takes the
    smallest of its current distance and the distance through any incoming edge. Converges
    once nothing improves.

    **Negative edge weights are supported on a directed graph**, with Bellman-Ford's
    negative-cycle check: a shortest path visits each node at most once, so on an
    `n`-node graph the distances are final within `n` rounds, and a relaxation still
    improving after that has found a negative cycle reachable from a source. That raises
    rather than returning distances that would keep falling forever. A negative weight on
    an undirected graph is refused up front, because walking the edge there and back is
    already a negative cycle.

    Args:
        g: The graph. One built with `directed=False` is walked both ways.
        sources: The nodes to start from, at distance 0.
        node: The column in `sources` holding the node id.
        max_iterations: The cap on relaxation rounds, or `None` (the default) to relax
            until nothing improves, which takes at most one round per node.

    Returns:
        A dataset of `node` and `distance`, holding only the nodes actually reached.

    Raises:
        PlanError: If an undirected graph has a negative edge weight, a negative cycle
            is reachable from a source, no source is in the graph, or a `max_iterations`
            you set stops the relaxation before the distances are final.

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
    negatives = g.edges.filter(bt.col(WEIGHT) < 0.0).count()
    if negatives and not g.directed:
        raise PlanError(
            f"shortest_path_lengths(): {negatives} edge(s) of an undirected graph have a "
            "negative weight. Walking such an edge there and back is a negative cycle, so "
            "no shortest path is defined. Build the graph with directed=True if each edge "
            "is meant to be walked one way."
        )
    g = walked(g)
    rounds = fixpoint_rounds(max_iterations, g.num_nodes() if max_iterations is None else 0)
    frontier = _seed_frontier(g, sources, node)
    dist = checkpoint(frontier.select(**{NODE: bt.col(NODE), "distance": bt.lit(0.0)}))
    edges = g.edges.cache()
    for _ in range(rounds):
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
                    "distance": bt.least(
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
    if negatives and max_iterations is None:
        # `rounds` is one more than the node count, which a simple path cannot need.
        raise PlanError(
            "shortest_path_lengths(): the distances were still falling after "
            f"{rounds} rounds, one more than the graph has nodes, so a negative cycle is "
            "reachable from the sources and no shortest path is defined."
        )
    require_fixpoint(False, "shortest_path_lengths", rounds)
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
    g: Graph, sources: Dataset, *, node: str = "node", max_depth: int | None = None
) -> Dataset:
    """Every node reachable from a set of sources, at any distance.

    Args:
        g: The graph.
        sources: The nodes to start from.
        node: The column in `sources` holding the node id.
        max_depth: A cap on hops, or `None` (the default) for every reachable node.

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


def _seeded_levels(
    g: Graph, sources: Dataset, node: str, max_depth: int | None, batch_size: int
) -> Iterator[tuple[int, Dataset]]:
    """Every (source, node) pair first reached at each depth, one source batch at a time.

    One breadth-first search per source, run for a whole batch at once: the state carries
    the source it started from, so a batch costs the search depth in queries rather than
    sources times depth, and holds one row per (source, reached node) pair. The batch size
    bounds that state, which is the memory each round passes through the driver.

    Yields:
        ``(depth, pairs)`` for each non-empty depth of each batch, where `pairs` has
        `_seed` and `node`.
    """
    if max_depth is not None and max_depth < 0:
        raise PlanError(f"max_depth must be non-negative, got {max_depth}")
    if batch_size < 1:
        raise PlanError(f"source_batch_size must be positive, got {batch_size}")
    seeds = _seed_frontier(g, sources, node).to_pydict()[NODE]
    edges = walked(g).edges.cache()
    for start in range(0, len(seeds), batch_size):
        batch = seeds[start : start + batch_size]
        frontier = checkpoint(
            bt.from_pydict({"_seed": batch}).select("_seed", **{NODE: bt.col("_seed")})
        )
        seen = frontier
        depth = 0
        while max_depth is None or depth < max_depth:
            depth += 1
            nxt = checkpoint(
                edges.join(frontier.select("_seed", **{SRC: bt.col(NODE)}), on=SRC, how="inner")
                .select("_seed", **{NODE: bt.col(DST)})
                .distinct()
                .join(seen, on=["_seed", NODE], how="anti")
            )
            if nxt.count() == 0:
                break
            yield depth, nxt
            seen = checkpoint(seen.union(nxt))
            frontier = nxt


def harmonic_centrality(
    g: Graph,
    sources: Dataset,
    *,
    node: str = "node",
    max_depth: int | None = None,
    source_batch_size: int = 64,
) -> Dataset:
    """How close each node is to a sample of sources, as a sum of reciprocal distances.

    Harmonic rather than plain closeness, because a node that some sources cannot reach
    has an infinite distance to them; harmonic centrality scores that as zero and keeps
    going, while closeness would be undefined. That makes it the measure that works on a
    disconnected graph, which is most of them.

    **This is an estimate over the sources you provide**, not the true centrality over all
    node pairs, which is quadratic and does not fit. Sample the sources (`Dataset.sample`)
    and the ranking converges quickly even when the values do not.

    Sources are searched in batches, each one breadth-first search whose state carries the
    source it started from, so a batch costs the search depth in queries rather than
    sources times depth. The state is one row per (source, reached node) pair and passes
    through the driver once per hop; `source_batch_size` bounds it.

    Args:
        g: The graph. One built with `directed=False` is walked both ways.
        sources: The nodes to measure distance from.
        node: The column in `sources` holding the node id.
        max_depth: A hop cap, or `None` (the default) for none; nodes beyond a cap
            contribute nothing.
        source_batch_size: How many sources search together. A larger batch trades
            memory for fewer rounds.

    Returns:
        A dataset of `node` and `harmonic_centrality`.

    Raises:
        PlanError: If `max_depth` is negative, `source_batch_size` is not positive, or
            no source is in the graph.

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
    summed: Dataset | None = None
    for depth, pairs in _seeded_levels(g, sources, node, max_depth, source_batch_size):
        arm = pairs.select(NODE, _s=bt.lit(1.0 / depth))
        merged = arm if summed is None else summed.union(arm)
        # Folded every level so the running total stays one row per node.
        summed = checkpoint(merged.group_by(NODE).agg(_s=bt.sum("_s")))
    if summed is None:
        return g.nodes().select(**{NODE: bt.col(NODE), "harmonic_centrality": bt.lit(0.0)})
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
    g: Graph,
    sources: Dataset,
    *,
    node: str = "node",
    max_depth: int | None = None,
    source_batch_size: int = 64,
) -> int:
    """The longest shortest path found from a sample of sources.

    A **lower bound** on the true diameter, and named for it: the real diameter is the
    maximum over all pairs, and this is the largest eccentricity among the sources you
    give it. Each source is searched on its own, so adding a source can only raise the
    estimate. Sampling a few dozen high-degree nodes gets within a hop or two of the truth
    on most real graphs, which is enough to size a `max_iterations` for the iterative
    algorithms.

    Args:
        g: The graph.
        sources: The nodes to search from.
        node: The column in `sources` holding the node id.
        max_depth: A hop cap, or `None` (the default) for none. A result equal to a cap
            means the search was truncated and the true diameter may be larger.
        source_batch_size: How many sources search together; see `harmonic_centrality`.

    Returns:
        The largest depth reached from any one source, or 0 when nothing is reachable.

    Raises:
        PlanError: If `max_depth` is negative, `source_batch_size` is not positive, or
            no source is in the graph.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, diameter_estimate
            >>> e = bt.from_pydict({"src": [1, 2, 3], "dst": [2, 3, 4]})
            >>> diameter_estimate(Graph.from_edges(e), bt.from_pydict({"node": [1]}))
            3
    """
    return max(
        (depth for depth, _ in _seeded_levels(g, sources, node, max_depth, source_batch_size)),
        default=0,
    )


def topological_order(g: Graph, *, max_iterations: int | None = None) -> Dataset:
    """Order the nodes so every edge points forward, by repeated source peeling.

    Kahn's algorithm: take every node with no incoming edge, emit them as one level,
    remove them, repeat. Nodes at the same level are mutually independent, which is more
    useful than a total order -- a build system or a task scheduler can run a whole level
    in parallel, and a flat sequence would hide that.

    Args:
        g: The graph. It must be acyclic; see the return value for what happens when it
            is not.
        max_iterations: The cap on peeling rounds, or `None` (the default) to peel until
            nothing more can be placed. A cap shorter than the longest chain raises
            rather than returning a partial order that `is_dag` would misread as a cycle.

    Returns:
        A dataset of `node` and `level`, holding **only the nodes that could be ordered**.
        A node inside or downstream of a cycle can never lose its last incoming edge, so
        it is absent -- which makes `topological_order(g).count() == g.num_nodes()` the
        acyclicity test, and is exactly what `is_dag` does.

    Raises:
        PlanError: If `max_iterations` is not positive, or is shorter than the longest
            chain of the graph.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, topological_order
            >>> # b and c both depend on a; d depends on both.
            >>> e = bt.from_pydict({"src": ["a", "a", "b", "c"], "dst": ["b", "c", "d", "d"]})
            >>> topological_order(Graph.from_edges(e)).sort("node").to_pydict()
            {'node': ['a', 'b', 'c', 'd'], 'level': [0, 1, 1, 2]}
    """
    remaining = checkpoint(g.nodes())
    rounds = fixpoint_rounds(max_iterations, remaining.count())
    edges = g.edges.cache()
    ordered: Dataset | None = None
    for level in range(rounds + 1):
        if remaining.count() == 0:
            break
        live = edges.join(remaining.select(**{SRC: bt.col(NODE)}), on=SRC, how="semi").join(
            remaining.select(**{DST: bt.col(NODE)}), on=DST, how="semi"
        )
        targets = live.select(**{NODE: bt.col(DST)}).distinct()
        sources = checkpoint(remaining.join(targets, on=NODE, how="anti"))
        if sources.count() == 0:
            # Everything left has an incoming edge, so everything left is in or after a
            # cycle. Stopping here is what makes the result the orderable subset.
            break
        # A level beyond the cap still had nodes to place, so the cap cut the order short.
        require_fixpoint(level < rounds, "topological_order", rounds)
        levelled = sources.select(**{NODE: bt.col(NODE), "level": bt.lit(level)})
        ordered = levelled if ordered is None else checkpoint(ordered.union(levelled))
        remaining = checkpoint(remaining.join(sources, on=NODE, how="anti"))
    if ordered is None:
        return g.nodes().select(**{NODE: bt.col(NODE), "level": bt.lit(0)}).limit(0)
    return ordered


def is_dag(g: Graph, *, max_iterations: int | None = None) -> bool:
    """Whether the graph is acyclic.

    Decided by whether every node can be peeled off in topological order, since a node in
    or downstream of a cycle never loses its last incoming edge.

    Args:
        g: The graph.
        max_iterations: The cap on peeling rounds, or `None` (the default) for none. A
            cap shorter than the longest chain raises `PlanError` rather than reporting a
            deep acyclic graph as cyclic.

    Returns:
        True when the graph has no directed cycle.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, is_dag
            >>> deps = bt.from_pydict({"src": ["a", "b"], "dst": ["b", "c"]})
            >>> cyclic = bt.from_pydict({"src": ["a", "b", "c"], "dst": ["b", "c", "a"]})
            >>> is_dag(Graph.from_edges(deps)), is_dag(Graph.from_edges(cyclic))
            (True, False)
    """
    return topological_order(g, max_iterations=max_iterations).count() == g.num_nodes()


def nodes_in_cycles(g: Graph, *, max_iterations: int | None = None) -> Dataset:
    """The nodes that a topological order cannot place: those in or after a cycle.

    The diagnostic that goes with `is_dag`. A boolean tells you a dependency graph is
    broken; this tells you which packages to look at.

    Args:
        g: The graph.
        max_iterations: The cap on peeling rounds, or `None` (the default) for none.

    Returns:
        A one-column dataset of `node`, empty for an acyclic graph.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, nodes_in_cycles
            >>> e = bt.from_pydict({"src": ["a", "b", "c", "c"], "dst": ["b", "c", "b", "d"]})
            >>> sorted(nodes_in_cycles(Graph.from_edges(e)).to_pydict()["node"])
            ['b', 'c', 'd']
    """
    placed = topological_order(g, max_iterations=max_iterations).select(NODE)
    return g.nodes().join(placed, on=NODE, how="anti")
