"""Betweenness: how much of the graph's shortest-path traffic flows through each node.

The centrality that finds *bridges* rather than hubs. A node joining two dense clusters
can have a small degree and a tiny PageRank while every path between the clusters goes
through it, which is exactly the node whose removal splits the graph. On an
infrastructure or supply graph it is the single-point-of-failure measure; on a social
graph it finds the brokers.

Exact betweenness is a shortest-path computation from **every** node, which is quadratic
in node count and does not fit on any graph large enough to need this engine. This is
Brandes' algorithm run from a set of sources you supply, which is the standard estimator:
the *ranking* stabilizes after a few dozen well-chosen sources long before the values do,
so sample and compare ranks rather than magnitudes.
"""

from __future__ import annotations

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.api.dataset import Dataset
from batcher.graph._graph import DST, NODE, SRC, Graph
from batcher.graph._iterate import checkpoint

__all__ = ["betweenness_centrality"]


def _shortest_path_counts(g: Graph, source: object, max_depth: int) -> list[Dataset]:
    """One dataset per BFS level: the nodes at that depth and how many shortest paths
    from `source` reach each.

    `sigma` is the path count, and it is what makes betweenness a *fraction* rather than
    a count: when two shortest paths reach a node, each carries half the credit. A
    version that tracked only distance would give every tie the full weight and rank the
    wrong nodes.
    """
    levels: list[Dataset] = []
    frontier = checkpoint(
        bt.from_pydict({NODE: [source]}).select(**{NODE: bt.col(NODE), "sigma": bt.lit(1.0)})
    )
    seen = checkpoint(frontier.select(NODE))
    levels.append(frontier)
    edges = g.edges.cache()
    for _ in range(max_depth):
        if frontier.count() == 0:
            break
        nxt = (
            edges.join(
                frontier.select(**{SRC: bt.col(NODE), "_s": bt.col("sigma")}),
                on=SRC,
                how="inner",
            )
            .select(**{NODE: bt.col(DST), "_s": bt.col("_s")})
            .group_by(NODE)
            .agg(sigma=bt.sum("_s"))
            # Only nodes not yet reached are at this depth; one already seen was reached
            # by a shorter path, and counting it again would inflate its sigma.
            .join(seen, on=NODE, how="anti")
        )
        nxt = checkpoint(nxt)
        if nxt.count() == 0:
            break
        levels.append(nxt)
        seen = checkpoint(seen.union(nxt.select(NODE)))
        frontier = nxt
    return levels


def betweenness_centrality(
    g: Graph, sources: Dataset, *, node: str = "node", max_depth: int = 10
) -> Dataset:
    """Estimate betweenness from shortest paths out of a set of source nodes.

    Brandes' algorithm: a forward pass counts the shortest paths from each source to
    every node, and a backward pass over the same levels accumulates each node's share of
    the traffic. Running it from every node would be exact and quadratic; running it from
    a sample is the standard estimator.

    Args:
        g: The graph. Direction is respected; symmetrize with `Graph.to_undirected` for
            the undirected measure, which is what most questions mean.
        sources: The nodes to compute shortest paths from. A few dozen high-degree nodes
            (`degree(g).sort(...).limit(n)`) converge the ranking fastest.
        node: The column in `sources` holding the node id.
        max_depth: The hop cap per source. Paths longer than this contribute nothing,
            which bounds the cost and is why a bounded search is affordable at all.

    Returns:
        A dataset of `node` and `betweenness`, summed over the sources and **not**
        normalized: the values scale with the number of sources, so compare ranks across
        runs rather than magnitudes.

    Raises:
        PlanError: If `max_depth` is not positive, or no source is in the graph.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, betweenness_centrality
            >>> # A bridge: everything from the left must pass through 'c'.
            >>> e = bt.from_pydict(
            ...     {"src": ["a", "b", "c", "d"], "dst": ["c", "c", "d", "e"]}
            ... )
            >>> g = Graph.from_edges(e)
            >>> out = betweenness_centrality(g, bt.from_pydict({"node": ["a", "b"]}))
            >>> out.sort("betweenness", descending=True).to_pydict()["node"][0]
            'c'
    """
    if max_depth < 1:
        raise PlanError(f"max_depth must be positive, got {max_depth}")
    seeds = (
        sources.select(**{NODE: bt.col(node)})
        .distinct()
        .join(g.nodes(), on=NODE, how="semi")
        .to_pydict()[NODE]
    )
    if not seeds:
        raise PlanError(
            "betweenness_centrality(): none of the source nodes appear in the graph, so "
            "there are no shortest paths to trace"
        )
    edges = g.edges.cache()
    totals: Dataset | None = None

    for source in seeds:
        levels = _shortest_path_counts(g, source, max_depth)
        if len(levels) < 2:
            continue
        # Backward pass. `delta[v]` is the share of shortest-path traffic through `v`;
        # a node at the deepest level has none, and each level hands its share back to
        # the level above in proportion to the paths it received from there.
        delta = checkpoint(levels[-1].select(**{NODE: bt.col(NODE), "delta": bt.lit(0.0)}))
        deeper = levels[-1]
        for depth in range(len(levels) - 2, -1, -1):
            above = levels[depth]
            contribution = (
                edges.join(
                    above.select(**{SRC: bt.col(NODE), "_sv": bt.col("sigma")}),
                    on=SRC,
                    how="inner",
                )
                .join(
                    deeper.select(**{DST: bt.col(NODE), "_sw": bt.col("sigma")}),
                    on=DST,
                    how="inner",
                )
                .join(
                    delta.select(**{DST: bt.col(NODE), "_dw": bt.col("delta")}),
                    on=DST,
                    how="inner",
                )
                .select(
                    **{
                        NODE: bt.col(SRC),
                        "_c": (bt.col("_sv") / bt.col("_sw")) * (bt.lit(1.0) + bt.col("_dw")),
                    }
                )
                .group_by(NODE)
                .agg(_c=bt.sum("_c"))
            )
            level_delta = checkpoint(
                above.select(NODE)
                .join(contribution, on=NODE, how="left")
                .select(**{NODE: bt.col(NODE), "delta": bt.coalesce(bt.col("_c"), bt.lit(0.0))})
            )
            # The source itself accumulates nothing: it is an endpoint of every path it
            # starts, and betweenness counts only the nodes a path passes *through*.
            if depth > 0:
                totals = level_delta if totals is None else checkpoint(totals.union(level_delta))
            delta = level_delta
            deeper = above

    if totals is None:
        return g.nodes().select(**{NODE: bt.col(NODE), "betweenness": bt.lit(0.0)})
    summed = totals.group_by(NODE).agg(betweenness=bt.sum("delta"))
    return (
        g.nodes()
        .join(summed, on=NODE, how="left")
        .select(
            **{
                NODE: bt.col(NODE),
                "betweenness": bt.coalesce(bt.col("betweenness"), bt.lit(0.0)),
            }
        )
    )
