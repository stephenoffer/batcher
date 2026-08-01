"""Degree: how many edges touch each node, and how those counts are distributed.

The cheapest thing you can ask a graph, and usually the first. A degree table is one
`group_by` over the edge list, so it costs a single shuffle and nothing else, and it
answers a surprising share of real questions: which accounts are hubs, which items are
never referenced, whether the graph is scale-free or regular.

It is also the input several other algorithms need. PageRank divides by out-degree,
clustering coefficient divides by the number of possible neighbour pairs, and both are
wrong on a graph with parallel edges unless `Graph.simple` ran first.
"""

from __future__ import annotations

import batcher as bt
from batcher.api.dataset import Dataset
from batcher.graph._graph import DST, NODE, SRC, WEIGHT, Graph

__all__ = [
    "average_degree",
    "degree",
    "degree_distribution",
    "in_degree",
    "isolated_nodes",
    "out_degree",
    "weighted_degree",
]


def _side_count(g: Graph, side: str, name: str) -> Dataset:
    """Count edges by one endpoint, then restore the nodes that endpoint never names."""
    counted = g.edges.group_by(side).agg(**{name: bt.count()})
    counted = counted.select(**{NODE: bt.col(side), name: bt.col(name)})
    # A left join from the full node set is what makes a zero appear for a node that has
    # no edge on this side. Without it a sink node is simply absent from an out-degree
    # table, and every downstream average is computed over the wrong denominator.
    return (
        g.nodes()
        .join(counted, on=NODE, how="left")
        .select(**{NODE: bt.col(NODE), name: bt.coalesce(bt.col(name), bt.lit(0))})
    )


def out_degree(g: Graph) -> Dataset:
    """The number of edges leaving each node.

    Args:
        g: The graph.

    Returns:
        A dataset of `node` and `out_degree`, with a zero row for every sink.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, out_degree
            >>> g = Graph.from_edges(bt.from_pydict({"src": [1, 1, 2], "dst": [2, 3, 3]}))
            >>> out_degree(g).sort("node").to_pydict()
            {'node': [1, 2, 3], 'out_degree': [2, 1, 0]}
    """
    return _side_count(g, SRC, "out_degree")


def in_degree(g: Graph) -> Dataset:
    """The number of edges arriving at each node.

    Args:
        g: The graph.

    Returns:
        A dataset of `node` and `in_degree`, with a zero row for every source.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, in_degree
            >>> g = Graph.from_edges(bt.from_pydict({"src": [1, 1, 2], "dst": [2, 3, 3]}))
            >>> in_degree(g).sort("node").to_pydict()
            {'node': [1, 2, 3], 'in_degree': [0, 1, 2]}
    """
    return _side_count(g, DST, "in_degree")


def degree(g: Graph) -> Dataset:
    """The total number of edge endpoints at each node.

    On a directed graph this is in-degree plus out-degree, so a self-loop contributes 2,
    which is the graph-theoretic convention and what makes the handshake identity
    (`sum of degrees == 2 * edge count`) hold.

    Args:
        g: The graph.

    Returns:
        A dataset of `node` and `degree`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, degree
            >>> g = Graph.from_edges(bt.from_pydict({"src": [1, 1, 2], "dst": [2, 3, 3]}))
            >>> degree(g).sort("node").to_pydict()
            {'node': [1, 2, 3], 'degree': [2, 2, 2]}
    """
    both = g.edges.select(**{NODE: bt.col(SRC)}).union(g.edges.select(**{NODE: bt.col(DST)}))
    counted = both.group_by(NODE).agg(degree=bt.count())
    return (
        g.nodes()
        .join(counted, on=NODE, how="left")
        .select(**{NODE: bt.col(NODE), "degree": bt.coalesce(bt.col("degree"), bt.lit(0))})
    )


def weighted_degree(g: Graph) -> Dataset:
    """The total edge weight at each node.

    The weighted analogue of `degree`, and the right measure when edges carry a count, a
    volume or a duration rather than existing or not. On an unweighted graph, where every
    edge weighs 1.0, it equals `degree`.

    Args:
        g: The graph.

    Returns:
        A dataset of `node` and `weighted_degree`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, weighted_degree
            >>> e = bt.from_pydict({"src": [1, 1], "dst": [2, 3], "w": [5.0, 0.5]})
            >>> g = Graph.from_edges(e, weight="w")
            >>> weighted_degree(g).sort("node").to_pydict()
            {'node': [1, 2, 3], 'weighted_degree': [5.5, 5.0, 0.5]}
    """
    both = g.edges.select(**{NODE: bt.col(SRC), "w": bt.col(WEIGHT)}).union(
        g.edges.select(**{NODE: bt.col(DST), "w": bt.col(WEIGHT)})
    )
    summed = both.group_by(NODE).agg(weighted_degree=bt.sum("w"))
    return (
        g.nodes()
        .join(summed, on=NODE, how="left")
        .select(
            **{
                NODE: bt.col(NODE),
                "weighted_degree": bt.coalesce(bt.col("weighted_degree"), bt.lit(0.0)),
            }
        )
    )


def degree_distribution(g: Graph) -> Dataset:
    """How many nodes have each degree.

    The shape of this table is the single most informative summary of a graph. A power-law
    tail (a few nodes with enormous degree, most with one or two) means a scale-free graph
    where hub-aware sampling matters; a narrow band means a regular one where it does not.

    Args:
        g: The graph.

    Returns:
        A dataset of `degree` and `nodes`, sorted by degree.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, degree_distribution
            >>> g = Graph.from_edges(bt.from_pydict({"src": [1, 1, 2], "dst": [2, 3, 3]}))
            >>> degree_distribution(g).to_pydict()
            {'degree': [2], 'nodes': [3]}
    """
    return degree(g).group_by("degree").agg(nodes=bt.count()).sort("degree")


def average_degree(g: Graph) -> float:
    """The mean number of edge endpoints per node.

    Args:
        g: The graph.

    Returns:
        The average degree, or 0.0 for a graph with no nodes.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, average_degree
            >>> g = Graph.from_edges(bt.from_pydict({"src": [1, 1, 2], "dst": [2, 3, 3]}))
            >>> average_degree(g)
            2.0
    """
    got = degree(g).agg(m=bt.mean("degree")).to_pydict()["m"]
    return float(got[0]) if got and got[0] is not None else 0.0


def isolated_nodes(g: Graph) -> Dataset:
    """The nodes with no edges at all.

    Only ever non-empty on a graph built with `Graph.with_nodes`: an edge table cannot
    express a node that has no edges, so without a node table there is nothing to find.

    Args:
        g: The graph.

    Returns:
        A one-column dataset of `node`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, isolated_nodes
            >>> g = Graph.from_edges(bt.from_pydict({"src": [1], "dst": [2]}))
            >>> g = g.with_nodes(bt.from_pydict({"node": [1, 2, 7]}))
            >>> isolated_nodes(g).to_pydict()
            {'node': [7]}
    """
    return degree(g).filter(bt.col("degree") == 0).select(NODE)
