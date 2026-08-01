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


#: The per-endpoint contribution being summed. Private, and never in a returned schema:
#: every function here aggregates it away in the same expression that introduces it.
_C = "_contribution"


def _totalled(g: Graph, contributions: Dataset, name: str, zero: object) -> Dataset:
    """Sum each node's contributions, keeping the nodes that contribute nothing.

    The zero rows are what make this correct rather than merely fast. A node with no edge
    on the counted side contributes no row at all, so without them a sink is simply absent
    from an out-degree table and every downstream average divides by the wrong denominator.

    Emitting them as another `union` arm rather than as an outer join from `nodes()` is
    deliberate. The two compute the same table, but the join form is
    `union -> distinct  LEFT JOIN  union -> group_by`, which has no distributed path and
    raises under `distributed=True`. This form is one `union` into one `group_by`: a single
    shuffle, no join build, no distinct, and it runs on a cluster.
    """
    extra = g.extra_nodes()
    if extra is not None:
        contributions = contributions.union(extra.select(**{NODE: bt.col(NODE), _C: bt.lit(zero)}))
    return contributions.group_by(NODE).agg(**{name: bt.sum(_C)})


def _side_count(g: Graph, side: str, name: str) -> Dataset:
    """Count edges by one endpoint, giving the nodes it never names a zero."""
    other = DST if side == SRC else SRC
    # Every endpoint on the *other* side contributes zero, which is what puts a node with
    # no edge on this side into the output without needing the distinct node set.
    counted = g.edges.select(**{NODE: bt.col(side), _C: bt.lit(1)}).union(
        g.edges.select(**{NODE: bt.col(other), _C: bt.lit(0)})
    )
    return _totalled(g, counted, name, 0)


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
    both = g.edges.select(**{NODE: bt.col(SRC), _C: bt.lit(1)}).union(
        g.edges.select(**{NODE: bt.col(DST), _C: bt.lit(1)})
    )
    return _totalled(g, both, "degree", 0)


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
    both = g.edges.select(**{NODE: bt.col(SRC), _C: bt.col(WEIGHT)}).union(
        g.edges.select(**{NODE: bt.col(DST), _C: bt.col(WEIGHT)})
    )
    return _totalled(g, both, "weighted_degree", 0.0)


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
