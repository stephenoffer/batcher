"""Graph-level numbers: the ones to look at before running anything else.

A graph you did not build yourself can be almost anything, and the algorithms in this
package behave very differently depending on which. `summarize` runs the cheap
diagnostics in one pass so you know what you have before spending anything: whether it is
one piece or a thousand, whether the degree distribution is flat or has a tail that will
dominate a triangle count, and whether the direction column means anything.
"""

from __future__ import annotations

import batcher as bt
from batcher.api.dataset import Dataset
from batcher.graph._graph import DST, NODE, SRC, Graph
from batcher.graph.degree import degree

__all__ = ["assortativity", "density", "reciprocity", "summarize"]


def density(g: Graph) -> float:
    """The fraction of possible edges that exist.

    1.0 is a complete graph, and a real graph of any size is far closer to 0: a social
    network with a million users and a hundred friends each has a density of 0.0001. That
    sparsity is the reason every algorithm here is a join over an edge list rather than a
    matrix operation.

    Args:
        g: The graph.

    Returns:
        The density in `[0, 1]`, or 0.0 for a graph with fewer than two nodes.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, density
            >>> # A complete directed triangle: every ordered pair exists.
            >>> e = bt.from_pydict({"src": [1, 2, 3, 1, 2, 3], "dst": [2, 3, 1, 3, 1, 2]})
            >>> density(Graph.from_edges(e))
            1.0
    """
    n = g.num_nodes()
    if n < 2:
        return 0.0
    possible = n * (n - 1)
    if not g.directed:
        possible //= 2
    return g.simple().without_self_loops().num_edges() / possible


def reciprocity(g: Graph) -> float:
    """The fraction of directed edges whose reverse also exists.

    Near 1 means the direction column carries no information and the graph is undirected
    in all but name, which is worth knowing before running anything that respects
    direction. Near 0 on a follow graph is the expected shape and tells you PageRank and
    HITS will say different things.

    Args:
        g: The graph.

    Returns:
        The reciprocity in `[0, 1]`, or 0.0 for a graph with no edges.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, reciprocity
            >>> e = bt.from_pydict({"src": [1, 2, 1], "dst": [2, 1, 3]})
            >>> round(reciprocity(Graph.from_edges(e)), 4)
            0.6667
    """
    simple = g.simple().without_self_loops().edges.cache()
    total = simple.count()
    if total == 0:
        return 0.0
    mutual = simple.join(
        simple.select(**{SRC: bt.col(DST), DST: bt.col(SRC)}), on=[SRC, DST], how="semi"
    ).count()
    return mutual / total


def assortativity(g: Graph) -> float:
    """Whether high-degree nodes attach to other high-degree nodes.

    The Pearson correlation between the degrees at the two ends of an edge. Positive means
    hubs connect to hubs, which is the shape of a collaboration network; negative means
    hubs connect to leaves, which is the shape of the internet's router graph and of most
    biological networks.

    Practically, it predicts how a sampling strategy will behave. On a disassortative
    graph a random-neighbour sample from a hub reaches mostly leaves, so a two-hop
    neighbourhood explodes; on an assortative one it stays inside the core.

    Args:
        g: The graph.

    Returns:
        The correlation in `[-1, 1]`, or 0.0 when the graph has no edges or every degree
        is the same (where the correlation is undefined rather than zero).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, assortativity
            >>> # A star: the hub attaches only to leaves, which is maximally disassortative.
            >>> e = bt.from_pydict({"src": [1, 2, 3, 4], "dst": [0, 0, 0, 0]})
            >>> assortativity(Graph.from_edges(e))
            -1.0
    """
    undirected = g.to_undirected().simple().without_self_loops()
    deg = degree(undirected).select(**{NODE: bt.col(NODE), "_k": bt.col("degree")})
    pairs = (
        undirected.edges.join(deg.select(**{SRC: bt.col(NODE), "_ks": bt.col("_k")}), on=SRC)
        .join(deg.select(**{DST: bt.col(NODE), "_kd": bt.col("_k")}), on=DST)
        .select(x=bt.col("_ks").cast("float64"), y=bt.col("_kd").cast("float64"))
    )
    if pairs.count() == 0:
        return 0.0
    got = pairs.agg(r=bt.corr("x", "y")).to_pydict()["r"]
    return float(got[0]) if got and got[0] is not None else 0.0


def summarize(g: Graph) -> Dataset:
    """One row of cheap diagnostics: size, density, degree shape, direction.

    Deliberately excludes anything expensive. Triangles, components and centrality each
    cost a shuffle or several, so they are separate calls you make once you know from this
    whether they are worth it.

    Args:
        g: The graph.

    Returns:
        A one-row dataset of `nodes`, `edges`, `density`, `reciprocity`,
        `average_degree`, `max_degree`, `isolated`, and `directed`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, summarize
            >>> e = bt.from_pydict({"src": [1, 2, 1], "dst": [2, 3, 3]})
            >>> row = summarize(Graph.from_edges(e)).to_pydict()
            >>> row["nodes"], row["edges"], row["max_degree"]
            ([3], [3], [2])
    """
    cached = g.cache()
    degrees = degree(cached).cache()
    stats = degrees.agg(
        _avg=bt.mean("degree"), _max=bt.max("degree"), _iso=bt.count_if(bt.col("degree") == 0)
    ).to_pydict()
    return bt.from_pydict(
        {
            "nodes": [cached.num_nodes()],
            "edges": [cached.num_edges()],
            "density": [density(cached)],
            "reciprocity": [reciprocity(cached)],
            "average_degree": [float(stats["_avg"][0] or 0.0)],
            "max_degree": [int(stats["_max"][0] or 0)],
            "isolated": [int(stats["_iso"][0] or 0)],
            "directed": [cached.directed],
        }
    )
