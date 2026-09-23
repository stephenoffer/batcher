"""Communities, triangles, and how tightly a neighbourhood closes on itself.

Triangles are the load-bearing measurement here, and the reason is that they are the
smallest structure that distinguishes a real social graph from a random one with the same
degrees. If your friends are friends with each other, the graph has triangles; a random
graph with identical degrees has almost none. Clustering coefficient is that observation
per node, and `modularity` scores a whole partition by the same instinct: a good community
has more edges inside it than chance would put there.

Triangle counting is the expensive operation in this package. It is a three-way join, so
its cost is driven by the highest-degree node rather than by the average, and on a
scale-free graph one celebrity account can dominate the whole run. `k_core` first, or a
degree cap, is the standard mitigation.
"""

from __future__ import annotations

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.api.dataset import Dataset
from batcher.graph._graph import DST, NODE, SRC, WEIGHT, Graph, ends, undirected_pairs
from batcher.graph._iterate import checkpoint, count_changed, iterate, settled

__all__ = [
    "average_clustering",
    "clustering_coefficient",
    "label_propagation",
    "modularity",
    "transitivity",
    "triangle_count",
    "triangles",
]


def _canonical_edges(g: Graph) -> Dataset:
    """Each undirected edge once, oriented from the smaller endpoint to the larger.

    Orienting by node id is what stops a triangle from being counted six times (once per
    ordering of its three vertices): with a consistent orientation only one of the six
    survives, so the join returns each triangle exactly once. `undirected_pairs` documents
    why the orientation is a swap rather than a filter. Self-loops are dropped here, since a
    self-loop closes no triangle and is not a neighbour pair.
    """
    return (
        undirected_pairs(g)
        .filter(bt.col("_lo") != bt.col("_hi"))
        .select(**{SRC: bt.col("_lo"), DST: bt.col("_hi")})
    )


def _per_node(g: Graph, credits: Dataset, columns: dict[str, int]) -> Dataset:
    """Sum `credits` per node, with a zero row for every node the credits never name.

    The zero rows come from more union arms, one per edge endpoint and one per declared
    node, rather than from an outer join against `nodes()`: that join is a `distinct` over
    a union feeding a join, which has no distributed path. Every per-node result in this
    module goes through here, so an isolated node and a node with only a self-loop get a
    row (and a clustering of 0.0, as networkx gives them) instead of disappearing.
    """
    zeros = {name: bt.lit(zero) for name, zero in columns.items()}
    arms = ends(undirected_pairs(g)).select(NODE, **zeros)
    extra = g.extra_nodes()
    if extra is not None:
        arms = arms.union(extra.select(NODE, **zeros))
    return credits.union(arms).group_by(NODE).agg(**{name: bt.sum(name) for name in columns})


def _triangle_credits(g: Graph, **columns: int) -> Dataset:
    """One row per (triangle, member): `node` plus `columns`, for summing per node."""
    tri = triangles(g).cache()
    return (
        tri.select(**{NODE: bt.col("a")}, **{k: bt.lit(v) for k, v in columns.items()})
        .union(tri.select(**{NODE: bt.col("b")}, **{k: bt.lit(v) for k, v in columns.items()}))
        .union(tri.select(**{NODE: bt.col("c")}, **{k: bt.lit(v) for k, v in columns.items()}))
    )


def triangles(g: Graph) -> Dataset:
    """Every triangle in the graph, once, as an ordered triple.

    Args:
        g: The graph. Direction is ignored and parallel edges are collapsed.

    Returns:
        A dataset of `a`, `b` and `c` with `a < b < c`, one row per triangle.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, triangles
            >>> e = bt.from_pydict({"src": [1, 2, 1, 3], "dst": [2, 3, 3, 4]})
            >>> triangles(Graph.from_edges(e)).to_pydict()
            {'a': [1], 'b': [2], 'c': [3]}
    """
    edges = _canonical_edges(g).cache()
    # a<b, b<c, and the closing a<c edge: three joins, one row per triangle.
    ab = edges.select(a=bt.col(SRC), b=bt.col(DST))
    bc = edges.select(b=bt.col(SRC), c=bt.col(DST))
    ac = edges.select(a=bt.col(SRC), c=bt.col(DST))
    closed = ab.join(bc, on="b", how="inner").join(ac, on=["a", "c"], how="inner")
    return closed.select("a", "b", "c")


def triangle_count(g: Graph) -> Dataset:
    """How many triangles each node takes part in.

    Args:
        g: The graph.

    Returns:
        A dataset of `node` and `triangles`, with a zero row for every node in none.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, triangle_count
            >>> e = bt.from_pydict({"src": [1, 2, 1, 3], "dst": [2, 3, 3, 4]})
            >>> triangle_count(Graph.from_edges(e)).sort("node").to_pydict()
            {'node': [1, 2, 3, 4], 'triangles': [1, 1, 1, 0]}
    """
    return _per_node(g, _triangle_credits(g, triangles=1), {"triangles": 0})


def clustering_coefficient(g: Graph) -> Dataset:
    """For each node, the fraction of its neighbour pairs that are themselves connected.

    1.0 means every pair of a node's neighbours knows each other, 0.0 means none do. A
    node with fewer than two neighbours has no pairs and scores 0.0 rather than null,
    which is networkx's convention and the one that makes `average_clustering` well
    defined. That includes an isolated node from `Graph.with_nodes` and a node whose only
    edge is a self-loop, which is not a neighbour.

    Args:
        g: The graph.

    Returns:
        A dataset of `node` and `clustering`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, clustering_coefficient
            >>> e = bt.from_pydict({"src": [1, 2, 1, 3], "dst": [2, 3, 3, 4]})
            >>> out = clustering_coefficient(Graph.from_edges(e)).sort("node")
            >>> [round(v, 4) for v in out.to_pydict()["clustering"]]
            [1.0, 1.0, 0.3333, 0.0]
    """
    # The neighbour count and the triangle count come from one aggregate over the
    # canonical undirected edge set, so the numerator and denominator can never describe
    # two different graphs, and a node with no neighbour pair still gets its 0.0 row.
    neighbours = ends(_canonical_edges(g).select(_lo=bt.col(SRC), _hi=bt.col(DST))).select(
        NODE, _k=bt.lit(1), _t=bt.lit(0)
    )
    credits = neighbours.union(_triangle_credits(g, _k=0, _t=1))
    k = bt.col("_k").cast("float64")
    return _per_node(g, credits, {"_k": 0, "_t": 0}).select(
        **{
            NODE: bt.col(NODE),
            "clustering": bt.when(bt.col("_k") >= 2)
            .then(bt.lit(2.0) * bt.col("_t").cast("float64") / (k * (k - 1.0)))
            .otherwise(bt.lit(0.0)),
        }
    )


def average_clustering(g: Graph) -> float:
    """The mean clustering coefficient over all nodes.

    Every node counts, including those with fewer than two neighbours, which score 0.0.
    That matches networkx's `average_clustering`: a triangle plus an isolated node
    averages 0.75, not 1.0.

    Args:
        g: The graph.

    Returns:
        The average, or 0.0 for a graph with no nodes.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, average_clustering
            >>> e = bt.from_pydict({"src": [1, 2, 1], "dst": [2, 3, 3]})
            >>> average_clustering(Graph.from_edges(e))
            1.0
    """
    # Materialized first: the coefficient is an aggregate over a union, which a cluster
    # cannot feed into a second aggregate. It is one row per node.
    per_node = checkpoint(clustering_coefficient(g))
    got = per_node.agg(m=bt.mean("clustering")).to_pydict()["m"]
    return float(got[0]) if got and got[0] is not None else 0.0


def transitivity(g: Graph) -> float:
    """The global clustering coefficient: closed triples over all connected triples.

    Different from `average_clustering`, and the difference matters on a scale-free graph.
    Averaging per-node coefficients weights every node equally, so a horde of low-degree
    nodes with perfect local clustering dominates; transitivity weights by the number of
    triples, so the hubs dominate. Report both, or say which you mean.

    Direction is ignored, parallel edges count once, and a self-loop is not a neighbour,
    so it neither closes a triangle nor forms a triple.

    Args:
        g: The graph.

    Returns:
        The ratio in `[0, 1]`, or 0.0 when the graph has no connected triples.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, transitivity
            >>> e = bt.from_pydict({"src": [1, 2, 1], "dst": [2, 3, 3]})
            >>> transitivity(Graph.from_edges(e))
            1.0
    """
    tri = triangles(g).count()
    k = bt.col("_k").cast("float64")
    got = (
        ends(_canonical_edges(g).select(_lo=bt.col(SRC), _hi=bt.col(DST)))
        .group_by(NODE)
        .agg(_k=bt.count())
        .agg(_triples=bt.sum(k * (k - 1.0) / 2.0))
        .to_pydict()["_triples"]
    )
    triples = float(got[0]) if got and got[0] is not None else 0.0
    return (3.0 * tri / triples) if triples else 0.0


def label_propagation(g: Graph, *, max_iterations: int = 20) -> Dataset:
    """Detect communities by repeatedly adopting the most common neighbouring label.

    Near-linear in the edge count and needs no parameter beyond a round cap, which makes
    it the community algorithm to try first. What it gives up is determinism and
    stability: ties are broken by the smallest label, so a run is reproducible, but a
    small change to the graph can reshape the partition. Score the result with
    `modularity` rather than trusting the label count.

    Args:
        g: The graph.
        max_iterations: The cap on rounds. Label propagation usually settles in under ten
            and does not always settle at all, which is why the cap is low by default.
            Hitting it returns the last labelling and warns with `ConvergenceWarning`.

    Returns:
        A dataset of `node` and `community`.

    Raises:
        PlanError: If `max_iterations` is not positive.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, label_propagation
            >>> # Two triangles joined by nothing.
            >>> e = bt.from_pydict(
            ...     {"src": [1, 2, 1, 4, 5, 4], "dst": [2, 3, 3, 5, 6, 6]}
            ... )
            >>> out = label_propagation(Graph.from_edges(e)).sort("node")
            >>> out.to_pydict()["community"]
            [1, 1, 1, 4, 4, 4]
    """
    if max_iterations < 1:
        raise PlanError(f"max_iterations must be at least 1, got {max_iterations}")
    undirected = g.to_undirected().cache()
    nodes = undirected.nodes().cache()
    initial = nodes.select(**{NODE: bt.col(NODE), "community": bt.col(NODE)})

    def step(labels: Dataset) -> Dataset:
        # Weight of each label arriving at each node, then the heaviest label per node
        # with the smallest id breaking ties.
        votes = (
            undirected.edges.join(
                labels.select(**{SRC: bt.col(NODE), "_l": bt.col("community")}),
                on=SRC,
                how="inner",
            )
            .group_by(DST, "_l")
            .agg(_w=bt.sum(WEIGHT))
        )
        # The heaviest label per node, with the smallest label breaking a tie. Written as
        # max-then-min rather than a sort-and-take-first because a tie broken by physical
        # row order would make the result depend on partitioning, and the same graph would
        # then produce different communities single-node and distributed.
        heaviest = votes.group_by(DST).agg(_w=bt.max("_w"))
        best = votes.join(heaviest, on=[DST, "_w"], how="inner").group_by(DST).agg(_l=bt.min("_l"))
        return labels.join(
            best.select(**{NODE: bt.col(DST), "_l": bt.col("_l")}), on=NODE, how="left"
        ).select(
            **{
                NODE: bt.col(NODE),
                "community": bt.coalesce(bt.col("_l"), bt.col("community")),
            }
        )

    result = iterate(
        initial,
        step,
        max_iterations=max_iterations,
        delta=count_changed(NODE, "community"),
        tolerance=0.0,
    )
    return settled(result, "label_propagation", max_iterations)


def modularity(g: Graph, communities: Dataset, *, community: str = "community") -> float:
    """Score a partition: how much more edge weight sits inside communities than chance.

    Ranges from -0.5 to 1. Above about 0.3 usually means real community structure; near
    zero means the partition explains nothing that the degree sequence does not already.

    This is the number to compare partitions with, and the reason `label_propagation` does
    not report a quality of its own: the algorithm cannot tell you whether the communities
    it found are meaningful, and this can.

    Args:
        g: The graph.
        communities: A dataset of `node` and a community label.
        community: The column holding the label.

    Returns:
        The modularity of the partition, or 0.0 for a graph with no edges.

    Raises:
        PlanError: If the community table has no `node` column or no label column.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, label_propagation, modularity
            >>> e = bt.from_pydict(
            ...     {"src": [1, 2, 1, 4, 5, 4], "dst": [2, 3, 3, 5, 6, 6]}
            ... )
            >>> g = Graph.from_edges(e)
            >>> round(modularity(g, label_propagation(g)), 4)
            0.5
    """
    for col_name in (NODE, community):
        if col_name not in communities.columns:
            raise PlanError(
                f"modularity(): community table has no {col_name!r} column; "
                f"available: {sorted(communities.columns)}"
            )
    # Every sum below is the one the symmetrized edge table would give, taken from the table
    # as stored: a non-loop edge counts once from each end, a self-loop once. Built without
    # `to_undirected`, whose `union` under the aggregates would have no distributed path.
    edges = g.edges
    loop = bt.col(SRC) == bt.col(DST)
    both_ways = (
        bt.col(WEIGHT)
        if g.symmetrized
        else bt.when(loop).then(bt.col(WEIGHT)).otherwise(bt.lit(2.0) * bt.col(WEIGHT))
    )
    # 2m: the total weight of the symmetrized edge table, which counts each undirected
    # edge twice by construction.
    total = edges.agg(w=bt.sum(both_ways)).to_pydict()["w"]
    two_m = float(total[0]) if total and total[0] else 0.0
    if two_m <= 0.0:
        return 0.0
    labels = communities.select(**{NODE: bt.col(NODE), "_c": bt.col(community)}).cache()

    inside = (
        edges.join(labels.select(**{SRC: bt.col(NODE), "_cs": bt.col("_c")}), on=SRC, how="inner")
        .join(labels.select(**{DST: bt.col(NODE), "_cd": bt.col("_c")}), on=DST, how="inner")
        .filter(bt.col("_cs") == bt.col("_cd"))
        .group_by("_cs")
        .agg(_in=bt.sum(both_ways))
    )
    # Each community's total strength: every edge endpoint's weight, a self-loop once. Not
    # `weighted_degree`, which counts a self-loop at both ends -- and modularity squares
    # this term, so an error here is not a scale factor that cancels: summing both ends of
    # an already-symmetrized table drove a perfect two-community split to -1.0 instead of
    # 0.5.
    endpoints = (
        edges.select(SRC, DST, WEIGHT, **{NODE: bt.array(bt.col(SRC), bt.col(DST))})
        .explode(NODE, index="_end")
        .filter(
            bt.col("_end") == bt.lit(0) if g.symmetrized else (bt.col("_end") == bt.lit(0)) | ~loop
        )
    )
    totals = endpoints.join(labels, on=NODE, how="inner").group_by("_c").agg(_tot=bt.sum(WEIGHT))
    per_community = (
        totals.join(inside.select(_c=bt.col("_cs"), _in=bt.col("_in")), on="_c", how="left")
        .select(
            q=bt.coalesce(bt.col("_in"), bt.lit(0.0)) / bt.lit(two_m)
            - (bt.col("_tot") / bt.lit(two_m)) * (bt.col("_tot") / bt.lit(two_m))
        )
        .agg(total=bt.sum("q"))
        .to_pydict()["total"]
    )
    return float(per_community[0]) if per_community and per_community[0] is not None else 0.0
