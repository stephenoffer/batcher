"""Building a graph out of data that is not already an edge list.

Most data that wants graph analysis does not arrive as edges. It arrives as embeddings,
as coordinates, or as a user-item interaction log, and the graph is something you decide
to construct from it. These four constructors are those decisions, made explicitly:

| Function | Edges connect | Typical use |
| --- | --- | --- |
| `knn_graph` | each vector to its `k` nearest | clustering and labelling over an embedding space |
| `threshold_graph` | vectors closer than a cut-off | deduplication, entity resolution |
| `spatial_graph` | positions within a radius on the Earth | catchment areas, spread modelling |
| `co_occurrence_graph` | items appearing together | recommendation, basket analysis |

All four are quadratic in the worst case, because "which pairs are close" is a question
about pairs. Each one says in its own documentation what bounds it and what to reach for
when the input is too large for the naive form. That is not a caveat bolted on: choosing
the blocking key *is* the engineering in every one of these.
"""

from __future__ import annotations

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.api.dataset import Dataset
from batcher.graph._graph import Graph

__all__ = [
    "co_occurrence_graph",
    "knn_graph",
    "spatial_graph",
    "threshold_graph",
]

#: The vector metrics these constructors understand, mapped to the accessor method that
#: computes them and whether a *higher* value means *closer*.
_METRICS = {
    "cosine": ("cosine_similarity", True),
    "dot": ("dot", True),
    "euclidean": ("l2_distance", False),
    "manhattan": ("l1_distance", False),
}


def _with_all_nodes(built: Graph, source: Dataset, node: str) -> Graph:
    """Declare every input row as a node, whether or not it earned an edge.

    A constructor that returned only the connected nodes would silently drop the rows
    that matched nothing -- and for entity resolution those are exactly the rows you need
    back, as singleton clusters. `connected_components` over the result then labels every
    input, not just the ones that paired up.
    """
    return built.with_nodes(source.select(node=bt.col(node)).distinct())


def _pairwise(vectors: Dataset, node: str, vector: str, metric: str, block: str | None) -> Dataset:
    """Every candidate pair with its score, blocked when a blocking key is given."""
    if metric not in _METRICS:
        raise PlanError(f"metric must be one of {sorted(_METRICS)}, got {metric!r}")
    for name in (node, vector):
        if name not in vectors.columns:
            raise PlanError(
                f"column {name!r} is not in the vector table; available: {sorted(vectors.columns)}"
            )
    method, _higher_is_closer = _METRICS[metric]
    left_cols = {"_a": bt.col(node), "_va": bt.col(vector)}
    right_cols = {"_b": bt.col(node), "_vb": bt.col(vector)}
    if block is not None:
        if block not in vectors.columns:
            raise PlanError(
                f"blocking column {block!r} is not in the vector table; available: "
                f"{sorted(vectors.columns)}"
            )
        left_cols["_blk"] = bt.col(block)
        right_cols["_blk"] = bt.col(block)
    left = vectors.select(**left_cols)
    right = vectors.select(**right_cols)
    joined = (
        left.join(right, on="_blk", how="inner") if block is not None else left.cross_join(right)
    )
    return joined.filter(bt.col("_a") != bt.col("_b")).select(
        _a=bt.col("_a"),
        _b=bt.col("_b"),
        _score=getattr(bt.col("_va").list, method)(bt.col("_vb")),
    )


def knn_graph(
    vectors: Dataset,
    k: int,
    *,
    node: str = "node",
    vector: str = "vector",
    metric: str = "cosine",
    block: str | None = None,
) -> Graph:
    """Connect each vector to its `k` nearest neighbours.

    The bridge from an embedding space to graph analysis: once vectors are a graph, the
    community, centrality and label-propagation algorithms all apply. Semi-supervised
    labelling in particular is just `label_propagation` over this graph.

    **This compares every pair unless you block it.** A million rows is a trillion
    comparisons. `block` restricts the comparison to rows sharing a key, which is the
    standard fix and is exact within each block: partition by a coarse cluster id, a
    date, a category, or a geohash prefix, and pairs across blocks are simply not
    considered.

    Args:
        vectors: A dataset of node ids and vector columns.
        k: How many neighbours to keep per node.
        node: The column holding the node id.
        vector: The column holding the embedding, as a list of floats.
        metric: `"cosine"`, `"dot"`, `"euclidean"` or `"manhattan"`.
        block: A column restricting comparison to rows that share its value.

    Returns:
        A directed graph whose edge weight is the similarity or distance. Directed
        because "nearest" is not symmetric: B can be A's nearest neighbour without A
        being B's. Call `Graph.to_undirected` when the analysis needs it to be.

    Raises:
        PlanError: If `k` is not positive, the metric is unknown, or a column is absent.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import knn_graph
            >>> vecs = bt.from_pydict(
            ...     {
            ...         "node": ["a", "b", "c"],
            ...         "vector": [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
            ...     }
            ... )
            >>> g = knn_graph(vecs, 1)
            >>> g.edges.sort("src").to_pydict()["dst"]
            ['b', 'a', 'b']
    """
    if k < 1:
        raise PlanError(f"k must be positive, got {k}")
    _, higher_is_closer = _METRICS.get(metric, (None, True))
    scored = _pairwise(vectors, node, vector, metric, block)
    # A window's ordering is ascending, so "nearest" is a *negated* score for a
    # similarity metric and the plain score for a distance one. Materializing that as its
    # own column keeps the rank ascending in both cases rather than needing two spellings.
    ordered = scored.with_columns(
        _order=-bt.col("_score") if higher_is_closer else bt.col("_score")
    )
    ranked = ordered.with_columns(
        _rank=bt.col("_order").rank().over(partition_by="_a", order_by="_order")
    )
    edges = ranked.filter(bt.col("_rank") <= bt.lit(k)).select(
        src=bt.col("_a"), dst=bt.col("_b"), weight=bt.col("_score")
    )
    return _with_all_nodes(Graph.from_edges(edges, weight="weight"), vectors, node)


def threshold_graph(
    vectors: Dataset,
    threshold: float,
    *,
    node: str = "node",
    vector: str = "vector",
    metric: str = "cosine",
    block: str | None = None,
) -> Graph:
    """Connect every pair of vectors closer than a cut-off.

    The deduplication and entity-resolution shape: build this graph over your records,
    take `connected_components`, and each component is a cluster of things that are the
    same thing. That transitive closure is the reason to use a graph rather than a
    pairwise threshold -- A matching B and B matching C puts all three together even when
    A and C do not match directly.

    The result is undirected, because "within a distance of" is symmetric in a way
    "nearest" is not.

    Args:
        vectors: A dataset of node ids and vector columns.
        threshold: The cut-off. For a similarity metric a pair is kept when it scores at
            or above this; for a distance metric, at or below.
        node: The column holding the node id.
        vector: The column holding the embedding.
        metric: `"cosine"`, `"dot"`, `"euclidean"` or `"manhattan"`.
        block: A column restricting comparison to rows that share its value.

    Returns:
        An undirected graph whose edge weight is the score.

    Raises:
        PlanError: If the metric is unknown or a column is absent.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import connected_components, threshold_graph
            >>> vecs = bt.from_pydict(
            ...     {
            ...         "node": ["a", "b", "c"],
            ...         "vector": [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]],
            ...     }
            ... )
            >>> g = threshold_graph(vecs, 0.9)
            >>> connected_components(g).sort("node").to_pydict()["component"]
            ['a', 'a', 'c']
            >>> # 'c' matched nothing, so it is its own cluster rather than absent.
    """
    _, higher_is_closer = _METRICS.get(metric, (None, True))
    scored = _pairwise(vectors, node, vector, metric, block)
    keep = (
        bt.col("_score") >= bt.lit(threshold)
        if higher_is_closer
        else bt.col("_score") <= bt.lit(threshold)
    )
    # `_a < _b` keeps one row per unordered pair; `to_undirected` then materializes both
    # directions once, rather than the cross join's two arriving as parallel edges.
    edges = (
        scored.filter(keep)
        .filter(bt.col("_a") < bt.col("_b"))
        .select(src=bt.col("_a"), dst=bt.col("_b"), weight=bt.col("_score"))
    )
    built = Graph.from_edges(edges, weight="weight").to_undirected()
    return _with_all_nodes(built, vectors, node)


def spatial_graph(
    points: Dataset,
    radius_metres: float,
    *,
    node: str = "node",
    geometry: str = "geometry",
    block: str | None = None,
) -> Graph:
    """Connect every pair of positions within a radius of each other on the Earth.

    Proximity as a graph: contact tracing, catchment areas, transmission modelling, "which
    of these sites are close enough to share a depot". Once it is a graph, the community
    and component algorithms give you the clusters directly.

    Distance is geodesic and in metres, so the radius means what it says regardless of
    latitude. The pair generator rejects on bounding boxes first, which is exact in the
    negative direction, but it is still a cross join: pass `block` with a coarse geohash
    prefix to bound it, which is the spatial analogue of blocking an embedding join.

    Args:
        points: A dataset of node ids and geometries.
        radius_metres: The maximum distance between connected positions.
        node: The column holding the node id.
        geometry: The column holding the geometry, in WKB or any text encoding.
        block: A column restricting comparison to rows that share its value. A geohash
            prefix from `bt.geohash_encode` is the usual choice.

    Returns:
        An undirected graph whose edge weight is the distance in metres.

    Raises:
        PlanError: If the radius is negative or a column is absent.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import spatial_graph
            >>> sites = bt.from_pydict(
            ...     {
            ...         "node": ["ferry", "pier", "opera"],
            ...         "geometry": [
            ...             "POINT(-122.3937 37.7955)",
            ...             "POINT(-122.3930 37.7960)",
            ...             "POINT(151.2153 -33.8568)",
            ...         ],
            ...     }
            ... )
            >>> g = spatial_graph(sites, 200.0)
            >>> g.edges.sort("src").to_pydict()["src"]
            ['ferry', 'pier']
    """
    if radius_metres < 0.0:
        raise PlanError(f"radius_metres must be non-negative, got {radius_metres}")
    for name in (node, geometry):
        if name not in points.columns:
            raise PlanError(
                f"column {name!r} is not in the point table; available: {sorted(points.columns)}"
            )
    left_cols = {"_a": bt.col(node), "_ga": bt.col(geometry)}
    right_cols = {"_b": bt.col(node), "_gb": bt.col(geometry)}
    if block is not None:
        if block not in points.columns:
            raise PlanError(
                f"blocking column {block!r} is not in the point table; available: "
                f"{sorted(points.columns)}"
            )
        left_cols["_blk"] = bt.col(block)
        right_cols["_blk"] = bt.col(block)
    left = points.select(**left_cols)
    right = points.select(**right_cols)
    joined = (
        left.join(right, on="_blk", how="inner") if block is not None else left.cross_join(right)
    )
    edges = (
        joined.filter(bt.col("_a") < bt.col("_b"))
        .filter(bt.st_dwithin_sphere(bt.col("_ga"), bt.col("_gb"), radius_metres))
        .select(
            src=bt.col("_a"),
            dst=bt.col("_b"),
            weight=bt.st_distance_sphere(bt.col("_ga"), bt.col("_gb")),
        )
    )
    built = Graph.from_edges(edges, weight="weight").to_undirected()
    return _with_all_nodes(built, points, node)


def co_occurrence_graph(
    interactions: Dataset,
    *,
    group: str = "user",
    item: str = "item",
    min_count: int = 1,
) -> Graph:
    """Connect items that appear together in the same group.

    The projection that turns a user-item log into an item-item graph, which is the
    classic collaborative-filtering signal: two items are related when the same people
    touched both. `pagerank` over it ranks items by how central they are to the catalogue,
    and `label_propagation` finds the categories the behaviour implies rather than the
    ones someone declared.

    **Cost is the sum of squared group sizes.** One group containing ten thousand items
    contributes fifty million pairs on its own, so cap or split the outsized groups
    first. `min_count` prunes the result but not the work.

    Args:
        interactions: A dataset with one row per (group, item) interaction.
        group: The column holding the grouping id, usually a user or a session.
        item: The column holding the item id.
        min_count: Drop pairs that co-occur fewer than this many times. Raising it is the
            cheapest way to cut the noise from incidental co-occurrence.

    Returns:
        An undirected graph whose edge weight is the co-occurrence count.

    Raises:
        PlanError: If `min_count` is not positive or a column is absent.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import co_occurrence_graph
            >>> log = bt.from_pydict(
            ...     {
            ...         "user": ["u1", "u1", "u2", "u2", "u3"],
            ...         "item": ["bread", "jam", "bread", "jam", "shovel"],
            ...     }
            ... )
            >>> g = co_occurrence_graph(log, min_count=2)
            >>> g.edges.sort("src").to_pydict()
            {'src': ['bread', 'jam'], 'dst': ['jam', 'bread'], 'weight': [2.0, 2.0]}
    """
    if min_count < 1:
        raise PlanError(f"min_count must be positive, got {min_count}")
    for name in (group, item):
        if name not in interactions.columns:
            raise PlanError(
                f"column {name!r} is not in the interaction table; available: "
                f"{sorted(interactions.columns)}"
            )
    pairs = interactions.select(_g=bt.col(group), _i=bt.col(item)).distinct()
    left = pairs.select(_g=bt.col("_g"), _a=bt.col("_i"))
    right = pairs.select(_g=bt.col("_g"), _b=bt.col("_i"))
    counted = (
        left.join(right, on="_g", how="inner")
        .filter(bt.col("_a") < bt.col("_b"))
        .group_by("_a", "_b")
        .agg(_n=bt.count())
    )
    edges = counted.filter(bt.col("_n") >= bt.lit(min_count)).select(
        src=bt.col("_a"), dst=bt.col("_b"), weight=bt.col("_n").cast("float64")
    )
    built = Graph.from_edges(edges, weight="weight").to_undirected()
    return _with_all_nodes(built, interactions, item)
