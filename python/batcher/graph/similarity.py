"""How alike two nodes are, judged by who they are connected to.

These are the link-prediction scores: given a graph, which pairs that are *not* connected
look like they should be. They are also the recommendation primitives, since "users who
share your neighbours" and "items co-purchased with yours" are the same computation.

All five are computed for *candidate pairs you supply*, never for all pairs. Scoring every
pair is quadratic in node count and does not fit; the standard candidate generator is
"pairs sharing at least one neighbour", which `candidate_pairs` produces and which is
where every real pipeline starts.

They differ in how they weight a shared neighbour:

| Score | A shared neighbour is worth |
| --- | --- |
| `common_neighbors` | 1, always |
| `jaccard_similarity` | 1, normalized by how many neighbours the pair has between them |
| `adamic_adar` | more when that neighbour is rare, because a hub connects everyone |
| `resource_allocation` | the same idea, weighted harder against hubs |
| `preferential_attachment` | nothing; it scores by degree alone |
"""

from __future__ import annotations

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.api.dataset import Dataset
from batcher.graph._graph import DST, NODE, SRC, Graph
from batcher.graph.degree import out_degree

__all__ = [
    "adamic_adar",
    "candidate_pairs",
    "common_neighbors",
    "jaccard_similarity",
    "preferential_attachment",
    "resource_allocation",
]


def _adjacency(g: Graph) -> Dataset:
    """The symmetrized, deduplicated `(node, neighbour)` table every score joins on."""
    return g.to_undirected().simple().edges.select(**{NODE: bt.col(SRC), "nbr": bt.col(DST)})


def candidate_pairs(g: Graph, *, max_degree: int | None = None) -> Dataset:
    """Every unordered pair of nodes sharing at least one neighbour, with the count.

    The candidate generator the scores below are meant to be fed. It is a self-join of the
    adjacency table, so its cost is the sum of squared degrees: one node with a million
    neighbours produces a trillion pairs on its own.

    `max_degree` is the standard defence. Dropping the highest-degree nodes from the
    *generator* loses few real candidates, because a neighbour shared with a hub is weak
    evidence anyway, which is exactly what `adamic_adar` formalizes.

    Args:
        g: The graph.
        max_degree: Exclude nodes with more neighbours than this from the pairing. `None`
            keeps everything, which is only safe on a graph you know is not scale-free.

    Returns:
        A dataset of `a`, `b` and `common`, with `a < b`.

    Raises:
        PlanError: If `max_degree` is not positive.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, candidate_pairs
            >>> # 1 and 3 both connect to 2, so they are a candidate.
            >>> e = bt.from_pydict({"src": [1, 3], "dst": [2, 2]})
            >>> candidate_pairs(Graph.from_edges(e)).to_pydict()
            {'a': [1], 'b': [3], 'common': [1]}
    """
    if max_degree is not None and max_degree < 1:
        raise PlanError(f"max_degree must be positive, got {max_degree}")
    adj = _adjacency(g)
    if max_degree is not None:
        hubs = (
            out_degree(g.to_undirected().simple())
            .filter(bt.col("out_degree") > bt.lit(max_degree))
            .select(NODE)
        )
        adj = adj.join(hubs.select(nbr=bt.col(NODE)), on="nbr", how="anti")
    adj = adj.cache()
    left = adj.select(a=bt.col(NODE), nbr=bt.col("nbr"))
    right = adj.select(b=bt.col(NODE), nbr=bt.col("nbr"))
    return (
        left.join(right, on="nbr", how="inner")
        .filter(bt.col("a") < bt.col("b"))
        .group_by("a", "b")
        .agg(common=bt.count())
    )


def _with_degrees(g: Graph, pairs: Dataset, a: str, b: str) -> Dataset:
    """Attach each endpoint's neighbour count to a pair table."""
    deg = out_degree(g.to_undirected().simple()).select(
        **{NODE: bt.col(NODE), "_k": bt.col("out_degree")}
    )
    return (
        pairs.join(deg.select(**{a: bt.col(NODE), "_ka": bt.col("_k")}), on=a, how="left")
        .join(deg.select(**{b: bt.col(NODE), "_kb": bt.col("_k")}), on=b, how="left")
        .with_columns(
            _ka=bt.coalesce(bt.col("_ka"), bt.lit(0)),
            _kb=bt.coalesce(bt.col("_kb"), bt.lit(0)),
        )
    )


def common_neighbors(g: Graph, pairs: Dataset, *, a: str = "a", b: str = "b") -> Dataset:
    """How many neighbours each candidate pair shares.

    The simplest link-prediction score and a surprisingly strong baseline. Its weakness is
    that it does not normalize: two celebrities share hundreds of neighbours without being
    related, which is what `jaccard_similarity` and `adamic_adar` each correct differently.

    Args:
        g: The graph.
        pairs: The candidate pairs to score.
        a: The column holding the first node of each pair.
        b: The column holding the second.

    Returns:
        The pair table with a `common_neighbors` column.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, common_neighbors
            >>> e = bt.from_pydict({"src": [1, 3, 1, 3], "dst": [2, 2, 4, 4]})
            >>> pairs = bt.from_pydict({"a": [1], "b": [3]})
            >>> common_neighbors(Graph.from_edges(e), pairs).to_pydict()
            {'a': [1], 'b': [3], 'common_neighbors': [2]}
    """
    adj = _adjacency(g).cache()
    shared = (
        pairs.join(adj.select(**{a: bt.col(NODE), "nbr": bt.col("nbr")}), on=a, how="inner")
        .join(adj.select(**{b: bt.col(NODE), "nbr": bt.col("nbr")}), on=[b, "nbr"], how="semi")
        .group_by(a, b)
        .agg(common_neighbors=bt.count())
    )
    return pairs.join(shared, on=[a, b], how="left").with_columns(
        common_neighbors=bt.coalesce(bt.col("common_neighbors"), bt.lit(0))
    )


def jaccard_similarity(g: Graph, pairs: Dataset, *, a: str = "a", b: str = "b") -> Dataset:
    """Shared neighbours over total distinct neighbours, in `[0, 1]`.

    Normalizing by the union is what stops two hubs from scoring highly just for being
    hubs. A pair with no neighbours at all scores 0.0 rather than dividing by zero.

    Args:
        g: The graph.
        pairs: The candidate pairs to score.
        a: The column holding the first node of each pair.
        b: The column holding the second.

    Returns:
        The pair table with a `jaccard` column.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, jaccard_similarity
            >>> e = bt.from_pydict({"src": [1, 3, 1], "dst": [2, 2, 4]})
            >>> pairs = bt.from_pydict({"a": [1], "b": [3]})
            >>> jaccard_similarity(Graph.from_edges(e), pairs).to_pydict()
            {'a': [1], 'b': [3], 'jaccard': [0.5]}
    """
    scored = common_neighbors(g, pairs, a=a, b=b)
    with_deg = _with_degrees(g, scored, a, b)
    union = bt.col("_ka") + bt.col("_kb") - bt.col("common_neighbors")
    return with_deg.select(
        *list(pairs.columns),
        jaccard=bt.when(union > 0)
        .then(bt.col("common_neighbors").cast("float64") / union.cast("float64"))
        .otherwise(bt.lit(0.0)),
    )


def _rare_neighbour_score(
    g: Graph, pairs: Dataset, a: str, b: str, name: str, discount: object
) -> Dataset:
    """Sum a per-shared-neighbour weight that falls with that neighbour's degree."""
    adj = _adjacency(g).cache()
    deg = out_degree(g.to_undirected().simple()).select(nbr=bt.col(NODE), _kn=bt.col("out_degree"))
    shared = (
        pairs.join(adj.select(**{a: bt.col(NODE), "nbr": bt.col("nbr")}), on=a, how="inner")
        .join(adj.select(**{b: bt.col(NODE), "nbr": bt.col("nbr")}), on=[b, "nbr"], how="semi")
        .join(deg, on="nbr", how="left")
        .select(*[a, b], _w=discount(bt.coalesce(bt.col("_kn"), bt.lit(0)).cast("float64")))
        .group_by(a, b)
        .agg(**{name: bt.sum("_w")})
    )
    return pairs.join(shared, on=[a, b], how="left").with_columns(
        **{name: bt.coalesce(bt.col(name), bt.lit(0.0))}
    )


def adamic_adar(g: Graph, pairs: Dataset, *, a: str = "a", b: str = "b") -> Dataset:
    """Shared neighbours, each weighted by `1 / log(its degree)`.

    The insight that makes it better than a raw count: sharing an obscure neighbour is
    strong evidence two nodes are related, and sharing a celebrity is almost none. A
    neighbour of degree 2 contributes far more than one of degree 20,000.

    A shared neighbour of degree 1 would divide by `log(1) = 0`, so its degree is floored
    at 2 rather than producing an infinity.

    Args:
        g: The graph.
        pairs: The candidate pairs to score.
        a: The column holding the first node of each pair.
        b: The column holding the second.

    Returns:
        The pair table with an `adamic_adar` column.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, adamic_adar
            >>> e = bt.from_pydict({"src": [1, 3], "dst": [2, 2]})
            >>> pairs = bt.from_pydict({"a": [1], "b": [3]})
            >>> out = adamic_adar(Graph.from_edges(e), pairs).to_pydict()
            >>> round(out["adamic_adar"][0], 4)
            1.4427
    """
    return _rare_neighbour_score(
        g,
        pairs,
        a,
        b,
        "adamic_adar",
        lambda k: bt.lit(1.0) / bt.max_horizontal(k, bt.lit(2.0)).log(),
    )


def resource_allocation(g: Graph, pairs: Dataset, *, a: str = "a", b: str = "b") -> Dataset:
    """Shared neighbours, each weighted by `1 / its degree`.

    The same idea as `adamic_adar` with a harsher penalty: a hub is discounted linearly
    rather than logarithmically. Usually the better of the two on graphs whose degree
    distribution has a very long tail, and slightly worse on flatter ones.

    Args:
        g: The graph.
        pairs: The candidate pairs to score.
        a: The column holding the first node of each pair.
        b: The column holding the second.

    Returns:
        The pair table with a `resource_allocation` column.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, resource_allocation
            >>> e = bt.from_pydict({"src": [1, 3], "dst": [2, 2]})
            >>> pairs = bt.from_pydict({"a": [1], "b": [3]})
            >>> resource_allocation(Graph.from_edges(e), pairs).to_pydict()
            {'a': [1], 'b': [3], 'resource_allocation': [0.5]}
    """
    return _rare_neighbour_score(
        g,
        pairs,
        a,
        b,
        "resource_allocation",
        lambda k: bt.lit(1.0) / bt.max_horizontal(k, bt.lit(1.0)),
    )


def preferential_attachment(g: Graph, pairs: Dataset, *, a: str = "a", b: str = "b") -> Dataset:
    """The product of the two nodes' degrees, ignoring who they connect to.

    The "rich get richer" score: it predicts that a new edge is most likely between two
    already-popular nodes, and it needs no shared neighbour at all. That makes it the one
    score here that can rank a pair with nothing in common, and the one to compare the
    others against: a neighbourhood score that does not beat this is not using the
    neighbourhood.

    Args:
        g: The graph.
        pairs: The candidate pairs to score.
        a: The column holding the first node of each pair.
        b: The column holding the second.

    Returns:
        The pair table with a `preferential_attachment` column.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, preferential_attachment
            >>> e = bt.from_pydict({"src": [1, 1, 3], "dst": [2, 4, 2]})
            >>> pairs = bt.from_pydict({"a": [1], "b": [3]})
            >>> preferential_attachment(Graph.from_edges(e), pairs).to_pydict()
            {'a': [1], 'b': [3], 'preferential_attachment': [2]}
    """
    with_deg = _with_degrees(g, pairs, a, b)
    return with_deg.select(
        *list(pairs.columns),
        preferential_attachment=bt.col("_ka") * bt.col("_kb"),
    )
