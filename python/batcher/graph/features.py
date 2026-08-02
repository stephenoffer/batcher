"""Turning a graph into a feature table a model can train on.

Two ways, and they compose. `aggregate_neighbors` is one round of message passing: every
node's features become a summary of its neighbours' features, which is the arithmetic
core of every graph neural network. Stacking it `k` times gives each node a view `k` hops
out, and the stack of those views is a perfectly good feature matrix for an ordinary
gradient-boosted model, without training a GNN at all.

`structural_features` is the other way: describe each node by what its position in the
graph looks like, with no node attributes required. On a fraud or abuse problem those
columns are frequently the strongest signal available, because the behaviour being
detected is a shape in the graph rather than a property of any single account.
"""

from __future__ import annotations

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.api.dataset import Dataset
from batcher.graph._graph import DST, NODE, SRC, WEIGHT, Graph
from batcher.graph._iterate import checkpoint

__all__ = ["aggregate_neighbors", "propagate_features", "structural_features"]

#: The aggregations one round of message passing can apply. `mean` is the GraphSAGE
#: default and the safest on a graph with a wide degree distribution; `sum` preserves
#: magnitude and lets a model see how many neighbours contributed; `max` is a pooling
#: layer and is what detects "any neighbour with this property".
_AGGREGATIONS = {
    "mean": bt.mean,
    "sum": bt.sum,
    "max": bt.max,
    "min": bt.min,
}


def aggregate_neighbors(
    g: Graph,
    node_features: Dataset,
    features: list[str],
    *,
    how: str = "mean",
    node: str = "node",
    weighted: bool = False,
) -> Dataset:
    """One round of message passing: summarize each node's neighbours' features.

    Args:
        g: The graph. Features flow along edge direction, from `src` to `dst`; symmetrize
            with `Graph.to_undirected` when a node should see everyone it touches.
        node_features: A dataset of `node` plus numeric feature columns.
        features: The feature columns to aggregate.
        how: `"mean"`, `"sum"`, `"max"` or `"min"`.
        node: The column in `node_features` holding the node id.
        weighted: Weight each neighbour's contribution by the edge weight. Only meaningful
            with `"mean"` and `"sum"`; a weighted max is not a max of anything.

    Returns:
        A dataset of `node` plus one `<feature>_<how>` column per input feature. A node
        with no incoming edges gets null, not zero: it has no neighbours, which is a
        different fact from having neighbours whose features sum to zero.

    Raises:
        PlanError: If `how` is not a supported aggregation, or a named feature is absent.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, aggregate_neighbors
            >>> e = bt.from_pydict({"src": [1, 2], "dst": [3, 3]})
            >>> feats = bt.from_pydict({"node": [1, 2, 3], "x": [10.0, 20.0, 0.0]})
            >>> out = aggregate_neighbors(Graph.from_edges(e), feats, ["x"])
            >>> out.sort("node").to_pydict()
            {'node': [1, 2, 3], 'x_mean': [None, None, 15.0]}
    """
    if how not in _AGGREGATIONS:
        raise PlanError(
            f"aggregate_neighbors(): how must be one of {sorted(_AGGREGATIONS)}, got {how!r}"
        )
    missing = [f for f in features if f not in node_features.columns]
    if missing:
        raise PlanError(
            f"aggregate_neighbors(): feature column(s) {missing} are not in the feature "
            f"table; available: {sorted(node_features.columns)}"
        )
    if not features:
        raise PlanError("aggregate_neighbors(): name at least one feature column")
    if weighted and how in {"max", "min"}:
        raise PlanError(
            f"aggregate_neighbors(): weighted={weighted} is meaningless with how={how!r} "
            "— a weighted maximum is not the maximum of anything. Use mean or sum."
        )

    src_side = node_features.select(
        **{SRC: bt.col(node)}, **{f: bt.col(f).cast("float64") for f in features}
    )
    messages = g.edges.join(src_side, on=SRC, how="inner")
    if weighted:
        messages = messages.with_columns(**{f: bt.col(f) * bt.col(WEIGHT) for f in features})
    agg = _AGGREGATIONS[how]
    gathered = (
        messages.select(**{NODE: bt.col(DST)}, **{f: bt.col(f) for f in features})
        .group_by(NODE)
        .agg(**{f"{f}_{how}": agg(f) for f in features})
    )
    all_nodes = node_features.select(**{NODE: bt.col(node)})
    return all_nodes.join(gathered, on=NODE, how="left")


def propagate_features(
    g: Graph,
    node_features: Dataset,
    features: list[str],
    hops: int,
    *,
    how: str = "mean",
    node: str = "node",
) -> Dataset:
    """Stack `hops` rounds of message passing, keeping every round's output.

    The output has one column per feature per hop, so a model sees the node's own value,
    its neighbourhood's, its two-hop neighbourhood's, and so on. That stack is what a GNN
    learns to weight; handing it to a gradient-boosted model instead is a strong baseline
    that trains in seconds and is far easier to explain.

    Args:
        g: The graph.
        node_features: A dataset of `node` plus numeric feature columns.
        features: The feature columns to propagate.
        hops: How many rounds.
        how: The aggregation each round applies.
        node: The column in `node_features` holding the node id.

    Returns:
        A dataset of `node`, the original features, and `<feature>_hop<k>` for each round.

    Raises:
        PlanError: If `hops` is not positive, or an argument fails
            `aggregate_neighbors`'s checks.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, propagate_features
            >>> e = bt.from_pydict({"src": [1, 2], "dst": [2, 3]})
            >>> feats = bt.from_pydict({"node": [1, 2, 3], "x": [1.0, 0.0, 0.0]})
            >>> out = propagate_features(Graph.from_edges(e), feats, ["x"], 2)
            >>> out.sort("node").to_pydict()["x_hop2"]
            [None, 0.0, 1.0]

            Node 1 has no in-neighbours at any hop, so its value stays null. Node 2's
            only neighbour is node 1, whose hop-1 value *was* null and is carried into
            the next round as ``0.0`` — the deliberate `coalesce` below, which stops one
            neighbourless node nulling every chain that runs through it.
    """
    if hops < 1:
        raise PlanError(f"hops must be positive, got {hops}")
    current = node_features.select(
        **{NODE: bt.col(node)}, **{f: bt.col(f).cast("float64") for f in features}
    )
    out = current
    for hop in range(1, hops + 1):
        stepped = aggregate_neighbors(g, current, features, how=how, node=NODE)
        renamed = stepped.select(
            **{NODE: bt.col(NODE)},
            **{f: bt.col(f"{f}_{how}") for f in features},
        )
        out = out.join(
            renamed.select(
                **{NODE: bt.col(NODE)},
                **{f"{f}_hop{hop}": bt.col(f) for f in features},
            ),
            on=NODE,
            how="left",
        )
        # A null propagates as "no neighbours"; zero it so the next hop still carries
        # signal from the nodes that do have neighbours rather than nulling the chain.
        current = checkpoint(
            renamed.select(
                **{NODE: bt.col(NODE)},
                **{f: bt.coalesce(bt.col(f), bt.lit(0.0)) for f in features},
            )
        )
    return out


def structural_features(g: Graph) -> Dataset:
    """A feature row per node describing its position, with no node attributes needed.

    Assembles degree, weighted degree, triangle participation, clustering and PageRank
    into one table. On an abuse or fraud problem these are frequently the strongest
    features available, because the behaviour is a shape in the graph rather than a
    property of any single account.

    This runs several algorithms, including triangle counting, so it is the expensive
    convenience in this package. Call it once and cache the result.

    Args:
        g: The graph.

    Returns:
        A dataset of `node`, `in_degree`, `out_degree`, `degree`, `weighted_degree`,
        `triangles`, `clustering` and `pagerank`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, structural_features
            >>> e = bt.from_pydict({"src": [1, 2, 1], "dst": [2, 3, 3]})
            >>> for name in sorted(structural_features(Graph.from_edges(e)).columns):
            ...     print(name)
            clustering
            degree
            in_degree
            node
            out_degree
            pagerank
            triangles
            weighted_degree
    """
    from batcher.graph.centrality import pagerank
    from batcher.graph.community import clustering_coefficient, triangle_count
    from batcher.graph.degree import degree, in_degree, out_degree, weighted_degree

    cached = g.cache()
    out = in_degree(cached)
    for table in (
        out_degree(cached),
        degree(cached),
        weighted_degree(cached),
        triangle_count(cached),
        clustering_coefficient(cached),
        pagerank(cached),
    ):
        out = out.join(table, on=NODE, how="left")
    return out
