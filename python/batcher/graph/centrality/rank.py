"""PageRank and the degree-based centralities.

The two most-used centralities, and the ones whose cost is a join per round rather than
an eigen-decomposition. `pagerank` is the default choice on any graph; the personalized
form is the recommendation and local-neighbourhood primitive.
"""

from __future__ import annotations

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.api.dataset import Dataset
from batcher.graph._graph import DST, NODE, SRC, WEIGHT, Graph
from batcher.graph._iterate import iterate, max_abs_change
from batcher.graph.degree import degree

__all__ = ["degree_centrality", "pagerank", "personalized_pagerank"]


def _check_iterations(max_iterations: int, tolerance: float) -> None:
    if max_iterations < 1:
        raise PlanError(f"max_iterations must be at least 1, got {max_iterations}")
    if tolerance < 0.0:
        raise PlanError(f"tolerance must be non-negative, got {tolerance}")


def _out_strength(g: Graph) -> Dataset:
    """Each node's total outgoing edge weight, which is what a rank is divided by."""
    summed = g.edges.group_by(SRC).agg(_out=bt.sum(WEIGHT))
    return (
        g.nodes()
        .join(summed.select(**{NODE: bt.col(SRC), "_out": bt.col("_out")}), on=NODE, how="left")
        .select(**{NODE: bt.col(NODE), "_out": bt.coalesce(bt.col("_out"), bt.lit(0.0))})
    )


def _spread(g: Graph, ranks: Dataset, out: Dataset, value: str) -> Dataset:
    """Push each node's value along its outgoing edges, weighted, and sum at the far end.

    The shared body of every propagating algorithm here. A node with no outgoing edges
    contributes nothing, which is what makes its mass "dangling" and is handled by the
    caller rather than swallowed here.
    """
    sending = (
        ranks.join(out, on=NODE, how="inner")
        .filter(bt.col("_out") > 0.0)
        .select(**{SRC: bt.col(NODE), "_share": bt.col(value) / bt.col("_out")})
    )
    return (
        g.edges.join(sending, on=SRC, how="inner")
        .select(**{NODE: bt.col(DST), "_in": bt.col("_share") * bt.col(WEIGHT)})
        .group_by(NODE)
        .agg(_in=bt.sum("_in"))
    )


def pagerank(
    g: Graph,
    *,
    damping: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> Dataset:
    """PageRank: the stationary distribution of a random surfer on the graph.

    A surfer follows a random outgoing edge with probability `damping` and teleports to a
    uniformly random node otherwise. A node's rank is the long-run fraction of time spent
    there, so ranks sum to 1 and a node is important when important nodes point at it and
    do not point at much else.

    **Dangling nodes are handled explicitly.** A node with no outgoing edges would
    otherwise absorb rank every round until the total no longer sums to 1, which is the
    classic way a hand-rolled PageRank comes out quietly wrong. Their mass is
    redistributed uniformly each round, which is the standard treatment.

    Edge weights are honoured: a node's rank is split in proportion to outgoing weight
    rather than evenly. Run `Graph.simple` first if the edge table has parallel edges,
    since two copies of an edge otherwise carry twice the mass.

    Args:
        g: The graph.
        damping: The probability of following an edge rather than teleporting. The
            conventional 0.85 comes from the original paper; lower values converge faster
            and localize the score more tightly.
        max_iterations: The cap on rounds.
        tolerance: Stop once no node's rank moves by more than this.

    Returns:
        A dataset of `node` and `pagerank`, summing to 1.

    Raises:
        PlanError: If `damping` is outside `[0, 1)`, or the iteration arguments are
            invalid.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, pagerank
            >>> # A star: everyone points at node 0.
            >>> e = bt.from_pydict({"src": [1, 2, 3], "dst": [0, 0, 0]})
            >>> out = pagerank(Graph.from_edges(e)).sort("pagerank", descending=True)
            >>> out.to_pydict()["node"][0]
            0
    """
    if not 0.0 <= damping < 1.0:
        raise PlanError(f"damping must be in [0, 1), got {damping}")
    _check_iterations(max_iterations, tolerance)

    nodes = g.nodes().cache()
    n = nodes.count()
    if n == 0:
        return nodes.select(**{NODE: bt.col(NODE), "pagerank": bt.lit(0.0)})
    out = _out_strength(g).cache()
    base = (1.0 - damping) / n
    initial = nodes.select(**{NODE: bt.col(NODE), "pagerank": bt.lit(1.0 / n)})

    def step(ranks: Dataset) -> Dataset:
        # Mass sitting on nodes with no outgoing edge has nowhere to go; without this it
        # simply disappears and the ranks stop summing to 1.
        #
        # Cross-joined as a one-row table rather than collected into a Python float: the
        # loop already pays a `collect` per round for the checkpoint and one for the
        # convergence test, and pulling this scalar out would make it three. Keeping it
        # in the plan is a third of the per-round cost on a small graph, where the fixed
        # cost of executing a plan dominates the work.
        leaked = (
            ranks.join(out, on=NODE, how="inner")
            .filter(bt.col("_out") <= 0.0)
            .agg(_leak=bt.sum("pagerank"))
        )
        incoming = _spread(g, ranks, out, "pagerank")
        return (
            nodes.join(incoming, on=NODE, how="left")
            .cross_join(leaked)
            .select(
                **{
                    NODE: bt.col(NODE),
                    "pagerank": bt.lit(base)
                    + bt.lit(damping / n) * bt.coalesce(bt.col("_leak"), bt.lit(0.0))
                    + bt.lit(damping) * bt.coalesce(bt.col("_in"), bt.lit(0.0)),
                }
            )
        )

    result = iterate(
        initial,
        step,
        max_iterations=max_iterations,
        delta=max_abs_change(NODE, "pagerank"),
        tolerance=tolerance,
    )
    return result.state


def personalized_pagerank(
    g: Graph,
    sources: Dataset,
    *,
    node: str = "node",
    damping: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> Dataset:
    """PageRank whose surfer teleports back to a chosen set rather than anywhere.

    The recommendation primitive: seed with the items one user touched and the ranking
    that comes back is "what else is close to those", measured through the whole graph
    rather than by direct neighbours only. Also the standard way to find the part of a
    large graph relevant to a small set of nodes.

    Args:
        g: The graph.
        sources: The nodes to teleport back to. Rank concentrates around these.
        node: The column in `sources` holding the node id.
        damping: The probability of following an edge rather than teleporting.
        max_iterations: The cap on rounds.
        tolerance: Stop once no node's rank moves by more than this.

    Returns:
        A dataset of `node` and `pagerank`, summing to 1.

    Raises:
        PlanError: If `sources` is empty or names no node in the graph.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, personalized_pagerank
            >>> e = bt.from_pydict({"src": [1, 2, 3], "dst": [2, 3, 4]})
            >>> seed = bt.from_pydict({"node": [1]})
            >>> out = personalized_pagerank(Graph.from_edges(e), seed)
            >>> out.sort("pagerank", descending=True).to_pydict()["node"][0]
            1
    """
    if not 0.0 <= damping < 1.0:
        raise PlanError(f"damping must be in [0, 1), got {damping}")
    _check_iterations(max_iterations, tolerance)

    nodes = g.nodes().cache()
    seeds = (
        sources.select(**{NODE: bt.col(node)}).distinct().join(nodes, on=NODE, how="semi").cache()
    )
    seed_count = seeds.count()
    if seed_count == 0:
        raise PlanError(
            "personalized_pagerank(): none of the source nodes appear in the graph, so "
            "there is nowhere to teleport back to"
        )
    out = _out_strength(g).cache()
    # The teleport vector: uniform over the seeds, zero everywhere else.
    teleport = nodes.join(
        seeds.select(**{NODE: bt.col(NODE), "_seed": bt.lit(1.0 / seed_count)}),
        on=NODE,
        how="left",
    ).select(**{NODE: bt.col(NODE), "_seed": bt.coalesce(bt.col("_seed"), bt.lit(0.0))})
    initial = teleport.select(**{NODE: bt.col(NODE), "pagerank": bt.col("_seed")})

    def step(ranks: Dataset) -> Dataset:
        leaked = (
            ranks.join(out, on=NODE, how="inner")
            .filter(bt.col("_out") <= 0.0)
            .agg(_leak=bt.sum("pagerank"))
        )
        incoming = _spread(g, ranks, out, "pagerank")
        # Dangling mass returns to the seeds too, not uniformly: teleporting anywhere
        # would leak the personalization away a little more each round.
        return (
            teleport.join(incoming, on=NODE, how="left")
            .cross_join(leaked)
            .select(
                **{
                    NODE: bt.col(NODE),
                    "pagerank": bt.lit(1.0 - damping) * bt.col("_seed")
                    + bt.lit(damping)
                    * bt.coalesce(bt.col("_leak"), bt.lit(0.0))
                    * bt.col("_seed")
                    + bt.lit(damping) * bt.coalesce(bt.col("_in"), bt.lit(0.0)),
                }
            )
        )

    result = iterate(
        initial,
        step,
        max_iterations=max_iterations,
        delta=max_abs_change(NODE, "pagerank"),
        tolerance=tolerance,
    )
    return result.state


def degree_centrality(g: Graph) -> Dataset:
    """Degree, scaled so the most connected possible node scores 1.

    The cheapest centrality and often the most useful. It is also the one that is
    trivially gamed, which is the reason PageRank exists: a node can raise its degree by
    adding edges, and cannot raise its PageRank without persuading important nodes to
    point at it.

    Args:
        g: The graph.

    Returns:
        A dataset of `node` and `degree_centrality` in `[0, 1]`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, degree_centrality
            >>> e = bt.from_pydict({"src": [1, 2, 3], "dst": [0, 0, 0]})
            >>> out = degree_centrality(Graph.from_edges(e)).sort("node")
            >>> out.to_pydict()["degree_centrality"]
            [1.0, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
    """
    n = g.num_nodes()
    scale = 1.0 / max(n - 1, 1)
    return degree(g).select(
        **{
            NODE: bt.col(NODE),
            "degree_centrality": bt.col("degree").cast("float64") * bt.lit(scale),
        }
    )
