"""Connected components, and the cores that survive peeling the graph down.

Both answer "what are the separate pieces of this graph", at two different strengths.
Components split it into parts with no edges between them, which is the first thing to
check on any graph you did not build yourself: a graph that is 90% one giant component
and 10% dust behaves nothing like one that is a thousand equal islands, and every
sampling and partitioning decision downstream turns on which you have.

`k_core` is the graduated version. Repeatedly removing every node with fewer than `k`
neighbours leaves the part of the graph that is densely interconnected, which is how you
find the real community inside a graph whose degree distribution is dominated by
one-edge stragglers.
"""

from __future__ import annotations

from dataclasses import replace

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.api.dataset import Dataset
from batcher.graph._graph import DST, NODE, SRC, Graph
from batcher.graph._iterate import checkpoint, count_changed, iterate
from batcher.graph.degree import out_degree

__all__ = [
    "component_sizes",
    "connected_components",
    "is_connected",
    "k_core",
    "largest_component",
    "strongly_connected_components",
]


def connected_components(g: Graph, *, max_iterations: int = 100) -> Dataset:
    """Label each node with the component it belongs to, by label propagation.

    Every node starts as its own component and repeatedly adopts the smallest label among
    itself and its neighbours, so a component converges to the smallest node id in it.
    That makes the labels stable and comparable across runs, unlike a scheme that hands
    out arbitrary component numbers.

    **Direction is ignored.** These are the *weakly* connected components: the graph is
    symmetrized first, so a chain `a -> b -> c` is one component even though nothing can
    reach `a`. That is what "connected" means for almost every practical question, and
    the strongly connected version needs a different algorithm entirely.

    Args:
        g: The graph.
        max_iterations: The cap on rounds. Label propagation needs about as many rounds
            as the graph's diameter, so the default covers any graph whose longest
            shortest path is under 100 hops.

    Returns:
        A dataset of `node` and `component`.

    Raises:
        PlanError: If `max_iterations` is not positive.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, connected_components
            >>> e = bt.from_pydict({"src": [1, 2, 8], "dst": [2, 3, 9]})
            >>> out = connected_components(Graph.from_edges(e)).sort("node")
            >>> out.to_pydict()
            {'node': [1, 2, 3, 8, 9], 'component': [1, 1, 1, 8, 8]}
    """
    if max_iterations < 1:
        raise PlanError(f"max_iterations must be at least 1, got {max_iterations}")
    undirected = g.to_undirected()
    nodes = undirected.nodes().cache()
    initial = nodes.select(**{NODE: bt.col(NODE), "component": bt.col(NODE)})

    def step(labels: Dataset) -> Dataset:
        # Every node offers its label to its neighbours; each node takes the smallest of
        # what it is offered and what it already holds.
        offered = (
            undirected.edges.join(
                labels.select(**{SRC: bt.col(NODE), "_l": bt.col("component")}),
                on=SRC,
                how="inner",
            )
            .select(**{NODE: bt.col(DST), "_l": bt.col("_l")})
            .group_by(NODE)
            .agg(_l=bt.min("_l"))
        )
        return labels.join(offered, on=NODE, how="left").select(
            **{
                NODE: bt.col(NODE),
                "component": bt.min_horizontal(
                    bt.col("component"), bt.coalesce(bt.col("_l"), bt.col("component"))
                ),
            }
        )

    return iterate(
        initial,
        step,
        max_iterations=max_iterations,
        delta=count_changed(NODE, "component"),
        tolerance=0.0,
    ).state


def component_sizes(g: Graph, *, max_iterations: int = 100) -> Dataset:
    """How many nodes each component holds, largest first.

    The shape of this table is the diagnosis. One row holding almost everything is a
    graph with a giant component; a long flat tail is a graph of islands.

    Args:
        g: The graph.
        max_iterations: The cap on label-propagation rounds.

    Returns:
        A dataset of `component` and `nodes`, sorted by size descending.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, component_sizes
            >>> e = bt.from_pydict({"src": [1, 2, 8], "dst": [2, 3, 9]})
            >>> component_sizes(Graph.from_edges(e)).to_pydict()
            {'component': [1, 8], 'nodes': [3, 2]}
    """
    return (
        connected_components(g, max_iterations=max_iterations)
        .group_by("component")
        .agg(nodes=bt.count())
        .sort("nodes", "component", descending=[True, False])
    )


def largest_component(g: Graph, *, max_iterations: int = 100) -> Graph:
    """The subgraph induced on the biggest connected component.

    The usual first step before running anything expensive: path-based measures are
    undefined across components, and running them on the whole graph wastes the work on
    pairs that can never reach each other.

    Args:
        g: The graph.
        max_iterations: The cap on label-propagation rounds.

    Returns:
        The induced subgraph. An empty graph is returned unchanged.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, largest_component
            >>> e = bt.from_pydict({"src": [1, 2, 8], "dst": [2, 3, 9]})
            >>> largest_component(Graph.from_edges(e)).num_edges()
            2
    """
    sizes = component_sizes(g, max_iterations=max_iterations).limit(1).to_pydict()
    if not sizes.get("component"):
        return g
    biggest = sizes["component"][0]
    labels = connected_components(g, max_iterations=max_iterations)
    keep = labels.filter(bt.col("component") == bt.lit(biggest)).select(NODE)
    return g.subgraph(keep)


def is_connected(g: Graph, *, max_iterations: int = 100) -> bool:
    """Whether the whole graph is one connected piece.

    Args:
        g: The graph.
        max_iterations: The cap on label-propagation rounds.

    Returns:
        True when every node reaches every other, ignoring direction. An empty graph is
        connected, vacuously.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, is_connected
            >>> is_connected(Graph.from_edges(bt.from_pydict({"src": [1], "dst": [2]})))
            True
    """
    return connected_components(g, max_iterations=max_iterations).n_unique("component") <= 1


def k_core(g: Graph, k: int, *, max_iterations: int = 100) -> Graph:
    """The largest subgraph in which every node has at least `k` neighbours.

    Found by repeatedly deleting every node below the threshold, which can cascade: a
    deletion lowers its neighbours' degrees and may take them below `k` too. The loop runs
    until nothing more is removed.

    The natural way to strip a graph down to its dense part. On a social graph, `k_core(g,
    3)` drops the accounts with one or two connections and leaves the interconnected
    community they were attached to.

    Args:
        g: The graph.
        k: The minimum degree to survive.
        max_iterations: The cap on peeling rounds.

    Returns:
        The induced subgraph on the surviving nodes, which may be empty.

    Raises:
        PlanError: If `k` is negative or `max_iterations` is not positive.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, k_core
            >>> # A triangle with a pendant node hanging off it.
            >>> e = bt.from_pydict({"src": [1, 2, 3, 1], "dst": [2, 3, 1, 9]})
            >>> sorted(k_core(Graph.from_edges(e), 2).nodes().to_pydict()["node"])
            [1, 2, 3]
    """
    if k < 0:
        raise PlanError(f"k must be non-negative, got {k}")
    if max_iterations < 1:
        raise PlanError(f"max_iterations must be at least 1, got {max_iterations}")
    current = g.to_undirected().simple()
    for _ in range(max_iterations):
        # `out_degree`, not `degree`: on a symmetrized edge table every edge is present in
        # both directions, so counting both endpoints reports twice the neighbour count
        # and `k_core(g, 2)` would keep the pendant nodes it exists to peel.
        survivors = checkpoint(
            out_degree(current).filter(bt.col("out_degree") >= bt.lit(k)).select(NODE)
        )
        kept = survivors.count()
        if kept == current.num_nodes():
            break
        if kept == 0:
            return current.subgraph(survivors)
        # Checkpoint the *edges* too, not just the survivor list. Each round's subgraph is
        # a join over the previous round's plan, so without cutting it the plan nests once
        # per peel and a deep enough graph overflows the engine's plan walk — which
        # surfaces as a segfault rather than an error, since the recursion is in Rust.
        peeled = current.subgraph(survivors)
        current = replace(peeled, edges=checkpoint(peeled.edges))
    return current


def _propagate_max_label(edges: Dataset, labels: Dataset, rounds: int) -> Dataset:
    """Push the largest label along `edges` to a fixpoint.

    Shared by the forward and backward halves of the colouring algorithm, which differ
    only in which edge table they walk -- writing it twice is how the two would drift.
    """
    current = checkpoint(labels)
    for _ in range(rounds):
        offered = (
            edges.join(
                current.select(**{SRC: bt.col(NODE), "_offer": bt.col("_c")}),
                on=SRC,
                how="inner",
            )
            .select(**{NODE: bt.col(DST), "_offer": bt.col("_offer")})
            .group_by(NODE)
            .agg(_offer=bt.max("_offer"))
        )
        nxt = checkpoint(
            current.join(offered, on=NODE, how="left").select(
                **{
                    NODE: bt.col(NODE),
                    "_c": bt.max_horizontal(
                        bt.col("_c"), bt.coalesce(bt.col("_offer"), bt.col("_c"))
                    ),
                }
            )
        )
        if count_changed(NODE, "_c")(current, nxt) == 0:
            return nxt
        current = nxt
    return current


def strongly_connected_components(g: Graph, *, max_iterations: int = 20) -> Dataset:
    """Label each node with the strongly connected component it belongs to.

    Two nodes are in the same component when each can reach the other *following edge
    direction*. That is a much stronger claim than `connected_components` makes, and the
    difference is the whole point on a directed graph: a chain `a -> b -> c` is one weak
    component and three strong ones, because nothing gets back to `a`.

    Found by the colouring algorithm rather than by Tarjan's, whose depth-first search has
    no relational form. Each round propagates the largest reachable label forward to a
    fixpoint, then propagates it backward; a node whose two labels agree is mutually
    reachable with that label's node, which is exactly the definition, so it settles into
    that component. What is left goes round again.

    Args:
        g: The graph. Direction is respected, which is the entire difference from
            `connected_components`.
        max_iterations: The cap on rounds, inner and outer. Each outer round settles at
            least one component, so a graph of many small components needs more of them;
            whatever the cap leaves over is reported as singletons rather than dropped.

    Returns:
        A dataset of `node` and `component`, labelled by the largest node id in each.

    Raises:
        PlanError: If `max_iterations` is not positive.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, strongly_connected_components
            >>> # A 3-cycle, plus a tail that cannot be returned to.
            >>> e = bt.from_pydict({"src": [1, 2, 3, 3], "dst": [2, 3, 1, 4]})
            >>> out = strongly_connected_components(Graph.from_edges(e)).sort("node")
            >>> out.to_pydict()
            {'node': [1, 2, 3, 4], 'component': [3, 3, 3, 4]}
    """
    if max_iterations < 1:
        raise PlanError(f"max_iterations must be at least 1, got {max_iterations}")
    reversed_g = g.reverse()
    remaining = checkpoint(g.nodes())
    settled: Dataset | None = None

    for _ in range(max_iterations):
        if remaining.count() == 0:
            break
        live = g.subgraph(remaining).edges.cache()
        live_reversed = reversed_g.subgraph(remaining).edges.cache()
        seed = remaining.select(**{NODE: bt.col(NODE), "_c": bt.col(NODE)})
        forward = _propagate_max_label(live, seed, max_iterations)
        backward = _propagate_max_label(live_reversed, seed, max_iterations)
        agreed = checkpoint(
            forward.select(**{NODE: bt.col(NODE), "_f": bt.col("_c")})
            .join(backward.select(**{NODE: bt.col(NODE), "_r": bt.col("_c")}), on=NODE)
            .filter(bt.col("_f") == bt.col("_r"))
            .select(**{NODE: bt.col(NODE), "component": bt.col("_f")})
        )
        if agreed.count() == 0:
            # No node agreed with itself, which can only happen once nothing is left to
            # settle; the rest are singletons.
            break
        settled = agreed if settled is None else checkpoint(settled.union(agreed))
        remaining = checkpoint(remaining.join(agreed.select(NODE), on=NODE, how="anti"))

    leftovers = remaining.select(**{NODE: bt.col(NODE), "component": bt.col(NODE)})
    if settled is None:
        return leftovers
    return settled.union(leftovers) if remaining.count() > 0 else settled
