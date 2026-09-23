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
from batcher.graph._graph import DST, NODE, SRC, WEIGHT, Graph, ends, walked
from batcher.graph._iterate import (
    checkpoint,
    count_changed,
    fixpoint_rounds,
    iterate,
    require_fixpoint,
)

__all__ = [
    "component_sizes",
    "connected_components",
    "is_connected",
    "k_core",
    "largest_component",
    "strongly_connected_components",
]


def connected_components(g: Graph, *, max_iterations: int | None = None) -> Dataset:
    """Label each node with the component it belongs to, by label propagation.

    Every node starts as its own component and repeatedly adopts the smallest label among
    itself and its neighbours, so a component converges to the smallest node id in it.
    That makes the labels stable and comparable across runs, unlike a scheme that hands
    out arbitrary component numbers.

    Each round also *jumps*: a node adopts the label its current label's node holds. A
    label is always the id of a node in the same component, so the jump is safe, and it is
    what lets a long chain settle in a number of rounds closer to the logarithm of its
    length than to the length itself.

    **Direction is ignored.** These are the *weakly* connected components: the graph is
    symmetrized first, so a chain `a -> b -> c` is one component even though nothing can
    reach `a`. That is what "connected" means for almost every practical question, and
    the strongly connected version needs a different algorithm entirely.

    Args:
        g: The graph.
        max_iterations: The cap on rounds, or `None` (the default) to run until no label
            changes, which never takes more rounds than there are nodes.

    Returns:
        A dataset of `node` and `component`.

    Raises:
        PlanError: If `max_iterations` is not positive, or stops the propagation before
            every label has settled. A truncated run would split components, so it is
            refused rather than returned.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, connected_components
            >>> e = bt.from_pydict({"src": [1, 2, 8], "dst": [2, 3, 9]})
            >>> out = connected_components(Graph.from_edges(e)).sort("node")
            >>> out.to_pydict()
            {'node': [1, 2, 3, 8, 9], 'component': [1, 1, 1, 8, 8]}
    """
    undirected = g.to_undirected()
    nodes = undirected.nodes().cache()
    rounds = fixpoint_rounds(max_iterations, nodes.count() if max_iterations is None else 0)
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
        jumped = labels.select(component=bt.col(NODE), _j=bt.col("component"))
        own = bt.col("component")
        return (
            labels.join(offered, on=NODE, how="left")
            .join(jumped, on="component", how="left")
            .select(
                **{
                    NODE: bt.col(NODE),
                    "component": bt.least(
                        own, bt.coalesce(bt.col("_l"), own), bt.coalesce(bt.col("_j"), own)
                    ),
                }
            )
        )

    result = iterate(
        initial,
        step,
        max_iterations=rounds,
        delta=count_changed(NODE, "component"),
        tolerance=0.0,
    )
    require_fixpoint(result.converged, "connected_components", rounds)
    return result.state


def component_sizes(g: Graph, *, max_iterations: int | None = None) -> Dataset:
    """How many nodes each component holds, largest first.

    The shape of this table is the diagnosis. One row holding almost everything is a
    graph with a giant component; a long flat tail is a graph of islands.

    Args:
        g: The graph.
        max_iterations: The cap on label-propagation rounds; see `connected_components`.

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


def largest_component(g: Graph, *, max_iterations: int | None = None) -> Graph:
    """The subgraph induced on the biggest connected component.

    The usual first step before running anything expensive: path-based measures are
    undefined across components, and running them on the whole graph wastes the work on
    pairs that can never reach each other.

    Args:
        g: The graph.
        max_iterations: The cap on label-propagation rounds; see `connected_components`.

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


def is_connected(g: Graph, *, max_iterations: int | None = None) -> bool:
    """Whether the whole graph is one connected piece.

    Args:
        g: The graph.
        max_iterations: The cap on label-propagation rounds; see `connected_components`.

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
    return connected_components(g, max_iterations=max_iterations).count_distinct("component") <= 1


def k_core(g: Graph, k: int, *, max_iterations: int | None = None) -> Graph:
    """The largest subgraph in which every node has at least `k` neighbours.

    Found by repeatedly deleting every node below the threshold, which can cascade: a
    deletion lowers its neighbours' degrees and may take them below `k` too. The loop runs
    until nothing more is removed.

    The natural way to strip a graph down to its dense part. On a social graph, `k_core(g,
    3)` drops the accounts with one or two connections and leaves the interconnected
    community they were attached to.

    Direction is ignored and parallel edges count once. A self-loop counts as one
    neighbour.

    Args:
        g: The graph.
        k: The minimum degree to survive.
        max_iterations: The cap on peeling rounds, or `None` (the default) to peel until
            nothing more is removed.

    Returns:
        The induced subgraph on the surviving nodes, symmetrized and with parallel edges
        merged (their weights summed), which may be empty.

    Raises:
        PlanError: If `k` is negative, or `max_iterations` is not positive or stops the
            peeling before it is finished.

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
    rounds = fixpoint_rounds(max_iterations, g.num_nodes() if max_iterations is None else 0)
    # Each undirected edge once, as (lo, hi), with its total weight. Built from the edge
    # table directly rather than from `to_undirected().simple()`: that form is a union
    # under an aggregate, which has no distributed path once another breaker sits on it.
    lo = bt.least(bt.col(SRC), bt.col(DST))
    hi = bt.greatest(bt.col(SRC), bt.col(DST))
    pairs = (
        g.edges.select(_lo=lo, _hi=hi, _w=bt.col(WEIGHT))
        .group_by("_lo", "_hi")
        .agg(_w=bt.sum("_w"))
    ).cache()
    survivors = checkpoint(_at_least(g, pairs, k))
    kept = survivors.count()
    # Every round restricts the *original* pair table by the latest survivor set, which is
    # data rather than a plan, so the plan stays the same depth however long the cascade.
    for _ in range(rounds):
        live = pairs.join(survivors.select(_lo=bt.col(NODE)), on="_lo", how="semi").join(
            survivors.select(_hi=bt.col(NODE)), on="_hi", how="semi"
        )
        nxt = checkpoint(_at_least(g, live, k))
        now = nxt.count()
        if now == kept:
            break
        survivors, kept = nxt, now
    else:
        require_fixpoint(False, "k_core", rounds)
    both = live.select(**{SRC: bt.col("_lo"), DST: bt.col("_hi"), WEIGHT: bt.col("_w")}).union(
        live.filter(bt.col("_lo") != bt.col("_hi")).select(
            **{SRC: bt.col("_hi"), DST: bt.col("_lo"), WEIGHT: bt.col("_w")}
        )
    )
    return replace(g._rebuild(both, survivors), directed=False, symmetrized=True)


def _at_least(g: Graph, pairs: Dataset, k: int) -> Dataset:
    """The nodes with at least `k` distinct neighbours in a `(_lo, _hi)` pair table.

    Each pair credits both endpoints and a self-loop credits its node once, which is what
    `ends` emits. A declared node is credited zero so that `k == 0` keeps it.
    """
    credits = ends(pairs).select(NODE, _k=bt.lit(1))
    extra = g.extra_nodes()
    if extra is not None:
        credits = credits.union(extra.select(NODE, _k=bt.lit(0)))
    counted = credits.group_by(NODE).agg(_k=bt.sum("_k"))
    return counted.filter(bt.col("_k") >= bt.lit(k)).select(NODE)


def _propagate_max_label(edges: Dataset, labels: Dataset, rounds: int) -> Dataset:
    """Push the largest label along `edges` to a fixpoint.

    Shared by the forward and backward halves of the colouring algorithm, which differ
    only in which edge table they walk -- writing it twice is how the two would drift.

    Raises:
        PlanError: If `rounds` rounds pass without the labels settling.
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
                    "_c": bt.greatest(bt.col("_c"), bt.coalesce(bt.col("_offer"), bt.col("_c"))),
                }
            )
        )
        if count_changed(NODE, "_c")(current, nxt) == 0:
            return nxt
        current = nxt
    require_fixpoint(False, "strongly_connected_components", rounds)
    return current


def strongly_connected_components(g: Graph, *, max_iterations: int | None = None) -> Dataset:
    """Label each node with the strongly connected component it belongs to.

    Two nodes are in the same component when each can reach the other *following edge
    direction*. That is a much stronger claim than `connected_components` makes, and the
    difference is the whole point on a directed graph: a chain `a -> b -> c` is one weak
    component and three strong ones, because nothing gets back to `a`.

    Found by the colouring algorithm rather than by Tarjan's, whose depth-first search has
    no relational form. Each round propagates the largest id forward to a fixpoint, so every
    node's colour is the largest node that reaches it. A node whose colour is its own id is
    a *root*, and the root's component is exactly the nodes of its colour that can reach it
    without leaving that colour, so a second propagation, backward and restricted to edges
    inside one colour, finds every root's component at once. Every round settles at least
    the component of the largest remaining node; what is left goes round again.

    Args:
        g: The graph. Direction is respected, which is the entire difference from
            `connected_components`. On a graph built with `directed=False` the answer is
            its connected components.
        max_iterations: The cap on rounds, inner and outer, or `None` (the default) to run
            to completion. A propagation takes up to one round per hop of the longest
            path it follows.

    Returns:
        A dataset of `node` and `component`, labelled by the largest node id in each.

    Raises:
        PlanError: If `max_iterations` is not positive, or is too small to finish. A
            truncated run would report real components as singletons, so it is refused.

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
    g = walked(g)
    remaining = checkpoint(g.nodes())
    left = remaining.count()
    rounds = fixpoint_rounds(max_iterations, left)
    found: list[Dataset] = []

    for _ in range(rounds):
        if left == 0:
            break
        live = g.subgraph(remaining).edges.cache()
        seed = remaining.select(**{NODE: bt.col(NODE), "_c": bt.col(NODE)})
        colour = _propagate_max_label(live, seed, rounds).cache()
        # Reversed edges whose two ends share a colour: the only ones a root's backward
        # search may use, since leaving the colour would reach a different component.
        inside = (
            live.join(colour.select(**{SRC: bt.col(NODE), "_cs": bt.col("_c")}), on=SRC)
            .join(colour.select(**{DST: bt.col(NODE), "_cd": bt.col("_c")}), on=DST)
            .filter(bt.col("_cs") == bt.col("_cd"))
            .select(**{SRC: bt.col(DST), DST: bt.col(SRC)})
        )
        back = _propagate_max_label(inside, seed, rounds)
        agreed = checkpoint(
            colour.select(**{NODE: bt.col(NODE), "component": bt.col("_c")})
            .join(back.select(**{NODE: bt.col(NODE), "_r": bt.col("_c")}), on=NODE)
            .filter(bt.col("component") == bt.col("_r"))
            .select(NODE, "component")
        )
        found.append(agreed)
        remaining = checkpoint(remaining.join(agreed.select(NODE), on=NODE, how="anti"))
        left = remaining.count()
    require_fixpoint(left == 0, "strongly_connected_components", rounds)
    if not found:
        return remaining.select(**{NODE: bt.col(NODE), "component": bt.col(NODE)})
    out = found[0]
    for part in found[1:]:
        out = out.union(part)
    return out
