"""Betweenness: how much of the graph's shortest-path traffic flows through each node.

The centrality that finds *bridges* rather than hubs. A node joining two dense clusters
can have a small degree and a tiny PageRank while every path between the clusters goes
through it, which is exactly the node whose removal splits the graph. On an
infrastructure or supply graph it is the single-point-of-failure measure; on a social
graph it finds the brokers.

Exact betweenness is a shortest-path computation from **every** node, which is quadratic
in node count and does not fit on any graph large enough to need this engine. This is
Brandes' algorithm run from a set of sources you supply, which is the standard estimator:
the *ranking* stabilizes after a few dozen well-chosen sources long before the values do,
so sample and compare ranks rather than magnitudes.

Sources are searched in batches rather than one at a time. Each batch is one breadth-first
search whose state carries the source it started from, so a batch of `b` sources costs the
depth in queries rather than `b` times it, and holds one row per (source, reached node)
pair. `source_batch_size` is the knob between the two: it bounds that state, and with it
the memory each round passes through the driver.
"""

from __future__ import annotations

from typing import Literal

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.api.dataset import Dataset
from batcher.graph._graph import DST, NODE, SRC, Graph, walked
from batcher.graph._iterate import checkpoint

__all__ = ["betweenness_centrality"]

_SEED = "_seed"
_SCALES = ("sum", "networkx", "normalized")


def _shortest_path_counts(
    edges: Dataset, seeds: list[object], max_depth: int | None
) -> list[Dataset]:
    """One dataset per BFS level: the (source, node) pairs at that depth and their path counts.

    `sigma` is the path count, and it is what makes betweenness a *fraction* rather than
    a count: when two shortest paths reach a node, each carries half the credit. A
    version that tracked only distance would give every tie the full weight and rank the
    wrong nodes. Every column is keyed by `_seed`, so the searches of a whole batch of
    sources share each round's joins without ever mixing their counts.
    """
    levels: list[Dataset] = []
    frontier = checkpoint(
        bt.from_pydict({_SEED: seeds}).select(_SEED, **{NODE: bt.col(_SEED), "sigma": bt.lit(1.0)})
    )
    seen = checkpoint(frontier.select(_SEED, NODE))
    levels.append(frontier)
    depth = 0
    while max_depth is None or depth < max_depth:
        depth += 1
        nxt = (
            edges.join(
                frontier.select(_SEED, **{SRC: bt.col(NODE), "_s": bt.col("sigma")}),
                on=SRC,
                how="inner",
            )
            .select(_SEED, **{NODE: bt.col(DST), "_s": bt.col("_s")})
            .group_by(_SEED, NODE)
            .agg(sigma=bt.sum("_s"))
            # Only pairs not yet reached are at this depth; one already seen was reached
            # by a shorter path, and counting it again would inflate its sigma.
            .join(seen, on=[_SEED, NODE], how="anti")
        )
        nxt = checkpoint(nxt)
        if nxt.count() == 0:
            break
        levels.append(nxt)
        seen = checkpoint(seen.union(nxt.select(_SEED, NODE)))
        frontier = nxt
    return levels


def betweenness_centrality(
    g: Graph,
    sources: Dataset,
    *,
    node: str = "node",
    max_depth: int | None = None,
    scale: Literal["sum", "networkx", "normalized"] = "sum",
    source_batch_size: int = 64,
) -> Dataset:
    """Estimate betweenness from shortest paths out of a set of source nodes.

    Brandes' algorithm: a forward pass counts the shortest paths from each source to
    every node, and a backward pass over the same levels accumulates each node's share of
    the traffic. Running it from every node would be exact and quadratic; running it from
    a sample is the standard estimator. Paths are counted in hops; edge weights are not
    read.

    Args:
        g: The graph. Direction is respected on a directed graph; one built with
            `directed=False` is walked both ways.
        sources: The nodes to compute shortest paths from. A few dozen high-degree nodes
            (`degree(g).sort(...).limit(n)`) converge the ranking fastest.
        node: The column in `sources` holding the node id.
        max_depth: The hop cap per source, or `None` for none. Paths longer than a cap
            contribute nothing, which bounds the cost but changes the answer.
        scale: How the per-source sums are reported. ``"sum"`` returns the raw Brandes
            sum over the sources, which counts an undirected pair from both ends.
            ``"networkx"`` and ``"normalized"`` apply networkx's rescaling for
            ``normalized=False`` and ``normalized=True``: the undirected pair counted once,
            a sample of `k` of the `n` nodes extrapolated to all of them, and for
            ``"normalized"`` a division by the pairs a node can lie between. With every
            node as a source they equal networkx's exact values; with a sample they equal
            ``betweenness_centrality(G, k=k)`` given the same sampled nodes. A single
            source leaves its own score undefined there, and it is NaN here too.
        source_batch_size: How many sources search together. A batch costs the search
            depth in queries and holds one row per (source, reached node) pair, so a
            larger batch trades memory for fewer rounds.

    Returns:
        A dataset of `node` and `betweenness`. Under the default ``"sum"`` the values
        scale with the number of sources, so compare ranks across runs rather than
        magnitudes.

    Raises:
        PlanError: If `max_depth` or `source_batch_size` is not positive, `scale` is not
            one of the three names, or no source is in the graph.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, betweenness_centrality
            >>> # A bridge: everything from the left must pass through 'c'.
            >>> e = bt.from_pydict(
            ...     {"src": ["a", "b", "c", "d"], "dst": ["c", "c", "d", "e"]}
            ... )
            >>> g = Graph.from_edges(e)
            >>> out = betweenness_centrality(g, bt.from_pydict({"node": ["a", "b"]}))
            >>> out.sort("betweenness", descending=True).to_pydict()["node"][0]
            'c'
            >>> every = g.nodes().select("node")
            >>> exact = betweenness_centrality(g, every, scale="normalized").sort("node")
            >>> [round(v, 4) for v in exact.to_pydict()["betweenness"]]
            [0.0, 0.0, 0.3333, 0.25, 0.0]
    """
    if max_depth is not None and max_depth < 1:
        raise PlanError(f"max_depth must be positive, got {max_depth}")
    if source_batch_size < 1:
        raise PlanError(f"source_batch_size must be positive, got {source_batch_size}")
    if scale not in _SCALES:
        raise PlanError(f"scale must be one of {list(_SCALES)}, got {scale!r}")
    seeds = (
        sources.select(**{NODE: bt.col(node)})
        .distinct()
        .join(g.nodes(), on=NODE, how="semi")
        .to_pydict()[NODE]
    )
    if not seeds:
        raise PlanError(
            "betweenness_centrality(): none of the source nodes appear in the graph, so "
            "there are no shortest paths to trace"
        )
    edges = walked(g).edges.cache()
    totals: Dataset | None = None
    for start in range(0, len(seeds), source_batch_size):
        levels = _shortest_path_counts(edges, seeds[start : start + source_batch_size], max_depth)
        if len(levels) >= 2:
            totals = _accumulate_backward(edges, levels, totals)
    raw = _finalize(g, totals)
    return raw if scale == "sum" else _rescale(g, raw, seeds, normalized=scale == "normalized")


def _accumulate_backward(edges: Dataset, levels: list[Dataset], totals: Dataset | None) -> Dataset:
    """Fold one batch's backward pass into `totals`, returning the new per-node sum."""
    # Backward pass. `delta[s, v]` is the share of source s's shortest-path traffic
    # through `v`; a pair at the deepest level has none, and each level hands its share
    # back to the level above in proportion to the paths it received from there.
    deeper = checkpoint(levels[-1].select(_SEED, NODE, "sigma", delta=bt.lit(0.0)))
    for depth in range(len(levels) - 2, -1, -1):
        above = levels[depth]
        contribution = (
            edges.join(
                above.select(_SEED, **{SRC: bt.col(NODE), "_sv": bt.col("sigma")}),
                on=SRC,
                how="inner",
            )
            .join(
                deeper.select(
                    _SEED, **{DST: bt.col(NODE), "_sw": bt.col("sigma"), "_dw": bt.col("delta")}
                ),
                on=[_SEED, DST],
                how="inner",
            )
            .select(
                _SEED,
                **{
                    NODE: bt.col(SRC),
                    "_c": (bt.col("_sv") / bt.col("_sw")) * (bt.lit(1.0) + bt.col("_dw")),
                },
            )
            .group_by(_SEED, NODE)
            .agg(_c=bt.sum("_c"))
        )
        level = checkpoint(
            above.join(contribution, on=[_SEED, NODE], how="left").select(
                _SEED, NODE, "sigma", delta=bt.coalesce(bt.col("_c"), bt.lit(0.0))
            )
        )
        # The source itself accumulates nothing: it is an endpoint of every path it
        # starts, and betweenness counts only the nodes a path passes *through*. Folded
        # per level, so the running total stays one row per node however many levels
        # and sources feed it.
        if depth > 0:
            arm = level.select(NODE, "delta")
            merged = arm if totals is None else totals.union(arm)
            totals = checkpoint(merged.group_by(NODE).agg(delta=bt.sum("delta")))
        deeper = level
    return totals if totals is not None else _empty_totals(edges)


def _empty_totals(edges: Dataset) -> Dataset:
    """A zero-row `(node, delta)` table typed like the edges' node ids."""
    return edges.select(**{NODE: bt.col(SRC), "delta": bt.lit(0.0)}).limit(0)


def _finalize(g: Graph, totals: Dataset | None) -> Dataset:
    """Sum each node's accumulated share and left-join it back onto the full node set."""
    if totals is None:
        return g.nodes().select(**{NODE: bt.col(NODE), "betweenness": bt.lit(0.0)})
    summed = totals.group_by(NODE).agg(betweenness=bt.sum("delta"))
    return (
        g.nodes()
        .join(summed, on=NODE, how="left")
        .select(
            **{
                NODE: bt.col(NODE),
                "betweenness": bt.coalesce(bt.col("betweenness"), bt.lit(0.0)),
            }
        )
    )


def _rescale(g: Graph, raw: Dataset, seeds: list[object], *, normalized: bool) -> Dataset:
    """Networkx's `_rescale` with `endpoints=False`, for `k` sampled sources of `n` nodes.

    A source cannot lie between itself and a target, so it sat in only `k - 1` of the
    searches that could credit it, and networkx scales it by `k - 1` where every other
    node gets `k`. With every node a source the two agree, which is the exact case.
    """
    n = g.num_nodes()
    pairs = n - 1
    if pairs < 2:
        return raw
    k = len(seeds)
    if normalized:
        per_source = 1.0 / ((k - 1) * (pairs - 1)) if k > 1 else float("nan")
        per_other = 1.0 / (k * (pairs - 1))
    else:
        correction = 1.0 if g.directed else 2.0
        per_source = pairs / ((k - 1) * correction) if k > 1 else float("nan")
        per_other = pairs / (k * correction)
    marks = bt.from_pydict({NODE: seeds}).select(NODE, _is_source=bt.lit(True))
    factor = (
        bt.when(bt.col("_is_source").is_not_null())
        .then(bt.lit(per_source))
        .otherwise(bt.lit(per_other))
    )
    return raw.join(marks, on=NODE, how="left").select(
        NODE, betweenness=bt.col("betweenness") * factor
    )
