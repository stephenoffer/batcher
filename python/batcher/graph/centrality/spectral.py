"""The eigen-family centralities: eigenvector, Katz, and HITS.

All three are power iterations over the adjacency matrix, differing in what they add at
each step. They share `_power_iteration`, which is also where the normalization lives:
without it eigenvector centrality either explodes or collapses depending on the leading
eigenvalue, and the convergence test would measure a scale rather than a shape.
"""

from __future__ import annotations

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.api.dataset import Dataset
from batcher.graph._graph import DST, NODE, SRC, WEIGHT, Graph
from batcher.graph._iterate import iterate, max_abs_change

__all__ = ["eigenvector_centrality", "hits", "katz_centrality"]


def _check_iterations(max_iterations: int, tolerance: float) -> None:
    if max_iterations < 1:
        raise PlanError(f"max_iterations must be at least 1, got {max_iterations}")
    if tolerance < 0.0:
        raise PlanError(f"tolerance must be non-negative, got {tolerance}")


def _power_iteration(
    g: Graph,
    value: str,
    initial_value: float,
    combine: object,
    max_iterations: int,
    tolerance: float,
) -> Dataset:
    """Shared body of eigenvector and Katz centrality: propagate, combine, normalize."""
    nodes = g.nodes().cache()
    n = nodes.count()
    if n == 0:
        return nodes.select(**{NODE: bt.col(NODE), value: bt.lit(0.0)})
    initial = nodes.select(**{NODE: bt.col(NODE), value: bt.lit(initial_value)})

    def step(state: Dataset) -> Dataset:
        incoming = (
            g.edges.join(
                state.select(**{SRC: bt.col(NODE), "_v": bt.col(value)}), on=SRC, how="inner"
            )
            .select(**{NODE: bt.col(DST), "_in": bt.col("_v") * bt.col(WEIGHT)})
            .group_by(NODE)
            .agg(_in=bt.sum("_in"))
        )
        raw = nodes.join(incoming, on=NODE, how="left").select(
            **{NODE: bt.col(NODE), "_raw": combine(bt.coalesce(bt.col("_in"), bt.lit(0.0)))}
        )
        # Normalize to unit L2 norm. Without it eigenvector centrality either explodes or
        # collapses to zero depending on the leading eigenvalue, and the convergence test
        # measures a scale rather than a shape.
        norm = raw.agg(s=bt.sum(bt.col("_raw") * bt.col("_raw"))).to_pydict()["s"]
        total = float(norm[0]) ** 0.5 if norm and norm[0] else 0.0
        factor = 1.0 / total if total > 0.0 else 0.0
        return raw.select(**{NODE: bt.col(NODE), value: bt.col("_raw") * bt.lit(factor)})

    return iterate(
        initial,
        step,
        max_iterations=max_iterations,
        delta=max_abs_change(NODE, value),
        tolerance=tolerance,
    ).state


def eigenvector_centrality(
    g: Graph, *, max_iterations: int = 100, tolerance: float = 1e-6
) -> Dataset:
    """The leading eigenvector of the adjacency matrix, by power iteration.

    PageRank without the damping, which makes it the cleaner definition and the less
    robust one. On a strongly connected graph it is exactly what you want. On a graph with
    a sink component it concentrates the entire score there and reports zero for
    everything else, which is correct and useless; `katz_centrality` is the standard fix.

    Args:
        g: The graph.
        max_iterations: The cap on rounds.
        tolerance: Stop once no node's score moves by more than this.

    Returns:
        A dataset of `node` and `eigenvector_centrality`, normalized to unit L2 norm.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, eigenvector_centrality
            >>> e = bt.from_pydict({"src": [1, 2, 0], "dst": [0, 0, 1]})
            >>> out = eigenvector_centrality(Graph.from_edges(e))
            >>> len(out.to_pydict()["node"])
            3
    """
    _check_iterations(max_iterations, tolerance)
    return _power_iteration(
        g,
        "eigenvector_centrality",
        1.0,
        lambda incoming: incoming,
        max_iterations,
        tolerance,
    )


def katz_centrality(
    g: Graph,
    *,
    attenuation: float = 0.1,
    baseline: float = 1.0,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> Dataset:
    """Katz centrality: every path counts, with longer paths attenuated.

    Each node starts with `baseline` and adds `attenuation` times the score arriving along
    its incoming edges. Because every node keeps a floor, a node with no incoming edges
    still scores, which is exactly the failure mode `eigenvector_centrality` has.

    `attenuation` must be smaller than the reciprocal of the graph's largest eigenvalue
    for the series to converge. There is no cheap way to know that value in advance, so
    the practical rule is to start small: 0.1 converges on most real graphs, and a result
    whose scores grow without bound between rounds means it was too large.

    Args:
        g: The graph.
        attenuation: How much of a neighbour's score propagates. Smaller is more local
            and more likely to converge.
        baseline: The score every node starts with and keeps.
        max_iterations: The cap on rounds.
        tolerance: Stop once no node's score moves by more than this.

    Returns:
        A dataset of `node` and `katz_centrality`, normalized to unit L2 norm.

    Raises:
        PlanError: If `attenuation` is not positive.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, katz_centrality
            >>> e = bt.from_pydict({"src": [1, 2, 3], "dst": [0, 0, 0]})
            >>> out = katz_centrality(Graph.from_edges(e)).sort("katz_centrality")
            >>> out.to_pydict()["node"][-1]
            0
    """
    if attenuation <= 0.0:
        raise PlanError(f"attenuation must be positive, got {attenuation}")
    _check_iterations(max_iterations, tolerance)
    return _power_iteration(
        g,
        "katz_centrality",
        baseline,
        lambda incoming: bt.lit(baseline) + bt.lit(attenuation) * incoming,
        max_iterations,
        tolerance,
    )


def hits(g: Graph, *, max_iterations: int = 100, tolerance: float = 1e-6) -> Dataset:
    """HITS: mutually reinforcing hub and authority scores.

    A good authority is pointed at by good hubs; a good hub points at good authorities.
    The distinction PageRank cannot make: on a citation or link graph, a survey paper and
    a seminal paper are both important, and they are important in opposite directions.

    Args:
        g: The graph.
        max_iterations: The cap on rounds.
        tolerance: Stop once no node's authority moves by more than this.

    Returns:
        A dataset of `node`, `hub` and `authority`, each normalized to unit L2 norm.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.graph import Graph, hits
            >>> e = bt.from_pydict({"src": [1, 2, 3], "dst": [0, 0, 0]})
            >>> out = hits(Graph.from_edges(e)).sort("authority", descending=True)
            >>> out.to_pydict()["node"][0]
            0
    """
    _check_iterations(max_iterations, tolerance)
    nodes = g.nodes().cache()
    if nodes.count() == 0:
        return nodes.select(**{NODE: bt.col(NODE), "hub": bt.lit(0.0), "authority": bt.lit(0.0)})
    initial = nodes.select(**{NODE: bt.col(NODE), "hub": bt.lit(1.0), "authority": bt.lit(1.0)})

    def normalize(state: Dataset, column: str) -> Dataset:
        got = state.agg(s=bt.sum(bt.col(column) * bt.col(column))).to_pydict()["s"]
        total = float(got[0]) ** 0.5 if got and got[0] else 0.0
        factor = 1.0 / total if total > 0.0 else 0.0
        return state.with_columns(**{column: bt.col(column) * bt.lit(factor)})

    def step(state: Dataset) -> Dataset:
        # Authority: sum of the hub scores pointing in.
        auth = (
            g.edges.join(
                state.select(**{SRC: bt.col(NODE), "_h": bt.col("hub")}), on=SRC, how="inner"
            )
            .select(**{NODE: bt.col(DST), "_a": bt.col("_h") * bt.col(WEIGHT)})
            .group_by(NODE)
            .agg(_a=bt.sum("_a"))
        )
        with_auth = nodes.join(auth, on=NODE, how="left").select(
            **{NODE: bt.col(NODE), "authority": bt.coalesce(bt.col("_a"), bt.lit(0.0))}
        )
        with_auth = normalize(with_auth, "authority")
        # Hub: sum of the authority scores pointed at.
        hub = (
            g.edges.join(
                with_auth.select(**{DST: bt.col(NODE), "_a": bt.col("authority")}),
                on=DST,
                how="inner",
            )
            .select(**{NODE: bt.col(SRC), "_h": bt.col("_a") * bt.col(WEIGHT)})
            .group_by(NODE)
            .agg(_h=bt.sum("_h"))
        )
        combined = with_auth.join(hub, on=NODE, how="left").select(
            **{
                NODE: bt.col(NODE),
                "hub": bt.coalesce(bt.col("_h"), bt.lit(0.0)),
                "authority": bt.col("authority"),
            }
        )
        return normalize(combined, "hub")

    return iterate(
        initial,
        step,
        max_iterations=max_iterations,
        delta=max_abs_change(NODE, "authority"),
        tolerance=tolerance,
    ).state
