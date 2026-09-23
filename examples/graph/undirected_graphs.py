"""An undirected graph, and the answers you can check by hand on it.

`Graph.from_edges(directed=False)` stores each edge once, written either way round, and every
algorithm walks it in both directions. That is the whole difference from a directed graph,
and on a three-node path it is easy to see: a search from either end reaches the other, and
PageRank gives the two ends the same score.

The numbers asserted below are worked out in the comments, so the example fails if an
algorithm ever reads the edge table one way only, or stops before it has finished.

    python examples/graph/undirected_graphs.py
"""

from __future__ import annotations

import warnings

import batcher as bt
from batcher import graph as bg


def path_of_three() -> None:
    # 1 - 2 - 3, with the second edge deliberately written from its far end.
    edges = bt.from_pydict({"src": [1, 3], "dst": [2, 2]})
    path = bg.Graph.from_edges(edges, directed=False)

    # From node 1, node 2 is one hop away and node 3 two. Read one way only, the search
    # would stop at node 2, because no row starts there.
    depth = bg.bfs(path, bt.from_pydict({"node": [1]})).sort("node").to_pydict()
    print("hops from 1:", depth)
    assert depth == {"node": [1, 2, 3], "depth": [0, 1, 2]}

    # PageRank by hand. The ends are symmetric, so p1 = p3, and with damping 0.85 over three
    # nodes:  p1 = 0.05 + 0.85 * p2 / 2  and  p2 = 0.05 + 0.85 * (p1 + p3).
    # Substituting gives p2 = 0.135 / 0.2775 = 0.48649 and p1 = p3 = 0.25676.
    ranks = bg.pagerank(path).sort("node").to_pydict()["pagerank"]
    print("pagerank:", [round(r, 5) for r in ranks])
    assert [round(r, 4) for r in ranks] == [0.2568, 0.4865, 0.2568]


def triangle_with_a_loop_and_a_stray() -> None:
    # A triangle, a self-loop on node 1, and node 9 declared with no edges at all.
    edges = bt.from_pydict({"src": [1, 2, 3, 1], "dst": [2, 3, 1, 1]})
    g = bg.Graph.from_edges(edges, directed=False).with_nodes(
        bt.from_pydict({"node": [1, 2, 3, 9]})
    )

    # A self-loop is not a neighbour, so the triangle is still fully closed: 3 closed
    # triples out of 3 connected ones.
    assert bg.transitivity(g) == 1.0

    # Node 9 has no neighbour pair, so its clustering is 0.0 rather than missing, and the
    # average is over all four nodes: (1 + 1 + 1 + 0) / 4.
    clustering = bg.clustering_coefficient(g).sort("node").to_pydict()
    print("clustering:", clustering)
    assert clustering == {"node": [1, 2, 3, 9], "clustering": [1.0, 1.0, 1.0, 0.0]}
    assert bg.average_clustering(g) == 0.75


def a_long_chain() -> None:
    # Exact algorithms run until they are finished, however long the graph. A 150-node chain
    # is one component, and its far end is 149 hops from the start.
    n = 150
    chain = bg.Graph.from_edges(
        bt.from_pydict({"src": list(range(n - 1)), "dst": list(range(1, n))}), directed=False
    )
    labels = bg.connected_components(chain)
    assert labels.count_distinct("component") == 1

    # A cap you set is a promise about the answer, so one too small for the graph raises
    # rather than returning a chain split in pieces.
    try:
        bg.connected_components(chain, max_iterations=2)
    except bt.PlanError as refused:
        print("refused:", str(refused).split(".")[0])
    else:
        raise AssertionError("a truncated component labelling was returned")

    # An approximate algorithm returns its last iterate instead, and says so.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bg.pagerank(chain, max_iterations=2)
    assert [w.category for w in caught] == [bg.ConvergenceWarning]


def main() -> None:
    path_of_three()
    triangle_with_a_loop_and_a_stray()
    a_long_chain()


if __name__ == "__main__":
    main()
