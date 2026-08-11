"""Shortest paths on a small weighted graph.

Distances have properties you can assert without knowing the answer: zero to yourself,
symmetric on an undirected graph, and the triangle inequality. Checking those is how you
find out an algorithm ran on the graph you meant rather than on an edge list with a typo.

    python examples/graph/shortest_paths.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import graph as bg


def main() -> None:
    edges = bt.from_pydict(
        {
            "src": ["a", "b", "c", "a", "d"],
            "dst": ["b", "c", "d", "d", "e"],
            "weight": [1.0, 1.0, 1.0, 5.0, 2.0],
        }
    )
    graph = bg.Graph.from_edges(edges, weight="weight")

    # The source is a Dataset of seed nodes, not a string — the same call does a
    # multi-source search, which is what you want for "distance to the nearest depot".
    seeds = bt.from_pydict({"node": ["a"]})
    distances = bg.shortest_path_lengths(graph, seeds).sort("node").to_pydict()
    print(distances)

    measure = next(name for name in distances if name != "node")
    reachable = dict(zip(distances["node"], distances[measure], strict=True))

    # Zero to yourself.
    assert reachable["a"] == 0.0

    # The direct a->d edge costs 5; a->b->c->d costs 3, so the algorithm must find 3.
    assert reachable["d"] == 3.0

    # And e is one more hop beyond d.
    assert reachable["e"] == 5.0

    # The triangle inequality holds along every edge.
    edge_rows = edges.to_pydict()
    for source, target, weight in zip(
        edge_rows["src"], edge_rows["dst"], edge_rows["weight"], strict=True
    ):
        if source in reachable and target in reachable:
            assert reachable[target] <= reachable[source] + weight + 1e-9

    # Every distance is non-negative.
    assert all(value >= 0 for value in distances[measure])


if __name__ == "__main__":
    main()
