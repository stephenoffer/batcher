"""Local structure: triangles and the clustering coefficient.

A triangle is three mutually connected nodes, and the clustering coefficient is how many of
a node's neighbours know each other. Together they say whether a graph has communities
worth detecting or is just a hub-and-spoke star.

    python examples/graph/triangles_and_clustering.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import graph as bg


def main() -> None:
    # Two triangles joined by a bridge, plus a pendant node.
    edges = bt.from_pydict(
        {
            "src": ["a", "b", "a", "d", "e", "d", "c", "a"],
            "dst": ["b", "c", "c", "e", "f", "f", "d", "z"],
        }
    )
    graph = bg.Graph.from_edges(edges)

    summary = bg.summarize(graph).to_pydict()
    print("summary:", summary)

    degrees = bg.degree(graph).sort("degree", descending=True).to_pydict()
    print("degrees:", dict(zip(degrees["node"], degrees["degree"], strict=True)))

    # `a` sits in a triangle and also carries the pendant, so it has the highest degree.
    by_node = dict(zip(degrees["node"], degrees["degree"], strict=True))
    assert by_node["a"] == 3
    assert by_node["z"] == 1

    # The handshake identity.
    assert sum(degrees["degree"]) == 2 * edges.count()

    # Components: everything is reachable, so there is one.
    components = bg.connected_components(graph).to_pydict()
    assert len(set(components["component"])) == 1
    assert len(components["node"]) == 7

    # Each triangle is counted once per member: a, b, c and d, e, f are in one each.
    triangles = bg.triangle_count(graph).sort("node").to_pydict()
    print("triangles:", triangles)
    assert triangles == {
        "node": ["a", "b", "c", "d", "e", "f", "z"],
        "triangles": [1, 1, 1, 1, 1, 1, 0],
    }

    # Clustering by hand. `a` has neighbours b, c, z: three pairs, one closed, so 1/3. `c`
    # and `d` each have three neighbours and one closed pair; `b`, `e`, `f` have two and
    # one. `z` has a single neighbour and so no pair, which scores 0.0.
    clustering = bg.clustering_coefficient(graph).sort("node").to_pydict()["clustering"]
    assert [round(v, 4) for v in clustering] == [0.3333, 1.0, 0.3333, 0.3333, 1.0, 1.0, 0.0]

    # Transitivity counts triples instead: 3+1+3+3+1+1+0 = 12 connected triples, of which
    # 3 x 2 triangles are closed, so 0.5. The average of the coefficients above is 0.5714,
    # and the gap between the two is the point of reporting both.
    assert bg.transitivity(graph) == 0.5
    assert round(bg.average_clustering(graph), 4) == 0.5714


if __name__ == "__main__":
    main()
