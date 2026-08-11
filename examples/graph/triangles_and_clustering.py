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

    # Triangle counting, where available.
    if hasattr(bg, "triangle_count"):
        triangles = bg.triangle_count(graph).to_pydict()
        print("triangles:", triangles)
        counts = dict(
            zip(
                triangles["node"],
                triangles[next(name for name in triangles if name != "node")],
                strict=True,
            )
        )
        # `a`, `b` and `c` form one; `d`, `e` and `f` form another.
        assert counts["a"] >= 1
        assert counts["z"] == 0
    else:
        print("triangle_count not available in this build")


if __name__ == "__main__":
    main()
