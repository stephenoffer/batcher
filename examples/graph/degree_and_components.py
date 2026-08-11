"""The cheap graph diagnostics that decide what is worth running next.

Degree distribution and connected components cost one pass each and tell you whether an
expensive algorithm will even be meaningful. A graph that is mostly isolated nodes does not
need PageRank; it needs a better edge definition.

    python examples/graph/degree_and_components.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import graph as bg


def main() -> None:
    # A real bipartite graph: customers linked to the nations they belong to.
    edges = (
        tpch("customer")
        .head(2_000)
        .select(
            src=bt.col("c_custkey").cast("string"),
            dst=bt.col("c_nationkey").cast("string"),
        )
    )

    graph = bg.Graph.from_edges(edges)

    summary = bg.summarize(graph).to_pydict()
    print(summary)

    degrees = bg.degree(graph).sort("degree", descending=True).to_pydict()
    print("highest degree nodes:", degrees["node"][:5])
    print("their degrees:", degrees["degree"][:5])

    assert degrees["degree"] == sorted(degrees["degree"], reverse=True)
    # Every customer has exactly one edge; the 25 nations absorb them all, so the top
    # degrees belong to nations.
    assert max(degrees["degree"]) > 25
    assert min(degrees["degree"]) == 1

    # The handshake identity: the degrees sum to twice the edge count.
    assert sum(degrees["degree"]) == 2 * edges.count()

    components = bg.connected_components(graph).to_pydict()
    distinct = len(set(components["component"]))
    print("connected components:", distinct)
    assert distinct >= 1
    assert len(components["node"]) == len(degrees["node"])


if __name__ == "__main__":
    main()
