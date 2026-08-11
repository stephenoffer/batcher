"""Ranking nodes by influence on a real bipartite graph.

PageRank is worth running only once the diagnostics say the graph is connected enough for it
to mean anything. The scores form a distribution, so the check is that they sum to one and
that the ranking is stable, not that any particular node wins.

    python examples/graph/pagerank_and_centrality.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import graph as bg


def main() -> None:
    # Customers linked to their nations: a real bipartite graph with strong hubs.
    edges = (
        tpch("customer")
        .head(3_000)
        .select(
            src=bt.col("c_custkey").cast("string"),
            dst=bt.col("c_nationkey").cast("string"),
        )
    )
    graph = bg.Graph.from_edges(edges)

    ranked = bg.pagerank(graph).sort("pagerank", descending=True).to_pydict()
    print("top nodes:", ranked["node"][:5])
    print("their scores:", [round(value, 6) for value in ranked["pagerank"][:5]])

    # A probability distribution over nodes.
    assert abs(sum(ranked["pagerank"]) - 1.0) < 1e-3
    assert all(value > 0 for value in ranked["pagerank"])
    assert ranked["pagerank"] == sorted(ranked["pagerank"], reverse=True)

    # The 25 nation nodes are hubs, so they should dominate the top of the ranking.
    nations = {str(value) for value in range(25)}
    top_twenty = set(ranked["node"][:20])
    assert len(top_twenty & nations) >= 15

    # Running it again gives the same ranking — it is deterministic.
    again = bg.pagerank(graph).sort("pagerank", descending=True).to_pydict()
    assert again["node"][:10] == ranked["node"][:10]


if __name__ == "__main__":
    main()
