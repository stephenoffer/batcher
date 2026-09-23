"""Ranking nodes by influence on a real bipartite graph.

PageRank is worth running only once the diagnostics say the graph is connected enough for it
to mean anything. On this graph the scores can also be derived by hand, so the example checks
the values against that derivation rather than only that they sum to one.

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
    # Customers linked to their nations: a real bipartite graph with strong hubs. The ids are
    # prefixed because customer 5 and nation 5 are different nodes.
    edges = (
        tpch("customer")
        .limit(3_000)
        .select(
            src=bt.concat_str(bt.lit("c"), bt.col("c_custkey").cast("string")),
            dst=bt.concat_str(bt.lit("n"), bt.col("c_nationkey").cast("string")),
        )
    )
    graph = bg.Graph.from_edges(edges)

    ranked = bg.pagerank(graph).sort("pagerank", descending=True).to_pydict()
    print("top nodes:", ranked["node"][:5])
    print("their scores:", [round(value, 6) for value in ranked["pagerank"][:5]])

    # A probability distribution over nodes, with the 25 nation hubs on top.
    assert abs(sum(ranked["pagerank"]) - 1.0) < 1e-6
    assert {node[0] for node in ranked["node"][:25]} == {"n"}

    # The values themselves can be worked out on paper. Every customer has one outgoing edge
    # and none incoming, so all customers share one score `c`: the teleport share plus a
    # share of the mass the nations, which have no outgoing edges, hand back. A nation gets
    # that same amount plus 0.85 of each of its customers' scores, so it scores exactly
    # c * (1 + 0.85 * customers). The default tolerance stops each score within about 1e-5
    # of its fixed point, so the check allows a relative 1e-3, which is still several times
    # tighter than what separates a nation with one more customer.
    scores = dict(zip(ranked["node"], ranked["pagerank"], strict=True))
    customers = dict(zip(*bg.in_degree(graph).to_pydict().values(), strict=True))
    customer_scores = [v for k, v in scores.items() if k.startswith("c")]
    c = customer_scores[0]
    assert max(customer_scores) - min(customer_scores) < 1e-12
    for nation in (k for k in scores if k.startswith("n")):
        expected = c * (1.0 + 0.85 * customers[nation])
        assert abs(scores[nation] - expected) < 1e-3 * expected, (nation, scores[nation])

    # Running it again gives the same ranking: it is deterministic.
    again = bg.pagerank(graph).sort("pagerank", descending=True).to_pydict()
    assert again["node"][:10] == ranked["node"][:10]


if __name__ == "__main__":
    main()
