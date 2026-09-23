"""Cut a graph down to the part an algorithm should actually run on.

Most graph work starts by narrowing: a PageRank over a whole interaction log answers a
question nobody asked, and a self-loop or an isolated node quietly changes the result. Three
reshaping operations do that narrowing, and each returns a new `Graph`, so they compose
before any algorithm reads the edges.

`subgraph` keeps the edges whose *both* endpoints are in a node set — the induced subgraph,
not the edges merely touching it. `without_self_loops` drops `a -> a`, which centrality
measures otherwise count as real endorsement. `extra_nodes` hands back the node roster
attached by `with_nodes`, and `None` when there is none, so a caller can tell an
edge-derived node set from a supplied one without comparing them.

    python examples/graph/reshaping_a_graph.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher.graph import Graph


def main() -> None:
    # 1->2, 2->3, 3->1 is a cycle; 3->3 is a self-loop; 4 appears only as a destination.
    edges = bt.from_pydict({"src": [1, 2, 3, 3, 1], "dst": [2, 3, 1, 3, 4]})
    g = Graph.from_edges(edges, src="src", dst="dst")
    assert g.num_edges() == 5

    # A self-loop is an edge like any other until you say otherwise, and it is the one that
    # flatters a node's own centrality.
    clean = g.without_self_loops()
    assert clean.num_edges() == 4
    remaining = clean.edges.to_pydict()
    assert not any(s == d for s, d in zip(remaining["src"], remaining["dst"], strict=True)), (
        remaining
    )

    # `subgraph` is *induced*: the 1->4 edge goes even though 1 is in the set, because 4 is
    # not. That is the difference from "edges touching these nodes" and it is usually the
    # one you want — the alternative leaves dangling endpoints the algorithms then count.
    core = clean.subgraph(bt.from_pydict({"node": [1, 2, 3]}))
    kept = core.edges.to_pydict()
    pairs = sorted(zip(kept["src"], kept["dst"], strict=True))
    assert pairs == [(1, 2), (2, 3), (3, 1)], pairs

    # The node set is derived from the edges, so an account with no edge is invisible to it:
    # node 99 exists in the roster and in no edge, and `nodes()` alone would never mention it.
    assert sorted(g.nodes().to_pydict()["node"]) == [1, 2, 3, 4]
    assert g.extra_nodes() is None, "an edge-derived graph carries no supplied roster"

    roster = bt.from_pydict({"node": [1, 2, 3, 4, 99]})
    with_isolated = g.with_nodes(roster)
    assert sorted(with_isolated.nodes().to_pydict()["node"]) == [1, 2, 3, 4, 99]

    # `extra_nodes` returns the roster itself, not the difference against the edges — the
    # signal is "this graph's nodes came from somewhere other than its edges", which is what
    # an algorithm needs to know before it reports a node count.
    extra = with_isolated.extra_nodes()
    assert extra is not None
    assert sorted(extra.to_pydict()["node"]) == [1, 2, 3, 4, 99], extra.to_pydict()

    print(
        f"edges {g.num_edges()} -> {clean.num_edges()} without self-loops "
        f"-> {core.num_edges()} in the induced core"
    )
    print("nodes from the edges:", sorted(g.nodes().to_pydict()["node"]))
    print("nodes with the roster:", sorted(with_isolated.nodes().to_pydict()["node"]))


if __name__ == "__main__":
    main()
