"""Which endpoint an undirected edge is written with must not change the answer.

An undirected edge ``{0, 2}`` can be stored as ``(0, 2)`` or ``(2, 0)``; the row order carries
no information. `_canonical_edges` is what normalizes that for the triangle algorithms, and it
used to *filter* ``src < dst`` rather than *orient* to it. Filtering is equivalent to orienting
only when both directions of every edge are present -- true after `to_undirected` symmetrizes
a **directed** graph, false for one the caller built with ``directed=False``, which
`to_undirected` returns unchanged. So every edge written larger-endpoint-first was silently
dropped and `triangle_count` answered zero for a graph full of triangles.

`clustering_coefficient` had the same cause by a second route: it took its neighbour count
from ``out_degree(g.to_undirected())``, which is the neighbour count only when that call
symmetrized.

networkx is the oracle here rather than DuckDB: these are graph algorithms with a standard
definition and an independent implementation to check against, which is a stronger check than
an internal one.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.graph import Graph, clustering_coefficient, triangle_count

nx = pytest.importorskip("networkx", reason="the independent graph oracle")

#: One triangle {0,1,2}, written three ways. The graph is identical in all three.
SPELLINGS = {
    "all_ascending": [(0, 1), (1, 2), (0, 2)],
    "one_descending": [(0, 1), (1, 2), (2, 0)],
    "all_descending": [(1, 0), (2, 1), (2, 0)],
}

#: Two components: a five-node blob with several triangles, and a disjoint triangle.
MIXED = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 0), (1, 4), (10, 11), (11, 12), (12, 10)]


def _graph(edges, *, directed):
    table = bt.from_pydict({"src": [a for a, _ in edges], "dst": [b for _, b in edges]})
    return Graph.from_edges(table, directed=directed)


def _nx(edges):
    g = nx.Graph()
    g.add_edges_from(edges)
    return g


@pytest.mark.parametrize("spelling", sorted(SPELLINGS))
@pytest.mark.parametrize("directed", [True, False], ids=["directed", "undirected"])
def test_triangle_count_ignores_how_each_edge_was_written(spelling, directed):
    """All three spellings are one triangle, so all three must count one per node."""
    got = triangle_count(_graph(SPELLINGS[spelling], directed=directed)).sort("node").to_pydict()
    assert got["triangles"] == [1, 1, 1], f"{spelling} lost its triangle"


@pytest.mark.parametrize("directed", [True, False], ids=["directed", "undirected"])
def test_triangle_count_matches_networkx_on_a_mixed_graph(directed):
    """The edge list mixes both spellings and spans two components."""
    got = triangle_count(_graph(MIXED, directed=directed)).sort("node").to_pydict()
    expected = _nx(MIXED)
    assert got["node"] == sorted(expected)
    assert got["triangles"] == [nx.triangles(expected)[n] for n in sorted(expected)]


@pytest.mark.parametrize("spelling", sorted(SPELLINGS))
def test_clustering_coefficient_ignores_how_each_edge_was_written(spelling):
    """Every node of a triangle has clustering 1.0, whichever way the edges were written."""
    got = (
        clustering_coefficient(_graph(SPELLINGS[spelling], directed=False)).sort("node").to_pydict()
    )
    assert [round(c, 6) for c in got["clustering"]] == [1.0, 1.0, 1.0]


@pytest.mark.parametrize("directed", [True, False], ids=["directed", "undirected"])
def test_clustering_coefficient_matches_networkx_on_a_mixed_graph(directed):
    got = clustering_coefficient(_graph(MIXED, directed=directed)).sort("node").to_pydict()
    expected = _nx(MIXED)
    assert [round(c, 6) for c in got["clustering"]] == [
        round(nx.clustering(expected)[n], 6) for n in sorted(expected)
    ]


def test_the_documented_examples_still_hold():
    """The directed default was always correct; the fix must not have moved it.

    Both docstrings publish these numbers, and `just docs` executes them -- so a change that
    fixed the undirected case by breaking the directed one would fail the build rather than
    this test. Pinning it here fails it faster and says why.
    """
    edges = bt.from_pydict({"src": [1, 2, 1, 3], "dst": [2, 3, 3, 4]})
    graph = Graph.from_edges(edges)
    assert triangle_count(graph).sort("node").to_pydict()["triangles"] == [1, 1, 1, 0]
    clustering = clustering_coefficient(graph).sort("node").to_pydict()["clustering"]
    assert [round(c, 4) for c in clustering] == [1.0, 1.0, 0.3333, 0.0]
