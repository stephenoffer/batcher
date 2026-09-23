"""`batcher.graph.degree_distribution` against the same histogram written in DuckDB SQL.

The result is sorted by degree, so every comparison here is ordered. The edge table carries
the shapes a degree count is easy to get wrong on: a self-loop (it touches its node twice),
a parallel edge (counted twice until `Graph.simple`), and, via `with_nodes`, a declared node
no edge names, which must appear with degree zero rather than vanish from the histogram.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from _harness import assert_same_ordered

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")
from batcher.graph import Graph, degree_distribution  # noqa: E402

EDGES = pa.table(
    {
        "src": pa.array([1, 1, 2, 3, 3, 4, 1], pa.int64()),
        "dst": pa.array([2, 3, 3, 3, 4, 1, 2], pa.int64()),  # 3->3 self-loop, 1->2 twice
    }
)

_HISTOGRAM = """
    SELECT degree, count(*) AS nodes FROM (
        SELECT n, sum(c) AS degree FROM (
            SELECT src AS n, 1 AS c FROM e UNION ALL SELECT dst, 1 FROM e
            {extra}
        ) GROUP BY n
    ) GROUP BY degree ORDER BY degree
"""


def test_degree_distribution_matches_duckdb(duck):
    duck.register("e", EDGES)
    out = degree_distribution(Graph.from_edges(bt.from_arrow(EDGES))).collect()
    assert out.column_names == ["degree", "nodes"]
    assert_same_ordered(out, duck.sql(_HISTOGRAM.format(extra="")))


def test_an_isolated_declared_node_lands_in_the_zero_bucket(duck):
    nodes = pa.table({"node": pa.array([1, 2, 3, 4, 9, 10], pa.int64())})
    duck.register("e", EDGES)
    duck.register("v", nodes)
    g = Graph.from_edges(bt.from_arrow(EDGES)).with_nodes(bt.from_arrow(nodes))
    out = degree_distribution(g).collect()
    expected = duck.sql(_HISTOGRAM.format(extra="UNION ALL SELECT node, 0 FROM v"))
    assert_same_ordered(out, expected)
    assert out.to_pydict()["degree"][0] == 0  # the positive control: the zero bucket exists


def test_degree_distribution_of_one_edge_and_of_no_edges():
    one = pa.table({"src": pa.array([5], pa.int64()), "dst": pa.array([6], pa.int64())})
    out = degree_distribution(Graph.from_edges(bt.from_arrow(one))).to_pydict()
    assert out == {"degree": [1], "nodes": [2]}

    empty = one.slice(0, 0)
    assert degree_distribution(Graph.from_edges(bt.from_arrow(empty))).collect().num_rows == 0
