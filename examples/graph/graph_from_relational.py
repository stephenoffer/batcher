"""Building a graph out of an ordinary relational table.

A graph is an edge table with two columns you have chosen to call source and target. Seeing
it that way is what makes graph analytics available to any dataset with a foreign key —
there is no separate ingestion.

Materialize the edge list before running the iterative algorithms. They re-read the edges
once per iteration, so handing them a join-and-distinct plan re-executes that plan on every
round — the cost is the whole derivation multiplied by the number of iterations.

    python examples/graph/graph_from_relational.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col
from batcher import graph as bg


def main() -> None:
    # Two orders are connected when the same part appears in both. The edge table is a
    # self-join on the part key.
    lines = tpch("lineitem").select("l_orderkey", "l_partkey").head(4_000)

    right = lines.select(
        col("l_orderkey").alias("other_order"), col("l_partkey").alias("other_part")
    )
    edges = (
        lines.join(right, left_on="l_partkey", right_on="other_part")
        .filter(col("l_orderkey") < col("other_order"))
        .select(
            src=col("l_orderkey").cast("string"),
            dst=col("other_order").cast("string"),
        )
        .distinct()
    )
    print("edges:", edges.count())
    assert edges.count() > 0

    # Materialize once; every algorithm below reads this rather than the plan.
    materialized = bt.from_pydict(edges.to_pydict())
    assert materialized.count() == edges.count()

    graph = bg.Graph.from_edges(materialized)
    summary = bg.summarize(graph).to_pydict()
    print("summary:", summary)

    degrees = bg.degree(graph).sort("degree", descending=True).to_pydict()
    print("most connected orders:", degrees["node"][:5])

    # The handshake identity holds on the constructed graph.
    assert sum(degrees["degree"]) == 2 * materialized.count()

    # Every node in the graph is an order key from the source table.
    order_keys = {
        str(value) for value in lines.select("l_orderkey").distinct().to_pydict()["l_orderkey"]
    }
    assert set(degrees["node"]) <= order_keys

    # Components: orders that share no part with anything are absent from the edge table
    # entirely, which is why the graph has fewer nodes than the table has orders.
    components = bg.connected_components(graph).to_pydict()
    print("components:", len(set(components["component"])))
    assert len(components["node"]) == len(degrees["node"])
    assert len(degrees["node"]) <= len(order_keys)


if __name__ == "__main__":
    main()
