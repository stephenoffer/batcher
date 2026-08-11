"""Naming intermediate results: CTEs in a query, views in a catalog.

A CTE is scoped to its query; a view outlives it. Both are names for a plan rather than for
data, so neither materializes anything until a terminal operation runs — which is why a view
over a huge table costs nothing to define.

    python examples/sql_queries/ctes_and_views.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch


def main() -> None:
    lineitem = tpch("lineitem")
    orders = tpch("orders")

    # Two CTEs, the second reading the first.
    result = bt.sql(
        """
        WITH per_order AS (
            SELECT l_orderkey, SUM(l_extendedprice) AS revenue, COUNT(*) AS lines
            FROM lineitem
            GROUP BY l_orderkey
        ),
        large AS (
            SELECT * FROM per_order WHERE lines >= 4
        )
        SELECT o.o_orderpriority, COUNT(*) AS orders, SUM(l.revenue) AS revenue
        FROM large l
        JOIN orders o ON o.o_orderkey = l.l_orderkey
        GROUP BY o.o_orderpriority
        ORDER BY o.o_orderpriority
        """,
        lineitem=lineitem,
        orders=orders,
    ).to_pydict()
    print(result)

    assert result["o_orderpriority"] == sorted(result["o_orderpriority"])
    assert sum(result["orders"]) > 0

    # A view: defined once, referenced later without re-binding the inputs.
    session = bt.Session()
    session.sql(
        "CREATE OR REPLACE VIEW big_orders AS "
        "SELECT o_orderkey, o_totalprice FROM orders WHERE o_totalprice > 250000",
        orders=orders,
    )
    from_view = session.sql("SELECT COUNT(*) AS n FROM big_orders").to_pydict()
    print("rows in the view:", from_view["n"][0])
    assert from_view["n"][0] > 0

    # Defining a view runs nothing: it is a name for a plan.
    again = session.sql("SELECT MAX(o_totalprice) AS top FROM big_orders").to_pydict()
    assert again["top"][0] > 250_000

    session.sql("DROP VIEW big_orders")


if __name__ == "__main__":
    main()
