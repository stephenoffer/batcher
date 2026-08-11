"""Joins, CTEs and subqueries in SQL over real TPC-H tables.

A CTE is not a materialization hint here: it names a subplan, and the optimizer is free to
inline it, push predicates through it, or reorder around it. Write the query for clarity
and let the plan decide.

    python examples/sql_queries/joins_and_subqueries.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch


def main() -> None:
    customer = tpch("customer")
    orders = tpch("orders")
    nation = tpch("nation")

    result = bt.sql(
        """
        WITH big_spenders AS (
            SELECT o_custkey, SUM(o_totalprice) AS spend
            FROM orders
            GROUP BY o_custkey
            HAVING SUM(o_totalprice) > 300000
        )
        SELECT n.n_name, COUNT(*) AS customers, SUM(b.spend) AS total_spend
        FROM big_spenders b
        JOIN customer c ON c.c_custkey = b.o_custkey
        JOIN nation n ON n.n_nationkey = c.c_nationkey
        GROUP BY n.n_name
        ORDER BY total_spend DESC
        """,
        customer=customer,
        orders=orders,
        nation=nation,
    ).to_pydict()

    print(result["n_name"][:5])
    print([round(value) for value in result["total_spend"][:5]])

    assert result["total_spend"] == sorted(result["total_spend"], reverse=True)
    assert all(count > 0 for count in result["customers"])

    # Every nation named is a real one.
    assert set(result["n_name"]) <= set(nation.to_pydict()["n_name"])

    # A scalar subquery in the WHERE clause.
    above_average = bt.sql(
        """
        SELECT COUNT(*) AS n
        FROM orders
        WHERE o_totalprice > (SELECT AVG(o_totalprice) FROM orders)
        """,
        orders=orders,
    ).to_pydict()
    print("orders above the mean:", above_average["n"][0])
    assert 0 < above_average["n"][0] < orders.count()


if __name__ == "__main__":
    main()
