"""Date and string functions in SQL over real data.

The function names follow DuckDB, which is the dialect the parser reads by default. When a
function is missing the error names it, and the DataFrame equivalent is usually one
accessor call away — so a gap is a detour, not a wall.

    python examples/sql_queries/date_and_string_functions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders")

    result = bt.sql(
        """
        SELECT
            EXTRACT(YEAR FROM o_orderdate) AS order_year,
            UPPER(o_orderstatus) AS status,
            COUNT(*) AS orders,
            ROUND(AVG(o_totalprice), 2) AS avg_price
        FROM orders
        GROUP BY order_year, status
        ORDER BY order_year, status
        """,
        orders=orders,
    ).to_pydict()
    print(result["order_year"][:4], result["status"][:4])

    assert sum(result["orders"]) == orders.count()
    assert all(value == value.upper() for value in result["status"])
    assert result["order_year"] == sorted(result["order_year"])

    # The DataFrame spelling gives the same answer.
    equivalent = (
        orders.with_columns(
            order_year=col("o_orderdate").dt.year(),
            status=col("o_orderstatus").str.to_uppercase(),
        )
        .group_by("order_year", "status")
        .agg(orders=bt.count(), avg_price=col("o_totalprice").mean().round(2))
        .sort("order_year", "status")
        .to_pydict()
    )
    assert equivalent["orders"] == result["orders"]
    assert equivalent["order_year"] == result["order_year"]

    # String predicates in SQL.
    matching = bt.sql(
        "SELECT COUNT(*) AS n FROM orders WHERE o_clerk LIKE 'Clerk#00000001%'",
        orders=orders,
    ).to_pydict()
    print("clerks matching the prefix:", matching["n"][0])
    direct = orders.filter(col("o_clerk").str.starts_with("Clerk#00000001")).count()
    assert matching["n"][0] == direct


if __name__ == "__main__":
    main()
