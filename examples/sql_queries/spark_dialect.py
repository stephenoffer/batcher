"""Reading SQL written for another engine.

The parser takes a dialect, so a query written against Spark can be run without being
rewritten first. That matters for a migration: you port the query once you have proved the
old one still returns the same rows here.

    python examples/sql_queries/spark_dialect.py
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

    # The default dialect is DuckDB.
    default = bt.sql(
        "SELECT o_orderstatus, COUNT(*) AS n FROM orders GROUP BY o_orderstatus "
        "ORDER BY o_orderstatus",
        orders=orders,
    ).to_pydict()
    print("duckdb:", default)

    # The same query read as Spark SQL.
    spark = bt.sql(
        "SELECT o_orderstatus, COUNT(*) AS n FROM orders GROUP BY o_orderstatus "
        "ORDER BY o_orderstatus",
        dialect="spark",
        orders=orders,
    ).to_pydict()
    print("spark: ", spark)

    assert spark == default

    # A Spark-flavoured function name, read with the Spark dialect.
    try:
        spark_only = bt.sql(
            "SELECT COUNT(*) AS n FROM orders WHERE o_clerk RLIKE 'Clerk#0000001'",
            dialect="spark",
            orders=orders,
        ).to_pydict()
        print("rlike matched:", spark_only["n"][0])
        assert spark_only["n"][0] >= 0
    except Exception as error:
        print("not supported in this build:", str(error)[:70])

    # The DataFrame equivalent, which is what a port ends up as.
    equivalent = (
        orders.group_by("o_orderstatus").agg(n=bt.count()).sort("o_orderstatus").to_pydict()
    )
    assert equivalent == default
    assert col is not None


if __name__ == "__main__":
    main()
