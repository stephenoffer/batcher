"""SQL over Datasets: bt.sql with named table bindings.

The query is parsed into the same logical plan the DataFrame API builds, so the result is
a lazy Dataset rather than a materialized table. That means SQL and method chaining are
the same thing spelled differently, and you can move between them mid-pipeline.

    python examples/sql_queries/basics.py
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
        SELECT o_orderstatus, COUNT(*) AS orders, SUM(o_totalprice) AS value
        FROM orders
        GROUP BY o_orderstatus
        ORDER BY o_orderstatus
        """,
        orders=orders,
    )

    # Still lazy: this is a plan, not a table.
    print(result.explain())

    rows = result.to_pydict()
    print(rows)
    assert sum(rows["orders"]) == orders.count()
    assert rows["o_orderstatus"] == sorted(rows["o_orderstatus"])

    # The DataFrame spelling of the same query, for comparison.
    equivalent = (
        orders.group_by("o_orderstatus")
        .agg(orders=bt.count(), value=col("o_totalprice").sum())
        .sort("o_orderstatus")
        .to_pydict()
    )
    assert equivalent["orders"] == rows["orders"]

    # And the result composes with the DataFrame API, because it is a Dataset.
    biggest = result.filter(col("orders") > 1_000).sort("value", descending=True).head(1)
    print(biggest.to_pydict())
    assert biggest.count() <= 1


if __name__ == "__main__":
    main()
