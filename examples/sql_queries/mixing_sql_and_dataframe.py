"""Moving between SQL and the DataFrame API mid-pipeline.

`bt.sql` returns a Dataset, and a Dataset can be bound back into another query. Neither
direction materializes anything, so the boundary is free — use whichever spelling makes the
step clearer rather than committing to one.

    python examples/sql_queries/mixing_sql_and_dataframe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")
    orders = tpch("orders")

    # Step one in SQL, because the aggregate reads well as SQL.
    per_order = bt.sql(
        """
        SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue
        FROM lineitem
        GROUP BY l_orderkey
        """,
        lineitem=lineitem,
    )

    # Step two with the DataFrame API, because the join and the window read better there.
    enriched = (
        per_order.join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .with_columns(
            rank=bt.row_number().over(
                partition_by=["o_orderpriority"], order_by=[("revenue", True)]
            )
        )
        .filter(col("rank") <= 3)
    )

    # Step three back in SQL, over the Dataset from step two.
    final = bt.sql(
        """
        SELECT o_orderpriority, COUNT(*) AS kept, MAX(revenue) AS best
        FROM enriched
        GROUP BY o_orderpriority
        ORDER BY o_orderpriority
        """,
        enriched=enriched,
    ).to_pydict()
    print(final)

    # Three per priority, no more.
    assert all(value <= 3 for value in final["kept"])
    assert len(final["o_orderpriority"]) == orders.n_unique("o_orderpriority")

    # Nothing was materialized along the way: the whole thing is one plan.
    plan = enriched.explain()
    assert "aggregate" in plan.lower()
    assert "join" in plan.lower()

    # `ds.sql` is the same thing scoped to one Dataset, which it calls `self`.
    scoped = per_order.sql("SELECT COUNT(*) AS n FROM self").to_pydict()
    assert scoped["n"][0] == per_order.count()


if __name__ == "__main__":
    main()
