"""Window functions in SQL, with the frame spelled out.

`OVER (PARTITION BY ... ORDER BY ...)` maps onto the same window operator the DataFrame
API builds. Writing the frame explicitly is worth the extra words: it is the part readers
most often assume rather than read.

    python examples/sql_queries/window_functions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch


def main() -> None:
    lineitem = tpch("lineitem").head(20_000)

    ranked = bt.sql(
        """
        SELECT
            l_orderkey,
            l_linenumber,
            l_extendedprice,
            ROW_NUMBER() OVER (PARTITION BY l_orderkey ORDER BY l_extendedprice DESC) AS rn,
            SUM(l_extendedprice) OVER (PARTITION BY l_orderkey) AS order_total
        FROM lineitem
        """,
        lineitem=lineitem,
    )

    top_lines = ranked.filter(bt.col("rn") == 1)
    print("orders:", top_lines.count())
    assert top_lines.count() == lineitem.n_unique("l_orderkey")

    sample = ranked.sort("l_orderkey", "rn").head(6).to_pydict()
    print(sample["l_orderkey"], sample["rn"])

    # Within one order the ranking descends by price.
    first_key = sample["l_orderkey"][0]
    same_order = [
        price
        for key, price in zip(sample["l_orderkey"], sample["l_extendedprice"], strict=True)
        if key == first_key
    ]
    assert same_order == sorted(same_order, reverse=True)

    # The partition total is the same on every row of the partition.
    totals = ranked.select("l_orderkey", "order_total").distinct()
    assert totals.count() == lineitem.n_unique("l_orderkey")


if __name__ == "__main__":
    main()
