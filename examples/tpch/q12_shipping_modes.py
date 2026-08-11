"""TPC-H Q12 — late deliveries split by order priority, using conditional sums.

Two counters over one scan, each a `when(...).then(1).otherwise(0).sum()`. The pattern
generalizes: any "count the rows matching X, grouped by Y" that would otherwise be a
second query is a masked sum instead.

    python examples/tpch/q12_shipping_modes.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")
    orders = tpch("orders")

    start = dt.date(1994, 1, 1)
    end = dt.date(1995, 1, 1)

    urgent = col("o_orderpriority").is_in(["1-URGENT", "2-HIGH"])

    result = (
        lineitem.filter(col("l_shipmode").is_in(["MAIL", "SHIP"]))
        # Committed before it was received, and shipped before it was committed: a
        # genuinely late delivery rather than a data-entry inversion.
        .filter(col("l_commitdate") < col("l_receiptdate"))
        .filter(col("l_shipdate") < col("l_commitdate"))
        .filter((col("l_receiptdate") >= bt.lit(start)) & (col("l_receiptdate") < bt.lit(end)))
        .join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .group_by("l_shipmode")
        .agg(
            high_line_count=bt.when(urgent).then(1).otherwise(0).sum(),
            low_line_count=bt.when(urgent).then(0).otherwise(1).sum(),
        )
        .sort("l_shipmode")
        .to_pydict()
    )

    print(result)

    assert set(result["l_shipmode"]) <= {"MAIL", "SHIP"}
    # The two counters partition the rows, so they can never overlap or lose one.
    assert all(
        high >= 0 and low >= 0
        for high, low in zip(result["high_line_count"], result["low_line_count"], strict=True)
    )


if __name__ == "__main__":
    main()
