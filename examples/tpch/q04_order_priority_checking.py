"""TPC-H Q4 — order priority, counted with a semi join.

The SQL spells this `EXISTS (SELECT ... FROM lineitem WHERE ...)`. The relational form
is a semi join: keep the order if it has at least one late line, add nothing from the
right side, and count each order once no matter how many late lines it has. Writing it
as an inner join instead is the classic way to over-count.

    python examples/tpch/q04_order_priority_checking.py
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
    orders = tpch("orders")
    lineitem = tpch("lineitem")

    start = dt.date(1993, 7, 1)
    end = dt.date(1993, 10, 1)

    late_lines = lineitem.filter(col("l_commitdate") < col("l_receiptdate")).select("l_orderkey")

    in_quarter = orders.filter(
        (col("o_orderdate") >= bt.lit(start)) & (col("o_orderdate") < bt.lit(end))
    )

    result = (
        in_quarter.join(late_lines, left_on="o_orderkey", right_on="l_orderkey", how="semi")
        .group_by("o_orderpriority")
        .agg(order_count=bt.count())
        .sort("o_orderpriority")
        .to_pydict()
    )

    print(result)

    assert result["o_orderpriority"] == sorted(result["o_orderpriority"])
    # The semi join counts orders, not lines, so the total can never exceed the number of
    # orders in the quarter. An inner join here would blow past this.
    assert sum(result["order_count"]) <= in_quarter.count()


if __name__ == "__main__":
    main()
