"""Each row's share of its group, without a join back.

Dividing a row by its group's total is the canonical reason windows exist. The grouped
alternative — aggregate, then join the total back on — computes the same thing with a
shuffle and a join the window does not need.

    python examples/windows/share_of_partition.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").head(20_000)

    shares = lineitem.select(
        "l_orderkey",
        "l_linenumber",
        "l_extendedprice",
        order_total=col("l_extendedprice").sum().over(partition_by=["l_orderkey"]),
    ).with_columns(share=col("l_extendedprice") / col("order_total"))

    result = shares.sort("l_orderkey", "l_linenumber").head(8).to_pydict()
    print([round(value, 4) for value in result["share"]])

    # Every share is a proportion.
    assert all(0.0 < value <= 1.0 for value in result["share"])

    # And the shares within an order sum to one.
    totals = shares.group_by("l_orderkey").agg(total_share=col("share").sum()).to_pydict()
    assert all(abs(value - 1.0) < 1e-9 for value in totals["total_share"])

    # The join-based equivalent gives the same numbers, at more cost.
    by_join = lineitem.join(
        lineitem.group_by("l_orderkey").agg(order_total=col("l_extendedprice").sum()),
        on="l_orderkey",
    ).with_columns(share=col("l_extendedprice") / col("order_total"))
    joined = by_join.sort("l_orderkey", "l_linenumber").head(8).to_pydict()
    assert [round(value, 9) for value in joined["share"]] == [
        round(value, 9) for value in result["share"]
    ]


if __name__ == "__main__":
    main()
