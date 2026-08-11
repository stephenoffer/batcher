"""TPC-H Q8 — one nation's share of a market, as a ratio of two conditional sums.

The final expression is a share: revenue from BRAZIL over revenue from everyone. Both
halves come out of the same scan, the numerator masked by a `when(...)`. Computing the
numerator as a second query over the same data is the mistake this shape avoids.

    python examples/tpch/q08_national_market_share.py
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
    part = tpch("part")
    supplier = tpch("supplier")
    lineitem = tpch("lineitem")
    orders = tpch("orders")
    customer = tpch("customer")
    nation = tpch("nation")
    region = tpch("region")

    america = region.filter(col("r_name") == "AMERICA")
    american_nations = nation.join(america, left_on="n_regionkey", right_on="r_regionkey").select(
        col("n_nationkey").alias("cust_nationkey")
    )
    supplier_nation = nation.select(
        col("n_nationkey").alias("supp_nationkey"), col("n_name").alias("supp_nation")
    )

    volume = (col("l_extendedprice") * (1 - col("l_discount"))).alias("volume")

    rows = (
        part.join(lineitem, left_on="p_partkey", right_on="l_partkey")
        .join(supplier, left_on="l_suppkey", right_on="s_suppkey")
        .join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .join(customer, left_on="o_custkey", right_on="c_custkey")
        .join(american_nations, left_on="c_nationkey", right_on="cust_nationkey")
        .join(supplier_nation, left_on="s_nationkey", right_on="supp_nationkey")
        .filter(
            (col("o_orderdate") >= bt.lit(dt.date(1995, 1, 1)))
            & (col("o_orderdate") <= bt.lit(dt.date(1996, 12, 31)))
        )
        .with_columns(order_year=col("o_orderdate").dt.year(), volume=volume)
    )

    result = (
        rows.group_by("order_year")
        .agg(
            brazil=bt.when(col("supp_nation") == "BRAZIL").then(col("volume")).otherwise(0.0).sum(),
            total=col("volume").sum(),
        )
        .with_columns(market_share=col("brazil") / col("total"))
        .sort("order_year")
        .to_pydict()
    )

    print(result)

    # A share is a proportion: bounded, and never larger than the whole it came from.
    assert all(0.0 <= share <= 1.0 for share in result["market_share"])
    assert all(
        part_value <= whole
        for part_value, whole in zip(result["brazil"], result["total"], strict=True)
    )


if __name__ == "__main__":
    main()
