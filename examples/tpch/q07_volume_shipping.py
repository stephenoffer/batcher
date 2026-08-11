"""TPC-H Q7 — trade volume between two nations, in both directions.

The query joins `nation` twice under different names, then keeps only the two ordered
pairs it cares about. Self-joins like this are where column-name collisions bite, so the
projection right after each join is not optional tidiness.

    python examples/tpch/q07_volume_shipping.py
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
    supplier = tpch("supplier")
    lineitem = tpch("lineitem")
    orders = tpch("orders")
    customer = tpch("customer")
    nation = tpch("nation")

    supplier_nation = nation.select(
        col("n_nationkey").alias("supp_nationkey"), col("n_name").alias("supp_nation")
    )
    customer_nation = nation.select(
        col("n_nationkey").alias("cust_nationkey"), col("n_name").alias("cust_nation")
    )

    shipped = (
        supplier.join(lineitem, left_on="s_suppkey", right_on="l_suppkey")
        .join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .join(customer, left_on="o_custkey", right_on="c_custkey")
        .join(supplier_nation, left_on="s_nationkey", right_on="supp_nationkey")
        .join(customer_nation, left_on="c_nationkey", right_on="cust_nationkey")
    )

    pair = (col("supp_nation") == "FRANCE") & (col("cust_nation") == "GERMANY")
    reverse = (col("supp_nation") == "GERMANY") & (col("cust_nation") == "FRANCE")

    result = (
        shipped.filter(pair | reverse)
        .filter(
            (col("l_shipdate") >= bt.lit(dt.date(1995, 1, 1)))
            & (col("l_shipdate") <= bt.lit(dt.date(1996, 12, 31)))
        )
        .with_columns(ship_year=col("l_shipdate").dt.year())
        .group_by("supp_nation", "cust_nation", "ship_year")
        .agg(revenue=(col("l_extendedprice") * (1 - col("l_discount"))).sum())
        .sort("supp_nation", "cust_nation", "ship_year")
        .to_pydict()
    )

    print(result)

    # Only the two ordered pairs asked for, and only years inside the window.
    assert set(zip(result["supp_nation"], result["cust_nation"], strict=True)) <= {
        ("FRANCE", "GERMANY"),
        ("GERMANY", "FRANCE"),
    }
    assert all(1995 <= year <= 1996 for year in result["ship_year"])


if __name__ == "__main__":
    main()
