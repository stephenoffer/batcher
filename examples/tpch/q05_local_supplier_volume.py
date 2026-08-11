"""TPC-H Q5 — revenue by nation, where customer and supplier share that nation.

Six tables in one query, and one join condition that is not a foreign key: the
supplier's nation must equal the *customer's* nation. That extra equality is what makes
this query a join-ordering problem rather than a scan.

    python examples/tpch/q05_local_supplier_volume.py
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
    customer = tpch("customer")
    orders = tpch("orders")
    lineitem = tpch("lineitem")
    supplier = tpch("supplier")
    nation = tpch("nation")
    region = tpch("region")

    start = dt.date(1994, 1, 1)
    end = dt.date(1995, 1, 1)

    asia = region.filter(col("r_name") == "ASIA")
    asian_nations = nation.join(asia, left_on="n_regionkey", right_on="r_regionkey").select(
        "n_nationkey", "n_name"
    )

    result = (
        customer.join(asian_nations, left_on="c_nationkey", right_on="n_nationkey")
        .join(orders, left_on="c_custkey", right_on="o_custkey")
        .filter((col("o_orderdate") >= bt.lit(start)) & (col("o_orderdate") < bt.lit(end)))
        .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .join(supplier, left_on="l_suppkey", right_on="s_suppkey")
        # The non-key equality: supplier nation == customer nation.
        .filter(col("s_nationkey") == col("c_nationkey"))
        .group_by("n_name")
        .agg(revenue=(col("l_extendedprice") * (1 - col("l_discount"))).sum())
        .sort("revenue", descending=True)
        .to_pydict()
    )

    print(result)

    assert result["revenue"] == sorted(result["revenue"], reverse=True)
    assert all(value >= 0 for value in result["revenue"])
    # Only Asian nations can appear, because the region filter is applied before the join.
    assert set(result["n_name"]) <= set(asian_nations.to_pydict()["n_name"])


if __name__ == "__main__":
    main()
