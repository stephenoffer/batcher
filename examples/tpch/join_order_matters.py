"""The same five-table query, written in two join orders.

Both orders return the same rows; the plan and the intermediate sizes differ. Applying the
selective dimension filters first is what keeps the intermediates small, and it is the
single most useful thing to get right by hand when the optimizer needs help.

    python examples/tpch/join_order_matters.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")
    orders = tpch("orders")
    customer = tpch("customer")
    nation = tpch("nation")
    region = tpch("region")

    def timed(label: str, build) -> dict:
        started = time.perf_counter()
        result = build().to_pydict()
        print(f"{label:<26} {(time.perf_counter() - started) * 1000:7.1f} ms")
        return result

    # Filters first: `region` and `nation` shrink to a handful of rows before the fact
    # table is touched.
    def narrow_first():
        asia = region.filter(col("r_name") == "ASIA").select("r_regionkey")
        asian = nation.join(asia, left_on="n_regionkey", right_on="r_regionkey").select(
            "n_nationkey", "n_name"
        )
        return (
            customer.join(asian, left_on="c_nationkey", right_on="n_nationkey")
            .join(orders, left_on="c_custkey", right_on="o_custkey")
            .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
            .group_by("n_name")
            .agg(revenue=col("l_extendedprice").sum())
            .sort("n_name")
        )

    # Filters last: the whole join is built, then most of it is thrown away.
    def filter_last():
        return (
            lineitem.join(orders, left_on="l_orderkey", right_on="o_orderkey")
            .join(customer, left_on="o_custkey", right_on="c_custkey")
            .join(nation, left_on="c_nationkey", right_on="n_nationkey")
            .join(region, left_on="n_regionkey", right_on="r_regionkey")
            .filter(col("r_name") == "ASIA")
            .group_by("n_name")
            .agg(revenue=col("l_extendedprice").sum())
            .sort("n_name")
        )

    early = timed("filters pushed early", narrow_first)
    late = timed("filters left late", filter_last)

    # Same answer, whatever the order.
    assert early["n_name"] == late["n_name"]
    assert all(abs(a - b) < 1e-3 for a, b in zip(early["revenue"], late["revenue"], strict=True))
    print(f"{len(early['n_name'])} Asian nations, identical revenue either way")
    assert bt is not None


if __name__ == "__main__":
    main()
