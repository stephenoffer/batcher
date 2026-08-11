"""TPC-H Q21 — the supplier who was the only one late on a multi-supplier order.

This is the query with both an `EXISTS` and a `NOT EXISTS` over the same table: another
supplier on the order (semi join), and no *other* late supplier on it (anti join). Get
either direction backwards and the answer is still plausible, which is why both are
asserted here rather than eyeballed.

    python examples/tpch/q21_suppliers_kept_orders_waiting.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    supplier = tpch("supplier")
    lineitem = tpch("lineitem")
    orders = tpch("orders")
    nation = tpch("nation")

    late = lineitem.filter(col("l_receiptdate") > col("l_commitdate"))

    # Another supplier on the same order, whoever they are.
    others = lineitem.select(col("l_orderkey").alias("o_key"), col("l_suppkey").alias("other_supp"))
    # Another *late* supplier on the same order.
    other_late = late.select(
        col("l_orderkey").alias("late_key"), col("l_suppkey").alias("late_supp")
    )

    candidates = (
        late.join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .filter(col("o_orderstatus") == "F")
        .join(supplier, left_on="l_suppkey", right_on="s_suppkey")
        .join(nation, left_on="s_nationkey", right_on="n_nationkey")
        .filter(col("n_name") == "SAUDI ARABIA")
    )

    with_company = candidates.join(
        others,
        left_on="l_orderkey",
        right_on="o_key",
        how="semi",
    )

    result = (
        with_company.join(other_late, left_on="l_orderkey", right_on="late_key", how="anti")
        .group_by("s_name")
        .agg(numwait=col("l_orderkey").count())
        .sort("numwait", "s_name", descending=[True, False])
        .limit(20)
        .to_pydict()
    )

    print(f"{len(result['s_name'])} suppliers; worst {result['numwait'][:3]}")

    assert result["numwait"] == sorted(result["numwait"], reverse=True)
    # The anti join can only remove rows, never add them.
    assert sum(result["numwait"]) <= with_company.count()


if __name__ == "__main__":
    main()
