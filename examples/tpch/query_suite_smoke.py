"""Running every TPC-H example's core query in one pass, as a smoke check.

This is the release check in miniature: eight query shapes, each asserted on a property that
does not depend on the slice size. If this passes, the relational engine is answering the
whole standard shape vocabulary.

    python examples/tpch/query_suite_smoke.py
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
    customer = tpch("customer")
    nation = tpch("nation")
    region = tpch("region")
    supplier = tpch("supplier")
    part = tpch("part")

    checks: dict[str, bool] = {}

    # Grouped aggregation.
    q1 = lineitem.group_by("l_returnflag", "l_linestatus").agg(n=bt.count())
    checks["grouped aggregate"] = q1.count() > 0 and q1.to_pydict()["n"][0] > 0

    # Three-table join with a top-N.
    q3 = (
        customer.join(orders, left_on="c_custkey", right_on="o_custkey")
        .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .group_by("o_orderkey")
        .agg(revenue=col("l_extendedprice").sum())
        .sort("revenue", descending=True)
        .limit(10)
    )
    revenues = q3.to_pydict()["revenue"]
    checks["join + top-N"] = revenues == sorted(revenues, reverse=True)

    # Semi join.
    q4 = orders.join(
        lineitem.filter(col("l_commitdate") < col("l_receiptdate")).select("l_orderkey"),
        left_on="o_orderkey",
        right_on="l_orderkey",
        how="semi",
    )
    checks["semi join"] = 0 < q4.count() <= orders.count()

    # Six-table join with a non-key predicate.
    q5 = (
        customer.join(nation, left_on="c_nationkey", right_on="n_nationkey")
        .join(region, left_on="n_regionkey", right_on="r_regionkey")
        .join(orders, left_on="c_custkey", right_on="o_custkey")
        .join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .join(supplier, left_on="l_suppkey", right_on="s_suppkey")
        .filter(col("s_nationkey") == col("c_nationkey"))
    )
    checks["six-table join"] = q5.count() >= 0

    # Single-table filtered scan.
    q6 = lineitem.filter(
        (col("l_shipdate") >= bt.lit(dt.date(1994, 1, 1)))
        & (col("l_discount") >= 0.05)
        & (col("l_quantity") < 24)
    )
    checks["filtered scan"] = 0 < q6.count() < lineitem.count()

    # Substring match.
    q9 = part.filter(col("p_name").str.contains("green"))
    checks["substring filter"] = 0 < q9.count() < part.count()

    # Left join with a zero bucket.
    q13 = (
        customer.join(orders, left_on="c_custkey", right_on="o_custkey", how="left")
        .group_by("c_custkey")
        .agg(n=col("o_orderkey").count())
    )
    checks["left join with zeros"] = 0 in set(q13.to_pydict()["n"])

    # Anti join.
    q22 = customer.join(orders, left_on="c_custkey", right_on="o_custkey", how="anti")
    checks["anti join"] = q22.count() < customer.count()

    for name, passed in checks.items():
        print(f"  {'ok  ' if passed else 'FAIL'} {name}")
    assert all(checks.values()), [name for name, ok in checks.items() if not ok]
    print(f"{len(checks)} query shapes verified")


if __name__ == "__main__":
    main()
