"""TPC-H Q20 — suppliers holding excess stock, through two levels of subquery.

Three nested `IN` clauses become three joins, innermost first. Building it bottom-up —
the shipped quantities, then the parts that qualify, then the suppliers — keeps each
step something you can print and check on its own.

    python examples/tpch/q20_potential_part_promotion.py
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
    nation = tpch("nation")
    partsupp = tpch("partsupp")
    part = tpch("part")
    lineitem = tpch("lineitem")

    start = dt.date(1994, 1, 1)
    end = dt.date(1995, 1, 1)

    # Innermost: how much of each part/supplier pair actually shipped that year.
    shipped = (
        lineitem.filter((col("l_shipdate") >= bt.lit(start)) & (col("l_shipdate") < bt.lit(end)))
        .group_by("l_partkey", "l_suppkey")
        .agg(shipped_qty=col("l_quantity").sum())
    )

    forest_parts = part.filter(col("p_name").str.starts_with("forest")).select("p_partkey")

    # Middle: stock above half of what shipped is "excess".
    excess = (
        partsupp.join(forest_parts, left_on="ps_partkey", right_on="p_partkey")
        .join(
            shipped,
            left_on=["ps_partkey", "ps_suppkey"],
            right_on=["l_partkey", "l_suppkey"],
        )
        .filter(col("ps_availqty") > 0.5 * col("shipped_qty"))
        .select("ps_suppkey")
    )

    canadian = nation.filter(col("n_name") == "CANADA").select("n_nationkey")

    result = (
        supplier.join(canadian, left_on="s_nationkey", right_on="n_nationkey")
        .join(excess, left_on="s_suppkey", right_on="ps_suppkey", how="semi")
        .select("s_name", "s_address")
        .sort("s_name")
        .to_pydict()
    )

    print(f"{len(result['s_name'])} suppliers with excess forest-part stock")
    assert result["s_name"] == sorted(result["s_name"])
    # A semi join returns each left row at most once, however many excess parts it has.
    assert len(set(result["s_name"])) == len(result["s_name"])


if __name__ == "__main__":
    main()
