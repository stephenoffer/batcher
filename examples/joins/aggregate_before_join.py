"""Shrinking a side before joining it.

Aggregating first turns a many-row side into a one-row-per-key side, which removes the
fan-out and shrinks what has to be shuffled. It is the same answer for strictly less work,
whenever the join's only purpose is to attach a summary.

    python examples/joins/aggregate_before_join.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_custkey", "o_totalprice")
    lineitem = tpch("lineitem").select("l_orderkey", "l_quantity", "l_extendedprice")

    # Join then aggregate: the join emits one row per line.
    join_first = (
        orders.join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .group_by("o_custkey")
        .agg(units=col("l_quantity").sum())
        .sort("o_custkey")
    )

    # Aggregate then join: the right side is one row per order before it is joined.
    per_order = lineitem.group_by("l_orderkey").agg(units=col("l_quantity").sum())
    aggregate_first = (
        orders.join(per_order, left_on="o_orderkey", right_on="l_orderkey")
        .group_by("o_custkey")
        .agg(units=col("units").sum())
        .sort("o_custkey")
    )

    left = join_first.to_pydict()
    right = aggregate_first.to_pydict()
    print(f"{len(left['o_custkey'])} customers")

    # The same answer.
    assert left["o_custkey"] == right["o_custkey"]
    assert all(abs(a - b) < 1e-6 for a, b in zip(left["units"], right["units"], strict=True))

    # But a much smaller intermediate: one row per order rather than one per line.
    fanned = orders.join(lineitem, left_on="o_orderkey", right_on="l_orderkey").count()
    reduced = orders.join(per_order, left_on="o_orderkey", right_on="l_orderkey").count()
    print(f"intermediate rows: {fanned} vs {reduced}")
    assert reduced < fanned
    assert bt is not None


if __name__ == "__main__":
    main()
