"""Filtering rows by a property of their group.

"Keep every line belonging to an order worth more than X" is not a `HAVING`: it keeps rows,
not groups. The shape is a group-by to compute the property, then a semi join back onto the
detail — or a window, which does both in one pass.

    python examples/relational/filtering_by_aggregate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").select("l_orderkey", "l_linenumber", "l_extendedprice")

    threshold = 100_000.0

    # Via a semi join against the qualifying keys.
    big_orders = (
        lineitem.group_by("l_orderkey")
        .agg(total=col("l_extendedprice").sum())
        .filter(col("total") > threshold)
        .select("l_orderkey")
    )
    by_join = lineitem.join(big_orders, on="l_orderkey", how="semi")

    # Via a window, which attaches the group total to every row.
    by_window = (
        lineitem.with_columns(total=col("l_extendedprice").sum().over(partition_by=["l_orderkey"]))
        .filter(col("total") > threshold)
        .drop("total")
    )

    print(f"{by_join.count()} lines belong to a qualifying order")
    assert by_join.count() == by_window.count()
    assert by_join.count() < lineitem.count()

    # Same rows, in the same order.
    left = by_join.sort("l_orderkey", "l_linenumber").to_pydict()
    right = by_window.sort("l_orderkey", "l_linenumber").to_pydict()
    assert left["l_orderkey"] == right["l_orderkey"]
    assert left["l_linenumber"] == right["l_linenumber"]

    # Whole orders survive or none of them do: the filter is at group granularity even
    # though the output is rows.
    kept_keys = set(left["l_orderkey"])
    for key in list(kept_keys)[:5]:
        original = lineitem.filter(col("l_orderkey") == key).count()
        surviving = by_join.filter(col("l_orderkey") == key).count()
        assert original == surviving

    # And every kept order really is over the threshold.
    totals = by_join.group_by("l_orderkey").agg(total=col("l_extendedprice").sum())
    assert all(value > threshold for value in totals.to_pydict()["total"])
    assert bt is not None


if __name__ == "__main__":
    main()
