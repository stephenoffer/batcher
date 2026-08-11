"""Top-N per group without a window, using a join against the group's threshold.

The window version is clearer and usually right. The threshold version reads the data twice
but never sorts within a partition, which wins when the groups are enormous and N is small —
the case where the window's per-partition sort is the cost.

    python examples/relational/window_free_top_n_per_group.py
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

    # The window version.
    windowed = (
        lineitem.with_columns(
            rank=bt.row_number().over(
                partition_by=["l_orderkey"], order_by=[("l_extendedprice", True)]
            )
        )
        .filter(col("rank") == 1)
        .select("l_orderkey", "l_extendedprice")
        .sort("l_orderkey")
    )

    # The threshold version: the group maximum, joined back.
    thresholds = lineitem.group_by("l_orderkey").agg(top=col("l_extendedprice").max())
    by_threshold = (
        lineitem.join(thresholds, on="l_orderkey")
        .filter(col("l_extendedprice") == col("top"))
        # Ties: keep one per key, deterministically.
        .with_columns(
            tie=bt.row_number().over(partition_by=["l_orderkey"], order_by=["l_linenumber"])
        )
        .filter(col("tie") == 1)
        .select("l_orderkey", "l_extendedprice")
        .sort("l_orderkey")
    )

    left = windowed.to_pydict()
    right = by_threshold.to_pydict()
    print(f"{len(left['l_orderkey'])} orders, one top line each")

    # One row per order, both ways.
    assert len(left["l_orderkey"]) == lineitem.n_unique("l_orderkey")
    assert left["l_orderkey"] == right["l_orderkey"]

    # And the same prices, because both pick the maximum.
    assert left["l_extendedprice"] == right["l_extendedprice"]

    # The kept price really is the group's maximum.
    maxima = dict(zip(*thresholds.sort("l_orderkey").to_pydict().values(), strict=True))
    assert all(
        maxima[key] == price
        for key, price in zip(left["l_orderkey"], left["l_extendedprice"], strict=True)
    )


if __name__ == "__main__":
    main()
