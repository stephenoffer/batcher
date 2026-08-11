"""Removing duplicates: whole-row distinct versus keyed deduplication.

`distinct` compares every column. `drop_duplicates(subset=...)` compares the columns you
name and keeps one arbitrary row per key — arbitrary, so if you care which one survives,
sort first or aggregate instead.

    python examples/relational/distinct_and_dedup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")

    # Whole-row distinct over a projection: the set of ship modes.
    modes = lineitem.select("l_shipmode").distinct().sort("l_shipmode").to_pydict()
    print("ship modes:", modes["l_shipmode"])
    assert len(modes["l_shipmode"]) == len(set(modes["l_shipmode"]))

    # Distinct over several columns is distinct over the *combination*.
    pairs = lineitem.select("l_returnflag", "l_linestatus").distinct()
    assert pairs.count() <= 3 * 2

    # Keyed deduplication: one row per order, whichever line arrived first.
    per_order = lineitem.drop_duplicates(subset=["l_orderkey"])
    assert per_order.count() == lineitem.select("l_orderkey").distinct().count()

    # `n_unique` counts without materializing the distinct set.
    assert lineitem.n_unique("l_shipmode") == len(modes["l_shipmode"])

    # When you *do* care which row survives, say so with an aggregate rather than
    # trusting dedup order.
    largest_line = (
        lineitem.group_by("l_orderkey").agg(top_price=col("l_extendedprice").max()).head(3)
    )
    print(largest_line.to_pydict())
    assert largest_line.count() == 3


if __name__ == "__main__":
    main()
