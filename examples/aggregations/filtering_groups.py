"""HAVING: filtering groups after the aggregate, and why order matters.

A `filter` before `group_by` removes rows; a `filter` after it removes groups. They are
different queries and usually give different answers, so the placement is a decision
rather than a style choice.

    python examples/aggregations/filtering_groups.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")

    # HAVING: keep orders whose *total* quantity is large.
    big_orders = (
        lineitem.group_by("l_orderkey")
        .agg(qty=col("l_quantity").sum(), lines=bt.count())
        .filter(col("qty") > 200)
    )
    print("orders over 200 units:", big_orders.count())
    assert all(qty > 200 for qty in big_orders.to_pydict()["qty"])

    # WHERE: keep large *lines*, then total them. A different question.
    big_lines = (
        lineitem.filter(col("l_quantity") > 40)
        .group_by("l_orderkey")
        .agg(qty=col("l_quantity").sum())
    )

    # The two answers differ, and neither is a filtered version of the other.
    print("orders with a line over 40 units:", big_lines.count())
    assert big_orders.count() != big_lines.count()

    # Filtering on the group key itself can move to either side and mean the same thing,
    # which is exactly the case the optimizer is free to rewrite.
    late = lineitem.filter(col("l_orderkey") > 100_000)
    before = late.group_by("l_orderkey").agg(n=bt.count()).count()
    after = (
        lineitem.group_by("l_orderkey")
        .agg(n=bt.count())
        .filter(col("l_orderkey") > 100_000)
        .count()
    )
    assert before == after


if __name__ == "__main__":
    main()
