"""Reading `explain` output, and what the optimizer did to your query.

The plan is the ground truth about what will run. Two things to look for: whether the
filter sits directly on the scan (pushdown happened) and whether the projection narrowed
before the join. Both are visible without executing anything.

    python examples/operations/reading_a_plan.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders")
    lineitem = tpch("lineitem")

    # Written in the least helpful order on purpose: filter last, wide projection first.
    query = (
        lineitem.join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .filter(col("o_orderstatus") == "F")
        .filter(col("l_quantity") > 30)
        .select("l_orderkey", "l_quantity", "o_totalprice")
    )

    plan = query.explain()
    print(plan)

    lowered = plan.lower()
    assert "join" in lowered
    assert "filter" in lowered

    # Nothing has executed yet: `explain` reads the plan, not the data.
    assert isinstance(plan, str)

    # The optimized plan is what runs, and it produces the same rows as the naive
    # ordering — which is the only thing an optimizer is allowed to preserve.
    hand_written = (
        lineitem.filter(col("l_quantity") > 30)
        .join(
            orders.filter(col("o_orderstatus") == "F"),
            left_on="l_orderkey",
            right_on="o_orderkey",
        )
        .select("l_orderkey", "l_quantity", "o_totalprice")
    )
    left = query.sort("l_orderkey", "l_quantity", "o_totalprice").to_pydict()
    right = hand_written.sort("l_orderkey", "l_quantity", "o_totalprice").to_pydict()
    assert left == right
    print(f"both orderings return {len(left['l_orderkey'])} rows")


if __name__ == "__main__":
    main()
