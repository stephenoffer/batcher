"""What a join looks like in the plan, and what the shape of the query tells the optimizer.

You do not pick the join algorithm; you give the optimizer the information to pick well. The
two things that actually help are accurate filters early and a projection that drops the
columns the join does not need.

    python examples/joins/join_hints_and_plans.py
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
    orders = tpch("orders")

    wide = lineitem.join(orders, left_on="l_orderkey", right_on="o_orderkey")
    narrow = lineitem.select("l_orderkey", "l_quantity").join(
        orders.select("o_orderkey", "o_orderstatus"),
        left_on="l_orderkey",
        right_on="o_orderkey",
    )

    print("wide join columns:", wide.width)
    print("narrow join columns:", narrow.width)
    assert narrow.width < wide.width
    assert narrow.count() == wide.count()

    plan = narrow.explain()
    print(plan)
    assert "join" in plan.lower()

    # The narrowed join carries the same rows, so any aggregate over shared columns agrees.
    left = wide.agg(q=col("l_quantity").sum()).to_pydict()["q"][0]
    right = narrow.agg(q=col("l_quantity").sum()).to_pydict()["q"][0]
    assert abs(left - right) < 1e-6

    # A selective filter on the build side is the other lever.
    selective = lineitem.select("l_orderkey", "l_quantity").join(
        orders.filter(col("o_orderstatus") == "P").select("o_orderkey"),
        left_on="l_orderkey",
        right_on="o_orderkey",
    )
    print("after a selective filter:", selective.count())
    assert selective.count() < narrow.count()
    assert bt is not None


if __name__ == "__main__":
    main()
