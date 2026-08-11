"""Reading a plan at different levels of detail.

The logical plan is what you wrote; the physical plan is what will run. Comparing them is
how you see what the optimizer did — and the estimated row counts beside each operator are
what it based those decisions on.

    python examples/operations/explain_modes.py
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

    query = (
        lineitem.filter(col("l_quantity") > 40)
        .join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .filter(col("o_orderstatus") == "F")
        .group_by("o_orderpriority")
        .agg(lines=bt.count(), revenue=col("l_extendedprice").sum())
        .sort("o_orderpriority")
    )

    plan = query.explain()
    print(plan)

    lowered = plan.lower()
    for operator in ("scan", "filter", "join", "aggregate", "sort"):
        assert operator in lowered, operator

    # The plan carries row estimates, which is what a join-order decision reads.
    assert "est" in lowered or "rows" in lowered

    # Nothing ran: `explain` is a plan operation.
    result = query.to_pydict()
    print("priorities:", result["o_orderpriority"])
    assert len(result["o_orderpriority"]) == orders.n_unique("o_orderpriority")

    # `profile` is the executed counterpart, and it needs the query to run.
    report = query.profile()
    assert report is not None

    # `info` and `glimpse` describe the data rather than the plan.
    query.select("o_orderpriority", "lines").glimpse()


if __name__ == "__main__":
    main()
