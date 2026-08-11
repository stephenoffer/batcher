"""Checking that a refactor did not change the plan.

Rewriting a query for readability should not change what runs. Capturing the plan before and
after is the cheapest possible regression test, and it catches the case where a "cosmetic"
change quietly removed a pushdown.

    python examples/operations/plan_stability.py
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

    # One query, two spellings.
    inline = (
        lineitem.filter(col("l_quantity") > 30)
        .join(
            orders.filter(col("o_orderstatus") == "F"), left_on="l_orderkey", right_on="o_orderkey"
        )
        .group_by("o_orderpriority")
        .agg(revenue=col("l_extendedprice").sum())
        .sort("o_orderpriority")
    )

    def big_lines(dataset: bt.Dataset) -> bt.Dataset:
        return dataset.filter(col("l_quantity") > 30)

    def closed(dataset: bt.Dataset) -> bt.Dataset:
        return dataset.filter(col("o_orderstatus") == "F")

    factored = (
        lineitem.pipe(big_lines)
        .join(orders.pipe(closed), left_on="l_orderkey", right_on="o_orderkey")
        .group_by("o_orderpriority")
        .agg(revenue=col("l_extendedprice").sum())
        .sort("o_orderpriority")
    )

    inline_plan = inline.explain()
    factored_plan = factored.explain()
    print(inline_plan)

    # Factoring into functions is free: the plan is identical, because a Dataset is a
    # plan and composition does not add a layer.
    assert inline_plan == factored_plan

    # And so is the answer.
    assert inline.to_pydict() == factored.to_pydict()

    # A change that *does* matter shows up in the plan: dropping a filter.
    without_filter = (
        lineitem.join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .group_by("o_orderpriority")
        .agg(revenue=col("l_extendedprice").sum())
        .sort("o_orderpriority")
    )
    assert without_filter.explain() != inline_plan
    assert without_filter.to_pydict() != inline.to_pydict()


if __name__ == "__main__":
    main()
