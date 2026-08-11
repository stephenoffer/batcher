"""A star-schema query: one fact table, several small dimensions.

Each dimension join is a filter in disguise — the point of joining `nation` is usually the
`n_name` predicate, not the column. Applying that predicate to the dimension *before* the
join is what keeps the fact-table probe small.

    python examples/joins/star_schema.py
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
    customer = tpch("customer")
    nation = tpch("nation")
    region = tpch("region")

    # Filter the dimensions first: five rows of `region` decide how much of the fact
    # table has to be probed at all.
    europe = region.filter(col("r_name") == "EUROPE").select("r_regionkey")
    european_nations = nation.join(europe, left_on="n_regionkey", right_on="r_regionkey").select(
        "n_nationkey", "n_name"
    )

    result = (
        lineitem.join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .join(customer, left_on="o_custkey", right_on="c_custkey")
        .join(european_nations, left_on="c_nationkey", right_on="n_nationkey")
        .group_by("n_name")
        .agg(
            revenue=(col("l_extendedprice") * (1 - col("l_discount"))).sum(),
            lines=bt.count(),
        )
        .sort("revenue", descending=True)
        .to_pydict()
    )
    print(result["n_name"], [round(value) for value in result["revenue"]])

    assert result["revenue"] == sorted(result["revenue"], reverse=True)
    assert set(result["n_name"]) <= set(european_nations.to_pydict()["n_name"])
    assert sum(result["lines"]) > 0

    # The plan shows the region filter sitting under the joins rather than above them.
    plan = (
        lineitem.join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .join(customer, left_on="o_custkey", right_on="c_custkey")
        .join(european_nations, left_on="c_nationkey", right_on="n_nationkey")
        .explain()
    )
    print(plan)
    assert "join" in plan.lower()


if __name__ == "__main__":
    main()
