"""TPC-H Q9 — profit by nation and year, from a substring match on part name.

The filter is `p_name like '%green%'`, which no index and no Parquet statistic can help
with: the engine has to look at the string. That makes Q9 the query where pushing the
*join* order around matters, because the only way to cut work is to apply the expensive
predicate to the smallest relation first.

    python examples/tpch/q09_product_type_profit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    part = tpch("part")
    supplier = tpch("supplier")
    lineitem = tpch("lineitem")
    partsupp = tpch("partsupp")
    orders = tpch("orders")
    nation = tpch("nation")

    green_parts = part.filter(col("p_name").str.contains("green")).select("p_partkey")

    amount = (
        col("l_extendedprice") * (1 - col("l_discount")) - col("ps_supplycost") * col("l_quantity")
    ).alias("amount")

    result = (
        lineitem.join(green_parts, left_on="l_partkey", right_on="p_partkey")
        .join(supplier, left_on="l_suppkey", right_on="s_suppkey")
        .join(
            partsupp,
            left_on=["l_partkey", "l_suppkey"],
            right_on=["ps_partkey", "ps_suppkey"],
        )
        .join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .join(nation, left_on="s_nationkey", right_on="n_nationkey")
        .with_columns(o_year=col("o_orderdate").dt.year(), amount=amount)
        .group_by("n_name", "o_year")
        .agg(profit=col("amount").sum())
        .sort("n_name", "o_year", descending=[False, True])
        .to_pydict()
    )

    print(f"{len(result['n_name'])} nation/year rows")
    for nation_name, year, profit in list(
        zip(result["n_name"], result["o_year"], result["profit"], strict=True)
    )[:5]:
        print(f"  {nation_name:<16} {year} {profit:>16,.2f}")

    # Nation ascending, year descending within it — a two-key sort with mixed direction,
    # which is easy to get wrong and easy to check.
    pairs = list(zip(result["n_name"], result["o_year"], strict=True))
    assert pairs == sorted(pairs, key=lambda pair: (pair[0], -pair[1]))


if __name__ == "__main__":
    main()
