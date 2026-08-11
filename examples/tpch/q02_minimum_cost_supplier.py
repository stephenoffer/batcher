"""TPC-H Q2 — the cheapest supplier for a part, via a correlated minimum.

The shape worth studying is the self-join at the end: "the supplier whose cost equals
the minimum cost for this part" is a group-by that produces the minimum, joined back to
the rows it came from. That rewrite is how you express a correlated subquery without a
subquery, and it is what the engine does internally anyway.

    python examples/tpch/q02_minimum_cost_supplier.py
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
    partsupp = tpch("partsupp")
    nation = tpch("nation")
    region = tpch("region")

    europe = region.filter(col("r_name") == "EUROPE")
    european_suppliers = (
        supplier.join(nation, left_on="s_nationkey", right_on="n_nationkey")
        .join(europe, left_on="n_regionkey", right_on="r_regionkey")
        .select("s_suppkey", "s_name", "s_acctbal", "n_name")
    )

    offers = partsupp.join(european_suppliers, left_on="ps_suppkey", right_on="s_suppkey")

    # The minimum cost per part, as its own relation.
    cheapest = offers.group_by("ps_partkey").agg(min_cost=col("ps_supplycost").min())

    # Join it back to keep only the offers that hit that minimum.
    winners = (
        offers.join(cheapest, on="ps_partkey")
        .filter(col("ps_supplycost") == col("min_cost"))
        .join(part.filter(col("p_size") == 15), left_on="ps_partkey", right_on="p_partkey")
        .select("s_acctbal", "s_name", "n_name", "ps_partkey", "p_mfgr", "ps_supplycost")
        .sort("s_acctbal", descending=True)
        .limit(20)
    )

    result = winners.to_pydict()
    print(f"{len(result['ps_partkey'])} cheapest-supplier rows")
    for name, balance in list(zip(result["s_name"], result["s_acctbal"], strict=True))[:5]:
        print(f"  {name} {balance:>12,.2f}")

    # Ordering is the contract of the query: account balances descend.
    assert result["s_acctbal"] == sorted(result["s_acctbal"], reverse=True)
    # Every surviving offer really is a minimum for its part, so re-deriving the minimum
    # from the full offer set must agree.
    costs = dict(
        zip(
            cheapest.to_pydict()["ps_partkey"],
            cheapest.to_pydict()["min_cost"],
            strict=True,
        )
    )
    assert all(
        costs[partkey] == cost
        for partkey, cost in zip(result["ps_partkey"], result["ps_supplycost"], strict=True)
    )


if __name__ == "__main__":
    main()
