"""Projecting a bipartite graph onto one of its sides.

Customers linked to nations is bipartite. "Which customers share a nation" is the projection
onto the customer side, and it is a self-join through the shared neighbour — which is also
why the projection of a hub explodes quadratically.

    python examples/graph/bipartite_projection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    # A small slice, because the projection is quadratic in the group size.
    customers = tpch("customer").select("c_custkey", "c_nationkey").head(200)

    per_nation = customers.group_by("c_nationkey").agg(members=bt.count()).sort("c_nationkey")
    print("nations:", per_nation.count())
    assert per_nation.count() > 1

    right = customers.select(
        col("c_custkey").alias("other"), col("c_nationkey").alias("other_nation")
    )
    pairs = customers.join(right, left_on="c_nationkey", right_on="other_nation").filter(
        col("c_custkey") < col("other")
    )

    print("co-national pairs:", pairs.count())

    # The pair count is the sum of n*(n-1)/2 over the groups.
    sizes = per_nation.to_pydict()["members"]
    expected = sum(size * (size - 1) // 2 for size in sizes)
    assert pairs.count() == expected

    # Every pair really shares a nation, and no customer is paired with itself.
    sample = pairs.select("c_custkey", "other", "c_nationkey").head(5).to_pydict()
    print(sample)
    assert all(
        left != right_key
        for left, right_key in zip(sample["c_custkey"], sample["other"], strict=True)
    )

    # The degree in the projection: how many co-nationals each customer has.
    degrees = pairs.group_by("c_custkey").agg(neighbours=bt.count()).to_pydict()
    assert max(degrees["neighbours"]) < customers.count()

    # And the quadratic blow-up, stated plainly: the largest nation dominates the total.
    largest = max(sizes)
    print(
        f"largest nation has {largest} members, contributing "
        f"{largest * (largest - 1) // 2} of {expected} pairs"
    )
    assert largest * (largest - 1) // 2 <= expected


if __name__ == "__main__":
    main()
