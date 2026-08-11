"""Working with a list column: length, indexing, and membership.

A list column keeps a variable number of values per row without leaving the columnar
layout, so these operations are vectorized rather than a Python loop over rows. Indexing
past the end gives null, not an error.

    python examples/expr_collections/list_basics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    # Real lists, built by collecting each order's part keys.
    per_order = (
        tpch("lineitem")
        .head(20_000)
        .group_by("l_orderkey")
        .agg(parts=bt.array_agg(col("l_partkey")))
        .sort("l_orderkey")
    )

    described = per_order.select(
        "l_orderkey",
        size=col("parts").list.len(),
        first=col("parts").list.first(),
        last=col("parts").list.last(),
        beyond=col("parts").list.get(20),
    )

    result = described.head(5).to_pydict()
    print(result)

    assert all(value >= 1 for value in result["size"])
    # No order has 21 lines, so index 20 is out of range everywhere: null, not an error.
    assert all(value is None for value in described.to_pydict()["beyond"])

    # A single-element list has the same first and last value.
    singles = described.filter(col("size") == 1).to_pydict()
    assert all(first == last for first, last in zip(singles["first"], singles["last"], strict=True))

    # Membership, without exploding the list into rows.
    target = result["first"][0]
    holders = per_order.filter(col("parts").list.contains(target))
    print(f"orders containing part {target}: {holders.count()}")
    assert holders.count() >= 1


if __name__ == "__main__":
    main()
