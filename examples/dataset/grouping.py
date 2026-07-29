"""Grouping: agg, multi-key rollups, and the cube/rollup/grouping-set variants.

``group_by().agg()`` is the workhorse. ``rollup`` and ``cube`` compute subtotals in the
same pass, which is how you build a report with per-region, per-product, and grand-total
rows without three separate queries and a union.

    python examples/dataset/grouping.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    sales = bt.from_pydict(
        {
            "region": ["us", "us", "eu", "eu"],
            "product": ["a", "b", "a", "b"],
            "amount": [10, 20, 30, 40],
        }
    )

    # One key, several aggregates in one pass.
    by_region = (
        sales.group_by("region")
        .agg(
            total=col("amount").sum(),
            average=col("amount").mean(),
            biggest=col("amount").max(),
            n=bt.count(),
            distinct_products=col("product").n_unique(),
        )
        .sort("region")
        .to_pydict()
    )
    print(by_region)
    assert by_region["region"] == ["eu", "us"]
    assert by_region["total"] == [70, 30]
    assert by_region["n"] == [2, 2]

    # Several keys.
    by_both = (
        sales.group_by("region", "product")
        .agg(total=col("amount").sum())
        .sort("region", "product")
        .to_pydict()
    )
    assert len(by_both["total"]) == 4

    # A derived key, computed inline.
    by_initial = (
        sales.group_by(initial=col("region").str.head(1))
        .agg(total=col("amount").sum())
        .sort("initial")
        .to_pydict()
    )
    print(by_initial)
    assert by_initial["initial"] == ["e", "u"]

    # `rollup` adds the subtotals along the key prefix, plus a grand total.
    rolled = sales.rollup("region", "product").agg(total=col("amount").sum()).to_pydict()
    print("rollup rows:", len(rolled["total"]))
    assert len(rolled["total"]) > 4  # the leaf groups plus subtotals

    # `cube` adds every combination of the keys.
    cubed = sales.cube("region", "product").agg(total=col("amount").sum()).to_pydict()
    assert len(cubed["total"]) >= len(rolled["total"])

    # `grouping_sets` when you want specific combinations rather than all of them. Each
    # set is a separate positional argument; `()` is the grand total.
    sets = (
        sales.grouping_sets(["region"], ["product"], ()).agg(total=col("amount").sum()).to_pydict()
    )
    print("grouping sets rows:", len(sets["total"]))
    # 2 regions + 2 products + 1 grand total.
    assert len(sets["total"]) == 5

    # Filtering after aggregation is just another filter on the result.
    big = (
        sales.group_by("region")
        .agg(total=col("amount").sum())
        .filter(col("total") > 50)
        .to_pydict()
    )
    assert big["region"] == ["eu"]


if __name__ == "__main__":
    main()
