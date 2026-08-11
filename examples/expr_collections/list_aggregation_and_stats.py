"""Reducing a list column without exploding it.

A list column supports its own reductions, so "the total of each order's line quantities"
needs no explode and no second group-by. Keeping the row intact is the point: the summary
lands beside the detail rather than replacing it.

    python examples/expr_collections/list_aggregation_and_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    per_order = (
        tpch("lineitem")
        .head(30_000)
        .group_by("l_orderkey")
        .agg(quantities=bt.array_agg(col("l_quantity")))
        .sort("l_orderkey")
    )

    reduced = per_order.select(
        "l_orderkey",
        lines=col("quantities").list.len(),
        total=col("quantities").list.sum(),
        biggest=col("quantities").list.max(),
        smallest=col("quantities").list.min(),
        average=col("quantities").list.mean(),
    )
    result = reduced.head(5).to_pydict()
    print(result)

    full = reduced.to_pydict()

    # The reductions are internally consistent, row by row.
    assert all(low <= high for low, high in zip(full["smallest"], full["biggest"], strict=True))
    assert all(
        abs(total / lines - mean) < 1e-9
        for total, lines, mean in zip(full["total"], full["lines"], full["average"], strict=True)
    )

    # And they agree with the grouped aggregate over the same data.
    direct = (
        tpch("lineitem")
        .head(30_000)
        .group_by("l_orderkey")
        .agg(total=col("l_quantity").sum(), lines=bt.count())
        .sort("l_orderkey")
        .to_pydict()
    )
    assert full["total"] == direct["total"]
    assert full["lines"] == direct["lines"]

    # Sorting and deduplicating inside the list, still without exploding.
    tidied = per_order.select(
        unique=col("quantities").list.set_union(col("quantities")).list.len(),
        length=col("quantities").list.len(),
    ).to_pydict()
    assert all(
        unique <= length for unique, length in zip(tidied["unique"], tidied["length"], strict=True)
    )


if __name__ == "__main__":
    main()
