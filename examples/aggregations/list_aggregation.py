"""Collecting a group's values into a list column.

`array_agg` is the escape hatch from the flat relational world: it keeps the members of a
group rather than reducing them. It is also unbounded — a group with a million rows makes
a list with a million entries — so it wants a bounded key, not a high-cardinality one.

    python examples/aggregations/list_aggregation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").head(20_000)

    per_order = (
        lineitem.group_by("l_orderkey")
        .agg(
            parts=bt.array_agg(col("l_partkey")),
            quantities=bt.array_agg(col("l_quantity")),
            lines=bt.count(),
        )
        .sort("l_orderkey")
        .head(5)
        .to_pydict()
    )
    print(per_order["l_orderkey"], per_order["lines"])
    print("first order parts:", per_order["parts"][0])

    # The list length is the group size, by construction.
    assert all(
        len(parts) == lines
        for parts, lines in zip(per_order["parts"], per_order["lines"], strict=True)
    )

    # A list column supports the usual list expressions, so the reduction can happen
    # after the collection rather than instead of it.
    rebuilt = (
        lineitem.group_by("l_orderkey")
        .agg(quantities=bt.array_agg(col("l_quantity")))
        .with_columns(total=col("quantities").list.sum())
        .sort("l_orderkey")
        .head(5)
        .to_pydict()
    )
    direct = (
        lineitem.group_by("l_orderkey")
        .agg(total=col("l_quantity").sum())
        .sort("l_orderkey")
        .head(5)
        .to_pydict()
    )
    assert rebuilt["total"] == direct["total"]


if __name__ == "__main__":
    main()
