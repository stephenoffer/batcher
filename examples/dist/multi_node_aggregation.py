"""A grouped aggregate across partitions, checked against the single-node answer.

Every aggregate used here is mergeable, so the partitioned result must equal the whole-table
one. Running the check for each aggregate separately is what makes a failure attributable to
one of them rather than to "the distributed path".

    python examples/dist/multi_node_aggregation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import resolve_distributed, tpch
from batcher import col


def main() -> None:
    distributed = resolve_distributed()
    lineitem = tpch("lineitem")

    aggregates = {
        "count": bt.count(),
        "sum": col("l_quantity").sum(),
        "min": col("l_extendedprice").min(),
        "max": col("l_extendedprice").max(),
        "mean": col("l_quantity").mean(),
        "n_unique": col("l_partkey").n_unique(),
        "approx_n_unique": bt.approx_n_unique(col("l_partkey")),
        "bool_or": bt.bool_or(col("l_quantity") > 40),
        "bit_or": bt.bit_or(col("l_linenumber")),
    }

    for name, aggregate in aggregates.items():
        query = (
            lineitem.group_by("l_shipmode")
            .agg(**{name: aggregate})
            .sort("l_shipmode")
        )
        single = query.collect(distributed=False, num_partitions=1).to_pydict()
        many = query.collect(distributed=distributed, num_partitions=8).to_pydict()

        assert single["l_shipmode"] == many["l_shipmode"], name
        left, right = single[name], many[name]
        if left and isinstance(left[0], float):
            assert all(
                abs(a - b) <= max(abs(a), 1.0) * 1e-12
                for a, b in zip(left, right, strict=True)
            ), name
        else:
            assert left == right, name
        print(f"  {name:<18} agrees across 1 and 8 partitions")

    print(f"{len(aggregates)} aggregates verified mergeable")


if __name__ == "__main__":
    main()
