"""Running a query that does not fit in memory.

Spilling is what keeps a large aggregation or sort alive under a bounded budget: state
goes to disk and comes back, rather than the process dying. The result is identical to the
in-memory run, which is the property that makes it safe to leave on.

    python examples/perf/spilling_under_a_budget.py
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

    # A high-cardinality group-by: one group per order key, so the hash table is large.
    query = (
        lineitem.group_by("l_orderkey")
        .agg(revenue=col("l_extendedprice").sum(), lines=bt.count())
        .sort("l_orderkey")
    )

    in_memory = query.collect()
    print("groups:", in_memory.num_rows)

    # The same query with spilling enabled.
    spilled = query.collect(spill=True)
    print("spilled run rows:", spilled.num_rows)

    # Identical: same schema, same row count, same values in the same order.
    assert spilled.schema == in_memory.schema
    assert spilled.num_rows == in_memory.num_rows

    left = in_memory.to_pydict()
    right = spilled.to_pydict()
    assert left["l_orderkey"] == right["l_orderkey"]
    assert left["lines"] == right["lines"]
    assert all(abs(a - b) < 1e-6 for a, b in zip(left["revenue"], right["revenue"], strict=True))

    # A sort spills too, and a sorted result compared order-independently would prove
    # nothing — so this compares position by position.
    top = lineitem.sort("l_extendedprice", descending=True).head(20)
    assert (
        top.collect(spill=True).to_pydict()["l_extendedprice"]
        == top.collect().to_pydict()["l_extendedprice"]
    )


if __name__ == "__main__":
    main()
