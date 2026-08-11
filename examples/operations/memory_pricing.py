"""Memory: what a result costs to hold, and when to hold it.

`memory_usage` prices a materialized result, which is the number to check before caching
something. Caching a result larger than the memory you have is how a pipeline that used to
spill starts failing instead.

    python examples/operations/memory_pricing.py
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

    wide = lineitem.memory_usage()
    narrow = lineitem.select("l_orderkey", "l_quantity").memory_usage()
    print("full table:", wide)
    print("two columns:", narrow)

    def total(usage) -> int:
        if isinstance(usage, dict):
            return sum(int(value) for value in usage.values() if isinstance(value, int | float))
        return int(usage)

    # A projection costs less to hold than the whole table.
    assert total(narrow) < total(wide)

    # A grouped summary costs far less than the detail it came from.
    summary = lineitem.group_by("l_shipmode").agg(lines=bt.count())
    assert summary.count() < lineitem.count()

    # Cache the small thing, not the big one.
    cached = summary.cache()
    assert cached.count() == summary.count()
    assert cached.to_pydict() == summary.to_pydict()

    # Caching changes cost, never the answer — check both branches agree.
    first = cached.filter(col("lines") > 1_000).count()
    second = summary.filter(col("lines") > 1_000).count()
    assert first == second

    # `persist` is the longer-lived form.
    kept = summary.persist()
    assert kept.to_pydict() == summary.to_pydict()


if __name__ == "__main__":
    main()
