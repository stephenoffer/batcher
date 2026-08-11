"""Projection and predicate pushdown: reading less of a file, not filtering after it.

Both optimizations happen in the reader. A projection means the column's bytes are never
fetched; a predicate means whole row groups are skipped using their stored statistics.
Over object storage that is the difference between a query and a download.

    python examples/io/parquet_pushdown.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch_path
from batcher import col


def main() -> None:
    path = tpch_path("lineitem")

    # Two of sixteen columns: the other fourteen are never decoded.
    narrow = bt.read.parquet(path).select("l_orderkey", "l_shipdate")
    plan = narrow.explain()
    print(plan)
    assert narrow.width == 2

    # The reader can also be told directly which columns to read.
    explicit = bt.read.parquet(path, columns=["l_orderkey", "l_shipdate"])
    assert explicit.columns == ["l_orderkey", "l_shipdate"]
    assert explicit.count() == narrow.count()

    # A predicate on a sorted-ish column lets whole row groups be skipped by statistics.
    recent = bt.read.parquet(path).filter(col("l_shipdate") >= bt.lit(dt.date(1998, 1, 1)))
    print("rows after the date filter:", recent.count())
    assert 0 < recent.count() < bt.read.parquet(path).count()

    # The filter sits against the scan in the plan, which is what makes it pushable.
    filtered_plan = recent.explain()
    print(filtered_plan)
    assert "filter" in filtered_plan.lower()
    assert "scan" in filtered_plan.lower()


if __name__ == "__main__":
    main()
