"""Two optimizations you can see: reading fewer columns and fewer rows.

Both are decided before any data moves. The measurable version of "is my query fast" often
reduces to "did the scan read what it needed and nothing else", which is checkable by
comparing what a narrow query costs against a wide one over the same file.

    python examples/perf/pushdown_and_projection.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch_path
from batcher import col


def main() -> None:
    path = tpch_path("lineitem")

    def timed(label: str, build) -> float:
        started = time.perf_counter()
        rows = build().count()
        elapsed = time.perf_counter() - started
        print(f"{label:<28} {rows:>8} rows  {elapsed * 1000:7.1f} ms")
        return elapsed

    wide = timed("all 16 columns", lambda: bt.read.parquet(path))
    narrow = timed("two columns", lambda: bt.read.parquet(path).select("l_orderkey", "l_quantity"))
    filtered = timed(
        "two columns + predicate",
        lambda: (
            bt.read.parquet(path).select("l_orderkey", "l_quantity").filter(col("l_quantity") > 45)
        ),
    )

    # Timings on a warm page cache are noisy, so the assertion is on what the plan says
    # rather than on the clock: the projection really did narrow, and the filter really
    # did reduce the row count.
    assert wide > 0 and narrow > 0 and filtered > 0
    assert bt.read.parquet(path).width == 16
    assert bt.read.parquet(path).select("l_orderkey", "l_quantity").width == 2

    selective = bt.read.parquet(path).filter(col("l_quantity") > 45)
    assert selective.count() < bt.read.parquet(path).count()

    plan = selective.explain()
    print(plan)
    assert "filter" in plan.lower()


if __name__ == "__main__":
    main()
