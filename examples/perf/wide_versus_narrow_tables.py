"""What column count costs, and why a projection is the first optimization.

A columnar reader pays per column, so a query touching two of sixteen columns should cost
about an eighth of one touching all of them. When it does not, the bottleneck is somewhere
else — which is itself the useful finding.

    python examples/perf/wide_versus_narrow_tables.py
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
    full = bt.read.parquet(path)
    print(f"{full.width} columns, {full.count()} rows")

    def timed(label: str, columns: list[str]) -> float:
        build = lambda: bt.read.parquet(path, columns=columns).count()  # noqa: E731
        build()
        started = time.perf_counter()
        build()
        elapsed = (time.perf_counter() - started) * 1000
        print(f"{label:<24} {len(columns):>2} columns  {elapsed:7.1f} ms")
        return elapsed

    one = timed("one column", ["l_orderkey"])
    few = timed("four columns", ["l_orderkey", "l_quantity", "l_discount", "l_tax"])
    everything = timed("all columns", full.columns)

    assert one > 0 and few > 0 and everything > 0

    # Whatever the timings, the row count is identical — a projection changes what is
    # read, never how many rows there are.
    assert bt.read.parquet(path, columns=["l_orderkey"]).count() == full.count()

    # The wide read materializes the string columns, which is where the cost is: the
    # comment column alone is larger than every numeric column together.
    numeric_only = bt.read.parquet(
        path, columns=["l_orderkey", "l_partkey", "l_suppkey", "l_quantity"]
    )
    with_comment = bt.read.parquet(path, columns=["l_orderkey", "l_comment"])
    assert numeric_only.count() == with_comment.count()

    # And the aggregate over a projected column is the same either way.
    projected = (
        bt.read.parquet(path, columns=["l_quantity"])
        .agg(t=col("l_quantity").sum())
        .to_pydict()["t"][0]
    )
    from_full = full.agg(t=col("l_quantity").sum()).to_pydict()["t"][0]
    assert projected == from_full


if __name__ == "__main__":
    main()
