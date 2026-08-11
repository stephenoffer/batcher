"""Q6 four ways, and what each costs.

Q6 is one filtered scan, so it isolates the read path better than any other TPC-H query.
Comparing a wide read against a projected one, and a filtered one against an unfiltered one,
shows exactly what pushdown buys.

    python examples/tpch/q06_variants_and_pushdown.py
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch_path
from batcher import col


def main() -> None:
    path = tpch_path("lineitem")
    start, end = dt.date(1994, 1, 1), dt.date(1995, 1, 1)

    predicate = (
        (col("l_shipdate") >= bt.lit(start))
        & (col("l_shipdate") < bt.lit(end))
        & (col("l_discount") >= 0.05)
        & (col("l_discount") <= 0.07)
        & (col("l_quantity") < 24)
    )
    revenue = (col("l_extendedprice") * col("l_discount")).alias("revenue")

    def timed(label: str, build) -> float:
        started = time.perf_counter()
        value = build().agg(revenue=revenue.sum()).to_pydict()["revenue"][0]
        print(f"{label:<34} {(time.perf_counter() - started) * 1000:7.1f} ms")
        return value

    wide = timed("all columns, filtered", lambda: bt.read.parquet(path).filter(predicate))
    projected = timed(
        "four columns, filtered",
        lambda: bt.read.parquet(
            path, columns=["l_shipdate", "l_discount", "l_quantity", "l_extendedprice"]
        ).filter(predicate),
    )
    late = timed(
        "filtered after a projection",
        lambda: (
            bt.read.parquet(path)
            .select("l_shipdate", "l_discount", "l_quantity", "l_extendedprice")
            .filter(predicate)
        ),
    )

    # Every spelling is the same query, so the answer must not move.
    assert abs(wide - projected) < 1e-6
    assert abs(wide - late) < 1e-6
    print(f"forgone revenue: {wide:,.2f}")
    assert wide > 0

    # The filter really is selective, which is what makes the pushdown worth having.
    kept = bt.read.parquet(path).filter(predicate).count()
    total = bt.read.parquet(path).count()
    print(f"predicate keeps {kept} of {total} ({kept / total:.2%})")
    assert kept < total * 0.05


if __name__ == "__main__":
    main()
