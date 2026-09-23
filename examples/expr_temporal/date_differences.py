"""Measuring the gap between two dates.

Subtracting one date column from another gives whole days, which is the direct spelling
and the one to reach for.

The `*_between` family is the named alternative. It reads a `date32` column as midnight
UTC, so `days_between` agrees with subtraction on dates and on timestamps alike, and an
explicit cast changes nothing.

    python examples/expr_temporal/date_differences.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").select("l_orderkey", "l_shipdate", "l_commitdate", "l_receiptdate")

    gaps = lineitem.select(
        "l_shipdate",
        "l_receiptdate",
        # Direct subtraction of two date columns: whole days.
        transit_days=col("l_receiptdate") - col("l_shipdate"),
        # The named form, on timestamps. `days_between(other)` counts from `other`.
        transit_named=col("l_receiptdate")
        .cast("timestamp")
        .dt.days_between(col("l_shipdate").cast("timestamp")),
        # The same call on the raw date columns, with no cast.
        transit_uncast=col("l_receiptdate").dt.days_between(col("l_shipdate")),
        lateness=col("l_receiptdate") - col("l_commitdate"),
    )

    sample = gaps.limit(5).to_pydict()
    print(sample)

    full = gaps.to_pydict()

    # All three spellings agree, cast or not.
    assert full["transit_days"] == full["transit_named"]
    assert full["transit_days"] == full["transit_uncast"]

    # A shipment is always received after it ships, so transit time is positive.
    stats = gaps.agg(
        min_transit=col("transit_days").min(),
        max_transit=col("transit_days").max(),
        mean_transit=col("transit_days").mean(),
    ).to_pydict()
    print(stats)
    assert stats["min_transit"][0] > 0

    # Lateness is signed: negative when the line arrived before it was due.
    early = gaps.filter(col("lateness") < 0).count()
    late = gaps.filter(col("lateness") > 0).count()
    print(f"{early} early, {late} late")
    assert early > 0 and late > 0


if __name__ == "__main__":
    main()
