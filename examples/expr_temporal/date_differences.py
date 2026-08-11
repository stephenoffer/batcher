"""Measuring the gap between two dates.

Subtracting one date column from another gives whole days, which is the direct spelling
and the one to reach for.

The `*_between` family is the named alternative, and it works on **timestamps**. Handed a
`date32` column it returns 0 rather than raising, so cast explicitly before using it —
that silent zero is much harder to notice than an error would be.

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
        # On the raw date columns the same call silently answers zero.
        transit_uncast=col("l_receiptdate").dt.days_between(col("l_shipdate")),
        lateness=col("l_receiptdate") - col("l_commitdate"),
    )

    sample = gaps.head(5).to_pydict()
    print(sample)

    full = gaps.to_pydict()

    # The two working spellings agree.
    assert full["transit_days"] == full["transit_named"]
    # The uncast one does not, which is the trap this example exists to name.
    assert set(full["transit_uncast"]) == {0}

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
