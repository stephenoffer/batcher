"""Bucketing elapsed times into service-level bands.

An SLA report is a histogram over a duration, and the bands are a business rule rather than a
statistic. Writing them as an explicit CASE keeps the boundaries visible and reviewable,
which is what a report of this kind has to be.

    python examples/expr_temporal/duration_bucketing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").select("l_shipdate", "l_receiptdate", "l_shipmode")

    banded = lineitem.with_columns(transit=col("l_receiptdate") - col("l_shipdate")).with_columns(
        band=bt.when(col("transit") <= 7)
        .then(bt.lit("within a week"))
        .when(col("transit") <= 14)
        .then(bt.lit("within two weeks"))
        .when(col("transit") <= 30)
        .then(bt.lit("within a month"))
        .otherwise(bt.lit("longer"))
    )

    report = (
        banded.group_by("band")
        .agg(lines=bt.count(), longest=col("transit").max(), shortest=col("transit").min())
        .sort("shortest")
    )
    result = report.to_pydict()
    for row in zip(
        result["band"], result["lines"], result["shortest"], result["longest"], strict=True
    ):
        print(f"  {row[0]:<18} {row[1]:>7} lines  {row[2]}-{row[3]} days")

    # The bands partition the data.
    assert sum(result["lines"]) == lineitem.count()
    assert len(set(result["band"])) == len(result["band"])

    # They are ordered and non-overlapping.
    assert result["shortest"] == sorted(result["shortest"])
    assert all(
        longest < next_shortest
        for longest, next_shortest in zip(result["longest"], result["shortest"][1:], strict=False)
    )

    # And each band's range respects its own boundary.
    by_band = dict(zip(result["band"], result["longest"], strict=True))
    assert by_band["within a week"] <= 7
    assert by_band.get("within two weeks", 0) <= 14

    # The SLA number the report exists for.
    within_a_week = next(
        count
        for band, count in zip(result["band"], result["lines"], strict=True)
        if band == "within a week"
    )
    print(f"within a week: {within_a_week / lineitem.count():.2%}")
    assert 0.0 < within_a_week / lineitem.count() < 1.0


if __name__ == "__main__":
    main()
