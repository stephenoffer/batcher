"""Three ways to compute the same summary, and what each costs.

A grouped aggregate, a window plus a distinct, and a join against a pre-aggregate all produce
the same table. They do not cost the same, and which is cheapest depends on the group count —
so measure rather than assume.

    python examples/perf/aggregation_strategies.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").select("l_shipmode", "l_extendedprice")

    def grouped():
        return (
            lineitem.group_by("l_shipmode")
            .agg(revenue=col("l_extendedprice").sum())
            .sort("l_shipmode")
        )

    def windowed():
        return (
            lineitem.with_columns(
                revenue=col("l_extendedprice").sum().over(partition_by=["l_shipmode"])
            )
            .select("l_shipmode", "revenue")
            .distinct()
            .sort("l_shipmode")
        )

    def joined():
        totals = lineitem.group_by("l_shipmode").agg(revenue=col("l_extendedprice").sum())
        return (
            lineitem.select("l_shipmode")
            .distinct()
            .join(totals, on="l_shipmode")
            .sort("l_shipmode")
        )

    reference = grouped().to_pydict()
    for name, build in (("grouped", grouped), ("windowed", windowed), ("joined", joined)):
        build()
        started = time.perf_counter()
        result = build().to_pydict()
        elapsed = (time.perf_counter() - started) * 1000
        print(f"{name:<10} {elapsed:7.1f} ms")

        # All three produce the same table, which is the precondition for comparing them.
        assert result["l_shipmode"] == reference["l_shipmode"], name
        assert all(
            abs(a - b) < 1e-3 for a, b in zip(reference["revenue"], result["revenue"], strict=True)
        ), name

    # The grouped form is the one that collapses early; the windowed form carries every
    # row to the end and then deduplicates, which is why it is the wrong default.
    assert grouped().count() == lineitem.n_unique("l_shipmode")
    assert lineitem.count() > grouped().count()
    print(f"{lineitem.count()} rows collapse to {grouped().count()} groups")
    assert bt is not None


if __name__ == "__main__":
    main()
