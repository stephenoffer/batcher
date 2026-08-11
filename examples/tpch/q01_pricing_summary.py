"""TPC-H Q1 — the pricing summary report over real `lineitem` data.

Q1 is the aggregation benchmark in one query: one scan, a date predicate that keeps
almost every row, and eight aggregates grouped by a two-column key of very low
cardinality. Nothing here is a trick, which is why it is the honest measure of raw
group-by throughput.

    python examples/tpch/q01_pricing_summary.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")

    cutoff = dt.date(1998, 12, 1) - dt.timedelta(days=90)

    disc_price = col("l_extendedprice") * (1 - col("l_discount"))
    charge = disc_price * (1 + col("l_tax"))

    report = (
        lineitem.filter(col("l_shipdate") <= bt.lit(cutoff))
        .group_by("l_returnflag", "l_linestatus")
        .agg(
            sum_qty=col("l_quantity").sum(),
            sum_base_price=col("l_extendedprice").sum(),
            sum_disc_price=disc_price.sum(),
            sum_charge=charge.sum(),
            avg_qty=col("l_quantity").mean(),
            avg_price=col("l_extendedprice").mean(),
            avg_disc=col("l_discount").mean(),
            count_order=bt.count(),
        )
        .sort("l_returnflag", "l_linestatus")
    )

    result = report.to_pydict()
    print(result["l_returnflag"], result["l_linestatus"], result["count_order"])

    # The flag/status pair is drawn from a tiny domain, so the grouped output is a
    # handful of rows no matter how much of the table you read.
    assert len(result["l_returnflag"]) <= 6
    assert all(count > 0 for count in result["count_order"])
    # Every discounted price is below its base price, group by group, because the
    # discount is a fraction in [0, 0.1].
    assert all(
        disc < base
        for disc, base in zip(result["sum_disc_price"], result["sum_base_price"], strict=True)
    )
    # ... and the tax puts the charge back above the discounted price.
    assert all(
        charge_total > disc
        for charge_total, disc in zip(result["sum_charge"], result["sum_disc_price"], strict=True)
    )


if __name__ == "__main__":
    main()
