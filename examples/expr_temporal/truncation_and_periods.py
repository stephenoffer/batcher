"""Rounding a date down to a period, which is how you group a time series.

`truncate` maps every date in a period to the same value, so grouping by it produces one
row per period. That is a different thing from extracting the month number: truncation
keeps the year, so January 1995 and January 1996 stay separate.

    python examples/expr_temporal/truncation_and_periods.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderdate", "o_totalprice")

    monthly = (
        orders.with_columns(period=col("o_orderdate").dt.truncate("month"))
        .group_by("period")
        .agg(orders=bt.count(), revenue=col("o_totalprice").sum())
        .sort("period")
    )
    result = monthly.to_pydict()
    print(result["period"][:4], result["orders"][:4])

    # One row per calendar month, and every truncated date is the first of its month.
    # Truncating a date yields a timestamp at midnight on the first of the month.
    assert all(value.day == 1 for value in result["period"])
    assert result["period"] == sorted(result["period"])
    assert sum(result["orders"]) == orders.count()

    # Extracting the month number is a *different* grouping: it folds years together.
    by_month_number = orders.with_columns(month=col("o_orderdate").dt.month()).n_unique("month")
    print(f"{len(result['period'])} months vs {by_month_number} month numbers")
    assert by_month_number <= 12
    assert len(result["period"]) > by_month_number

    # Period boundaries, without the group-by.
    edges = (
        orders.select(
            "o_orderdate",
            month_start=col("o_orderdate").dt.month_start(),
            month_end=col("o_orderdate").dt.month_end(),
            quarter_start=col("o_orderdate").dt.quarter_start(),
            year_start=col("o_orderdate").dt.year_start(),
        )
        .head(3)
        .to_pydict()
    )
    print(edges)
    # Note the asymmetry: the `*_start` helpers return a timestamp (they are truncations)
    # while `month_end` returns a date. Normalize before comparing them, or Python raises
    # on the mixed comparison.
    assert all(
        start.date() <= date <= end
        for date, start, end in zip(
            edges["o_orderdate"], edges["month_start"], edges["month_end"], strict=True
        )
    )


if __name__ == "__main__":
    main()
