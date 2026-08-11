"""Generating a date range, and using it to find the gaps in a series.

A group-by only produces rows for days that had events. Joining against a generated
calendar is what turns that into a dense series — and the rows the join adds are exactly
the days nothing happened.

    python examples/expr_temporal/date_sequences.py
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

    daily = (
        orders.group_by("o_orderdate")
        .agg(orders=bt.count(), revenue=col("o_totalprice").sum())
        .sort("o_orderdate")
    )
    observed = daily.to_pydict()
    first, last = observed["o_orderdate"][0], observed["o_orderdate"][-1]
    print(f"{daily.count()} days with orders between {first} and {last}")

    # A dense calendar over the same span.
    calendar = bt.date_range(first, last, interval="1d")
    calendar_column = calendar.columns[0]
    span_days = (last - first).days + 1
    print("calendar days:", calendar.count())
    assert calendar.count() == span_days

    # The join is what makes the series dense.
    dense = calendar.join(
        daily, left_on=calendar_column, right_on="o_orderdate", how="left"
    ).with_columns(orders=bt.coalesce(col("orders"), bt.lit(0)))

    assert dense.count() == span_days
    filled = dense.filter(col("orders") == 0).count()
    print(f"{filled} days had no orders")
    assert filled == span_days - daily.count()

    # And the totals are unchanged by densifying.
    total = dense.agg(t=col("orders").sum()).to_pydict()["t"][0]
    assert total == orders.count()


if __name__ == "__main__":
    main()
