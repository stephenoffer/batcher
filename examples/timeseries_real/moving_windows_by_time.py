"""Windows measured in days rather than in rows.

A row-count frame assumes every period is present. When days are missing, "the last seven
rows" is not "the last seven days". Densifying the series first is what makes the two the
same thing — and is usually cheaper than a range frame.

    python examples/timeseries_real/moving_windows_by_time.py
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

    sparse = (
        orders.group_by("o_orderdate")
        .agg(revenue=col("o_totalprice").sum())
        .sort("o_orderdate")
        .head(60)
    )
    values = sparse.to_pydict()
    first, last = values["o_orderdate"][0], values["o_orderdate"][-1]
    span = (last - first).days + 1
    print(f"{sparse.count()} days with orders across {span} calendar days")

    # A row frame over the sparse series is not a seven-day window.
    row_framed = sparse.with_columns(
        seven_rows=col("revenue").mean().over(order_by=["o_orderdate"], frame=(-6, 0))
    ).sort("o_orderdate")

    # Densify, then the row frame *is* a day frame.
    calendar = bt.date_range(first, last, interval="1d")
    day_column = calendar.columns[0]
    dense = (
        calendar.join(sparse, left_on=day_column, right_on="o_orderdate", how="left")
        .with_columns(revenue=bt.coalesce(col("revenue"), bt.lit(0.0)))
        .sort(day_column)
    )
    assert dense.count() == span

    day_framed = dense.with_columns(
        seven_days=col("revenue").mean().over(order_by=[day_column], frame=(-6, 0))
    ).sort(day_column)

    sparse_result = row_framed.to_pydict()
    dense_result = day_framed.to_pydict()
    print("row-framed  first 3:", [round(v) for v in sparse_result["seven_rows"][:3]])
    print("day-framed  first 3:", [round(v) for v in dense_result["seven_days"][:3]])

    # Both are valid means; they differ whenever a day is missing, which is the point.
    assert len(dense_result["seven_days"]) >= len(sparse_result["seven_rows"])
    assert all(value >= 0 for value in dense_result["seven_days"])

    # The dense total still matches the sparse one — densifying adds zeros, not data.
    assert abs(sum(dense_result["revenue"]) - sum(sparse_result["revenue"])) < 1e-3


if __name__ == "__main__":
    main()
