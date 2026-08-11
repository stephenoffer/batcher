"""Cumulative sums: an ordered window with an unbounded preceding frame.

A running total is an aggregate over "every row up to this one", which is a frame, not a
partition. Leaving the frame off gives the partition total on every row instead — a
different and much less useful number.

    python examples/windows/running_totals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders")

    daily = (
        orders.group_by("o_orderdate")
        .agg(revenue=col("o_totalprice").sum())
        .sort("o_orderdate")
        .head(30)
    )

    cumulative = daily.with_columns(
        # From the start of the partition to the current row.
        running=col("revenue").sum().over(order_by=["o_orderdate"], frame=(None, 0)),
        # No frame: the total for the whole partition, repeated.
        grand_total=col("revenue").sum().over(),
    ).sort("o_orderdate")

    result = cumulative.to_pydict()
    print([round(value) for value in result["running"][:5]])

    # A running total is non-decreasing when the values are positive.
    assert all(
        earlier <= later
        for earlier, later in zip(result["running"], result["running"][1:], strict=False)
    )
    # It starts at the first value and ends at the total.
    assert abs(result["running"][0] - result["revenue"][0]) < 1e-6
    assert abs(result["running"][-1] - sum(result["revenue"])) < 1e-3
    # The unframed version is that same total on every row.
    assert all(abs(value - result["running"][-1]) < 1e-3 for value in result["grand_total"])


if __name__ == "__main__":
    main()
