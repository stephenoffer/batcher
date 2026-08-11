"""Sliding windows: a moving average over a bounded frame.

`frame=(-6, 0)` is "this row and the six before it". The first rows of a partition have
fewer than seven predecessors, so the window is short there rather than null — which is
what you want for a chart and worth knowing before you compare it to a lagged series.

    python examples/windows/moving_averages.py
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
        .head(40)
    )

    smoothed = daily.with_columns(
        weekly_mean=col("revenue").mean().over(order_by=["o_orderdate"], frame=(-6, 0)),
        weekly_max=col("revenue").max().over(order_by=["o_orderdate"], frame=(-6, 0)),
        centered=col("revenue").mean().over(order_by=["o_orderdate"], frame=(-3, 3)),
    ).sort("o_orderdate")

    result = smoothed.to_pydict()
    print([round(value) for value in result["weekly_mean"][:5]])

    # The first row's window holds only itself.
    assert abs(result["weekly_mean"][0] - result["revenue"][0]) < 1e-6

    # A mean over a window can never exceed the maximum over the same window.
    assert all(
        mean <= maximum + 1e-6
        for mean, maximum in zip(result["weekly_mean"], result["weekly_max"], strict=True)
    )

    # The seventh row onwards is a genuine seven-day mean, checkable by hand.
    window = result["revenue"][0:7]
    assert abs(result["weekly_mean"][6] - sum(window) / 7) < 1e-6

    # A centred window looks forwards as well as back, so it differs from the trailing one.
    assert result["centered"][10] != result["weekly_mean"][10]


if __name__ == "__main__":
    main()
