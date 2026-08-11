"""Rolling statistics beyond the mean: standard deviation, min and max over a frame.

`sum`, `mean`, `min` and `max` all take an explicit frame. `stddev` does not — the engine
refuses rather than silently computing it over the partition — so a rolling deviation is
built from the framed pieces: E[x^2] minus E[x]^2 over the same window.

    python examples/windows/rolling_statistics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    daily = (
        tpch("orders")
        .group_by("o_orderdate")
        .agg(revenue=col("o_totalprice").sum())
        .sort("o_orderdate")
        .head(60)
    )

    frame = (-9, 0)
    rolling = (
        daily.with_columns(
            mean=col("revenue").mean().over(order_by=["o_orderdate"], frame=frame),
            low=col("revenue").min().over(order_by=["o_orderdate"], frame=frame),
            high=col("revenue").max().over(order_by=["o_orderdate"], frame=frame),
            mean_square=(col("revenue") * col("revenue"))
            .mean()
            .over(order_by=["o_orderdate"], frame=frame),
        )
        .with_columns(
            # Variance as E[x^2] - E[x]^2, then its square root. Clamp at zero: the identity
            # is exact in real arithmetic and can land a hair below zero in floating point.
            spread=(col("mean_square") - col("mean") * col("mean")).clip(0.0, None).sqrt(),
        )
        .sort("o_orderdate")
    )

    result = rolling.to_pydict()
    print([round(value) for value in result["mean"][:4]])

    # The window's mean is bracketed by its own extremes.
    assert all(
        low <= mean <= high + 1e-6
        for mean, low, high in zip(result["mean"], result["low"], result["high"], strict=True)
    )

    # The tenth row onwards spans a full ten days, checkable by hand.
    window = result["revenue"][0:10]
    assert abs(result["mean"][9] - sum(window) / 10) < 1e-6
    assert result["low"][9] == min(window)
    assert result["high"][9] == max(window)

    # The first row's window holds one value, so its spread is zero.
    assert abs(result["spread"][0]) < 1e-6

    # The tenth row's deviation matches the one computed by hand over the same ten values.
    mean_ten = sum(window) / 10
    by_hand = (sum((value - mean_ten) ** 2 for value in window) / 10) ** 0.5
    assert abs(result["spread"][9] - by_hand) < 1e-3

    # A control chart: which days sit more than two rolling deviations from the trend.
    outliers = rolling.filter(
        col("spread").is_not_null() & ((col("revenue") - col("mean")).abs() > 2 * col("spread"))
    )
    print("days outside two rolling deviations:", outliers.count())
    assert outliers.count() < daily.count()
    assert bt is not None


if __name__ == "__main__":
    main()
