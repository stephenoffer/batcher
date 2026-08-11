"""The baselines any forecast has to beat.

Naive (yesterday's value) and seasonal-naive (the same day last week) are the two baselines
worth computing before anything else. A model that cannot beat them is not a model, and both
are one window each.

    python examples/timeseries_real/forecast_baseline.py
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
        .agg(actual=col("o_totalprice").sum())
        .sort("o_orderdate")
        .head(200)
    )

    forecast = daily.with_columns(
        naive=col("actual").shift(1).over(order_by=["o_orderdate"]),
        seasonal=col("actual").shift(7).over(order_by=["o_orderdate"]),
        trailing=col("actual").mean().over(order_by=["o_orderdate"], frame=(-7, -1)),
    ).sort("o_orderdate")

    scored = forecast.filter(
        col("naive").is_not_null() & col("seasonal").is_not_null() & col("trailing").is_not_null()
    ).with_columns(
        naive_error=(col("actual") - col("naive")).abs(),
        seasonal_error=(col("actual") - col("seasonal")).abs(),
        trailing_error=(col("actual") - col("trailing")).abs(),
    )

    errors = scored.agg(
        naive=col("naive_error").mean(),
        seasonal=col("seasonal_error").mean(),
        trailing=col("trailing_error").mean(),
        scale=col("actual").mean(),
    ).to_pydict()
    print({name: round(value[0], 1) for name, value in errors.items()})

    # Every baseline produces a finite error smaller than the series' own scale.
    for name in ("naive", "seasonal", "trailing"):
        assert errors[name][0] > 0
        assert errors[name][0] < errors["scale"][0]

    # The trailing mean smooths, so it usually beats the single-lag baselines on a series
    # with no strong trend. Asserting the ordering would be asserting a property of this
    # data, so what is asserted is that they are genuinely different forecasts.
    assert errors["naive"][0] != errors["seasonal"][0]
    assert errors["trailing"][0] != errors["naive"][0]

    # The first rows have no baseline, which is correct rather than an error.
    dropped = forecast.count() - scored.count()
    print(f"{dropped} rows have no baseline to score against")
    assert dropped >= 7
    assert bt is not None


if __name__ == "__main__":
    main()
