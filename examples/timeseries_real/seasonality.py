"""Finding a weekly and monthly pattern in a real order series.

Seasonality is a group-by on a date part. Comparing each period against the overall mean
turns the counts into an index, which is what makes a Tuesday in one series comparable to a
Tuesday in another.

    python examples/timeseries_real/seasonality.py
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
    overall = orders.count() / 7.0

    by_weekday = (
        orders.with_columns(day=col("o_orderdate").dt.day_name())
        .group_by("day")
        .agg(orders=bt.count(), revenue=col("o_totalprice").sum())
        .with_columns(index=col("orders") / overall)
        .sort("orders", descending=True)
        .to_pydict()
    )

    for day, count, index in zip(
        by_weekday["day"], by_weekday["orders"], by_weekday["index"], strict=True
    ):
        print(f"  {day:<10} {count:>7} orders  index {index:.3f}")

    assert len(by_weekday["day"]) == 7
    assert sum(by_weekday["orders"]) == orders.count()

    # The index is centred on 1 by construction.
    assert abs(sum(by_weekday["index"]) - 7.0) < 1e-6

    # TPC-H order dates are uniform, so no weekday stands out much — which is the correct
    # reading and the reason to compute the index rather than eyeball the counts.
    spread = max(by_weekday["index"]) - min(by_weekday["index"])
    print(f"weekday index spread: {spread:.4f}")
    assert spread < 0.2

    # Monthly, for comparison.
    by_month = (
        orders.with_columns(month=col("o_orderdate").dt.month())
        .group_by("month")
        .agg(orders=bt.count())
        .sort("month")
        .to_pydict()
    )
    assert by_month["month"] == list(range(1, 13))
    assert sum(by_month["orders"]) == orders.count()


if __name__ == "__main__":
    main()
