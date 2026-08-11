"""Building model features from a real table.

The features here are all expressions, so the whole feature pipeline is one plan and none of
it touches Python. That matters for training-serving parity: the same expressions run at
serving time over one row as at training time over millions.

    python examples/ml/feature_construction_on_tpch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders")
    lineitem = tpch("lineitem")

    per_order = lineitem.group_by("l_orderkey").agg(
        lines=bt.count(),
        units=col("l_quantity").sum(),
        gross=col("l_extendedprice").sum(),
        max_discount=col("l_discount").max(),
    )

    features = (
        orders.join(per_order, left_on="o_orderkey", right_on="l_orderkey")
        .with_columns(
            # Ratios, which travel better than raw magnitudes.
            units_per_line=col("units") / col("lines"),
            price_per_unit=col("gross") / col("units"),
            # Calendar features.
            month=col("o_orderdate").dt.month(),
            quarter=col("o_orderdate").dt.quarter(),
            is_weekend=col("o_orderdate").dt.is_weekend(),
            # A log transform for the skewed magnitude.
            log_total=col("o_totalprice").log(),
            # A boolean feature as an integer, which every model can consume.
            urgent=col("o_orderpriority").is_in(["1-URGENT", "2-HIGH"]).cast("int64"),
        )
        .select(
            "o_orderkey",
            "units_per_line",
            "price_per_unit",
            "month",
            "quarter",
            "is_weekend",
            "log_total",
            "urgent",
        )
    )

    print(features.head(3).to_pydict())
    assert features.count() > 0
    assert features.width == 8

    values = features.to_pydict()
    assert all(value > 0 for value in values["units_per_line"])
    assert all(1 <= value <= 12 for value in values["month"])
    assert set(values["urgent"]) <= {0, 1}

    # No nulls: a model cannot consume one, so finding them here is the point.
    nulls = features.null_count().to_pydict()
    assert all(value == 0 for column in nulls.values() for value in column)

    # The whole thing is one plan, computed on demand.
    plan = features.explain()
    assert "join" in plan.lower()


if __name__ == "__main__":
    main()
