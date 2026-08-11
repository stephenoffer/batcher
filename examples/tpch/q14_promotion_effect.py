"""TPC-H Q14 — what share of a month's revenue came from promotional parts.

One masked sum over another, times 100. The join is a plain foreign key and the window
is a single month, which makes this the query that most rewards partition pruning when
the data is laid out by date.

    python examples/tpch/q14_promotion_effect.py
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
    part = tpch("part")

    start = dt.date(1995, 9, 1)
    end = dt.date(1995, 10, 1)

    revenue = (col("l_extendedprice") * (1 - col("l_discount"))).alias("revenue")

    result = (
        lineitem.filter((col("l_shipdate") >= bt.lit(start)) & (col("l_shipdate") < bt.lit(end)))
        .join(part, left_on="l_partkey", right_on="p_partkey")
        .with_columns(revenue=revenue)
        .agg(
            promo=bt.when(col("p_type").str.starts_with("PROMO"))
            .then(col("revenue"))
            .otherwise(0.0)
            .sum(),
            total=col("revenue").sum(),
        )
        .with_columns(promo_revenue=100.0 * col("promo") / col("total"))
        .to_pydict()
    )

    print(f"promotional share: {result['promo_revenue'][0]:.4f}%")

    share = result["promo_revenue"][0]
    assert 0.0 <= share <= 100.0
    assert result["promo"][0] <= result["total"][0]


if __name__ == "__main__":
    main()
