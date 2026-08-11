"""Skewness and kurtosis: is this distribution lopsided, and how heavy are its tails.

Both are worth computing before you trust a mean. A near-zero skew on `l_quantity` says
the mean is a fair summary; a large positive skew on `l_extendedprice` says it is not,
and that the median is the number to report.

    python examples/aggregations/distribution_shape.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")

    shape = lineitem.agg(
        qty_skew=bt.skewness(col("l_quantity")),
        qty_kurtosis=bt.kurtosis(col("l_quantity")),
        price_skew=bt.skewness(col("l_extendedprice")),
        price_kurtosis=bt.kurtosis(col("l_extendedprice")),
        qty_mean=col("l_quantity").mean(),
        qty_median=bt.median(col("l_quantity")),
        price_mean=col("l_extendedprice").mean(),
        price_median=bt.median(col("l_extendedprice")),
    ).to_pydict()
    print({name: round(value[0], 4) for name, value in shape.items()})

    # Quantity is uniform over 1..50: symmetric, and flatter than a normal distribution.
    assert abs(shape["qty_skew"][0]) < 0.1
    assert shape["qty_kurtosis"][0] < 0.0
    assert abs(shape["qty_mean"][0] - shape["qty_median"][0]) < 1.0

    # Price is right-skewed, so its mean sits above its median.
    assert shape["price_skew"][0] > 0.3
    assert shape["price_mean"][0] > shape["price_median"][0]


if __name__ == "__main__":
    main()
