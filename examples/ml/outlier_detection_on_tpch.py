"""Finding outliers, and deciding what to do about them.

An outlier is not a defect — it is a row that does not fit the model you were about to
build. Flag them first and look, because dropping them silently is how a real signal gets
deleted.

    python examples/ml/outlier_detection_on_tpch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col, ml


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_totalprice")

    bounds = ml.outlier_bounds(orders, "o_totalprice")
    print("bounds:", bounds)

    flagged = ml.flag_outliers(orders, "o_totalprice")
    flag_column = next(name for name in flagged.columns if name not in orders.columns)
    count = flagged.filter(col(flag_column)).count()
    rate = count / orders.count()
    print(f"{count} outliers ({rate:.4%})")

    assert 0 < count < orders.count()

    # `count_outliers` is the summary form, keyed by column so it can take several at
    # once. It agrees with the row-level flag.
    counted = ml.count_outliers(orders, "o_totalprice")
    print("count_outliers:", counted)
    assert counted["o_totalprice"] == count

    # The flagged rows really are extreme.
    extremes = (
        flagged.filter(col(flag_column))
        .agg(low=col("o_totalprice").min(), high=col("o_totalprice").max())
        .to_pydict()
    )
    typical = orders.agg(median=bt.median(col("o_totalprice"))).to_pydict()["median"][0]
    print("outlier range:", extremes, "median:", round(typical, 2))
    assert extremes["high"][0] > typical

    # Clipping keeps the rows and bounds the values, which preserves the row count.
    clipped = ml.OutlierClipper("o_totalprice").fit(orders).transform(orders)
    assert clipped.count() == orders.count()
    assert (
        clipped.agg(m=col("o_totalprice").max()).to_pydict()["m"][0]
        <= orders.agg(m=col("o_totalprice").max()).to_pydict()["m"][0]
    )


if __name__ == "__main__":
    main()
