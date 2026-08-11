"""Effect size: how big a difference is, not just whether it exists.

With enough rows every difference is statistically significant, so significance stops being
informative. Cohen's d divides the difference by the pooled spread, which makes it comparable
across metrics and across sample sizes.

    python examples/statistics/effect_sizes.py
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

    def moments(dataset: bt.Dataset, column: str) -> tuple[float, float, int]:
        row = dataset.agg(mean=col(column).mean(), sd=bt.std(col(column)), n=bt.count()).to_pydict()
        return row["mean"][0], row["sd"][0], row["n"][0]

    def cohens_d(left: bt.Dataset, right: bt.Dataset, column: str) -> float:
        left_mean, left_sd, left_n = moments(left, column)
        right_mean, right_sd, right_n = moments(right, column)
        pooled = (
            ((left_n - 1) * left_sd**2 + (right_n - 1) * right_sd**2) / (left_n + right_n - 2)
        ) ** 0.5
        return (left_mean - right_mean) / pooled

    # A comparison with a real difference: discounted lines cost more, because a bigger
    # order attracts a discount.
    discounted = lineitem.filter(col("l_discount") > 0.05)
    plain = lineitem.filter(col("l_discount") <= 0.05)
    price_effect = cohens_d(discounted, plain, "l_extendedprice")
    print(f"price, discounted vs not: d = {price_effect:.4f}")

    # A comparison with essentially none: quantity is assigned independently of ship mode.
    air = lineitem.filter(col("l_shipmode") == "AIR")
    ship = lineitem.filter(col("l_shipmode") == "SHIP")
    quantity_effect = cohens_d(air, ship, "l_quantity")
    print(f"quantity, AIR vs SHIP:    d = {quantity_effect:.4f}")

    # Both are finite and signed.
    assert abs(price_effect) < 10
    assert abs(quantity_effect) < 10

    # The independent comparison has a negligible effect size, by the usual convention.
    assert abs(quantity_effect) < 0.2

    # A group compared against itself has an effect size of exactly zero.
    assert abs(cohens_d(air, air, "l_quantity")) < 1e-12

    # And the sample sizes are large enough that the negligible difference would still be
    # "significant" on a naive test — which is the whole reason to report the effect size.
    print(f"AIR n={air.count()}, SHIP n={ship.count()}")
    assert air.count() > 1_000
    assert ship.count() > 1_000


if __name__ == "__main__":
    main()
