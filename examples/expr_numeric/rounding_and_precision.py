"""Rounding: to a place, toward zero, and away from it.

`round` goes to the nearest and takes a digit count. `floor` and `ceil` are directional
and take none. Money wants `round(2)` at the point of presentation and full precision
everywhere before it, because rounding early accumulates.

    python examples/expr_numeric/rounding_and_precision.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").select("l_extendedprice", "l_discount")

    revenue = col("l_extendedprice") * (1 - col("l_discount"))
    rounded = lineitem.select(
        exact=revenue,
        pennies=revenue.round(2),
        whole=revenue.round(0),
        down=revenue.floor(),
        up=revenue.ceil(),
    )

    sample = rounded.head(5).to_pydict()
    print(sample)

    full = rounded.to_pydict()
    # floor <= exact <= ceil, and the two differ by at most one.
    assert all(
        low <= value <= high
        for value, low, high in zip(full["exact"], full["down"], full["up"], strict=True)
    )
    assert all(high - low <= 1.0 for low, high in zip(full["down"], full["up"], strict=True))

    # Rounding to two places changes the value by less than half a penny.
    assert all(
        abs(value - pennies) <= 0.005 + 1e-9
        for value, pennies in zip(full["exact"], full["pennies"], strict=True)
    )

    # Rounding early accumulates: the sum of rounded values drifts from the rounded sum.
    totals = rounded.agg(
        sum_exact=col("exact").sum(),
        sum_rounded=col("pennies").sum(),
    ).to_pydict()
    drift = abs(totals["sum_exact"][0] - totals["sum_rounded"][0])
    print(f"drift from rounding early: {drift:.4f}")
    assert drift > 0.0


if __name__ == "__main__":
    main()
