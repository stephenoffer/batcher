"""Least-squares regression as an aggregate, not a model fit.

`regr_slope` and friends are single-pass aggregates: no iteration, no library, and they
work inside a `group_by` so you get a separate fit per group for the same cost. That is
the cheap way to ask "does this relationship hold in every segment".

    python examples/aggregations/regression.py
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

    y = col("l_extendedprice")
    x = col("l_quantity")

    fit = lineitem.agg(
        slope=bt.regr_slope(y, x),
        intercept=bt.regr_intercept(y, x),
        r_squared=bt.regr_r2(y, x),
        pairs=bt.regr_count(y, x),
        mean_x=bt.regr_avgx(y, x),
        mean_y=bt.regr_avgy(y, x),
    ).to_pydict()
    print({name: round(value[0], 4) for name, value in fit.items()})

    # Price rises with quantity, and the fit explains most of the variance.
    assert fit["slope"][0] > 0
    assert 0.0 <= fit["r_squared"][0] <= 1.0
    assert fit["r_squared"][0] > 0.5
    assert fit["pairs"][0] == lineitem.count()

    # The fitted line passes through the point of means, which is the identity that
    # catches a slope and intercept computed from different row sets.
    predicted = fit["slope"][0] * fit["mean_x"][0] + fit["intercept"][0]
    assert abs(predicted - fit["mean_y"][0]) < 1e-3

    # One fit per ship mode, same single pass.
    per_mode = (
        lineitem.group_by("l_shipmode")
        .agg(slope=bt.regr_slope(y, x), r2=bt.regr_r2(y, x))
        .sort("l_shipmode")
        .to_pydict()
    )
    print(per_mode["l_shipmode"], [round(value, 2) for value in per_mode["slope"]])
    assert all(value > 0 for value in per_mode["slope"])


if __name__ == "__main__":
    main()
