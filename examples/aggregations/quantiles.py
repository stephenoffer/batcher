"""Quantiles: exact, named, and sketch-approximated.

An exact quantile has to see every value, so it is a pipeline breaker with memory
proportional to the data. The `approx_` family uses a mergeable sketch instead: bounded
memory, and — because it is mergeable — the same answer whether it ran on one core or a
cluster.

    python examples/aggregations/quantiles.py
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

    spread = lineitem.agg(
        p50=bt.median(col("l_extendedprice")),
        p90=bt.quantile(col("l_extendedprice"), 0.9),
        first_quartile=bt.q1(col("l_extendedprice")),
        third_quartile=bt.q3(col("l_extendedprice")),
        inter_quartile=bt.iqr(col("l_extendedprice")),
        approx_p50=bt.approx_median(col("l_extendedprice")),
        approx_p90=bt.approx_quantile(col("l_extendedprice"), 0.9),
    ).to_pydict()
    print({name: round(value[0], 2) for name, value in spread.items()})

    # Quantiles are monotone in their argument.
    assert spread["first_quartile"][0] < spread["p50"][0] < spread["third_quartile"][0]
    assert spread["p50"][0] < spread["p90"][0]

    # The IQR is exactly the gap between the quartiles.
    gap = spread["third_quartile"][0] - spread["first_quartile"][0]
    assert abs(spread["inter_quartile"][0] - gap) < 1e-6

    # The sketch lands within a small relative error of the exact value.
    for exact_name, approx_name in [("p50", "approx_p50"), ("p90", "approx_p90")]:
        exact = spread[exact_name][0]
        approx = spread[approx_name][0]
        assert abs(approx - exact) / exact < 0.02, (exact_name, exact, approx)


if __name__ == "__main__":
    main()
