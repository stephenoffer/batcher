"""Proportions and their uncertainty.

A rate from ten observations and a rate from a million are the same number and carry very
different weight. The standard error of a proportion says how different, and it is one
expression over the count and the total.

    python examples/statistics/binomial_proportions.py
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

    per_mode = (
        lineitem.group_by("l_shipmode")
        .agg(n=bt.count(), returned=bt.count_if(col("l_returnflag") == "R"))
        .with_columns(rate=col("returned") / col("n"))
        .with_columns(
            # sqrt(p(1-p)/n), the standard error of a proportion.
            error=((col("rate") * (1 - col("rate"))) / col("n")).sqrt()
        )
        .sort("rate", descending=True)
    )

    result = per_mode.to_pydict()
    for index, mode in enumerate(result["l_shipmode"]):
        rate, error, n = result["rate"][index], result["error"][index], result["n"][index]
        print(f"  {mode:<9} n={n:>6} rate={rate:.4f} +/- {1.96 * error:.4f}")

    # Rates are proportions and errors are positive.
    assert all(0.0 <= value <= 1.0 for value in result["rate"])
    assert all(value > 0 for value in result["error"])

    # The error shrinks as the sample grows: the largest group has the tightest interval.
    largest = max(range(len(result["n"])), key=lambda index: result["n"][index])
    smallest = min(range(len(result["n"])), key=lambda index: result["n"][index])
    assert result["error"][largest] <= result["error"][smallest]

    # The pooled rate is the ratio of the totals, not the mean of the rates.
    pooled = sum(result["returned"]) / sum(result["n"])
    mean_of_rates = sum(result["rate"]) / len(result["rate"])
    print(f"pooled {pooled:.6f} vs mean of rates {mean_of_rates:.6f}")
    global_rate = lineitem.agg(r=bt.count_if(col("l_returnflag") == "R") / bt.count()).to_pydict()[
        "r"
    ][0]
    assert abs(pooled - global_rate) < 1e-9

    # Every group's interval covers the global rate, because the groups are not actually
    # different populations here.
    covering = sum(
        1
        for index in range(len(result["rate"]))
        if abs(result["rate"][index] - global_rate) <= 3 * result["error"][index]
    )
    print(f"{covering} of {len(result['rate'])} intervals cover the global rate")
    assert covering >= len(result["rate"]) - 1


if __name__ == "__main__":
    main()
