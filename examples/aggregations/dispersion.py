"""Spread: sample versus population, and the scale-free summaries.

`std`/`var` are the *sample* statistics with the n-1 denominator; `stddev_pop`/`var_pop`
use n. On 200,000 rows the difference is invisible, which is exactly why picking the
wrong one never gets caught — so pick deliberately.

    python examples/aggregations/dispersion.py
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
    rows = lineitem.count()

    spread = lineitem.agg(
        sample_var=bt.var(col("l_quantity")),
        population_var=bt.var_pop(col("l_quantity")),
        sample_std=bt.std(col("l_quantity")),
        population_std=bt.stddev_pop(col("l_quantity")),
        standard_error=bt.sem(col("l_quantity")),
        coefficient=bt.cv(col("l_quantity")),
        root_mean_square=bt.rms(col("l_quantity")),
        span=bt.value_range(col("l_quantity")),
        middle=bt.midrange(col("l_quantity")),
    ).to_pydict()
    print({name: round(value[0], 4) for name, value in spread.items()})

    # The two variances differ by exactly the Bessel factor.
    ratio = spread["sample_var"][0] / spread["population_var"][0]
    assert abs(ratio - rows / (rows - 1)) < 1e-9

    # Standard deviation is the square root of variance.
    assert abs(spread["sample_std"][0] ** 2 - spread["sample_var"][0]) < 1e-6

    # The standard error shrinks with the square root of the sample size.
    assert abs(spread["standard_error"][0] - spread["sample_std"][0] / rows**0.5) < 1e-9

    # The range and midrange are determined by the extremes alone.
    extremes = lineitem.agg(low=col("l_quantity").min(), high=col("l_quantity").max()).to_pydict()
    assert spread["span"][0] == extremes["high"][0] - extremes["low"][0]
    assert abs(spread["middle"][0] - (extremes["high"][0] + extremes["low"][0]) / 2) < 1e-9


if __name__ == "__main__":
    main()
