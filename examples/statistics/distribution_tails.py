"""Where the extreme values are, and how many there are.

Percentiles describe the tail without assuming a distribution. That matters because the
usual "mean plus three standard deviations" rule assumes normality, and a right-skewed
revenue column is not normal — it flags far more rows than 0.1%.

    python examples/statistics/distribution_tails.py
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
    total = lineitem.count()

    cuts = lineitem.agg(
        p50=bt.quantile(col("l_extendedprice"), 0.5),
        p90=bt.quantile(col("l_extendedprice"), 0.9),
        p99=bt.quantile(col("l_extendedprice"), 0.99),
        mean=col("l_extendedprice").mean(),
        std=bt.std(col("l_extendedprice")),
    ).to_pydict()
    print({name: round(value[0], 1) for name, value in cuts.items()})

    assert cuts["p50"][0] < cuts["p90"][0] < cuts["p99"][0]

    # A percentile cut selects exactly the share it names.
    above_p99 = lineitem.filter(col("l_extendedprice") > cuts["p99"][0]).count()
    share = above_p99 / total
    print(f"above p99: {above_p99} rows ({share:.4%})")
    assert 0.005 < share < 0.015

    # The three-sigma rule on a skewed column flags a very different number.
    threshold = cuts["mean"][0] + 3 * cuts["std"][0]
    sigma_flagged = lineitem.filter(col("l_extendedprice") > threshold).count()
    print(f"above mean+3sd: {sigma_flagged} rows ({sigma_flagged / total:.4%})")

    # On a normal distribution that would be about 0.13%. Here it is not, which is the
    # point: the rule of thumb assumes a shape this column does not have.
    assert sigma_flagged != above_p99


if __name__ == "__main__":
    main()
