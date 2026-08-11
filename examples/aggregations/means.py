"""Four kinds of average, and when each is the right one.

The arithmetic mean is the default and is wrong for rates and ratios. Use the geometric
mean for growth factors, the harmonic mean for averaging rates, and the weighted mean
whenever the rows do not deserve equal say.

    python examples/aggregations/means.py
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

    averages = lineitem.agg(
        arithmetic=col("l_quantity").mean(),
        geometric=bt.geometric_mean(col("l_quantity")),
        harmonic=bt.harmonic_mean(col("l_quantity")),
        # Average unit price, weighted by how much of each line shipped.
        weighted=bt.weighted_mean(col("l_extendedprice") / col("l_quantity"), col("l_quantity")),
        unweighted=(col("l_extendedprice") / col("l_quantity")).mean(),
    ).to_pydict()
    print({name: round(value[0], 4) for name, value in averages.items()})

    # For positive values the three means are ordered: harmonic <= geometric <= arithmetic.
    assert averages["harmonic"][0] <= averages["geometric"][0] <= averages["arithmetic"][0]

    # Weighting by quantity moves the answer, which is the entire point of weighting.
    assert averages["weighted"][0] != averages["unweighted"][0]

    # The weighted mean is the ratio of two sums, so it is reproducible by hand.
    parts = lineitem.agg(
        numerator=col("l_extendedprice").sum(),
        denominator=col("l_quantity").sum(),
    ).to_pydict()
    by_hand = parts["numerator"][0] / parts["denominator"][0]
    assert abs(by_hand - averages["weighted"][0]) < 1e-6


if __name__ == "__main__":
    main()
