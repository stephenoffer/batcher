"""Statistics where the rows are not equally important.

A weighted mean is the ratio of two sums, which is why it composes across partitions when a
plain mean of means does not. Reach for it whenever the rows already carry a natural weight
— a quantity, a duration, a population.

    python examples/statistics/weighted_statistics.py
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

    unit_price = col("l_extendedprice") / col("l_quantity")

    summary = lineitem.agg(
        unweighted=unit_price.mean(),
        weighted=bt.weighted_mean(unit_price, col("l_quantity")),
        gross=col("l_extendedprice").sum(),
        units=col("l_quantity").sum(),
    ).to_pydict()
    print({name: round(value[0], 4) for name, value in summary.items()})

    # The weighted mean is the ratio of the two totals, exactly.
    by_hand = summary["gross"][0] / summary["units"][0]
    assert abs(by_hand - summary["weighted"][0]) < 1e-6

    # It differs from the unweighted mean, which is the reason to use it.
    assert abs(summary["weighted"][0] - summary["unweighted"][0]) > 1e-6

    # And it rolls up across groups, where the unweighted one does not.
    per_mode = (
        lineitem.group_by("l_shipmode")
        .agg(gross=col("l_extendedprice").sum(), units=col("l_quantity").sum())
        .to_pydict()
    )
    rolled = sum(per_mode["gross"]) / sum(per_mode["units"])
    print(f"global {summary['weighted'][0]:.6f} vs rolled up {rolled:.6f}")
    assert abs(rolled - summary["weighted"][0]) < 1e-9

    # A weighted variance for completeness.
    spread = lineitem.agg(weighted_var=bt.weighted_var(unit_price, col("l_quantity"))).to_pydict()
    print("weighted variance:", round(spread["weighted_var"][0], 4))
    assert spread["weighted_var"][0] > 0


if __name__ == "__main__":
    main()
