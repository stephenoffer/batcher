"""Ratios of aggregates, not aggregates of ratios.

The average of per-row margins is not the overall margin, and the difference grows with the
spread of the denominators. Compute the numerator and denominator separately and divide
once — that is the only version that composes across groups.

    python examples/aggregations/weighted_and_ratio_metrics.py
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
        .agg(
            gross=col("l_extendedprice").sum(),
            discount_value=(col("l_extendedprice") * col("l_discount")).sum(),
            # The wrong version: an unweighted mean of per-row rates.
            mean_of_rates=col("l_discount").mean(),
        )
        .with_columns(rate_of_totals=col("discount_value") / col("gross"))
        .sort("l_shipmode")
    )

    result = per_mode.to_pydict()
    for index, mode in enumerate(result["l_shipmode"]):
        print(
            f"{mode:<9} weighted={result['rate_of_totals'][index]:.6f} "
            f"unweighted={result['mean_of_rates'][index]:.6f}"
        )

    # Both are proportions, and they are not the same number.
    assert all(0.0 <= value <= 1.0 for value in result["rate_of_totals"])
    assert any(
        abs(left - right) > 1e-9
        for left, right in zip(result["rate_of_totals"], result["mean_of_rates"], strict=True)
    )

    # The weighted version rolls up correctly: the global rate is the ratio of the sums,
    # which is *not* the mean of the per-group rates.
    overall = lineitem.agg(
        gross=col("l_extendedprice").sum(),
        discount_value=(col("l_extendedprice") * col("l_discount")).sum(),
    ).to_pydict()
    global_rate = overall["discount_value"][0] / overall["gross"][0]
    rolled = sum(result["discount_value"]) / sum(result["gross"])
    print(f"global {global_rate:.8f} vs rolled-up {rolled:.8f}")
    assert abs(global_rate - rolled) < 1e-9
    assert bt is not None


if __name__ == "__main__":
    main()
