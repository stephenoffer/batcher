"""Turning a continuous feature into bins, three ways.

Equal-width bins are easy to explain and useless on a skewed column, because nearly every
row lands in the first bin. Quantile bins are equal-population by construction, which is
what you usually want.

    python examples/ml/binning_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col, ml


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_totalprice")
    total = orders.count()

    uniform = ml.KBinsDiscretizer("o_totalprice", n_bins=5, strategy="uniform").fit(orders)
    quantile = ml.KBinsDiscretizer("o_totalprice", n_bins=5, strategy="quantile").fit(orders)

    def sizes(name: str, fitted) -> list[int]:
        binned = fitted.transform(orders)
        counts = binned.value_counts("o_totalprice").sort("o_totalprice").to_pydict()
        print(f"{name:<10} {counts['count']}")
        assert sum(counts["count"]) == total
        return counts["count"]

    uniform_sizes = sizes("uniform", uniform)
    quantile_sizes = sizes("quantile", quantile)

    # Quantile bins are far more even than equal-width ones on a skewed column.
    uniform_spread = max(uniform_sizes) - min(uniform_sizes)
    quantile_spread = max(quantile_sizes) - min(quantile_sizes)
    print(f"spread: uniform {uniform_spread}, quantile {quantile_spread}")
    assert quantile_spread < uniform_spread

    # A quantile bin holds about a fifth of the rows.
    assert all(abs(size - total / 5) < total * 0.02 for size in quantile_sizes)

    # `width_bucket` is the expression form when the bounds are a business rule rather
    # than a fitted statistic.
    manual = orders.with_columns(band=bt.width_bucket(col("o_totalprice"), 0.0, 300_000.0, 3))
    manual_counts = manual.value_counts("band").sort("band").to_pydict()
    print("manual bands:", manual_counts)
    assert sum(manual_counts["count"]) == total


if __name__ == "__main__":
    main()
