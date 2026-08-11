"""Normalizing a column with expressions rather than a fitted preprocessor.

A preprocessor remembers the training statistics, which is what you want for a model. An
expression recomputes them from whatever it is given, which is what you want for a report.
Using the wrong one is a leak in one direction and a wrong chart in the other.

    python examples/expr_numeric/normalization_expressions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_totalprice")

    bounds = orders.agg(
        low=col("o_totalprice").min(),
        high=col("o_totalprice").max(),
        mean=col("o_totalprice").mean(),
        std=bt.std(col("o_totalprice")),
    ).to_pydict()
    low, high = bounds["low"][0], bounds["high"][0]
    mean, std = bounds["mean"][0], bounds["std"][0]

    normalized = orders.select(
        "o_totalprice",
        unit=(col("o_totalprice") - low) / (high - low),
        zscore=(col("o_totalprice") - mean) / std,
        share=col("o_totalprice") / col("o_totalprice").sum().over(),
    )
    result = normalized.to_pydict()
    print({name: [round(v, 6) for v in column[:3]] for name, column in result.items()})

    # Min-max lands in [0, 1] with both ends attained.
    assert min(result["unit"]) == 0.0
    assert abs(max(result["unit"]) - 1.0) < 1e-9

    # The z-score has mean zero and unit standard deviation.
    check = normalized.agg(m=col("zscore").mean(), s=bt.std(col("zscore"))).to_pydict()
    assert abs(check["m"][0]) < 1e-9
    assert abs(check["s"][0] - 1.0) < 1e-9

    # The shares sum to one, which is what makes them shares.
    assert abs(sum(result["share"]) - 1.0) < 1e-9

    # The window form recomputes over whatever it is given — a filtered subset gets its
    # own denominator, which is right for a report and wrong for a model feature.
    subset = orders.filter(col("o_totalprice") > mean).select(
        share=col("o_totalprice") / col("o_totalprice").sum().over()
    )
    assert abs(sum(subset.to_pydict()["share"]) - 1.0) < 1e-9


if __name__ == "__main__":
    main()
