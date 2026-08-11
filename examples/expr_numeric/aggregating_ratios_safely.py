"""Ratios that survive being aggregated.

Store the numerator and the denominator, not the ratio. That single rule is what makes a
metric composable: it rolls up to any grain, survives a partition change, and cannot be
averaged wrongly because there is nothing to average.

    python examples/expr_numeric/aggregating_ratios_safely.py
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

    # Store the pieces.
    pieces = lineitem.group_by("l_shipmode", "l_returnflag").agg(
        discounted=(col("l_extendedprice") * col("l_discount")).sum(),
        gross=col("l_extendedprice").sum(),
    )

    # The metric at the fine grain.
    fine = pieces.with_columns(rate=col("discounted") / col("gross"))
    assert all(0.0 <= value <= 1.0 for value in fine.to_pydict()["rate"])

    # Roll up to a coarser grain: sum the pieces, then divide.
    coarse = (
        pieces.group_by("l_shipmode")
        .agg(discounted=col("discounted").sum(), gross=col("gross").sum())
        .with_columns(rate=col("discounted") / col("gross"))
        .sort("l_shipmode")
    )
    rolled = coarse.to_pydict()

    # Compare against computing it directly at the coarse grain.
    direct = (
        lineitem.group_by("l_shipmode")
        .agg(
            discounted=(col("l_extendedprice") * col("l_discount")).sum(),
            gross=col("l_extendedprice").sum(),
        )
        .with_columns(rate=col("discounted") / col("gross"))
        .sort("l_shipmode")
        .to_pydict()
    )
    assert all(abs(a - b) < 1e-12 for a, b in zip(rolled["rate"], direct["rate"], strict=True))
    print("rolled-up rate matches the direct one exactly")

    # Averaging the fine-grained rates does not.
    naive = fine.group_by("l_shipmode").agg(rate=col("rate").mean()).sort("l_shipmode").to_pydict()
    differences = [abs(a - b) for a, b in zip(rolled["rate"], naive["rate"], strict=True)]
    print("mean-of-rates error:", [round(value, 6) for value in differences])
    assert max(differences) > 1e-9
    assert bt is not None


if __name__ == "__main__":
    main()
