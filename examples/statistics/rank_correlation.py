"""Rank correlation: agreement that survives a non-linear relationship.

Pearson correlation measures a linear relationship, so it understates a monotonic but curved
one. Spearman is Pearson over the ranks, which is why computing it here is a window followed
by the ordinary correlation aggregate.

    python examples/statistics/rank_correlation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").select("l_quantity", "l_extendedprice").head(20_000)

    # A monotonic but curved relationship: price against the square of quantity.
    curved = lineitem.with_columns(curved=col("l_quantity") ** 3)

    linear = curved.agg(pearson=bt.corr(col("l_quantity"), col("curved"))).to_pydict()["pearson"][0]

    ranked = curved.with_columns(
        rank_x=bt.row_number().over(order_by=["l_quantity"]),
        rank_y=bt.row_number().over(order_by=["curved"]),
    )
    spearman = ranked.agg(
        rho=bt.corr(col("rank_x").cast("float64"), col("rank_y").cast("float64"))
    ).to_pydict()["rho"][0]

    print(f"pearson {linear:.6f}, spearman {spearman:.6f}")

    # Both are correlations, so both are bounded.
    assert -1.0 - 1e-9 <= linear <= 1.0 + 1e-9
    assert -1.0 - 1e-9 <= spearman <= 1.0 + 1e-9

    # The relationship is perfectly monotonic, so the rank correlation is essentially 1
    # while the linear one is not — which is the whole reason to compute it.
    assert spearman > 0.99
    assert spearman > linear

    # A genuinely linear pair scores high on both.
    both = lineitem.agg(pearson=bt.corr(col("l_quantity"), col("l_extendedprice"))).to_pydict()[
        "pearson"
    ][0]
    print(f"quantity vs price, pearson {both:.6f}")
    assert both > 0.9


if __name__ == "__main__":
    main()
