"""Correlation and covariance between two columns.

Covariance carries the units of both inputs, so its magnitude says nothing on its own.
Correlation is the normalized form and is the one to report. Both come from the same
single pass over the pair.

    python examples/aggregations/correlation.py
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

    together = lineitem.agg(
        correlation=bt.corr(col("l_quantity"), col("l_extendedprice")),
        sample_cov=bt.covar_samp(col("l_quantity"), col("l_extendedprice")),
        population_cov=bt.covar_pop(col("l_quantity"), col("l_extendedprice")),
        qty_std=bt.std(col("l_quantity")),
        price_std=bt.std(col("l_extendedprice")),
    ).to_pydict()
    print({name: round(value[0], 4) for name, value in together.items()})

    # Extended price is quantity times a unit price, so the two are strongly related.
    assert 0.9 < together["correlation"][0] <= 1.0

    # Correlation is covariance divided by the two standard deviations. Checking the
    # identity proves the two aggregates saw the same rows.
    derived = together["sample_cov"][0] / (together["qty_std"][0] * together["price_std"][0])
    assert abs(derived - together["correlation"][0]) < 1e-6

    # A column is perfectly correlated with itself.
    self_corr = lineitem.agg(r=bt.corr(col("l_quantity"), col("l_quantity"))).to_pydict()
    assert abs(self_corr["r"][0] - 1.0) < 1e-9

    # The frame-level helper computes the whole matrix in one pass.
    matrix = lineitem.select("l_quantity", "l_extendedprice", "l_discount").corr_matrix()
    print(matrix.to_pydict())


if __name__ == "__main__":
    main()
