"""A correlation matrix over several columns at once.

One pass produces every pair. Reading it is the first step before any modelling: two
features correlated at 0.99 are one feature, and a linear model handed both will produce
unstable coefficients that flip sign between runs.

    python examples/statistics/correlation_matrix.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    numeric = tpch("lineitem").select("l_quantity", "l_extendedprice", "l_discount", "l_tax")

    matrix = numeric.corr_matrix().to_pydict()
    print(matrix)

    columns = numeric.columns
    label = next(name for name in matrix if not isinstance(matrix[name][0], float))
    assert matrix[label] == columns

    # The diagonal is 1 and the matrix is symmetric.
    for index, name in enumerate(columns):
        assert abs(matrix[name][index] - 1.0) < 1e-9
    for i, left in enumerate(columns):
        for j, right in enumerate(columns):
            assert abs(matrix[left][j] - matrix[right][i]) < 1e-9

    # Every entry is a correlation, so it is bounded.
    for name in columns:
        assert all(-1.0 - 1e-9 <= value <= 1.0 + 1e-9 for value in matrix[name])

    # Quantity and extended price are near-collinear; that pair is the one to notice.
    price_row = matrix["l_extendedprice"][columns.index("l_quantity")]
    print("quantity vs extended price:", round(price_row, 4))
    assert price_row > 0.9

    # The covariance matrix has the same shape but carries units.
    covariance = numeric.cov_matrix().to_pydict()
    assert len(covariance[label]) == len(columns)
    assert col is not None


if __name__ == "__main__":
    main()
