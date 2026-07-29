"""How strongly does one column relate to another?

Correlation is for two numeric columns. When one side is a category or a binary outcome
you need a different measure, and reaching for Pearson anyway is how a "no signal" result
gets reported on a variable that clearly has signal.

    python examples/statistics/association.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    data = bt.from_pydict(
        {
            "score": [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0],
            "spend": [2.0, 4.0, 6.0, 8.0, 20.0, 22.0, 24.0, 26.0],
            "churned": [0, 0, 0, 0, 1, 1, 1, 1],
            "weight": [1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0],
        }
    )

    churn = col("churned") == 1

    assoc = data.select(
        # Two numeric columns.
        corr=bt.corr("score", "spend"),
        covar=bt.covar_pop("score", "spend"),
        # Numeric against a binary outcome.
        point_biserial=bt.point_biserial("score", churn),
        signal_ratio=bt.signal_ratio("score", churn),
        # Weighted versions, when rows carry different importance.
        w_corr=bt.weighted_correlation("score", "spend", "weight"),
        w_cov=bt.weighted_covariance("score", "spend", "weight"),
    ).to_pydict()

    print(assoc)

    # `spend` is exactly 2x `score`, so they are perfectly correlated.
    assert abs(assoc["corr"][0] - 1.0) < 1e-9
    assert abs(assoc["w_corr"][0] - 1.0) < 1e-9
    assert assoc["covar"][0] > 0.0
    # Churners score much higher, so the point-biserial correlation is strongly positive.
    assert assoc["point_biserial"][0] > 0.8
    assert assoc["signal_ratio"][0] > 0.0

    # Ordinary least squares, as aggregates: spend = slope * score + intercept.
    fit = data.select(
        slope=bt.regr_slope("spend", "score"),
        intercept=bt.regr_intercept("spend", "score"),
        r2=bt.regr_r2("spend", "score"),
        n=bt.regr_count("spend", "score"),
        avg_x=bt.regr_avgx("spend", "score"),
        avg_y=bt.regr_avgy("spend", "score"),
    ).to_pydict()
    print(fit)

    assert abs(fit["slope"][0] - 2.0) < 1e-9
    assert abs(fit["intercept"][0]) < 1e-9
    assert abs(fit["r2"][0] - 1.0) < 1e-9
    assert fit["n"] == [8]


if __name__ == "__main__":
    main()
