"""Regression error metrics: absolute, squared, percentage, and robust.

Picking the metric is the modelling decision. MAE treats every miss equally, RMSE punishes
big misses, MAPE is scale-free but explodes near zero, and Huber sits between MAE and MSE.
All of them are aggregates here, so you can compute several in one pass.

    python examples/metrics/regression_errors.py
"""

from __future__ import annotations

import batcher as bt


def main() -> None:
    # Residuals are +1, -1, +2, -4: one large miss dominates the squared metrics.
    preds = bt.from_pydict(
        {
            "y_true": [10.0, 20.0, 30.0, 40.0],
            "y_pred": [11.0, 19.0, 32.0, 36.0],
        }
    )

    errs = preds.select(
        mae=bt.mae("y_true", "y_pred"),
        mse=bt.mse("y_true", "y_pred"),
        rmse=bt.rmse("y_true", "y_pred"),
        medae=bt.medae("y_true", "y_pred"),
        max_error=bt.max_error("y_true", "y_pred"),
        # Percentage-scaled.
        mape=bt.mape("y_true", "y_pred"),
        smape=bt.smape("y_true", "y_pred"),
        wape=bt.wape("y_true", "y_pred"),
        normalized_rmse=bt.normalized_rmse("y_true", "y_pred"),
        # Log-scaled, for targets that span orders of magnitude.
        msle=bt.msle("y_true", "y_pred"),
        rmsle=bt.rmsle("y_true", "y_pred"),
        # Robust and quantile losses.
        huber=bt.huber_loss("y_true", "y_pred", delta=1.0),
        pinball=bt.pinball_loss("y_true", "y_pred", quantile=0.9),
        # Fit quality and bias.
        r2=bt.r2("y_true", "y_pred"),
        explained_variance=bt.explained_variance("y_true", "y_pred"),
        mean_bias=bt.mean_bias("y_true", "y_pred"),
        mpe=bt.mean_percentage_error("y_true", "y_pred"),
    ).to_pydict()

    print(errs)

    # |1| + |-1| + |2| + |-4| = 8, over 4 rows.
    assert errs["mae"] == [2.0]
    # The squared residuals average to 5.5.
    assert errs["mse"] == [5.5]
    assert abs(errs["rmse"][0] - 5.5**0.5) < 1e-12
    # The single large residual pulls RMSE above MAE; the median is untouched by it.
    assert errs["rmse"][0] > errs["mae"][0]
    assert errs["medae"] == [1.5]
    assert errs["max_error"] == [4.0]
    # R^2 is a fraction of variance explained.
    assert 0.0 < errs["r2"][0] <= 1.0
    # Predictions are low on balance (+1 -1 +2 -4 sums to -2).
    assert errs["mean_bias"][0] != 0.0


if __name__ == "__main__":
    main()
