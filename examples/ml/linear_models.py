"""Regularized linear regression: Ridge, Lasso, and ElasticNet.

Every estimator follows the same two-step shape: ``fit(ds)`` returns a fitted model and
``predict(ds)`` returns a new Dataset with the prediction column appended. Fitting reads
the data through the engine, so the training set never has to fit in memory as a NumPy
array.

    python examples/ml/linear_models.py
"""

from __future__ import annotations

import batcher as bt
from batcher import ml


def main() -> None:
    # y = 3*x1 + 2*x2 exactly, so a well-fit model recovers those coefficients.
    train = bt.from_pydict(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "x2": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0],
            "y": [7.0, 8.0, 17.0, 18.0, 27.0, 28.0, 37.0, 38.0],
        }
    )

    # Ridge: L2 penalty. `alpha` trades fit against coefficient size.
    ridge = ml.Ridge(["x1", "x2"], "y", alpha=0.01).fit(train)
    print("ridge coef:", ridge.coef_, "intercept:", ridge.intercept_)
    assert len(ridge.coef_) == 2
    assert abs(ridge.coef_[0] - 3.0) < 0.2
    assert abs(ridge.coef_[1] - 2.0) < 0.2

    scored = ridge.predict(train).to_pydict()
    print(scored["prediction"][:4])
    assert "prediction" in scored
    assert len(scored["prediction"]) == 8

    # Lasso: L1 penalty, which drives weak coefficients to exactly zero.
    lasso = ml.Lasso(["x1", "x2"], "y", alpha=0.01).fit(train)
    print("lasso coef:", lasso.coef_)
    assert len(lasso.coef_) == 2

    # ElasticNet: a blend, controlled by `l1_ratio`.
    enet = ml.ElasticNet(["x1", "x2"], "y", alpha=0.01, l1_ratio=0.5).fit(train)
    print("enet coef:", enet.coef_)
    assert len(enet.coef_) == 2

    # Rename the output when you want several models' predictions side by side.
    both = ml.Ridge(["x1", "x2"], "y", alpha=1.0, output_column="ridge_pred").fit(train)
    out = both.predict(train).to_pydict()
    assert "ridge_pred" in out

    # Score the fit with the metric aggregates, in the engine.
    quality = (
        ridge.predict(train)
        .select(r2=bt.r2("y", "prediction"), rmse=bt.rmse("y", "prediction"))
        .to_pydict()
    )
    print(quality)
    assert quality["r2"][0] > 0.99


if __name__ == "__main__":
    main()
