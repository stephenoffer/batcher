"""Cross-validation, learning curves, and feature importance -- all in the engine.

``cross_val_score`` takes a ``fit`` and a ``predict`` callable, so it works with the
built-in estimators or with anything you wrap. Pass ``key=`` when rows share a group that
must not straddle a fold: that is the difference between an honest score and a leak.

    python examples/ml/model_selection.py
"""

from __future__ import annotations

import batcher as bt
from batcher import ml


def main() -> None:
    # y = 2*x + 1, with a group column that must not straddle folds. Keep the sample
    # comfortably larger than k -- a fold with too few rows leaves the metric undefined.
    n = 200
    data = bt.from_pydict(
        {
            "x": [float(i) for i in range(n)],
            "y": [2.0 * i + 1.0 for i in range(n)],
            "group": [f"g{i // 10}" for i in range(n)],
        }
    )

    def fit(train: bt.Dataset):
        return ml.Ridge(["x"], "y", alpha=0.001).fit(train)

    def predict(model, test: bt.Dataset) -> bt.Dataset:
        return model.predict(test)

    # A metric receives the *scored dataset* plus the two column names, so wrap the
    # aggregate expression into that shape.
    def r2(scored: bt.Dataset, y_true: str, prediction: str) -> float:
        return scored.select(v=bt.r2(y_true, prediction)).to_pydict()["v"][0]

    # K-fold scores, one per fold.
    scores = ml.cross_val_score(data, fit, predict, y_true="y", metric=r2, k=4, seed=0)
    print("cv r2:", [round(s, 4) for s in scores])
    assert len(scores) == 4
    assert all(s > 0.9 for s in scores)

    # Group-aware folds: no group appears in both train and test.
    grouped = ml.cross_val_score(
        data, fit, predict, y_true="y", metric=r2, k=4, seed=0, key="group"
    )
    print("grouped cv r2:", [round(s, 4) for s in grouped])
    assert len(grouped) == 4

    # Out-of-fold predictions, for stacking or for an honest residual plot.
    oof = ml.cross_val_predict(data, fit, predict, k=4, seed=0)
    out = oof.to_pydict()
    assert "prediction" in out
    assert len(out["prediction"]) == n

    # A learning curve: does more data still help? It reports the fraction, the row
    # count behind it, and the score.
    curve = ml.learning_curve(
        data, fit, predict, y_true="y", metric=r2, fractions=[0.25, 0.5, 1.0], seed=0
    ).to_pydict()
    print("learning curve:", curve)
    assert curve["train_fraction"] == [0.25, 0.5, 1.0]
    assert curve["train_rows"][0] < curve["train_rows"][-1]
    assert all(s > 0.9 for s in curve["score"])

    # Feature importance by permutation, using a fitted model as the predictor.
    model = fit(data)
    importance = ml.permutation_importance(
        data, model.predict, ["x"], y_true="y", metric=r2, n_repeats=2, seed=0
    ).to_pydict()
    print("importance:", importance)
    assert len(importance[next(iter(importance))]) == 1

    # Partial dependence: how does the prediction move as one feature sweeps?
    pd = ml.partial_dependence(data, model.predict, "x", grid_points=5).to_pydict()
    print("partial dependence:", pd)
    assert len(pd[next(iter(pd))]) == 5


if __name__ == "__main__":
    main()
