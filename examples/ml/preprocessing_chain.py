"""Chaining preprocessors into one fitted pipeline.

A ``Chain`` fits its steps in order and applies them in order, so the whole feature
pipeline is a single object you fit on train and apply to everything else. That is what
stops a validation set being scaled by its own statistics.

    python examples/ml/preprocessing_chain.py
"""

from __future__ import annotations

import batcher as bt
from batcher import ml


def main() -> None:
    train = bt.from_pydict(
        {
            "city": ["nyc", "sf", "nyc", "la"],
            "income": [100.0, None, 140.0, 60.0],
            "age": [30.0, 40.0, 50.0, 60.0],
            "label": [1, 0, 1, 0],
        }
    )
    holdout = bt.from_pydict(
        {
            "city": ["sf", "berlin"],
            "income": [None, 90.0],
            "age": [35.0, 45.0],
            "label": [0, 1],
        }
    )

    pipeline = ml.Chain(
        # 1. Record which rows were missing, before filling them.
        ml.MissingIndicator("income"),
        # 2. Fill the gap with the training median.
        ml.SimpleImputer("income", strategy="median"),
        # 3. Encode the categorical.
        ml.OrdinalEncoder("city"),
        # 4. Put the numerics on one scale.
        ml.StandardScaler(["income", "age"]),
    )

    fitted = pipeline.fit(train)
    out = fitted.transform(train).to_pydict()
    print(sorted(out))

    assert "income_missing" in out
    assert None not in out["income"]
    assert set(out["city"]) <= {0, 1, 2}
    # Standardized on the training statistics: mean ~0.
    assert abs(sum(out["age"]) / len(out["age"])) < 1e-9

    # The same fitted pipeline applies to unseen data. Nothing is refit, so the holdout
    # is scaled by the *training* statistics -- the whole point of the fit/transform split.
    applied = fitted.transform(holdout).to_pydict()
    print(applied)
    assert None not in applied["income"]
    # An unseen category maps to the encoder's unknown value rather than exploding.
    assert applied["city"][1] == -1
    # The holdout mean is not zero, because it was not fit on.
    assert abs(sum(applied["age"]) / len(applied["age"])) > 1e-9

    # `fit_transform` when you only need the training output.
    once = ml.Chain(ml.StandardScaler("age")).fit_transform(train).to_pydict()
    assert abs(sum(once["age"]) / len(once["age"])) < 1e-9

    # A fitted pipeline feeds an estimator directly.
    features = fitted.transform(train)
    model = ml.LogisticRegression(["income", "age", "city"], "label", max_iter=200).fit(features)
    scored = model.predict(features)
    acc = scored.select(a=bt.accuracy("label", "prediction")).to_pydict()
    print("accuracy:", acc)
    assert 0.0 <= acc["a"][0] <= 1.0


if __name__ == "__main__":
    main()
