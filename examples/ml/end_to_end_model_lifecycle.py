"""The classical-ML lifecycle over real data, end to end.

Split, fit the preprocessing on the training half only, fit the model, score the holdout,
evaluate, and check for drift. Each step is short; the value is in the order, because every
common leak is a step done in the wrong one.

    python examples/ml/end_to_end_model_lifecycle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col, ml


def main() -> None:
    lineitem = tpch("lineitem").select(
        "l_quantity", "l_discount", "l_tax", "l_extendedprice", "l_shipmode"
    )

    # 1. Split before looking at anything.
    train, holdout = lineitem.ml.train_test_split(test_size=0.2, seed=99)
    print(f"train {train.count()}, holdout {holdout.count()}")
    assert train.count() + holdout.count() == lineitem.count()

    # 2. Fit preprocessing on the training half only.
    preprocessing = ml.Chain(
        ml.StandardScaler("l_quantity"),
        ml.StandardScaler("l_discount"),
        ml.OrdinalEncoder("l_shipmode"),
    ).fit(train)

    prepared_train = preprocessing.transform(train)
    prepared_holdout = preprocessing.transform(holdout)

    # The training half centres; the holdout does not, which proves no refit happened.
    train_mean = prepared_train.agg(m=col("l_quantity").mean()).to_pydict()["m"][0]
    holdout_mean = prepared_holdout.agg(m=col("l_quantity").mean()).to_pydict()["m"][0]
    assert abs(train_mean) < 1e-6
    assert holdout_mean != 0.0

    # 3. Fit.
    features = ["l_quantity", "l_discount", "l_tax", "l_shipmode"]
    model = ml.Ridge(features, "l_extendedprice", alpha=1.0).fit(prepared_train)
    print("coefficients:", [round(value, 2) for value in model.coef_])
    assert len(model.coef_) == len(features)

    # 4. Score the holdout.
    scored = model.predict(prepared_holdout)
    assert scored.count() == holdout.count()

    # 5. Evaluate.
    errors = (
        scored.select(
            absolute=(col("prediction") - col("l_extendedprice")).abs(),
            signed=col("prediction") - col("l_extendedprice"),
        )
        .agg(
            mae=col("absolute").mean(),
            bias=col("signed").mean(),
            rmse=(col("signed") ** 2).mean().sqrt(),
        )
        .to_pydict()
    )
    scale = holdout.agg(m=col("l_extendedprice").mean()).to_pydict()["m"][0]
    print({name: round(value[0], 1) for name, value in errors.items()})
    assert errors["mae"][0] < scale
    assert abs(errors["bias"][0]) < scale * 0.1
    assert errors["rmse"][0] >= errors["mae"][0]

    # 6. Drift: does the holdout look like the training data.
    def profile(dataset: bt.Dataset) -> tuple[float, float]:
        row = dataset.agg(m=col("l_quantity").mean(), s=bt.std(col("l_quantity"))).to_pydict()
        return row["m"][0], row["s"][0]

    train_profile = profile(train)
    holdout_profile = profile(holdout)
    shift = abs(train_profile[0] - holdout_profile[0]) / train_profile[0]
    print(f"mean shift between the halves: {shift:.4%}")
    assert shift < 0.05


if __name__ == "__main__":
    main()
