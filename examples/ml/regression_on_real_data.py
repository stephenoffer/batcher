"""Fitting a regression on real TPC-H lineitems, end to end.

The features come out of the engine and the fit reads them through it, so the training set
never becomes a NumPy array in Python. That is the whole reason to fit here rather than
pulling to pandas first: the data can be larger than the machine.

    python examples/ml/regression_on_real_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col, ml


def main() -> None:
    lineitem = tpch("lineitem").select("l_quantity", "l_extendedprice", "l_discount", "l_tax")

    # Hold out a test set before looking at anything.
    train, test = lineitem.ml.train_test_split(test_size=0.2, seed=17)
    print("train:", train.count(), "test:", test.count())
    assert train.count() + test.count() == lineitem.count()

    model = ml.Ridge(["l_quantity", "l_discount", "l_tax"], "l_extendedprice", alpha=1.0).fit(train)
    print("coefficients:", [round(value, 3) for value in model.coef_])
    assert len(model.coef_) == 3

    # Price rises with quantity, so that coefficient is positive and dominant.
    assert model.coef_[0] > 0

    scored = model.predict(test)
    assert "prediction" in scored.columns
    assert scored.count() == test.count()

    # Score it honestly, on the held-out rows.
    errors = (
        scored.select(
            error=col("prediction") - col("l_extendedprice"),
            absolute=(col("prediction") - col("l_extendedprice")).abs(),
        )
        .agg(
            mae=col("absolute").mean(),
            bias=col("error").mean(),
            rmse=(col("error") ** 2).mean().sqrt(),
        )
        .to_pydict()
    )
    print({name: round(value[0], 2) for name, value in errors.items()})

    # A least-squares fit is unbiased on its own training distribution, so the mean
    # residual is small relative to the scale of the target.
    scale = test.agg(mean=col("l_extendedprice").mean()).to_pydict()["mean"][0]
    assert abs(errors["bias"][0]) < scale * 0.05
    assert errors["mae"][0] < scale


if __name__ == "__main__":
    main()
