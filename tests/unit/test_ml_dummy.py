"""Baseline predictors (dummy regressor and classifier).

Both ignore the features and predict a constant, so they are checked against scikit-learn's
`DummyRegressor` and `DummyClassifier`: the learned constant and every prediction match.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.dummy import DummyClassifier, DummyRegressor

pytestmark = pytest.mark.unit

sk_dummy = pytest.importorskip("sklearn.dummy")


@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_dummy_regressor_matches_sklearn(strategy: str) -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(5, 2, 100)
    ds = bt.from_pydict({"y": y.tolist()})
    model = DummyRegressor("y", strategy=strategy).fit(ds)
    sk = sk_dummy.DummyRegressor(strategy=strategy).fit(np.zeros((100, 1)), y)
    assert model.constant_ == pytest.approx(sk.predict(np.zeros((1, 1)))[0])
    assert all(
        v == pytest.approx(model.constant_) for v in model.predict(ds).to_pydict()["prediction"]
    )


def test_dummy_classifier_matches_sklearn() -> None:
    rng = np.random.default_rng(1)
    labels = (rng.random(100) < 0.7).astype(int)
    ds = bt.from_pydict({"y": labels.tolist()})
    model = DummyClassifier("y").fit(ds)
    sk = sk_dummy.DummyClassifier(strategy="most_frequent").fit(np.zeros((100, 1)), labels)
    assert model.constant_ == sk.predict(np.zeros((1, 1)))[0]


def test_dummy_classifier_predicts_the_majority_class() -> None:
    ds = bt.from_pydict({"y": ["a", "a", "a", "b"]})
    assert DummyClassifier("y").fit(ds).predict(ds).to_pydict()["prediction"] == [
        "a",
        "a",
        "a",
        "a",
    ]


def test_reject_unknown_strategies() -> None:
    with pytest.raises(PlanError, match="strategy must be"):
        DummyRegressor("y", strategy="constant")
    with pytest.raises(PlanError, match="strategy must be"):
        DummyClassifier("y", strategy="uniform")
