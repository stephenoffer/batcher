"""The hyperparameter validation curve.

`validation_curve` sweeps one hyperparameter and cross-validates at each value. The tests pin
its structure (one scored row per value, ordered) and its behavior on a task with a known
trade-off: a deeper tree fits a nonlinear target better than a stump, so the score rises with
depth.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.metrics import evaluate
from batcher.ml.model_selection import validation_curve

pytestmark = pytest.mark.unit

sk_tree = pytest.importorskip("sklearn.tree")


def _r2(scored: bt.Dataset, y_true: str, prediction: str) -> float:
    return evaluate(scored, y_true, y_pred=prediction, task="regression", metrics=["r2"])["r2"]


def _dataset() -> bt.Dataset:
    return bt.from_pydict(
        {"x": [float(i % 20) for i in range(300)], "y": [float((i % 20) ** 2) for i in range(300)]}
    )


def _fit(train: bt.Dataset, depth: int):
    frame = train.to_pydict()
    return sk_tree.DecisionTreeRegressor(max_depth=depth, random_state=0).fit(
        [[v] for v in frame["x"]], frame["y"]
    )


def test_returns_one_row_per_value_ordered() -> None:
    curve = validation_curve(
        _dataset(),
        _fit,
        lambda m, d: d.ml.predict(m, features=["x"]),
        y_true="y",
        metric=_r2,
        param_values=[5, 1, 3],
        param_name="max_depth",
        k=3,
        key="x",
    )
    got = curve.to_pydict()
    assert got["max_depth"] == [1, 3, 5]
    assert len(got["score"]) == 3


def test_score_improves_with_capacity() -> None:
    curve = validation_curve(
        _dataset(),
        _fit,
        lambda m, d: d.ml.predict(m, features=["x"]),
        y_true="y",
        metric=_r2,
        param_values=[1, 8],
        param_name="max_depth",
        k=3,
        key="x",
    )
    scores = curve.sort("max_depth").to_pydict()["score"]
    assert scores[1] > scores[0]


def test_rejects_too_few_folds() -> None:
    with pytest.raises(PlanError, match="at least 2 folds"):
        validation_curve(
            _dataset(),
            _fit,
            lambda m, d: d,
            y_true="y",
            metric=_r2,
            param_values=[1],
            k=1,
        )
