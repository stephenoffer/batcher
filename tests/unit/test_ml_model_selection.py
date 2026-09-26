"""Cross-validated scoring, out-of-fold prediction, and learning curves.

These tie the fold splitter, a fitted model, and a metric into one loop, so the tests use a
real scikit-learn model on a near-linear target where the *answer* is known: a linear model
should score near-perfectly on every fold, cover every row exactly once out-of-fold, and show
a learning curve that does not fall as more data is added.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.metrics import evaluate
from batcher.ml.model_selection import cross_val_predict, cross_val_score, learning_curve

pytestmark = pytest.mark.unit

pytest.importorskip("sklearn.linear_model", reason="scikit-learn supplies the fit callable")


@pytest.fixture(scope="module")
def problem() -> bt.Dataset:
    """A near-linear regression problem with a stable row id for hashing."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(300, 2))
    target = 3.0 * features[:, 0] - 2.0 * features[:, 1] + rng.normal(0, 0.05, 300)
    return bt.from_pydict(
        {
            "a": features[:, 0].tolist(),
            "b": features[:, 1].tolist(),
            "y": target.tolist(),
            "id": list(range(300)),
        }
    )


def _fit(train: bt.Dataset) -> Any:
    from sklearn.linear_model import LinearRegression

    frame = train.to_pydict()
    return LinearRegression().fit(list(zip(frame["a"], frame["b"], strict=True)), frame["y"])


def _predict(model: Any, ds: bt.Dataset) -> bt.Dataset:
    return ds.ml.predict(model, features=["a", "b"])


def _r2(ds: bt.Dataset, y_true: str, y_pred: str) -> float:
    return evaluate(ds, y_true, y_pred=y_pred, task="regression", metrics=["r2"])["r2"]


# --- cross_val_score -------------------------------------------------------------------


def test_cross_val_score_returns_one_score_per_fold(problem) -> None:
    scores = cross_val_score(problem, _fit, _predict, y_true="y", metric=_r2, k=5, key="id")
    assert len(scores) == 5


def test_a_linear_model_scores_near_perfectly_on_a_linear_target(problem) -> None:
    scores = cross_val_score(problem, _fit, _predict, y_true="y", metric=_r2, k=5, key="id")
    assert all(s > 0.99 for s in scores)


def test_cross_val_score_is_reproducible(problem) -> None:
    first = cross_val_score(problem, _fit, _predict, y_true="y", metric=_r2, k=4, seed=7, key="id")
    second = cross_val_score(problem, _fit, _predict, y_true="y", metric=_r2, k=4, seed=7, key="id")
    assert first == pytest.approx(second)


def test_cross_val_score_rejects_a_single_fold(problem) -> None:
    with pytest.raises(PlanError, match="at least 2 folds"):
        cross_val_score(problem, _fit, _predict, y_true="y", metric=_r2, k=1)


def test_cross_val_score_can_stratify() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    y = (x > 0).astype(int)
    ds = bt.from_pydict({"x": x.tolist(), "y": y.tolist(), "id": list(range(200))})

    def fit(train: bt.Dataset) -> Any:
        from sklearn.linear_model import LogisticRegression

        frame = train.to_pydict()
        return LogisticRegression().fit([[v] for v in frame["x"]], frame["y"])

    def accuracy(d: bt.Dataset, t: str, p: str) -> float:
        return evaluate(d, t, y_pred=p, metrics=["accuracy"])["accuracy"]

    scores = cross_val_score(
        ds,
        fit,
        lambda m, d: d.ml.predict(m, features=["x"]),
        y_true="y",
        metric=accuracy,
        k=4,
        key="id",
        stratify="y",
    )
    assert all(s > 0.9 for s in scores)


# --- cross_val_predict -----------------------------------------------------------------


def test_cross_val_predict_covers_every_row_exactly_once(problem) -> None:
    oof = cross_val_predict(problem, _fit, _predict, k=5, key="id")
    ids = oof.to_pydict()["id"]
    assert sorted(ids) == list(range(300))


def test_out_of_fold_predictions_are_accurate_on_a_linear_target(problem) -> None:
    oof = cross_val_predict(problem, _fit, _predict, k=5, key="id")
    assert _r2(oof, "y", "prediction") > 0.99


def test_cross_val_predict_rejects_a_single_fold(problem) -> None:
    with pytest.raises(PlanError, match="at least 2 folds"):
        cross_val_predict(problem, _fit, _predict, k=1)


# --- learning_curve --------------------------------------------------------------------


def test_learning_curve_reports_one_row_per_fraction(problem) -> None:
    curve = learning_curve(
        problem, _fit, _predict, y_true="y", metric=_r2, fractions=[0.5, 1.0], key="id"
    ).to_pydict()
    assert curve["train_fraction"] == [0.5, 1.0]
    assert len(curve["score"]) == 2


def test_learning_curve_train_rows_grow_with_the_fraction(problem) -> None:
    curve = learning_curve(
        problem, _fit, _predict, y_true="y", metric=_r2, fractions=[0.25, 0.5, 1.0], key="id"
    ).to_pydict()
    assert curve["train_rows"] == sorted(curve["train_rows"])


def test_more_data_does_not_lower_the_score_on_a_learnable_target(problem) -> None:
    # A linear model on a linear target: the score should not fall as data is added.
    curve = learning_curve(
        problem, _fit, _predict, y_true="y", metric=_r2, fractions=[0.3, 1.0], key="id"
    ).to_pydict()
    assert curve["score"][-1] >= curve["score"][0] - 0.01


def test_learning_curve_rejects_a_bad_holdout(problem) -> None:
    with pytest.raises(PlanError, match="holdout"):
        learning_curve(problem, _fit, _predict, y_true="y", metric=_r2, holdout=1.5)


def test_learning_curve_rejects_a_bad_fraction(problem) -> None:
    with pytest.raises(PlanError, match="fraction"):
        learning_curve(problem, _fit, _predict, y_true="y", metric=_r2, fractions=[1.5])
