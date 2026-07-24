"""Quadratic discriminant analysis.

QDA has no iterative freedom — each class is a mean and a covariance — so the bar is an exact
prediction match with scikit-learn in and out of sample, on data where the classes differ in
spread and orientation (the case a diagonal-covariance classifier gets wrong).
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.discriminant import QuadraticDiscriminantAnalysis

pytestmark = pytest.mark.unit

sk_da = pytest.importorskip("sklearn.discriminant_analysis")


@pytest.fixture(scope="module")
def shaped() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(0)
    x = np.vstack(
        [
            rng.normal([0, 0], [1, 2], (200, 2)),
            rng.normal([4, 4], [2, 1], (200, 2)),
            rng.normal([0, 5], [1, 1], (200, 2)),
        ]
    )
    y = np.repeat([0, 1, 2], 200)
    ds = bt.from_pydict({"a": x[:, 0].tolist(), "b": x[:, 1].tolist(), "y": y.tolist()})
    return x, y, ds


def test_matches_sklearn_in_sample(shaped) -> None:
    x, y, ds = shaped
    got = np.array(
        QuadraticDiscriminantAnalysis(["a", "b"], "y").fit(ds).predict(ds).to_pydict()["prediction"]
    )
    assert (
        got == sk_da.QuadraticDiscriminantAnalysis().fit(x, y).predict(x)
    ).mean() == pytest.approx(1.0)


def test_matches_sklearn_out_of_sample(shaped) -> None:
    x, y, ds = shaped
    model = QuadraticDiscriminantAnalysis(["a", "b"], "y").fit(ds)
    rng = np.random.default_rng(9)
    test = rng.normal([2, 3], 3, (120, 2))
    dst = bt.from_pydict({"a": test[:, 0].tolist(), "b": test[:, 1].tolist()})
    got = np.array(model.predict(dst).to_pydict()["prediction"])
    ref = sk_da.QuadraticDiscriminantAnalysis().fit(x, y).predict(test)
    assert (got == ref).mean() == pytest.approx(1.0)


def test_separates_classes_that_differ_only_in_spread() -> None:
    # Both classes centered at origin; class 1 is far more spread out. A location-only classifier
    # cannot separate them, but QDA's per-class covariance can.
    rng = np.random.default_rng(2)
    tight = rng.normal([0, 0], 0.5, (200, 2))
    wide = rng.normal([0, 0], 4.0, (200, 2))
    x = np.vstack([tight, wide])
    y = np.repeat([0, 1], 200)
    ds = bt.from_pydict({"a": x[:, 0].tolist(), "b": x[:, 1].tolist(), "y": y.tolist()})
    model = QuadraticDiscriminantAnalysis(["a", "b"], "y").fit(ds)
    ref = sk_da.QuadraticDiscriminantAnalysis().fit(x, y).predict(x)
    got = np.array(model.predict(ds).to_pydict()["prediction"])
    assert (got == ref).mean() == pytest.approx(1.0)


def test_rejects_no_features() -> None:
    with pytest.raises(PlanError, match="at least one feature"):
        QuadraticDiscriminantAnalysis([], "y")


def test_names_a_missing_column(shaped) -> None:
    _, _, ds = shaped
    with pytest.raises(ColumnNotFoundError):
        QuadraticDiscriminantAnalysis(["a", "nope"], "y").fit(ds)


def test_predict_before_fit_raises() -> None:
    ds = bt.from_pydict({"a": [1.0], "b": [1.0], "y": [0]})
    with pytest.raises(PlanError, match="must be fitted"):
        QuadraticDiscriminantAnalysis(["a", "b"], "y").predict(ds)
