"""Gaussian naive Bayes.

The whole fit is a grouped aggregate and the classification is a closed-form argmax, so the
test is that the predictions match scikit-learn's `GaussianNB` on held-out data — the model has
no iterative freedom, so an exact prediction match is the right bar.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.naive_bayes import GaussianNB

pytestmark = pytest.mark.unit

sk_nb = pytest.importorskip("sklearn.naive_bayes")


@pytest.fixture(scope="module")
def blobs() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(0)
    x = np.vstack(
        [
            rng.normal([0, 0], 1, (100, 2)),
            rng.normal([3, 3], 1.5, (100, 2)),
            rng.normal([0, 4], 1, (100, 2)),
        ]
    )
    y = np.repeat([0, 1, 2], 100)
    ds = bt.from_pydict({"x0": x[:, 0].tolist(), "x1": x[:, 1].tolist(), "y": y.tolist()})
    return x, y, ds


def test_predictions_match_sklearn_in_sample(blobs) -> None:
    x, y, ds = blobs
    got = np.array(GaussianNB(["x0", "x1"], "y").fit(ds).predict(ds).to_pydict()["prediction"])
    assert (got == sk_nb.GaussianNB().fit(x, y).predict(x)).mean() == pytest.approx(1.0)


def test_predictions_match_sklearn_out_of_sample(blobs) -> None:
    x, y, ds = blobs
    model = GaussianNB(["x0", "x1"], "y").fit(ds)
    rng = np.random.default_rng(9)
    test = rng.normal([1.5, 2.0], 1.5, (80, 2))
    dst = bt.from_pydict({"x0": test[:, 0].tolist(), "x1": test[:, 1].tolist()})
    got = np.array(model.predict(dst).to_pydict()["prediction"])
    assert (got == sk_nb.GaussianNB().fit(x, y).predict(test)).mean() == pytest.approx(1.0)


def test_learns_all_classes(blobs) -> None:
    _, _, ds = blobs
    assert sorted(GaussianNB(["x0", "x1"], "y").fit(ds).classes_) == [0, 1, 2]


def test_string_labels_work() -> None:
    ds = bt.from_pydict({"x": [0.0, 0.5, 5.0, 5.5], "y": ["a", "a", "b", "b"]})
    model = GaussianNB(["x"], "y").fit(ds)
    assert model.predict(bt.from_pydict({"x": [0.2, 5.2]})).to_pydict()["prediction"] == ["a", "b"]


def test_priors_sum_to_one(blobs) -> None:
    _, _, ds = blobs
    model = GaussianNB(["x0", "x1"], "y").fit(ds)
    assert sum(model.priors_.values()) == pytest.approx(1.0)


def test_rejects_no_features() -> None:
    with pytest.raises(PlanError, match="at least one feature"):
        GaussianNB([], "y")


def test_names_a_missing_column(blobs) -> None:
    _, _, ds = blobs
    with pytest.raises(ColumnNotFoundError):
        GaussianNB(["x0", "nope"], "y").fit(ds)


def test_predict_before_fit_raises() -> None:
    ds = bt.from_pydict({"x": [1.0], "y": [0]})
    with pytest.raises(PlanError, match="must be fitted"):
        GaussianNB(["x"], "y").predict(ds)
