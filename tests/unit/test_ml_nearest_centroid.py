"""Nearest-centroid classification.

The model has no free parameters — a class is just its mean — so the bar is an exact prediction
match with scikit-learn's `NearestCentroid` in and out of sample.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.cluster import NearestCentroid

pytestmark = pytest.mark.unit

sk_neighbors = pytest.importorskip("sklearn.neighbors")


@pytest.fixture(scope="module")
def blobs() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(0)
    x = np.vstack(
        [
            rng.normal([0, 0], 1, (80, 2)),
            rng.normal([5, 5], 1, (80, 2)),
            rng.normal([0, 6], 1, (80, 2)),
        ]
    )
    y = np.repeat([0, 1, 2], 80)
    ds = bt.from_pydict({"a": x[:, 0].tolist(), "b": x[:, 1].tolist(), "y": y.tolist()})
    return x, y, ds


def test_matches_sklearn_in_sample(blobs) -> None:
    x, y, ds = blobs
    got = np.array(NearestCentroid(["a", "b"], "y").fit(ds).predict(ds).to_pydict()["prediction"])
    assert (got == sk_neighbors.NearestCentroid().fit(x, y).predict(x)).mean() == pytest.approx(1.0)


def test_matches_sklearn_out_of_sample(blobs) -> None:
    x, y, ds = blobs
    model = NearestCentroid(["a", "b"], "y").fit(ds)
    rng = np.random.default_rng(4)
    test = rng.normal([2, 3], 2, (60, 2))
    dst = bt.from_pydict({"a": test[:, 0].tolist(), "b": test[:, 1].tolist()})
    got = np.array(model.predict(dst).to_pydict()["prediction"])
    assert (got == sk_neighbors.NearestCentroid().fit(x, y).predict(test)).mean() == pytest.approx(
        1.0
    )


def test_string_labels_work() -> None:
    ds = bt.from_pydict({"x": [0.0, 0.5, 5.0, 5.5], "y": ["a", "a", "b", "b"]})
    model = NearestCentroid(["x"], "y").fit(ds)
    assert model.predict(bt.from_pydict({"x": [0.2, 5.2]})).to_pydict()["prediction"] == ["a", "b"]


def test_rejects_no_features() -> None:
    with pytest.raises(PlanError, match="at least one feature"):
        NearestCentroid([], "y")


def test_names_a_missing_column(blobs) -> None:
    _, _, ds = blobs
    with pytest.raises(ColumnNotFoundError):
        NearestCentroid(["a", "nope"], "y").fit(ds)


def test_predict_before_fit_raises() -> None:
    ds = bt.from_pydict({"x": [1.0], "y": [0]})
    with pytest.raises(PlanError, match="must be fitted"):
        NearestCentroid(["x"], "y").predict(ds)
