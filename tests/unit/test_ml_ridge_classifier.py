"""Ridge classification (one-vs-rest ridge regression, argmax).

A closed-form fit with no iterative freedom, so the bar is an exact prediction match with
scikit-learn's `RidgeClassifier` across penalty strengths, in and out of sample.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.linear import RidgeClassifier

pytestmark = pytest.mark.unit

sk_linear = pytest.importorskip("sklearn.linear_model")


@pytest.fixture(scope="module")
def blobs() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(0)
    x = np.vstack(
        [
            rng.normal([0, 0], 1, (100, 2)),
            rng.normal([3, 3], 1, (100, 2)),
            rng.normal([0, 4], 1, (100, 2)),
        ]
    )
    y = np.repeat([0, 1, 2], 100)
    ds = bt.from_pydict({"a": x[:, 0].tolist(), "b": x[:, 1].tolist(), "y": y.tolist()})
    return x, y, ds


@pytest.mark.parametrize("alpha", [0.5, 1.0, 5.0])
def test_matches_sklearn_in_sample(blobs, alpha: float) -> None:
    x, y, ds = blobs
    got = np.array(
        RidgeClassifier(["a", "b"], "y", alpha=alpha).fit(ds).predict(ds).to_pydict()["prediction"]
    )
    assert (
        got == sk_linear.RidgeClassifier(alpha=alpha).fit(x, y).predict(x)
    ).mean() == pytest.approx(1.0)


def test_matches_sklearn_out_of_sample(blobs) -> None:
    x, y, ds = blobs
    model = RidgeClassifier(["a", "b"], "y").fit(ds)
    rng = np.random.default_rng(9)
    test = rng.normal([1.5, 2.0], 2, (120, 2))
    dst = bt.from_pydict({"a": test[:, 0].tolist(), "b": test[:, 1].tolist()})
    got = np.array(model.predict(dst).to_pydict()["prediction"])
    ref = sk_linear.RidgeClassifier().fit(x, y).predict(test)
    assert (got == ref).mean() == pytest.approx(1.0)


def test_rejects_no_features() -> None:
    with pytest.raises(PlanError, match="at least one feature"):
        RidgeClassifier([], "y")


def test_rejects_negative_alpha() -> None:
    with pytest.raises(PlanError, match="alpha must be non-negative"):
        RidgeClassifier(["x"], "y", alpha=-1.0)


def test_names_a_missing_column(blobs) -> None:
    _, _, ds = blobs
    with pytest.raises(ColumnNotFoundError):
        RidgeClassifier(["a", "b"], "nope").fit(ds)
