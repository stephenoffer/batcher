"""Linear discriminant analysis.

LDA pools one covariance across all classes and has a closed-form solution, so the bar is an
exact prediction match with scikit-learn (whichever solver it uses), in and out of sample.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.discriminant import LinearDiscriminantAnalysis

pytestmark = pytest.mark.unit

sk_da = pytest.importorskip("sklearn.discriminant_analysis")


@pytest.fixture(scope="module")
def blobs() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(0)
    x = np.vstack(
        [
            rng.normal([0, 0], 1, (200, 2)),
            rng.normal([3, 3], 1, (200, 2)),
            rng.normal([0, 4], 1, (200, 2)),
        ]
    )
    y = np.repeat([0, 1, 2], 200)
    ds = bt.from_pydict({"a": x[:, 0].tolist(), "b": x[:, 1].tolist(), "y": y.tolist()})
    return x, y, ds


@pytest.mark.parametrize("solver", ["svd", "lsqr"])
def test_matches_sklearn_in_sample(blobs, solver: str) -> None:
    x, y, ds = blobs
    got = np.array(
        LinearDiscriminantAnalysis(["a", "b"], "y").fit(ds).predict(ds).to_pydict()["prediction"]
    )
    ref = sk_da.LinearDiscriminantAnalysis(solver=solver).fit(x, y).predict(x)
    assert (got == ref).mean() == pytest.approx(1.0)


def test_matches_sklearn_out_of_sample(blobs) -> None:
    x, y, ds = blobs
    model = LinearDiscriminantAnalysis(["a", "b"], "y").fit(ds)
    rng = np.random.default_rng(9)
    test = rng.normal([1.5, 2.0], 2, (120, 2))
    dst = bt.from_pydict({"a": test[:, 0].tolist(), "b": test[:, 1].tolist()})
    got = np.array(model.predict(dst).to_pydict()["prediction"])
    ref = sk_da.LinearDiscriminantAnalysis().fit(x, y).predict(test)
    assert (got == ref).mean() == pytest.approx(1.0)


def test_rejects_no_features() -> None:
    with pytest.raises(PlanError, match="at least one feature"):
        LinearDiscriminantAnalysis([], "y")


def test_names_a_missing_column(blobs) -> None:
    _, _, ds = blobs
    with pytest.raises(ColumnNotFoundError):
        LinearDiscriminantAnalysis(["a", "nope"], "y").fit(ds)


def test_predict_before_fit_raises() -> None:
    ds = bt.from_pydict({"a": [1.0], "b": [1.0], "y": [0]})
    with pytest.raises(PlanError, match="must be fitted"):
        LinearDiscriminantAnalysis(["a", "b"], "y").predict(ds)
