"""Native logistic regression by IRLS.

Fitting logistic regression has no closed form, so the test is that the in-engine Newton
iteration converges to the same place scikit-learn's optimizer does: the unpenalized
coefficients, intercept, and predicted probabilities match, and the 0/1 labels agree.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.linear import LogisticRegression

pytestmark = pytest.mark.unit

sk_linear = pytest.importorskip("sklearn.linear_model")


@pytest.fixture(scope="module")
def classification() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (500, 3))
    logits = x @ np.array([1.2, -0.8, 0.5]) + 0.3
    y = (rng.random(500) < 1 / (1 + np.exp(-logits))).astype(int)
    ds = bt.from_pydict({**{f"x{i}": x[:, i].tolist() for i in range(3)}, "y": y.tolist()})
    return x, y, ds


def _sk(x: np.ndarray, y: np.ndarray):
    return sk_linear.LogisticRegression(penalty=None, tol=1e-10, max_iter=1000).fit(x, y)


def test_coefficients_match_sklearn(classification) -> None:
    x, y, ds = classification
    model = LogisticRegression(["x0", "x1", "x2"], "y").fit(ds)
    sk = _sk(x, y)
    assert np.allclose(model.coef_, sk.coef_[0], atol=1e-4)
    assert model.intercept_ == pytest.approx(sk.intercept_[0], abs=1e-4)


def test_probabilities_match_sklearn(classification) -> None:
    x, y, ds = classification
    model = LogisticRegression(["x0", "x1", "x2"], "y").fit(ds)
    got = np.array(model.predict_proba(ds).to_pydict()["prediction"])
    assert np.allclose(got, _sk(x, y).predict_proba(x)[:, 1], atol=1e-4)


def test_labels_match_sklearn(classification) -> None:
    x, y, ds = classification
    model = LogisticRegression(["x0", "x1", "x2"], "y").fit(ds)
    got = np.array(model.predict(ds).to_pydict()["prediction"])
    assert (got == _sk(x, y).predict(x)).mean() > 0.99


def test_converges_quickly(classification) -> None:
    _, _, ds = classification
    model = LogisticRegression(["x0", "x1", "x2"], "y").fit(ds)
    assert model.n_iter_ < 20


def test_separable_data_learns_a_positive_slope() -> None:
    ds = bt.from_pydict({"x": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], "y": [0, 0, 0, 1, 1, 1]})
    model = LogisticRegression(["x"], "y").fit(ds)
    assert model.coef_[0] > 0
    assert model.predict(bt.from_pydict({"x": [-5.0, 5.0]})).to_pydict()["prediction"] == [0, 1]


def test_rejects_no_features() -> None:
    with pytest.raises(PlanError, match="at least one feature"):
        LogisticRegression([], "y")


def test_names_a_missing_column(classification) -> None:
    _, _, ds = classification
    with pytest.raises(ColumnNotFoundError):
        LogisticRegression(["x0", "nope"], "y").fit(ds)


def test_predict_before_fit_raises() -> None:
    ds = bt.from_pydict({"x": [1.0], "y": [1]})
    with pytest.raises(PlanError, match="must be fitted"):
        LogisticRegression(["x"], "y").predict(ds)
