"""Gamma regression by Fisher-scoring IRLS.

The log-link gamma GLM has no closed form, so the test is convergence to scikit-learn's
`GammaRegressor` coefficients across penalty strengths, with the predicted mean staying positive.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.glm import GammaRegressor

pytestmark = pytest.mark.unit

sk_linear = pytest.importorskip("sklearn.linear_model")


@pytest.fixture(scope="module")
def amounts() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (400, 3))
    mu = np.exp(x @ np.array([0.4, -0.2, 0.3]) + 1.0)
    y = rng.gamma(2.0, mu / 2.0)
    ds = bt.from_pydict({**{f"x{i}": x[:, i].tolist() for i in range(3)}, "y": y.tolist()})
    return x, y, ds


@pytest.mark.parametrize("alpha", [0.0, 1.0, 5.0])
def test_matches_sklearn(amounts, alpha: float) -> None:
    x, y, ds = amounts
    model = GammaRegressor(["x0", "x1", "x2"], "y", alpha=alpha).fit(ds)
    sk = sk_linear.GammaRegressor(alpha=alpha, tol=1e-10, max_iter=1000).fit(x, y)
    assert np.allclose(model.coef_, sk.coef_, atol=1e-4)
    assert model.intercept_ == pytest.approx(sk.intercept_, abs=1e-4)


def test_predictions_are_positive(amounts) -> None:
    _, _, ds = amounts
    model = GammaRegressor(["x0", "x1", "x2"], "y", alpha=1.0).fit(ds)
    assert all(v > 0 for v in model.predict(ds).to_pydict()["prediction"])


def test_predictions_match_sklearn(amounts) -> None:
    x, y, ds = amounts
    model = GammaRegressor(["x0", "x1", "x2"], "y", alpha=1.0).fit(ds)
    got = np.array(model.predict(ds).to_pydict()["prediction"])
    sk = sk_linear.GammaRegressor(alpha=1.0, tol=1e-10, max_iter=1000).fit(x, y)
    assert np.allclose(got, sk.predict(x), rtol=1e-3)


def test_rejects_no_features() -> None:
    with pytest.raises(PlanError, match="at least one feature"):
        GammaRegressor([], "y")


def test_rejects_negative_alpha() -> None:
    with pytest.raises(PlanError, match="alpha must be non-negative"):
        GammaRegressor(["x"], "y", alpha=-1.0)


def test_names_a_missing_column(amounts) -> None:
    _, _, ds = amounts
    with pytest.raises(ColumnNotFoundError):
        GammaRegressor(["x0", "nope"], "y").fit(ds)


def test_predict_before_fit_raises() -> None:
    ds = bt.from_pydict({"x": [1.0], "y": [1.0]})
    with pytest.raises(PlanError, match="must be fitted"):
        GammaRegressor(["x"], "y").predict(ds)
