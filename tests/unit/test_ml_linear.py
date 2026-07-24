"""Native ordinary and ridge linear regression.

Both fit from the feature/target moments in one scan, so the tests check that they reproduce
scikit-learn's coefficients, intercept, and predictions exactly, and that ridge shrinks the
coefficients relative to ordinary least squares as its penalty rises.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.linear import LinearRegression, Ridge

pytestmark = pytest.mark.unit

sk_linear = pytest.importorskip("sklearn.linear_model")


@pytest.fixture(scope="module")
def regression() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (400, 3))
    y = x @ np.array([1.5, -2.0, 0.7]) + 3.0 + rng.normal(0, 0.1, 400)
    ds = bt.from_pydict({**{f"x{i}": x[:, i].tolist() for i in range(3)}, "y": y.tolist()})
    return x, y, ds


def test_ols_matches_sklearn(regression) -> None:
    x, y, ds = regression
    model = LinearRegression(["x0", "x1", "x2"], "y").fit(ds)
    sk = sk_linear.LinearRegression().fit(x, y)
    assert np.allclose(model.coef_, sk.coef_, atol=1e-7)
    assert model.intercept_ == pytest.approx(sk.intercept_, abs=1e-7)


def test_ols_prediction_matches_sklearn(regression) -> None:
    x, y, ds = regression
    model = LinearRegression(["x0", "x1", "x2"], "y").fit(ds)
    sk = sk_linear.LinearRegression().fit(x, y)
    got = np.array(model.predict(ds).to_pydict()["prediction"])
    assert np.allclose(got, sk.predict(x), atol=1e-6)


@pytest.mark.parametrize("alpha", [0.1, 1.0, 50.0])
def test_ridge_matches_sklearn(regression, alpha: float) -> None:
    x, y, ds = regression
    model = Ridge(["x0", "x1", "x2"], "y", alpha=alpha).fit(ds)
    sk = sk_linear.Ridge(alpha=alpha).fit(x, y)
    assert np.allclose(model.coef_, sk.coef_, atol=1e-6)
    assert model.intercept_ == pytest.approx(sk.intercept_, abs=1e-6)


def test_ridge_shrinks_the_coefficients(regression) -> None:
    _, _, ds = regression
    ols = LinearRegression(["x0", "x1", "x2"], "y").fit(ds)
    ridge = Ridge(["x0", "x1", "x2"], "y", alpha=100.0).fit(ds)
    assert np.linalg.norm(ridge.coef_) < np.linalg.norm(ols.coef_)


def test_alpha_zero_ridge_is_ols(regression) -> None:
    _, _, ds = regression
    ols = LinearRegression(["x0", "x1", "x2"], "y").fit(ds)
    ridge = Ridge(["x0", "x1", "x2"], "y", alpha=0.0).fit(ds)
    assert np.allclose(ols.coef_, ridge.coef_, atol=1e-9)


def test_rejects_no_features() -> None:
    with pytest.raises(PlanError, match="at least one feature"):
        LinearRegression([], "y")


def test_rejects_negative_alpha() -> None:
    with pytest.raises(PlanError, match="alpha must be non-negative"):
        Ridge(["x"], "y", alpha=-1.0)


def test_names_a_missing_column(regression) -> None:
    _, _, ds = regression
    with pytest.raises(ColumnNotFoundError):
        LinearRegression(["x0", "nope"], "y").fit(ds)


def test_predict_before_fit_raises() -> None:
    ds = bt.from_pydict({"x": [1.0], "y": [2.0]})
    with pytest.raises(PlanError, match="must be fitted"):
        LinearRegression(["x"], "y").predict(ds)
