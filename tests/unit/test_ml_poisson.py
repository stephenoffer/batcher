"""Poisson regression by IRLS.

The log-link GLM has no closed form, so the test is that the in-engine Newton iteration reaches
the same coefficients scikit-learn's optimizer does, across several penalty strengths, and that
the predicted rate stays positive.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.glm import PoissonRegressor

pytestmark = pytest.mark.unit

sk_linear = pytest.importorskip("sklearn.linear_model")


@pytest.fixture(scope="module")
def counts() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (400, 3))
    mu = np.exp(x @ np.array([0.5, -0.3, 0.2]) + 0.4)
    y = rng.poisson(mu).astype(float)
    ds = bt.from_pydict({**{f"x{i}": x[:, i].tolist() for i in range(3)}, "y": y.tolist()})
    return x, y, ds


@pytest.mark.parametrize("alpha", [0.0, 1.0, 5.0])
def test_matches_sklearn(counts, alpha: float) -> None:
    x, y, ds = counts
    model = PoissonRegressor(["x0", "x1", "x2"], "y", alpha=alpha).fit(ds)
    sk = sk_linear.PoissonRegressor(alpha=alpha, tol=1e-10, max_iter=1000).fit(x, y)
    assert np.allclose(model.coef_, sk.coef_, atol=1e-4)
    assert model.intercept_ == pytest.approx(sk.intercept_, abs=1e-4)


def test_predictions_are_positive(counts) -> None:
    _, _, ds = counts
    model = PoissonRegressor(["x0", "x1", "x2"], "y", alpha=1.0).fit(ds)
    assert all(v > 0 for v in model.predict(ds).to_pydict()["prediction"])


def test_predictions_match_sklearn(counts) -> None:
    x, y, ds = counts
    model = PoissonRegressor(["x0", "x1", "x2"], "y", alpha=1.0).fit(ds)
    got = np.array(model.predict(ds).to_pydict()["prediction"])
    sk = sk_linear.PoissonRegressor(alpha=1.0, tol=1e-10, max_iter=1000).fit(x, y)
    assert np.allclose(got, sk.predict(x), rtol=1e-3)


def test_rejects_no_features() -> None:
    with pytest.raises(PlanError, match="at least one feature"):
        PoissonRegressor([], "y")


def test_rejects_negative_alpha() -> None:
    with pytest.raises(PlanError, match="alpha must be non-negative"):
        PoissonRegressor(["x"], "y", alpha=-1.0)


def test_names_a_missing_column(counts) -> None:
    _, _, ds = counts
    with pytest.raises(ColumnNotFoundError):
        PoissonRegressor(["x0", "nope"], "y").fit(ds)


def test_predict_before_fit_raises() -> None:
    ds = bt.from_pydict({"x": [1.0], "y": [1.0]})
    with pytest.raises(PlanError, match="must be fitted"):
        PoissonRegressor(["x"], "y").predict(ds)
