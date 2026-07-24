"""Tweedie regression, and its Poisson/Gamma special cases.

`TweedieRegressor` with a log link is checked against scikit-learn across powers and penalties on
compound-shaped data. Because `PoissonRegressor` (power 1) and `GammaRegressor` (power 2) are now
thin subclasses, the test also confirms Tweedie at those powers reproduces them exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.glm import GammaRegressor, PoissonRegressor, TweedieRegressor

pytestmark = pytest.mark.unit

sk_linear = pytest.importorskip("sklearn.linear_model")


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (400, 3))
    mu = np.exp(x @ np.array([0.4, -0.2, 0.3]) + 0.5)
    y = rng.gamma(2.0, mu / 2.0)
    ds = bt.from_pydict({**{f"x{i}": x[:, i].tolist() for i in range(3)}, "y": y.tolist()})
    return x, y, ds


@pytest.mark.parametrize("power", [1.2, 1.5, 1.8])
@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_matches_sklearn(data, power: float, alpha: float) -> None:
    x, y, ds = data
    model = TweedieRegressor(["x0", "x1", "x2"], "y", power=power, alpha=alpha).fit(ds)
    sk = sk_linear.TweedieRegressor(
        power=power, alpha=alpha, link="log", tol=1e-10, max_iter=1000
    ).fit(x, y)
    assert np.allclose(model.coef_, sk.coef_, atol=1e-4)
    assert model.intercept_ == pytest.approx(sk.intercept_, abs=1e-4)


def test_power_one_equals_poisson(data) -> None:
    _, _, ds = data
    tweedie = TweedieRegressor(["x0", "x1", "x2"], "y", power=1.0, alpha=0.5).fit(ds)
    poisson = PoissonRegressor(["x0", "x1", "x2"], "y", alpha=0.5).fit(ds)
    assert np.allclose(tweedie.coef_, poisson.coef_, atol=1e-10)


def test_power_two_equals_gamma(data) -> None:
    _, _, ds = data
    tweedie = TweedieRegressor(["x0", "x1", "x2"], "y", power=2.0, alpha=0.5).fit(ds)
    gamma = GammaRegressor(["x0", "x1", "x2"], "y", alpha=0.5).fit(ds)
    assert np.allclose(tweedie.coef_, gamma.coef_, atol=1e-10)


def test_predictions_are_positive(data) -> None:
    _, _, ds = data
    model = TweedieRegressor(["x0", "x1", "x2"], "y", power=1.5, alpha=1.0).fit(ds)
    assert all(v > 0 for v in model.predict(ds).to_pydict()["prediction"])


def test_rejects_power_outside_range() -> None:
    with pytest.raises(PlanError, match="power must be in"):
        TweedieRegressor(["x"], "y", power=0.5)
    with pytest.raises(PlanError, match="power must be in"):
        TweedieRegressor(["x"], "y", power=2.5)


def test_rejects_no_features() -> None:
    with pytest.raises(PlanError, match="at least one feature"):
        TweedieRegressor([], "y")
