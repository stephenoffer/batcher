"""L1-regularized linear models (lasso and elastic net).

The elastic-net objective is strictly convex, so it has a unique minimizer and any correct solver
reaches it: the coordinate descent here matches scikit-learn's coefficients, intercept, and — the
point of an L1 penalty — its exact sparsity pattern, across penalty strengths and mixing ratios.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.sparse_linear import ElasticNet, Lasso

pytestmark = pytest.mark.unit

sk_linear = pytest.importorskip("sklearn.linear_model")


@pytest.fixture(scope="module")
def sparse() -> tuple[np.ndarray, np.ndarray, list[str], bt.Dataset]:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (300, 5))
    true = np.array([1.5, 0.0, -2.0, 0.0, 0.5])
    y = x @ true + rng.normal(0, 0.3, 300)
    names = [f"x{i}" for i in range(5)]
    ds = bt.from_pydict({**{n: x[:, i].tolist() for i, n in enumerate(names)}, "y": y.tolist()})
    return x, y, names, ds


@pytest.mark.parametrize(("alpha", "l1_ratio"), [(0.1, 1.0), (0.1, 0.5), (0.5, 0.5), (0.05, 0.9)])
def test_elastic_net_matches_sklearn(sparse, alpha: float, l1_ratio: float) -> None:
    x, y, names, ds = sparse
    model = ElasticNet(names, "y", alpha=alpha, l1_ratio=l1_ratio).fit(ds)
    sk = sk_linear.ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(x, y)
    assert np.allclose(model.coef_, sk.coef_, atol=1e-4)
    assert model.intercept_ == pytest.approx(sk.intercept_, abs=1e-4)


def test_lasso_matches_sklearn_including_sparsity(sparse) -> None:
    x, y, names, ds = sparse
    model = Lasso(names, "y", alpha=0.1).fit(ds)
    sk = sk_linear.Lasso(alpha=0.1).fit(x, y)
    assert np.allclose(model.coef_, sk.coef_, atol=1e-4)
    assert np.array_equal(np.array(model.coef_) == 0, sk.coef_ == 0)


def test_lasso_zeros_the_noise_features(sparse) -> None:
    _, _, names, ds = sparse
    model = Lasso(names, "y", alpha=0.1).fit(ds)
    # Features x1 and x3 have a true coefficient of zero and should be dropped.
    assert model.coef_[1] == 0.0
    assert model.coef_[3] == 0.0


def test_prediction_matches_sklearn(sparse) -> None:
    x, y, names, ds = sparse
    model = ElasticNet(names, "y", alpha=0.1, l1_ratio=0.5).fit(ds)
    got = np.array(model.predict(ds).to_pydict()["prediction"])
    sk = sk_linear.ElasticNet(alpha=0.1, l1_ratio=0.5).fit(x, y)
    assert np.allclose(got, sk.predict(x), atol=1e-3)


def test_rejects_bad_arguments() -> None:
    with pytest.raises(PlanError, match="at least one feature"):
        ElasticNet([], "y")
    with pytest.raises(PlanError, match="l1_ratio must be"):
        ElasticNet(["x"], "y", l1_ratio=2.0)
    with pytest.raises(PlanError, match="alpha must be non-negative"):
        Lasso(["x"], "y", alpha=-1.0)


def test_names_a_missing_column(sparse) -> None:
    _, _, names, ds = sparse
    with pytest.raises(ColumnNotFoundError):
        ElasticNet([*names, "nope"], "y").fit(ds)
