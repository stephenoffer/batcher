"""The Box-Cox power transform.

Box-Cox has a well-defined MLE, so the fit is checked against SciPy's `boxcox` two ways: the
fitted lambda lands within the grid resolution of SciPy's optimizer, and the transform applied
at a given lambda reproduces SciPy's `boxcox(x, lmbda=...)` exactly. The positivity contract is
pinned because a silent NaN on a non-positive column is the failure mode that matters.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.preprocessors import BoxCoxTransformer

pytestmark = pytest.mark.unit

scipy_stats = pytest.importorskip("scipy.stats")


@pytest.fixture(scope="module")
def skewed() -> tuple[np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(0)
    x = rng.gamma(2.0, 2.0, 500) + 0.5
    return x, bt.from_pydict({"x": x.tolist()})


def test_fitted_lambda_is_near_scipy(skewed) -> None:
    x, ds = skewed
    pre = BoxCoxTransformer("x", standardize=False).fit(ds)
    _, sk_lambda = scipy_stats.boxcox(x)
    # The engine searches a 0.1 grid; SciPy optimizes continuously, so allow the grid step.
    assert abs(pre.lambdas_["x"] - sk_lambda) <= 0.1


def test_transform_matches_scipy_at_the_fitted_lambda(skewed) -> None:
    x, ds = skewed
    pre = BoxCoxTransformer("x", standardize=False).fit(ds)
    out = np.array(pre.transform(ds).to_pydict()["x"])
    ref = scipy_stats.boxcox(x, lmbda=pre.lambdas_["x"])
    assert np.allclose(out, ref, atol=1e-9)


def test_standardized_output_is_zero_mean(skewed) -> None:
    _, ds = skewed
    out = np.array(BoxCoxTransformer("x").fit_transform(ds).to_pydict()["x"])
    assert abs(out.mean()) < 1e-9


def test_rejects_a_nonpositive_column() -> None:
    ds = bt.from_pydict({"x": [1.0, -2.0, 3.0]})
    with pytest.raises(PlanError, match="strictly positive"):
        BoxCoxTransformer("x").fit(ds)


def test_rejects_a_zero_value() -> None:
    ds = bt.from_pydict({"x": [0.0, 1.0, 2.0]})
    with pytest.raises(PlanError, match="strictly positive"):
        BoxCoxTransformer("x").fit(ds)


def test_lambda_one_leaves_a_uniform_column_roughly_linear() -> None:
    # A column that is already Gaussian-ish should get lambda near 1 (no reshaping needed).
    rng = np.random.default_rng(3)
    x = rng.normal(50, 5, 2000)
    ds = bt.from_pydict({"x": x.tolist()})
    lam = BoxCoxTransformer("x", standardize=False).fit(ds).lambdas_["x"]
    assert 0.5 <= lam <= 1.5


def test_serving_reuses_training_lambda() -> None:
    train = bt.from_pydict({"x": [1.0, 2.0, 4.0, 8.0, 16.0]})
    pre = BoxCoxTransformer("x", standardize=False).fit(train)
    lam = pre.lambdas_["x"]
    serve = bt.from_pydict({"x": [3.0]})
    got = pre.transform(serve).to_pydict()["x"][0]
    ref = scipy_stats.boxcox(np.array([3.0]), lmbda=lam)[0]
    assert got == pytest.approx(ref)
