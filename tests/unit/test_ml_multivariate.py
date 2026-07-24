"""Correlation/covariance matrices and partial correlation.

The matrices are checked against numpy's `corrcoef` and `cov` cell for cell. The partial
correlation is checked against its independent residual-regression definition (regress each
variable on the controls, then correlate the residuals), which is the honest oracle since the
closed form and the residual form are the same statistic arrived at two ways.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher.ml.stats import correlation_matrix, covariance_matrix, partial_correlation

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def three() -> tuple[np.ndarray, np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 300)
    b = 2 * a + rng.normal(0, 1, 300)
    c = rng.normal(0, 1, 300)
    ds = bt.from_pydict({"a": a.tolist(), "b": b.tolist(), "c": c.tolist()})
    return a, b, c, ds


def test_correlation_matrix_matches_numpy(three) -> None:
    a, b, c, ds = three
    got = correlation_matrix(ds, ["a", "b", "c"]).to_pydict()
    mine = np.array([got[name] for name in ["a", "b", "c"]]).T
    assert np.allclose(mine, np.corrcoef([a, b, c]), atol=1e-9)


def test_correlation_matrix_is_labeled_and_square(three) -> None:
    _, _, _, ds = three
    got = correlation_matrix(ds, ["a", "b", "c"]).to_pydict()
    assert got["column"] == ["a", "b", "c"]
    assert got["a"][0] == pytest.approx(1.0)


def test_covariance_matrix_matches_numpy(three) -> None:
    a, b, c, ds = three
    got = covariance_matrix(ds, ["a", "b", "c"]).to_pydict()
    mine = np.array([got[name] for name in ["a", "b", "c"]]).T
    assert np.allclose(mine, np.cov([a, b, c]), atol=1e-9)


def _resid(u: np.ndarray, controls: np.ndarray) -> np.ndarray:
    design = np.column_stack([controls, np.ones(len(u))])
    coef, *_ = np.linalg.lstsq(design, u, rcond=None)
    return u - design @ coef


def test_partial_correlation_one_control_matches_residuals() -> None:
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1, 400)
    x = z + rng.normal(0, 1, 400)
    y = z + rng.normal(0, 1, 400)
    ds = bt.from_pydict({"x": x.tolist(), "y": y.tolist(), "z": z.tolist()})
    ref = np.corrcoef(_resid(x, z[:, None]), _resid(y, z[:, None]))[0, 1]
    assert partial_correlation(ds, "x", "y", "z") == pytest.approx(ref, abs=1e-6)


def test_partial_correlation_several_controls_matches_residuals() -> None:
    rng = np.random.default_rng(2)
    z1, z2 = rng.normal(0, 1, 400), rng.normal(0, 1, 400)
    x = z1 + z2 + rng.normal(0, 1, 400)
    y = z1 + z2 + rng.normal(0, 1, 400)
    ds = bt.from_pydict({"x": x.tolist(), "y": y.tolist(), "z1": z1.tolist(), "z2": z2.tolist()})
    controls = np.column_stack([z1, z2])
    ref = np.corrcoef(_resid(x, controls), _resid(y, controls))[0, 1]
    assert partial_correlation(ds, "x", "y", ["z1", "z2"]) == pytest.approx(ref, abs=1e-6)


def test_partial_correlation_removes_a_confounder() -> None:
    # x and y correlate only because both track z; controlling for z collapses it toward 0.
    rng = np.random.default_rng(3)
    z = rng.normal(0, 1, 500)
    x = z + rng.normal(0, 0.3, 500)
    y = z + rng.normal(0, 0.3, 500)
    ds = bt.from_pydict({"x": x.tolist(), "y": y.tolist(), "z": z.tolist()})
    marginal = correlation_matrix(ds, ["x", "y"]).to_pydict()["x"][1]
    assert marginal > 0.8
    assert abs(partial_correlation(ds, "x", "y", "z")) < 0.2
