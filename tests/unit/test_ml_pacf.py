"""The partial autocorrelation function (Yule-Walker).

The Durbin-Levinson recursion is checked against an independent Yule-Walker Toeplitz solve of the
same estimator, and against the structural property that the PACF of an AR(1) process cuts off
after lag 1.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.timeseries import partial_autocorrelation, partial_autocorrelations

pytestmark = pytest.mark.unit

scipy_linalg = pytest.importorskip("scipy.linalg")


def _acf(x: np.ndarray, k: int) -> float:
    xbar = x.mean()
    denom = ((x - xbar) ** 2).sum()
    return float(((x[k:] - xbar) * (x[:-k] - xbar)).sum() / denom)


def test_matches_yule_walker_toeplitz_solve() -> None:
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 1, 300))
    ds = bt.from_pydict({"t": list(range(300)), "x": x.tolist()})
    lags = 8
    r = [1.0] + [_acf(x, k) for k in range(1, lags + 1)]
    reference = [
        np.linalg.solve(
            scipy_linalg.toeplitz([r[i] for i in range(k)]),
            np.array([r[i] for i in range(1, k + 1)]),
        )[-1]
        for k in range(1, lags + 1)
    ]
    got = partial_autocorrelations(ds, "x", lags, order_by="t")
    for k in range(1, lags + 1):
        assert got[k] == pytest.approx(reference[k - 1], abs=1e-9)


def test_ar1_pacf_cuts_off_after_lag_one() -> None:
    # Generate an AR(1) process: the PACF should be large at lag 1 and near zero after.
    rng = np.random.default_rng(1)
    n = 2000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + rng.normal(0, 1)
    ds = bt.from_pydict({"t": list(range(n)), "x": x.tolist()})
    pacf = partial_autocorrelations(ds, "x", 5, order_by="t")
    assert pacf[1] > 0.6
    assert all(abs(pacf[k]) < 0.1 for k in range(2, 6))


def test_single_lag_matches_the_full_function() -> None:
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.normal(0, 1, 200))
    ds = bt.from_pydict({"t": list(range(200)), "x": x.tolist()})
    full = partial_autocorrelations(ds, "x", 4, order_by="t")
    assert partial_autocorrelation(ds, "x", 3, order_by="t") == pytest.approx(full[3])


def test_rejects_a_nonpositive_lag() -> None:
    ds = bt.from_pydict({"t": [0, 1], "x": [1.0, 2.0]})
    with pytest.raises(PlanError, match="lag must be positive"):
        partial_autocorrelation(ds, "x", 0, order_by="t")
