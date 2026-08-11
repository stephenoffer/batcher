"""Time-series diagnostics — autocorrelation and the tests built on it.

The autocorrelation is the plumbing that matters here: lag the column over a time order, keep
the overlap, reduce it. So the tests feed a deliberately shuffled input to prove the ordering
is honored, and pin each statistic to an independent numpy computation of the Box-Jenkins
formula rather than to a library (statsmodels is not a dependency).
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.timeseries import (
    autocorrelation,
    autocorrelations,
    durbin_watson,
    ljung_box,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def walk() -> tuple[np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 1, 300))
    return x, bt.from_pydict({"t": list(range(300)), "x": x.tolist()})


def _acf_ref(x: np.ndarray, k: int) -> float:
    xbar = x.mean()
    denom = ((x - xbar) ** 2).sum()
    return float(((x[k:] - xbar) * (x[:-k] - xbar)).sum() / denom)


def test_autocorrelation_matches_the_box_jenkins_formula(walk) -> None:
    x, ds = walk
    for k in (1, 2, 3, 10):
        assert autocorrelation(ds, "x", k, order_by="t") == pytest.approx(_acf_ref(x, k))


def test_autocorrelation_honors_the_time_order_not_row_order() -> None:
    # Rows are shuffled; ordering by t must reconstruct the monotone series (acf1 high).
    rng = np.random.default_rng(1)
    n = 100
    x = np.arange(n, dtype=float)
    order = rng.permutation(n)
    ds = bt.from_pydict({"t": order.tolist(), "x": x[order].tolist()})
    # In time order the series is 0,1,2,... so lag-1 autocorrelation is strongly positive.
    assert autocorrelation(ds, "x", 1, order_by="t") > 0.9


def test_autocorrelation_rejects_a_nonpositive_lag(walk) -> None:
    _, ds = walk
    with pytest.raises(PlanError, match="lag must be positive"):
        autocorrelation(ds, "x", 0, order_by="t")


def test_autocorrelations_returns_every_lag(walk) -> None:
    _, ds = walk
    acf = autocorrelations(ds, "x", 5, order_by="t")
    assert list(acf) == [1, 2, 3, 4, 5]


def test_ljung_box_matches_the_reference(walk) -> None:
    x, ds = walk
    n, lags = 300, 5
    q_ref = n * (n + 2) * sum(_acf_ref(x, k) ** 2 / (n - k) for k in range(1, lags + 1))
    got = ljung_box(ds, "x", lags, order_by="t")
    assert got.statistic == pytest.approx(q_ref)
    assert got.df == float(lags)


def test_ljung_box_rejects_a_trending_series(walk) -> None:
    _, ds = walk
    # A random walk is heavily autocorrelated, so the null of no autocorrelation is rejected.
    assert ljung_box(ds, "x", 5, order_by="t").pvalue < 0.001


def test_ljung_box_keeps_white_noise() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 500)
    ds = bt.from_pydict({"t": list(range(500)), "x": x.tolist()})
    assert ljung_box(ds, "x", 10, order_by="t").pvalue > 0.05


def test_durbin_watson_matches_the_reference() -> None:
    rng = np.random.default_rng(2)
    e = rng.normal(0, 1, 200)
    ds = bt.from_pydict({"t": list(range(200)), "e": e.tolist()})
    dw_ref = (np.diff(e) ** 2).sum() / (e**2).sum()
    assert durbin_watson(ds, "e", order_by="t") == pytest.approx(dw_ref)


def test_durbin_watson_is_near_two_for_independent_residuals() -> None:
    rng = np.random.default_rng(3)
    e = rng.normal(0, 1, 1000)
    ds = bt.from_pydict({"t": list(range(1000)), "e": e.tolist()})
    assert durbin_watson(ds, "e", order_by="t") == pytest.approx(2.0, abs=0.2)


def test_the_acf_costs_one_window_however_many_lags() -> None:
    """Every lag shares one `Window`, which is the whole point of computing them together.

    Lag by lag, an ACF costs three executions *and a full global sort* each — a 40-lag
    seasonal diagnostic meant 40 sorts of the entire series. Building the lag columns in one
    `with_columns` collapses them onto a single `Window` node over one ordering. A correctness
    test cannot see the difference (the numbers are identical either way), so the plan shape
    is what has to be pinned.
    """
    from batcher.plan.expr_ir.constructors import col
    from batcher.plan.expr_ir.nodes import lag

    ds = bt.from_pydict({"t": list(range(50)), "x": [float(i % 7) for i in range(50)]})
    lagged = ds.with_columns(
        **{f"__l{k}": lag(col("x"), k).over(order_by=["t"]) for k in range(1, 41)}
    )
    plan = lagged.agg(**{f"r{k}": col(f"__l{k}").sum() for k in range(1, 41)}).explain()
    assert plan.lower().count("window") == 1, plan


def test_the_acf_is_unchanged_by_computing_the_lags_together() -> None:
    """The fused form must equal the per-lag form exactly, not merely closely."""
    rng = np.random.default_rng(7)
    series = [0.0]
    for _ in range(299):
        series.append(0.7 * series[-1] + float(rng.normal()))
    ds = bt.from_pydict({"t": list(range(300)), "x": series})

    fused = autocorrelations(ds, "x", 12, order_by="t")
    one_at_a_time = {k: autocorrelation(ds, "x", k, order_by="t") for k in range(1, 13)}
    assert fused == one_at_a_time


def test_a_lag_longer_than_the_series_is_undefined_not_zero() -> None:
    """No overlapping pair means no correlation to report, so it is NaN rather than 0.0.

    Summing over an empty overlap gives null, and reading that as a number used to raise a
    `TypeError` from inside `float()` — a message about a conversion for a question that
    simply has no answer.
    """
    ds = bt.from_pydict({"t": [0, 1, 2], "x": [1.0, 5.0, 3.0]})
    assert np.isnan(autocorrelations(ds, "x", 5, order_by="t")[4])


def test_autocorrelations_rejects_a_non_positive_lag_count() -> None:
    ds = bt.from_pydict({"t": [0, 1, 2], "x": [1.0, 2.0, 3.0]})
    with pytest.raises(PlanError, match="positive lag count"):
        autocorrelations(ds, "x", 0, order_by="t")
