"""The order-statistic and association functions in `plan.functions.analysis`.

Thirteen public names here had no test that so much as mentioned them: `midhinge`, `trimean`,
`bowley_skew`, `quartile_dispersion`, `interdecile_range`, `decile_ratio`, `moors_kurtosis`,
`geometric_std`, `pearson_mode_skew`, `correlation_ratio`, `point_biserial`, `signal_ratio` and
`weighted_covariance`. Each states its closed form in its own docstring, so that formula in
numpy is the oracle, and the bounds each one claims are asserted separately -- a ratio of order
statistics is the kind of thing that stays plausible while being wrong.

All thirteen were already correct. These tests exist so they stay that way.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

import batcher as bt

pytestmark = pytest.mark.unit

_RAMP = np.arange(1.0, 101.0)


@pytest.fixture(scope="module")
def skewed() -> np.ndarray:
    rng = np.random.default_rng(31)
    return np.abs(rng.lognormal(1.0, 0.6, 400)) + 0.5


def _one(values: np.ndarray, expr) -> float:
    return bt.from_pydict({"x": values.tolist()}).agg(r=expr).to_pydict()["r"][0]


def _q(x: np.ndarray, p: float) -> float:
    return float(np.quantile(x, p))


@pytest.mark.parametrize("data", ["ramp", "skewed"])
def test_quartile_summaries_match_their_closed_forms(data, skewed) -> None:
    x = _RAMP if data == "ramp" else skewed
    q1, q2, q3 = _q(x, 0.25), _q(x, 0.5), _q(x, 0.75)
    p10, p90 = _q(x, 0.10), _q(x, 0.90)

    assert _one(x, bt.midhinge("x")) == pytest.approx((q1 + q3) / 2.0)
    assert _one(x, bt.trimean("x")) == pytest.approx((q1 + 2 * q2 + q3) / 4.0)
    assert _one(x, bt.bowley_skew("x")) == pytest.approx((q3 + q1 - 2 * q2) / (q3 - q1))
    assert _one(x, bt.quartile_dispersion("x")) == pytest.approx((q3 - q1) / (q3 + q1))
    assert _one(x, bt.interdecile_range("x")) == pytest.approx(p90 - p10)
    assert _one(x, bt.decile_ratio("x")) == pytest.approx(p90 / p10)


@pytest.mark.parametrize("data", ["ramp", "skewed"])
def test_moors_kurtosis_is_the_octile_ratio(data, skewed) -> None:
    x = _RAMP if data == "ramp" else skewed
    o = [_q(x, k / 8.0) for k in range(9)]
    want = ((o[7] - o[5]) + (o[3] - o[1])) / (o[6] - o[2])
    assert _one(x, bt.moors_kurtosis("x")) == pytest.approx(want)


def test_moors_kurtosis_is_one_on_a_uniform_ramp() -> None:
    """Evenly spaced octiles give ``(2/8 + 2/8) / (4/8)``; the docstring's normal value is 1.23."""
    assert _one(_RAMP, bt.moors_kurtosis("x")) == pytest.approx(1.0)


@pytest.mark.parametrize("data", ["ramp", "skewed"])
def test_geometric_std_is_exp_of_the_log_spread(data, skewed) -> None:
    x = _RAMP if data == "ramp" else skewed
    got = _one(x, bt.geometric_std("x"))
    ln = np.log(x)
    assert got == pytest.approx(float(np.exp(ln.std(ddof=1))))


@pytest.mark.parametrize(
    "data",
    ["ramp", "skewed", "two spikes", "heavy tail"],
)
def test_bowley_skew_stays_inside_its_documented_bounds(data, skewed) -> None:
    rng = np.random.default_rng(7)
    x = {
        "ramp": _RAMP,
        "skewed": skewed,
        "two spikes": np.concatenate([np.ones(200), np.full(200, 100.0)]),
        "heavy tail": np.concatenate([rng.normal(10, 1, 390), np.full(10, 1e6)]),
    }[data]
    got = _one(x, bt.bowley_skew("x"))
    assert not math.isnan(got)
    assert -1.0 <= got <= 1.0, f"quartile skewness is bounded in [-1, 1], got {got}"


def test_pearson_mode_skew_is_the_mean_minus_mode_over_the_spread() -> None:
    rng = np.random.default_rng(31)
    values = np.concatenate([np.full(100, 5.0), rng.normal(8, 2, 300)])
    mode = Counter(values.tolist()).most_common(1)[0][0]
    assert mode == 5.0
    got = _one(values, bt.pearson_mode_skew("x"))
    assert got == pytest.approx(float((values.mean() - mode) / values.std(ddof=1)))


# --------------------------------------------------------------------------- #
# Association: correlation_ratio, point_biserial, signal_ratio
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def grouped() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(31)
    group = rng.integers(0, 4, 300)
    y = group * 3.0 + rng.normal(0, 1, 300)
    means = np.array([y[group == g].mean() for g in group])
    return y, means, bt.from_pydict({"y": y.tolist(), "m": means.tolist()})


def test_correlation_ratio_is_the_variance_share(grouped) -> None:
    y, means, ds = grouped
    got = ds.agg(e=bt.correlation_ratio("y", "m")).to_pydict()["e"][0]
    assert got == pytest.approx(float(means.var(ddof=1) / y.var(ddof=1)))
    assert 0.0 <= got <= 1.0


def test_correlation_ratio_is_zero_when_the_grouping_explains_nothing(grouped) -> None:
    y, _, _ = grouped
    ds = bt.from_pydict({"y": y.tolist(), "m": [float(y.mean())] * len(y)})
    assert ds.agg(e=bt.correlation_ratio("y", "m")).to_pydict()["e"][0] == pytest.approx(0.0)


def test_correlation_ratio_is_one_when_the_grouping_explains_everything(grouped) -> None:
    y, _, _ = grouped
    ds = bt.from_pydict({"y": y.tolist(), "m": y.tolist()})
    assert ds.agg(e=bt.correlation_ratio("y", "m")).to_pydict()["e"][0] == pytest.approx(1.0)


@pytest.fixture(scope="module")
def binary_outcome() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(31)
    hit = rng.random(300) < 0.4
    x = np.where(hit, rng.normal(9, 2, 300), rng.normal(5, 2, 300))
    return x, hit, bt.from_pydict({"x": x.tolist(), "hit": hit.tolist()})


def test_point_biserial_is_pearson_with_the_outcome_coded_zero_one(binary_outcome) -> None:
    x, hit, ds = binary_outcome
    got = ds.agg(r=bt.point_biserial("x", bt.col("hit"))).to_pydict()["r"][0]
    assert got == pytest.approx(float(np.corrcoef(x, hit.astype(float))[0, 1]))
    assert -1.0 <= got <= 1.0


def test_signal_ratio_is_the_mean_gap_in_standard_deviations(binary_outcome) -> None:
    x, hit, ds = binary_outcome
    got = ds.agg(s=bt.signal_ratio("x", bt.col("hit"))).to_pydict()["s"][0]
    gap = abs(x[hit].mean() - x[~hit].mean())
    assert got == pytest.approx(float(gap / x.std(ddof=1)))
    assert got >= 0.0, "an absolute difference is never negative"


# --------------------------------------------------------------------------- #
# weighted_covariance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("weights", ["unit", "integer", "fractional"])
def test_weighted_covariance_matches_its_definition(weights: str) -> None:
    rng = np.random.default_rng(31)
    n = 300
    x = rng.normal(0.0, 1.0, n)
    y = 0.7 * x + rng.normal(0.0, 1.0, n)
    w = {
        "unit": np.ones(n),
        "integer": rng.integers(1, 5, n).astype(float),
        "fractional": rng.random(n) + 0.1,
    }[weights]
    ds = bt.from_pydict({"x": x.tolist(), "y": y.tolist(), "w": w.tolist()})
    got = ds.agg(c=bt.weighted_covariance("x", "y", "w")).to_pydict()["c"][0]

    total = w.sum()
    want = (w * x * y).sum() / total - (w * x).sum() / total * (w * y).sum() / total
    assert got == pytest.approx(float(want))


def test_weighted_covariance_with_unit_weights_is_the_population_covariance() -> None:
    rng = np.random.default_rng(31)
    x = rng.normal(0.0, 1.0, 300)
    y = 0.7 * x + rng.normal(0.0, 1.0, 300)
    ds = bt.from_pydict({"x": x.tolist(), "y": y.tolist(), "w": [1.0] * 300})
    got = ds.agg(c=bt.weighted_covariance("x", "y", "w")).to_pydict()["c"][0]
    assert got == pytest.approx(float(np.cov(x, y, ddof=0)[0, 1]))


# --------------------------------------------------------------------------- #
# Degenerate input: a zero spread must not become a confident number
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["bowley_skew", "moors_kurtosis"])
def test_a_constant_column_gives_no_shape_estimate(name: str) -> None:
    """Every order statistic coincides, so these are 0/0 and NaN is the honest answer."""
    got = _one(np.full(50, 4.0), getattr(bt, name)("x"))
    assert got is None or math.isnan(got), f"{name} on a constant column returned {got}"


def test_a_constant_column_has_zero_quartile_dispersion() -> None:
    """Not 0/0: ``(q3 - q1) / (q3 + q1)`` is ``0 / 8`` here, and zero spread is the answer."""
    assert _one(np.full(50, 4.0), bt.quartile_dispersion("x")) == pytest.approx(0.0)
    assert _one(np.full(50, 4.0), bt.decile_ratio("x")) == pytest.approx(1.0)
    assert _one(np.full(50, 4.0), bt.interdecile_range("x")) == pytest.approx(0.0)
