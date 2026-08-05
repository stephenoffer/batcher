"""`HuberRegressor` — a fit a few wild target values cannot dominate.

Squared error grows with the square of the residual, so one row off by a hundred weighs as
much as ten thousand rows off by one. A single mistyped price or a stuck sensor visibly tilts
an ordinary least-squares fit, and nothing reports it: the coefficients move, the residuals
get worse everywhere, and the model is simply wrong.

The first test pins that OLS really is pulled on this data, so the rest are not defending a
fix to a problem nobody showed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, DataWarning, PlanError
from batcher.ml import HuberRegressor, LinearRegression

pytestmark = pytest.mark.unit

TRUE_SLOPE = 3.0


@pytest.fixture(scope="module")
def contaminated() -> bt.Dataset:
    """200 rows on a line of slope 3, with the first eight shifted far off it."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = TRUE_SLOPE * x + rng.normal(scale=0.5, size=200)
    y[:8] += 60.0
    return bt.from_pydict({"x": x.tolist(), "y": y.tolist()})


@pytest.fixture(scope="module")
def clean() -> bt.Dataset:
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    y = TRUE_SLOPE * x + rng.normal(scale=0.5, size=200)
    return bt.from_pydict({"x": x.tolist(), "y": y.tolist()})


def test_least_squares_really_is_pulled_by_the_outliers(contaminated) -> None:
    """The premise. Four percent of rows, and the slope moves by nearly a whole unit."""
    ordinary = LinearRegression(["x"], "y").fit(contaminated).coef_[0]
    assert abs(ordinary - TRUE_SLOPE) > 0.5, f"expected OLS to be pulled, got {ordinary}"


def test_huber_recovers_the_slope_the_outliers_hid(contaminated) -> None:
    robust = HuberRegressor(["x"], "y").fit(contaminated).coef_[0]
    assert robust == pytest.approx(TRUE_SLOPE, abs=0.1)


def test_huber_is_closer_to_the_truth_than_least_squares(contaminated) -> None:
    ordinary = LinearRegression(["x"], "y").fit(contaminated).coef_[0]
    robust = HuberRegressor(["x"], "y").fit(contaminated).coef_[0]
    assert abs(robust - TRUE_SLOPE) < abs(ordinary - TRUE_SLOPE)


def test_on_clean_data_it_agrees_with_least_squares(clean) -> None:
    """Robustness must not cost accuracy when there is nothing to be robust against."""
    ordinary = LinearRegression(["x"], "y").fit(clean)
    robust = HuberRegressor(["x"], "y").fit(clean)
    assert robust.coef_[0] == pytest.approx(ordinary.coef_[0], abs=0.05)
    assert robust.intercept_ == pytest.approx(ordinary.intercept_, abs=0.05)


def test_a_smaller_epsilon_is_more_robust(contaminated) -> None:
    """`epsilon` is where the loss turns linear, so a smaller one discounts sooner."""
    tight = HuberRegressor(["x"], "y", epsilon=1.1).fit(contaminated).coef_[0]
    loose = HuberRegressor(["x"], "y", epsilon=50.0).fit(contaminated).coef_[0]
    assert abs(tight - TRUE_SLOPE) <= abs(loose - TRUE_SLOPE)


def test_a_very_large_epsilon_falls_back_towards_least_squares(contaminated) -> None:
    """With the cutoff far past every residual, no row is down-weighted."""
    ordinary = LinearRegression(["x"], "y").fit(contaminated).coef_[0]
    loose = HuberRegressor(["x"], "y", epsilon=1e6, max_iter=5).fit(contaminated).coef_[0]
    assert loose == pytest.approx(ordinary, abs=0.05)


def test_it_matches_sklearn_closely(contaminated) -> None:
    """Not bit-identical - sklearn fits the scale jointly - but the same answer."""
    sk = pytest.importorskip("sklearn.linear_model")
    table = contaminated.to_pydict()
    x = np.array(table["x"]).reshape(-1, 1)
    y = np.array(table["y"])
    theirs = sk.HuberRegressor(epsilon=1.35, alpha=0.0).fit(x, y)
    ours = HuberRegressor(["x"], "y").fit(contaminated)
    assert ours.coef_[0] == pytest.approx(float(theirs.coef_[0]), abs=0.1)


def test_multiple_features(contaminated) -> None:
    rng = np.random.default_rng(4)
    n = 300
    a, b = rng.normal(size=n), rng.normal(size=n)
    y = 2.0 * a - 1.5 * b + rng.normal(scale=0.4, size=n)
    y[:10] += 80.0
    ds = bt.from_pydict({"a": a.tolist(), "b": b.tolist(), "y": y.tolist()})
    model = HuberRegressor(["a", "b"], "y").fit(ds)
    assert model.coef_[0] == pytest.approx(2.0, abs=0.15)
    assert model.coef_[1] == pytest.approx(-1.5, abs=0.15)


def test_a_union_of_partitions_fits_what_one_partition_does(contaminated) -> None:
    """Every step is an aggregate, so partitioning must not change the answer."""
    table = contaminated.to_pydict()
    half = len(table["y"]) // 2
    left = bt.from_pydict({k: v[:half] for k, v in table.items()})
    right = bt.from_pydict({k: v[half:] for k, v in table.items()})
    whole = HuberRegressor(["x"], "y").fit(contaminated)
    parted = HuberRegressor(["x"], "y").fit(left.union(right))
    assert whole.coef_ == pytest.approx(parted.coef_, abs=1e-6)
    assert whole.intercept_ == pytest.approx(parted.intercept_, abs=1e-6)


def test_predict_appends_the_linear_score(contaminated) -> None:
    model = HuberRegressor(["x"], "y").fit(contaminated)
    scored = model.predict(contaminated).to_pydict()
    table = contaminated.to_pydict()
    expected = [model.intercept_ + model.coef_[0] * v for v in table["x"]]
    assert scored["prediction"] == pytest.approx(expected, abs=1e-9)


def test_it_records_the_scale_it_fitted_against(contaminated) -> None:
    model = HuberRegressor(["x"], "y").fit(contaminated)
    assert model.scale_ > 0
    assert model.converged_ is True


def test_hitting_the_iteration_cap_warns_rather_than_passing_silently() -> None:
    """On an exactly-fitting subset the scale chases zero and the weights never settle.

    That is a real property of reweighting against a re-estimated scale, so the fit says it
    stopped on the cap rather than presenting the last iterate as an optimum.
    """
    ds = bt.from_pydict(
        {"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "y": [2.0, 4.0, 6.0, 8.0, 10.0, 400.0]}
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = HuberRegressor(["x"], "y").fit(ds)
    assert model.converged_ is False
    assert any(issubclass(c.category, DataWarning) for c in caught)
    # It is still the right answer, which is why this warns rather than raising.
    assert model.coef_[0] == pytest.approx(2.0, abs=0.05)


# --------------------------------------------------------------------------------------
# Rejections
# --------------------------------------------------------------------------------------


def test_no_features_is_rejected() -> None:
    with pytest.raises(PlanError, match="at least one feature"):
        HuberRegressor([], "y")


@pytest.mark.parametrize("value", [1.0, 0.5, 0.0, -1.0])
def test_an_epsilon_at_or_below_one_is_rejected(value: float) -> None:
    with pytest.raises(PlanError, match="epsilon"):
        HuberRegressor(["x"], "y", epsilon=value)


def test_a_negative_alpha_is_rejected() -> None:
    with pytest.raises(PlanError, match="alpha"):
        HuberRegressor(["x"], "y", alpha=-1.0)


def test_a_missing_column_is_named(contaminated) -> None:
    with pytest.raises(ColumnNotFoundError):
        HuberRegressor(["nope"], "y").fit(contaminated)


def test_a_string_feature_is_named(contaminated) -> None:
    with pytest.raises(PlanError, match="'label'"):
        HuberRegressor(["label"], "y").fit(contaminated.with_columns(label=bt.lit("x")))


def test_predicting_before_fitting_is_rejected(contaminated) -> None:
    with pytest.raises(PlanError, match="must be fitted"):
        HuberRegressor(["x"], "y").predict(contaminated)
