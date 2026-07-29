"""Fitting an estimator on data that cannot support it says so, in the model's terms.

Every one of these came back as a message from NumPy or the `math` module —
``float() argument must be a string or a real number, not 'NoneType'`` for an empty
dataset, ``LinAlgError: Singular matrix`` for collinear features, ``math domain error`` for
zero-variance ones, ``SVD did not converge`` for a class with one example. Each names the
solver step that failed and none names the data problem that caused it, which is the thing
the caller can act on.

Where a remedy is suggested, these check the remedy actually works.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml import (
    ElasticNet,
    GammaRegressor,
    GaussianNB,
    Lasso,
    LinearDiscriminantAnalysis,
    LinearRegression,
    LogisticRegression,
    PoissonRegressor,
    QuadraticDiscriminantAnalysis,
    Ridge,
    TweedieRegressor,
)


@pytest.fixture
def usable() -> bt.Dataset:
    return bt.from_pydict(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
            "y": [0, 1, 0, 1, 0, 1],
            "yr": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


@pytest.fixture
def empty(usable: bt.Dataset) -> bt.Dataset:
    return usable.filter(bt.col("a") > 99)


@pytest.mark.parametrize(
    ("cls", "target"),
    [
        (LinearRegression, "yr"),
        (Ridge, "yr"),
        (Lasso, "yr"),
        (ElasticNet, "yr"),
        (LogisticRegression, "y"),
        (PoissonRegressor, "yr"),
        (GammaRegressor, "yr"),
        (TweedieRegressor, "yr"),
    ],
)
def test_fitting_on_an_empty_dataset_names_the_row_count(empty: bt.Dataset, cls, target) -> None:
    with pytest.raises(PlanError, match=r"row\(s\)"):
        cls(features=["a", "b"], target=target).fit(empty)


def test_collinear_features_are_named_as_the_problem() -> None:
    """A duplicated column or a full one-hot is the ordinary way to hit this."""
    collinear = bt.from_pydict({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})
    collinear = collinear.with_columns(y=bt.col("a"))
    with pytest.raises(PlanError, match="linearly dependent"):
        LinearRegression(features=["a", "b"], target="y").fit(collinear)


def test_the_suggested_ridge_remedy_actually_works() -> None:
    """The error points at Ridge, so Ridge had better fit the same data."""
    collinear = bt.from_pydict({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})
    collinear = collinear.with_columns(y=bt.col("a"))
    model = Ridge(features=["a", "b"], target="y", alpha=1.0).fit(collinear)
    assert len(model.coef_) == 2


def test_gaussian_nb_rejects_features_with_no_variance() -> None:
    """`var_smoothing` is a fraction of the widest variance, so it is zero when all are."""
    constant = bt.from_pydict({"a": [5.0] * 6, "b": [5.0] * 6, "y": [0, 1, 0, 1, 0, 1]})
    with pytest.raises(PlanError, match="zero variance"):
        GaussianNB(features=["a", "b"], target="y").fit(constant)


def test_gaussian_nb_still_fits_ordinary_data(usable: bt.Dataset) -> None:
    model = GaussianNB(features=["a", "b"], target="y").fit(usable)
    assert len(model.predict(usable).to_pydict()["prediction"]) == 6


def test_qda_names_the_class_that_is_too_rare() -> None:
    """A class with one example is the ordinary shape of imbalanced data."""
    rare = bt.from_pydict({"a": [1.0, 2.0, 3.0, 9.0], "b": [2.0, 1.0, 4.0, 9.0], "y": [0, 0, 0, 1]})
    with pytest.raises(PlanError, match="class 1 has 1"):
        QuadraticDiscriminantAnalysis(features=["a", "b"], target="y").fit(rare)


def test_the_suggested_lda_remedy_actually_works() -> None:
    """The error points at LDA's pooled covariance, so LDA had better fit the same data."""
    rare = bt.from_pydict({"a": [1.0, 2.0, 3.0, 9.0], "b": [2.0, 1.0, 4.0, 9.0], "y": [0, 0, 0, 1]})
    model = LinearDiscriminantAnalysis(features=["a", "b"], target="y").fit(rare)
    assert model.predict(rare).to_pydict()["prediction"] == [0, 0, 0, 1]


def test_usable_data_still_fits_everywhere(usable: bt.Dataset) -> None:
    """The guards must not have moved the floor above ordinary training data."""
    assert LinearRegression(features=["a", "b"], target="yr").fit(usable).coef_
    assert LogisticRegression(features=["a", "b"], target="y").fit(usable).coef_
    assert TweedieRegressor(["a"], "yr").fit(usable).n_iter_ > 0
