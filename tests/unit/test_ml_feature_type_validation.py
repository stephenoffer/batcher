"""Every native estimator rejects an unusable feature column by name.

Before this, a string feature produced whatever the data plane happened to say first, and it
said something different depending on which aggregate ran: ``aggregate mean is not supported
for column type Utf8``, ``Cast error: Cannot cast string 'a' to value of Float64``, ``Invalid
arithmetic operation: Float64``, or ``could not convert string to float: 'b'``. Four messages
for one mistake, none naming the column and none saying what to do.

The check reads the schema rather than the data, so it costs no pass and fires before any work
is scheduled. These tests pin it across the whole estimator surface, because a validation that
covers eight of fifteen estimators is the kind of thing that reads as done and is not.
"""

from __future__ import annotations

import datetime

import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit


def _build(name: str, features: list[str]):
    """One unfitted instance of `name`, with the constructor arguments it actually takes."""
    import batcher.ml as ml

    klass = getattr(ml, name)
    if name == "KMeans":
        return klass(features, n_clusters=2)
    if name == "GaussianMixture":
        return klass(features, n_components=2)
    return klass(features, "y")


#: Every estimator that fits from aggregates over named feature columns.
ESTIMATORS = [
    "LinearRegression",
    "Ridge",
    "LogisticRegression",
    "RidgeClassifier",
    "GaussianNB",
    "MultinomialNB",
    "BernoulliNB",
    "KMeans",
    "NearestCentroid",
    "LinearDiscriminantAnalysis",
    "QuadraticDiscriminantAnalysis",
    "Lasso",
    "ElasticNet",
    "PoissonRegressor",
    "GaussianMixture",
]


@pytest.mark.parametrize("name", ESTIMATORS)
def test_a_string_feature_is_rejected_by_name(name: str) -> None:
    ds = bt.from_pydict({"x": ["a", "b", "c", "d"], "z": [1.0, 2.0, 3.0, 4.0], "y": [0, 0, 1, 1]})
    with pytest.raises(PlanError) as caught:
        _build(name, ["x", "z"]).fit(ds)
    message = str(caught.value)
    assert "'x'" in message, f"{name} must name the offending column"
    assert name in message, f"{name} must name itself"
    assert "OrdinalEncoder" in message, "the message must name a way forward"


@pytest.mark.parametrize("name", ESTIMATORS)
def test_a_boolean_feature_is_rejected_by_name(name: str) -> None:
    """A flag column is a natural 0/1 feature that no estimator here can actually take."""
    ds = bt.from_pydict(
        {"x": [True, False, True, False], "z": [1.0, 2.0, 3.0, 4.0], "y": [0, 0, 1, 1]}
    )
    with pytest.raises(PlanError) as caught:
        _build(name, ["x", "z"]).fit(ds)
    message = str(caught.value)
    assert "'x'" in message and "boolean" in message
    assert 'cast("int64")' in message, "the message must give the one-line fix"


def test_the_cast_the_boolean_message_recommends_actually_works() -> None:
    """A remedy nobody ran is a remedy that does not work. This runs it."""
    from batcher.ml import LinearRegression

    ds = bt.from_pydict(
        {"x": [True, False, True, False], "z": [1.0, 2.0, 3.0, 4.0], "y": [0.0, 1.0, 2.0, 3.0]}
    )
    fixed = ds.with_columns(x=bt.col("x").cast("int64"))
    model = LinearRegression(["x", "z"], "y").fit(fixed)
    assert model.predict(fixed).count() == 4


@pytest.mark.parametrize("name", ESTIMATORS)
def test_an_all_null_feature_is_rejected_by_name(name: str) -> None:
    ds = bt.from_pydict(
        {"x": [None, None, None, None], "z": [1.0, 2.0, 3.0, 4.0], "y": [0, 0, 1, 1]}
    )
    with pytest.raises(PlanError) as caught:
        _build(name, ["x", "z"]).fit(ds)
    message = str(caught.value)
    assert "'x'" in message and "null" in message
    assert "SimpleImputer" in message, "the message must name the way to fill it"


def test_a_date_feature_is_rejected_by_name() -> None:
    from batcher.ml import LinearRegression

    day = datetime.date(2024, 1, 1)
    ds = bt.from_pydict({"x": [day] * 4, "z": [1.0, 2.0, 3.0, 4.0], "y": [0.0, 1.0, 2.0, 3.0]})
    with pytest.raises(PlanError, match="date32"):
        LinearRegression(["x", "z"], "y").fit(ds)


@pytest.mark.parametrize("values", [[1, 4, 2, 9], [1.0, 4.0, 2.0, 9.0]])
def test_integer_and_float_features_are_accepted(values: list) -> None:
    """The guard must not fire on the types that work, or it breaks every model there is."""
    from batcher.ml import LinearRegression

    ds = bt.from_pydict({"x": values, "z": [1.0, 2.0, 3.0, 4.0], "y": [0.0, 1.0, 2.0, 3.0]})
    assert LinearRegression(["x", "z"], "y").fit(ds).coef_ is not None


def test_a_missing_column_still_reports_missing_rather_than_untyped() -> None:
    """The type check must defer to the existence check, not mask it with a type message."""
    from batcher._internal.errors import ColumnNotFoundError
    from batcher.ml import LinearRegression

    ds = bt.from_pydict({"z": [1.0, 2.0, 3.0], "y": [0.0, 1.0, 2.0]})
    with pytest.raises(ColumnNotFoundError):
        LinearRegression(["nope", "z"], "y").fit(ds)


def test_the_check_reads_the_schema_without_executing_the_plan() -> None:
    """Validation that costs a pass would double the cost of every fit that fails."""
    from batcher.ml._estimator import require_numeric

    ds = bt.from_pydict({"x": ["a"], "z": [1.0]})
    seen: list[str] = []
    original = type(ds).collect

    def spy(self, *args, **kwargs):
        seen.append("collect")
        return original(self, *args, **kwargs)

    type(ds).collect = spy
    try:
        with pytest.raises(PlanError):
            require_numeric("Probe", ds, ["x"])
    finally:
        type(ds).collect = original
    assert seen == [], "the type check must not execute the plan"
