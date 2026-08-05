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


def test_an_untyped_column_is_not_rejected_because_empty_data_looks_the_same() -> None:
    """An all-null column and an *empty* dataset are the same thing to a schema.

    Rejecting the untyped case broke fitting on a frame filtered down to nothing, which is a
    legitimate input and, in a distributed job, an ordinary empty partition. The check defers
    to the row-count guards rather than guessing, so this must reach one of those instead.
    """
    from batcher.ml import StandardScaler

    empty = bt.from_pydict({"x": [], "z": []})
    scaler = StandardScaler(["x"])
    try:
        scaler.fit(empty)
    except PlanError as exc:  # a row-count message is fine; a type message is not
        assert "null" not in str(exc), f"the type check must not fire on empty data: {exc}"


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


#: The estimators that predict a number, and so need a numeric target as well as features.
REGRESSORS = [
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "PoissonRegressor",
    "GammaRegressor",
    "TweedieRegressor",
]


@pytest.mark.parametrize("name", REGRESSORS)
def test_a_string_target_is_rejected_by_name(name: str) -> None:
    """A regressor cannot average a string any more than it can a string feature."""
    import batcher.ml as ml

    ds = bt.from_pydict({"x": [1.0, 4.0, 2.0, 9.0], "y": ["a", "b", "c", "d"]})
    with pytest.raises(PlanError) as caught:
        getattr(ml, name)(["x"], "y").fit(ds)
    message = str(caught.value)
    assert "'y'" in message, "the message must name the target column"
    assert "target" in message, "the message must say it is the target, not a feature"


@pytest.mark.parametrize("name", ["GaussianNB", "RidgeClassifier", "NearestCentroid"])
def test_a_classifier_still_accepts_a_string_target(name: str) -> None:
    """A class label is legitimately a string, so the target check must not reach classifiers."""
    import batcher.ml as ml

    ds = bt.from_pydict({"x": [0.0, 1.0, 9.0, 10.0], "y": ["a", "a", "b", "b"]})
    assert sorted(getattr(ml, name)(["x"], "y").fit(ds).classes_) == ["a", "b"]


def test_a_numeric_target_is_still_accepted() -> None:
    from batcher.ml import LinearRegression

    ds = bt.from_pydict({"x": [1.0, 4.0, 2.0, 9.0], "y": [1.0, 2.0, 3.0, 4.0]})
    assert LinearRegression(["x"], "y").fit(ds).coef_ is not None


#: Every preprocessor whose transform is arithmetic, so a string column cannot work.
NUMERIC_PREPROCESSORS = [
    "Binarizer",
    "BoxCoxTransformer",
    "Clipper",
    "GaussianRandomProjection",
    "KBinsDiscretizer",
    "LogTransformer",
    "MaxAbsScaler",
    "MinMaxScaler",
    "Normalizer",
    "Nystroem",
    "PolynomialFeatures",
    "PowerTransformer",
    "QuantileTransformer",
    "RBFSampler",
    "RankTransformer",
    "RobustScaler",
    "SparseRandomProjection",
    "StandardScaler",
    "VarianceThreshold",
]


@pytest.mark.parametrize("name", NUMERIC_PREPROCESSORS)
def test_a_numeric_preprocessor_names_a_string_column(name: str) -> None:
    """Twenty of these failed from inside the engine, in ten different ways."""
    from batcher.ml import preprocessors as pp

    ds = bt.from_pydict({"x": ["a", "b", "c", "d"], "z": [1.0, 2.0, 3.0, 4.0]})
    pre = getattr(pp, name)(["x"])
    with pytest.raises(PlanError) as caught:
        pre.fit(ds).transform(ds).collect()
    message = str(caught.value)
    assert "'x'" in message, f"{name} must name the column"
    assert name in message, f"{name} must name itself"


@pytest.mark.parametrize("name", NUMERIC_PREPROCESSORS)
def test_a_numeric_preprocessor_still_accepts_numbers(name: str) -> None:
    """The guard must not fire on the input these are for."""
    from batcher.ml import preprocessors as pp

    ds = bt.from_pydict({"z": [1.0, 4.0, 2.0, 9.0]})
    assert getattr(pp, name)(["z"]).fit(ds).transform(ds).count() == 4


@pytest.mark.parametrize("strategy", ["most_frequent", "constant"])
def test_the_imputer_strategies_that_exist_for_strings_still_take_them(strategy: str) -> None:
    """`numeric_only` is a property of the strategy here, not of the class.

    Filling with the most frequent value is exactly how a categorical column is imputed, so a
    blanket rule on `SimpleImputer` would have broken the one imputation strings need.
    """
    from batcher.ml.preprocessors import SimpleImputer

    ds = bt.from_pydict({"x": ["a", None, "c", "a"]})
    kwargs = {"fill_value": "z"} if strategy == "constant" else {}
    filled = SimpleImputer(["x"], strategy=strategy, **kwargs).fit(ds).transform(ds)
    assert None not in filled.to_pydict()["x"]


@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_the_arithmetic_imputer_strategies_reject_a_string_column(strategy: str) -> None:
    from batcher.ml.preprocessors import SimpleImputer

    ds = bt.from_pydict({"x": ["a", None, "c", "a"]})
    with pytest.raises(PlanError, match="'x'"):
        SimpleImputer(["x"], strategy=strategy).fit(ds)


def test_an_encoder_still_takes_a_string_column() -> None:
    """The encoders are the reason `numeric_only` is opt-in rather than the default."""
    from batcher.ml.preprocessors import OrdinalEncoder

    ds = bt.from_pydict({"x": ["a", "b", "a", "c"]})
    assert OrdinalEncoder(["x"]).fit(ds).transform(ds).to_pydict()["x"] == [0, 1, 0, 2]


#: The classes a first pass over this surface missed, found by re-sweeping rather than by
#: reading: decomposition, the distance-based models, and the two clipping/imputing helpers.
LATER_ADDITIONS = [
    "PCA",
    "TruncatedSVD",
    "OutlierClipper",
    "IterativeImputer",
    "KNNImputer",
    "InteractionFeatures",
]


@pytest.mark.parametrize("name", LATER_ADDITIONS)
def test_the_rest_of_the_surface_names_a_string_column(name: str) -> None:
    """A validation covering most of a surface reads as done and is not.

    These eight were still failing from inside the engine after the first pass, and only a
    re-sweep over every exported class with a `fit` found them.
    """
    import batcher.ml as ml

    ds = bt.from_pydict({"x": ["a", "b", "c", "d"], "z": [1.0, 2.0, 3.0, 4.0]})
    obj = getattr(ml, name)(["x", "z"])
    with pytest.raises(PlanError) as caught:
        obj.fit(ds).transform(ds).collect()
    assert "'x'" in str(caught.value), f"{name} must name the column"


@pytest.mark.parametrize("name", ["KNeighborsClassifier", "KNeighborsRegressor"])
def test_the_neighbour_models_name_a_string_feature(name: str) -> None:
    import batcher.ml as ml

    ds = bt.from_pydict({"x": ["a", "b", "c", "d"], "z": [1.0, 2.0, 3.0, 4.0], "y": [0, 0, 1, 1]})
    with pytest.raises(PlanError) as caught:
        getattr(ml, name)(["x", "z"], "y").fit(ds)
    assert "'x'" in str(caught.value)


@pytest.mark.parametrize("name", LATER_ADDITIONS)
def test_the_rest_of_the_surface_still_takes_numbers(name: str) -> None:
    import batcher.ml as ml

    ds = bt.from_pydict({"a": [1.0, 4.0, 2.0, 9.0], "b": [2.0, 1.0, 5.0, 3.0]})
    assert getattr(ml, name)(["a", "b"]).fit(ds).transform(ds).count() == 4


@pytest.mark.parametrize("name", ["DummyClassifier", "DummyRegressor"])
def test_a_baseline_rejects_a_feature_list_where_a_target_belongs(name: str) -> None:
    """It took the list, then raised ``TypeError: unhashable type: 'list'`` from inside fit.

    A baseline has no features, so writing it like every other estimator is the natural
    mistake; the message it produced named neither the argument nor the shape expected.
    """
    import batcher.ml as ml

    with pytest.raises(PlanError, match="one target column"):
        getattr(ml, name)(["x", "z"])


@pytest.mark.parametrize("name", ["DummyClassifier", "DummyRegressor"])
def test_a_baseline_still_takes_a_target_name(name: str) -> None:
    import batcher.ml as ml

    ds = bt.from_pydict({"y": [1.0, 2.0, 3.0, 4.0]})
    assert getattr(ml, name)("y").fit(ds).predict(ds).count() == 4


def test_the_rare_category_encoder_names_a_numeric_column() -> None:
    """The mirror image: its replacement is a string, so it is *numbers* it cannot rewrite.

    The engine's message for this was `arguments need to have the same data type`, from a
    `case` whose branches disagreed. It named neither column nor either type.
    """
    from batcher.ml import RareCategoryEncoder

    ds = bt.from_pydict({"x": ["a", "a", "a", "b"], "z": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(PlanError) as caught:
        RareCategoryEncoder(["x", "z"]).fit(ds)
    message = str(caught.value)
    assert "'z'" in message, "it must name the numeric column, not the string one"
    assert "KBinsDiscretizer" in message, "it must offer a way to use a numeric column"


def test_the_rare_category_encoder_still_takes_strings() -> None:
    from batcher.ml import RareCategoryEncoder

    ds = bt.from_pydict({"x": ["a", "a", "a", "b"]})
    out = RareCategoryEncoder(["x"], min_frequency=0.5).fit(ds).transform(ds).to_pydict()
    assert out["x"] == ["a", "a", "a", "__rare__"]
