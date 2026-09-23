"""Edge-case contracts for the preprocessors and k-NN models, held against scikit-learn.

Each test here pins a defect an audit found, and each one failed before its fix:

- k-NN ``weights="distance"`` weighted by ``1 / d**2`` instead of ``1 / d``;
- `GroupImputer` / `GroupStatEncoder` lost their fitted state on `save` / `to_dict`;
- a NaN in a numeric column poisoned every fit statistic (``mean_ == nan``), and
  `SimpleImputer` / `MissingIndicator` did not treat NaN as missing;
- the power transforms standardized with the sample std and picked lambda from a 0.1 grid;
- `SplineTransformer` used a clamped knot vector and returned all-zero rows out of range;
- `SimpleImputer(strategy="most_frequent")` broke ties by engine order.

scikit-learn 1.7 is the oracle wherever the two libraries mean the same thing.
"""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml import KNeighborsClassifier, KNeighborsRegressor

pytestmark = pytest.mark.unit

sklearn_neighbors = pytest.importorskip("sklearn.neighbors")


# --------------------------------------------------------------------------------------
# k-NN distance weighting
# --------------------------------------------------------------------------------------


def _knn_data(seed: int = 1) -> tuple[bt.Dataset, bt.Dataset, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train = rng.normal(size=(200, 3))
    query = rng.normal(size=(100, 3))
    # Two queries coincide with training rows, so the d == 0 rule is exercised too.
    query[0] = train[7]
    query[1] = train[42]
    target = train @ np.array([1.0, 2.0, 3.0])
    names = ["a", "b", "c"]
    train_ds = bt.from_pydict(
        {n: train[:, i].tolist() for i, n in enumerate(names)}
        | {"r": target.tolist(), "y": (train[:, 0] + train[:, 1] > 0).astype(int).tolist()}
    )
    query_ds = bt.from_pydict({n: query[:, i].tolist() for i, n in enumerate(names)})
    return train_ds, query_ds, train, query


def test_distance_weighted_regressor_matches_sklearn_off_the_training_set() -> None:
    train_ds, query_ds, train, query = _knn_data()
    got = (
        KNeighborsRegressor(["a", "b", "c"], "r", k=5, weights="distance")
        .fit(train_ds)
        .predict(query_ds)
        .to_pydict()["prediction"]
    )
    theirs = sklearn_neighbors.KNeighborsRegressor(n_neighbors=5, weights="distance")
    want = theirs.fit(train, train @ np.array([1.0, 2.0, 3.0])).predict(query)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def test_distance_weighted_classifier_matches_sklearn_off_the_training_set() -> None:
    train_ds, query_ds, train, query = _knn_data()
    labels = (train[:, 0] + train[:, 1] > 0).astype(int)
    got = (
        KNeighborsClassifier(["a", "b", "c"], "y", k=5, weights="distance")
        .fit(train_ds)
        .predict(query_ds)
        .to_pydict()["prediction"]
    )
    theirs = sklearn_neighbors.KNeighborsClassifier(n_neighbors=5, weights="distance")
    assert got == theirs.fit(train, labels).predict(query).tolist()


def test_an_exact_match_takes_the_whole_distance_weighted_vote() -> None:
    """sklearn: rows at d == 0 share the vote equally, every other neighbour gets zero."""
    train = bt.from_pydict({"x": [1.0, 1.0, 2.0, 5.0], "y": [4.0, 6.0, 100.0, 1000.0]})
    model = KNeighborsRegressor(["x"], "y", k=3, weights="distance").fit(train)
    got = model.predict(bt.from_pydict({"x": [1.0]})).to_pydict()["prediction"]
    want = (
        sklearn_neighbors.KNeighborsRegressor(n_neighbors=3, weights="distance")
        .fit([[1.0], [1.0], [2.0], [5.0]], [4.0, 6.0, 100.0, 1000.0])
        .predict([[1.0]])
    )
    assert got == [pytest.approx(5.0, abs=0.0)]
    assert got[0] == pytest.approx(float(want[0]), abs=0.0)


# --------------------------------------------------------------------------------------
# Persistence: every exported preprocessor round-trips, or refuses at save()
# --------------------------------------------------------------------------------------


def _mixed() -> bt.Dataset:
    """Twelve rows with a column of every kind some preprocessor consumes."""
    import datetime as dt

    base = dt.datetime(2024, 1, 1, 6)
    return bt.from_pydict(
        {
            "id": list(range(12)),
            "a": [1.0, 2.0, 3.5, 4.0, 5.5, 6.0, 7.0, 8.5, 9.0, 10.0, 12.0, 15.0],
            "b": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 10.0, 9.0, 12.0, 11.0],
            "c": ["x", "y", "x", "z", "x", "y", "z", "x", "y", "x", "z", "y"],
            "y": [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
            "s": [0.1, 0.4, 0.8, 0.3, 0.9, 0.7, 0.2, 0.85, 0.35, 0.15, 0.95, 0.25],
            "t": ["a red cat", "a dog", "cat dog", "red dog dog", "cat", "big red cat"] * 2,
            "lst": [["p"], ["q", "p"], [], ["r"], ["p", "r"], ["q"]] * 2,
            "ts": [base + dt.timedelta(hours=13 * i) for i in range(12)],
        }
    )


def _halve(expression):  # a named function, so FunctionTransformer pickles
    return expression / 2.0


def _fit_linear(ds, features):
    from batcher.ml import LinearRegression

    return LinearRegression(list(features), "a").fit(ds)


def _preprocessor_factories() -> dict[str, object]:
    """One suitable construction of every exported preprocessor class, by class name."""
    import batcher.ml as ml
    from batcher.ml import preprocessors as pp

    return {
        "Binarizer": lambda: pp.Binarizer(["a", "b"], threshold=5.0),
        "BinaryEncoder": lambda: pp.BinaryEncoder("c"),
        "BoxCoxTransformer": lambda: pp.BoxCoxTransformer("a"),
        "Chain": lambda: pp.Chain(pp.SimpleImputer("a"), pp.StandardScaler(["a", "b"])),
        "Clipper": lambda: pp.Clipper("a", lower=0.1, upper=0.9),
        "ColumnDropper": lambda: pp.ColumnDropper("t"),
        "ColumnSelector": lambda: pp.ColumnSelector(["id", "a"]),
        "Concatenator": lambda: pp.Concatenator(["a", "b"]),
        "CountVectorizer": lambda: pp.CountVectorizer("t", dense=True),
        "CyclicalEncoder": lambda: pp.CyclicalEncoder("ts"),
        "DateTimeFeaturizer": lambda: pp.DateTimeFeaturizer("ts"),
        "DropCorrelated": lambda: pp.DropCorrelated(["a", "b", "s"], threshold=0.9),
        "FrequencyEncoder": lambda: pp.FrequencyEncoder("c"),
        "FunctionTransformer": lambda: pp.FunctionTransformer("a", _halve),
        "GaussianRandomProjection": lambda: pp.GaussianRandomProjection(["a", "b"], n_components=2),
        "GroupImputer": lambda: pp.GroupImputer("a", by="c"),
        "GroupStatEncoder": lambda: pp.GroupStatEncoder(
            "a", by="c", statistics=["mean", "std", "count"]
        ),
        "HashingEncoder": lambda: pp.HashingEncoder("c", n_buckets=8),
        "HashingVectorizer": lambda: pp.HashingVectorizer("t", n_features=16),
        "InteractionFeatures": lambda: pp.InteractionFeatures(["a", "b"]),
        "IsotonicCalibrator": lambda: pp.IsotonicCalibrator("s", "y"),
        "IterativeImputer": lambda: pp.IterativeImputer(["a", "b"], max_iter=2),
        "JamesSteinEncoder": lambda: pp.JamesSteinEncoder("c", "y"),
        "KBinsDiscretizer": lambda: pp.KBinsDiscretizer("a", n_bins=3),
        "KNNImputer": lambda: ml.KNNImputer(["a", "b"], k=2),
        "LabelBinarizer": lambda: pp.LabelBinarizer("c"),
        "LabelEncoder": lambda: pp.LabelEncoder("c"),
        "LagFeaturizer": lambda: pp.LagFeaturizer("a", order_by="id"),
        "LeaveOneOutEncoder": lambda: pp.LeaveOneOutEncoder("c", "y"),
        "LogTransformer": lambda: pp.LogTransformer("a"),
        "MaxAbsScaler": lambda: pp.MaxAbsScaler(["a", "b"]),
        "MinMaxScaler": lambda: pp.MinMaxScaler(["a", "b"]),
        "MissingIndicator": lambda: pp.MissingIndicator("a"),
        "MultiHotEncoder": lambda: pp.MultiHotEncoder("lst"),
        "MultiLabelBinarizer": lambda: pp.MultiLabelBinarizer("lst"),
        "Normalizer": lambda: pp.Normalizer(["a", "b"]),
        "Nystroem": lambda: pp.Nystroem(["a", "b"], n_components=3, gamma=0.1),
        "OneHotEncoder": lambda: pp.OneHotEncoder("c"),
        "OrdinalEncoder": lambda: pp.OrdinalEncoder("c"),
        "OutlierClipper": lambda: ml.OutlierClipper("a"),
        "PCA": lambda: pp.PCA(["a", "b", "s"], n_components=2),
        "PlattCalibrator": lambda: pp.PlattCalibrator("s", "y"),
        "PolynomialFeatures": lambda: pp.PolynomialFeatures(["a", "b"]),
        "PowerTransformer": lambda: pp.PowerTransformer("a"),
        "QuantileTransformer": lambda: pp.QuantileTransformer("a", n_quantiles=5),
        "RBFSampler": lambda: pp.RBFSampler(["a", "b"], n_components=3),
        "RFE": lambda: pp.RFE(_fit_linear, features=["b", "s"], n_features=1),
        "RankTransformer": lambda: pp.RankTransformer("a"),
        "RareCategoryEncoder": lambda: pp.RareCategoryEncoder("c", min_frequency=0.3),
        "RatioFeatures": lambda: pp.RatioFeatures([("a", "b")]),
        "RobustScaler": lambda: pp.RobustScaler(["a", "b"]),
        "RollingFeaturizer": lambda: pp.RollingFeaturizer("a", order_by="id", window=3),
        "SelectFromModel": lambda: pp.SelectFromModel({"a": 1.0, "b": 0.0}),
        "SelectKBest": lambda: pp.SelectKBest("y", k=1, features=["a", "s"]),
        "SelectPercentile": lambda: pp.SelectPercentile("y", percentile=50.0, features=["a", "s"]),
        "SimpleImputer": lambda: pp.SimpleImputer("a", strategy="median"),
        "SparseRandomProjection": lambda: pp.SparseRandomProjection(["a", "b"], n_components=2),
        "SplineTransformer": lambda: pp.SplineTransformer("a", n_knots=4),
        "StandardScaler": lambda: pp.StandardScaler(["a", "b"]),
        "TargetEncoder": lambda: pp.TargetEncoder("c", "y"),
        "TextStatFeaturizer": lambda: pp.TextStatFeaturizer("t"),
        "TfidfVectorizer": lambda: pp.TfidfVectorizer("t"),
        "Tokenizer": lambda: pp.Tokenizer("t", str.split),
        "TruncatedSVD": lambda: pp.TruncatedSVD(["a", "b", "s"], n_components=2),
        "VarianceThreshold": lambda: pp.VarianceThreshold(["a", "b", "s"], threshold=0.5),
        "WOEEncoder": lambda: pp.WOEEncoder("c", "y"),
    }


#: Classes whose constructor takes a Python callable: they cannot be written as JSON, so
#: `save` must refuse them up front instead of writing a file `load` cannot rebuild.
_UNSAVEABLE = {"FunctionTransformer", "RFE", "Tokenizer"}


def _exported_preprocessor_classes() -> list[str]:
    import batcher.ml as ml
    from batcher.ml.preprocessors import Preprocessor

    return sorted(
        name
        for name in ml.__all__
        if isinstance(getattr(ml, name, None), type)
        and issubclass(getattr(ml, name), Preprocessor)
        and name != "Preprocessor"
    )


def test_the_sweep_covers_every_exported_preprocessor() -> None:
    """A new export must be added to the sweep, or this fails naming it."""
    exported = _exported_preprocessor_classes()
    assert len(exported) > 60
    assert set(exported) == set(_preprocessor_factories())


def _rows(ds: bt.Dataset) -> list[dict]:
    """The transform output as rows in `id` order; a join may return them in any order."""
    table = ds.to_pydict()
    count = len(next(iter(table.values())))
    rows = [{k: v[i] for k, v in table.items()} for i in range(count)]
    return sorted(rows, key=lambda r: r["id"]) if "id" in table else rows


def _same(left: object, right: object) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return left == right or (math.isnan(left) and math.isnan(right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_same(a, b) for a, b in zip(left, right, strict=True))
    return left == right


@pytest.mark.parametrize("name", sorted(_preprocessor_factories()))
def test_every_preprocessor_round_trips_or_refuses_at_save(name: str, tmp_path) -> None:
    from batcher.ml.preprocessors import Preprocessor, from_dict, to_dict

    ds = _mixed()
    fitted = _preprocessor_factories()[name]().fit(ds)
    want = _rows(fitted.transform(ds))
    assert want, f"{name} produced no rows to compare"

    pickled = pickle.loads(pickle.dumps(fitted))
    assert _same(_rows(pickled.transform(ds)), want), f"{name}: pickle changed the output"

    path = str(tmp_path / f"{name}.json")
    if name in _UNSAVEABLE:
        with pytest.raises(PlanError, match=f"{name} cannot be saved"):
            fitted.save(path)
        with pytest.raises(PlanError, match=f"{name} cannot be saved"):
            to_dict(fitted)
        return

    fitted.save(path)
    loaded = Preprocessor.load(path)
    assert type(loaded) is type(fitted)
    assert loaded.is_fitted
    assert _same(_rows(loaded.transform(ds)), want), f"{name}: save/load changed the output"
    rebuilt = from_dict(to_dict(fitted))
    assert _same(_rows(rebuilt.transform(ds)), want), f"{name}: to_dict changed the output"


def test_group_encoders_keep_their_learned_table_through_save(tmp_path) -> None:
    """The audit's repro: the reloaded encoder used to fail with a bare AssertionError."""
    from batcher.ml.preprocessors import GroupImputer, GroupStatEncoder, Preprocessor, to_dict

    train = bt.from_pydict({"g": ["a", "a", "b"], "v": [2.0, 4.0, None]})
    serve = bt.from_pydict({"g": ["a", "b", "c"], "v": [None, None, None]})
    imputer = GroupImputer("v", by="g").fit(train)
    encoder = GroupStatEncoder("v", by="g").fit(train)
    assert to_dict(imputer)["state"] != {}
    assert to_dict(encoder)["state"] != {}
    for fitted in (imputer, encoder):
        path = str(tmp_path / f"{type(fitted).__name__}.json")
        fitted.save(path)
        again = Preprocessor.load(path)
        want = fitted.transform(serve).sort("g").to_pydict()
        assert again.transform(serve).sort("g").to_pydict() == want
    assert imputer.transform(serve).sort("g").to_pydict()["v"] == [3.0, 3.0, 3.0]


# --------------------------------------------------------------------------------------
# NaN is missing: skipped by every fit statistic, flagged and filled like a null
# --------------------------------------------------------------------------------------

sklearn_preprocessing = pytest.importorskip("sklearn.preprocessing")
sklearn_impute = pytest.importorskip("sklearn.impute")

_WITH_NAN = [1.0, float("nan"), 3.0, None, 10.0, 4.0, float("nan"), 7.0]


def _nan_column() -> tuple[bt.Dataset, np.ndarray]:
    array = np.array([np.nan if v is None else v for v in _WITH_NAN], dtype=float)[:, None]
    return bt.from_pydict({"a": _WITH_NAN}), array


@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        ("StandardScaler", lambda: sklearn_preprocessing.StandardScaler()),
        ("MinMaxScaler", lambda: sklearn_preprocessing.MinMaxScaler()),
        ("MaxAbsScaler", lambda: sklearn_preprocessing.MaxAbsScaler()),
        ("RobustScaler", lambda: sklearn_preprocessing.RobustScaler()),
        ("PowerTransformer", lambda: sklearn_preprocessing.PowerTransformer()),
        ("BoxCoxTransformer", lambda: sklearn_preprocessing.PowerTransformer(method="box-cox")),
    ],
)
def test_nan_is_skipped_by_the_fit_like_sklearn(ours: str, theirs) -> None:
    from batcher.ml import preprocessors as pp

    ds, array = _nan_column()
    got = np.array(getattr(pp, ours)("a").fit_transform(ds).to_pydict()["a"], dtype=float)
    want = theirs().fit_transform(array)[:, 0]
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3, equal_nan=True)


def test_the_scaler_audit_repro_no_longer_learns_nan() -> None:
    from batcher.ml.preprocessors import StandardScaler

    fitted = StandardScaler("a").fit(bt.from_pydict({"a": [1.0, float("nan"), 3.0]}))
    assert fitted.mean_ == {"a": 2.0}
    assert fitted.scale_ == {"a": 1.0}


def test_quantile_based_fits_ignore_nan() -> None:
    from batcher.ml.preprocessors import Clipper, KBinsDiscretizer, QuantileTransformer

    clean = bt.from_pydict({"a": [v for v in _WITH_NAN if v is not None and v == v]})
    dirty = bt.from_pydict({"a": _WITH_NAN})
    for make in (
        lambda: QuantileTransformer("a", n_quantiles=4),
        lambda: KBinsDiscretizer("a", n_bins=3, strategy="uniform"),
        lambda: Clipper("a", lower=0.1, upper=0.9),
    ):
        from batcher.ml.preprocessors import to_dict

        assert to_dict(make().fit(dirty))["state"] == to_dict(make().fit(clean))["state"]


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_simple_imputer_fills_nan_and_null_like_sklearn(strategy: str) -> None:
    from batcher.ml.preprocessors import SimpleImputer

    values = [2.0, float("nan"), 1.0, None, 1.0, 2.0, 5.0, float("nan")]
    array = np.array([np.nan if v is None else v for v in values], dtype=float)[:, None]
    got = SimpleImputer("a", strategy=strategy).fit_transform(bt.from_pydict({"a": values}))
    want = sklearn_impute.SimpleImputer(strategy=strategy).fit_transform(array)[:, 0]
    np.testing.assert_allclose(got.to_pydict()["a"], want)


def test_missing_indicator_flags_nan_as_well_as_null() -> None:
    from batcher.ml.preprocessors import MissingIndicator

    ds = bt.from_pydict({"a": [1.0, float("nan"), None], "k": [1, None, 3]})
    out = MissingIndicator(["a", "k"]).fit_transform(ds).to_pydict()
    assert out["a_missing"] == [False, True, True]
    assert out["k_missing"] == [False, True, False]


def test_group_imputer_treats_nan_as_missing() -> None:
    from batcher.ml.preprocessors import GroupImputer

    ds = bt.from_pydict({"g": ["a", "a", "a", "b"], "v": [2.0, float("nan"), 4.0, None]})
    out = GroupImputer("v", by="g").fit_transform(ds).sort("g", "v").to_pydict()["v"]
    assert out == [2.0, 3.0, 4.0, 3.0]


# --------------------------------------------------------------------------------------
# Power transforms: population std, refined lambda
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["yeo-johnson", "box-cox"])
@pytest.mark.parametrize("seed", [0, 1])
def test_power_transforms_match_sklearn_lambda_and_output(method: str, seed: int) -> None:
    from batcher.ml.preprocessors import BoxCoxTransformer, PowerTransformer

    rng = np.random.default_rng(seed)
    values = rng.lognormal(size=300) if method == "box-cox" else rng.normal(1.0, 3.0, 300)
    ours = (PowerTransformer if method == "yeo-johnson" else BoxCoxTransformer)("a")
    ds = bt.from_pydict({"a": values.tolist()})
    fitted = ours.fit(ds)
    theirs = sklearn_preprocessing.PowerTransformer(method=method).fit(values[:, None])
    assert fitted.lambdas_["a"] == pytest.approx(theirs.lambdas_[0], abs=1e-3)
    np.testing.assert_allclose(
        fitted.transform(ds).to_pydict()["a"],
        theirs.transform(values[:, None])[:, 0],
        atol=1e-3,
    )


def test_the_power_audit_repro_matches_sklearn() -> None:
    """Four rows, where ddof=1 against ddof=0 moved every output by 15%."""
    from batcher.ml.preprocessors import PowerTransformer

    values = np.array([1.0, 2.0, 3.0, 10.0])
    fitted = PowerTransformer("a").fit(bt.from_pydict({"a": values.tolist()}))
    theirs = sklearn_preprocessing.PowerTransformer().fit(values[:, None])
    assert fitted.lambdas_["a"] == pytest.approx(-0.6916, abs=1e-3)
    got = fitted.transform(bt.from_pydict({"a": values.tolist()})).to_pydict()["a"]
    np.testing.assert_allclose(got, theirs.transform(values[:, None])[:, 0], atol=1e-3)


# --------------------------------------------------------------------------------------
# SplineTransformer: sklearn's default basis, constant extrapolation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("knots", ["uniform", "quantile"])
@pytest.mark.parametrize(("n_knots", "degree"), [(5, 3), (4, 1), (6, 2)])
def test_spline_basis_matches_sklearn_including_out_of_range(
    knots: str, n_knots: int, degree: int
) -> None:
    from batcher.ml.preprocessors import SplineTransformer

    rng = np.random.default_rng(0)
    train = rng.normal(size=200)
    query = np.r_[train[:40], [-10.0, 10.0, train.min(), train.max()]]
    fitted = SplineTransformer("a", n_knots=n_knots, degree=degree, knots=knots).fit(
        bt.from_pydict({"a": train.tolist()})
    )
    out = fitted.transform(bt.from_pydict({"a": query.tolist()})).to_pydict()
    got = np.column_stack([out[f"a_sp{i}"] for i in range(n_knots + degree - 1)])
    theirs = sklearn_preprocessing.SplineTransformer(n_knots=n_knots, degree=degree, knots=knots)
    want = theirs.fit(train[:, None]).transform(query[:, None])
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_spline_default_is_sklearns_default() -> None:
    from batcher.ml.preprocessors import SplineTransformer

    assert SplineTransformer("a").knots == sklearn_preprocessing.SplineTransformer().knots


# --------------------------------------------------------------------------------------
# SimpleImputer most_frequent: ties go to the smallest value
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values",
    [[2.0, None, 1.0, 1.0, 2.0], [3.0, 3.0, 9.0, 9.0, 5.0, 5.0], [7, 7, 1, 1, 4]],
)
def test_most_frequent_breaks_ties_by_the_smallest_value(values: list) -> None:
    from batcher.ml.preprocessors import SimpleImputer

    fitted = SimpleImputer("a", strategy="most_frequent").fit(bt.from_pydict({"a": values}))
    array = np.array([np.nan if v is None else v for v in values], dtype=float)[:, None]
    want = sklearn_impute.SimpleImputer(strategy="most_frequent").fit(array).statistics_[0]
    assert fitted.statistics_["a"] == want


def test_most_frequent_tie_break_holds_for_strings() -> None:
    from batcher.ml.preprocessors import SimpleImputer

    ds = bt.from_pydict({"c": ["pear", "apple", "pear", "apple", None, "fig"]})
    assert SimpleImputer("c", strategy="most_frequent").fit(ds).statistics_["c"] == "apple"
