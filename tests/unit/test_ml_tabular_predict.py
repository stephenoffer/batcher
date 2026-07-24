"""Tabular batch inference — `ds.ml.predict` over XGBoost / LightGBM / scikit-learn.

The contract this pins: a fitted tabular model scores a `Dataset` with the model loaded
once per worker, the features assembled from Arrow columns in a fixed order, nulls
becoming the framework's missing sentinel, and the output schema known *before* the query
runs. The failure modes that motivated each guard are the silent ones — a re-ordered
feature list produces confident nonsense, and an undeclared output column is invisible to
every operator above the UDF.

scikit-learn is a dev dependency so those paths always run; XGBoost and LightGBM are
optional and their tests skip when absent.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.tabular import (
    detect_framework,
    feature_matrix,
    prediction_columns,
    resolve_features,
    tabular_predictor,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def sample() -> tuple[np.ndarray, np.ndarray]:
    """A small deterministic binary-classification problem."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 3))
    labels = (features[:, 0] + 0.5 * features[:, 1] > 0).astype(int)
    return features, labels


@pytest.fixture(scope="module")
def dataset(sample: tuple[np.ndarray, np.ndarray]) -> bt.Dataset:
    """The same problem as a `Dataset` with columns ``a``, ``b``, ``c``."""
    features, _ = sample
    return bt.from_pydict(
        {
            "a": features[:, 0].tolist(),
            "b": features[:, 1].tolist(),
            "c": features[:, 2].tolist(),
        }
    )


def _logistic(sample: tuple[np.ndarray, np.ndarray]):
    from sklearn.linear_model import LogisticRegression

    features, labels = sample
    return LogisticRegression().fit(features, labels)


# --- the feature matrix ------------------------------------------------------------


def test_feature_matrix_orders_columns_as_asked() -> None:
    batch = pa.RecordBatch.from_pydict({"a": [1.0, 2.0], "b": [10.0, 20.0]})
    assert feature_matrix(batch, ["b", "a"]).tolist() == [[10.0, 1.0], [20.0, 2.0]]


def test_feature_matrix_nulls_become_nan_by_default() -> None:
    batch = pa.RecordBatch.from_pydict({"a": [1.0, None]})
    assert np.isnan(feature_matrix(batch, ["a"])[1, 0])


def test_feature_matrix_missing_sentinel_is_honored() -> None:
    batch = pa.RecordBatch.from_pydict({"a": [1.0, None]})
    assert feature_matrix(batch, ["a"], missing=-999.0)[1, 0] == -999.0


def test_feature_matrix_casts_integers_and_booleans() -> None:
    batch = pa.RecordBatch.from_pydict({"i": [1, 2], "f": [True, False]})
    assert feature_matrix(batch, ["i", "f"]).tolist() == [[1.0, 1.0], [2.0, 0.0]]


def test_feature_matrix_rejects_a_string_column() -> None:
    batch = pa.RecordBatch.from_pydict({"s": ["x"]})
    with pytest.raises(PlanError, match="cannot score"):
        feature_matrix(batch, ["s"])


def test_feature_matrix_rejects_an_unknown_dtype() -> None:
    batch = pa.RecordBatch.from_pydict({"a": [1.0]})
    with pytest.raises(PlanError, match="dtype"):
        feature_matrix(batch, ["a"], dtype="float16")


def test_resolve_features_rejects_a_duplicate() -> None:
    with pytest.raises(PlanError, match="twice"):
        resolve_features(["a", "a"], ["a", "b"])


def test_resolve_features_rejects_an_unknown_column() -> None:
    with pytest.raises(ColumnNotFoundError):
        resolve_features(["z"], ["a", "b"])


def test_resolve_features_defaults_to_every_column_in_order() -> None:
    assert resolve_features(None, ["a", "b"]) == ["a", "b"]


# --- output shaping ----------------------------------------------------------------


def test_prediction_columns_single_output_keeps_the_base_name() -> None:
    cols = prediction_columns([1.0, 2.0], output_column="p")
    assert list(cols) == ["p"]


def test_prediction_columns_multi_output_is_suffixed() -> None:
    cols = prediction_columns([[0.1, 0.9]], output_column="p")
    assert list(cols) == ["p_0", "p_1"]


def test_prediction_columns_as_list_collapses_to_one_column() -> None:
    cols = prediction_columns([[0.1, 0.9]], output_column="p", as_list=True)
    assert cols["p"].to_pylist() == [[0.1, 0.9]]


def test_prediction_columns_rejects_a_wrong_width_name_list() -> None:
    with pytest.raises(PlanError, match="output_columns"):
        prediction_columns([[0.1, 0.9]], output_column="p", output_columns=["only_one"])


def test_prediction_columns_flattens_a_3d_output() -> None:
    values = np.arange(8, dtype="float64").reshape(2, 2, 2)
    cols = prediction_columns(values, output_column="p")
    assert list(cols) == ["p_0", "p_1", "p_2", "p_3"]


# --- scikit-learn ------------------------------------------------------------------


def test_sklearn_regressor_matches_the_estimator(dataset: bt.Dataset) -> None:
    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit([[0.0], [1.0], [2.0]], [0.0, 2.0, 4.0])
    ds = bt.from_pydict({"x": [3.0, 4.0]})
    got = ds.ml.predict(model, features=["x"]).to_pydict()["prediction"]
    assert got == pytest.approx(model.predict([[3.0], [4.0]]).tolist())


def test_sklearn_classifier_predict_matches(sample, dataset: bt.Dataset) -> None:
    model = _logistic(sample)
    got = dataset.ml.predict(model, features=["a", "b", "c"]).to_pydict()["prediction"]
    assert got == model.predict(sample[0]).tolist()


def test_sklearn_predict_proba_becomes_one_column_per_class(sample, dataset) -> None:
    model = _logistic(sample)
    out = dataset.ml.predict(model, features=["a", "b", "c"], method="predict_proba").to_pydict()
    assert "prediction_0" in out and "prediction_1" in out
    assert out["prediction_1"] == pytest.approx(model.predict_proba(sample[0])[:, 1].tolist())


def test_sklearn_raw_uses_the_decision_function(sample, dataset) -> None:
    model = _logistic(sample)
    out = dataset.ml.predict(model, features=["a", "b", "c"], method="raw").to_pydict()
    assert out["prediction"] == pytest.approx(model.decision_function(sample[0]).tolist())


def test_float32_is_available_when_speed_beats_the_last_digits(sample, dataset) -> None:
    model = _logistic(sample)
    out = dataset.ml.predict(
        model, features=["a", "b", "c"], method="raw", dtype="float32"
    ).to_pydict()
    assert out["prediction"] == pytest.approx(model.decision_function(sample[0]).tolist(), rel=1e-5)


def test_as_list_gives_one_list_column(sample, dataset) -> None:
    model = _logistic(sample)
    out = dataset.ml.predict(
        model, features=["a", "b", "c"], method="predict_proba", as_list=True
    ).to_pydict()
    assert len(out["prediction"][0]) == 2


def test_predict_proba_on_a_regressor_is_an_actionable_error() -> None:
    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit([[0.0], [1.0]], [0.0, 1.0])
    ds = bt.from_pydict({"x": [1.0]})
    with pytest.raises(PlanError, match="predict_proba"):
        ds.ml.predict(model, features=["x"], method="predict_proba").to_pydict()


def test_a_pipeline_is_scored_end_to_end(sample, dataset) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features, labels = sample
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(features, labels)
    got = dataset.ml.predict(model, features=["a", "b", "c"]).to_pydict()["prediction"]
    assert got == model.predict(features).tolist()


# --- schema, nulls, empties --------------------------------------------------------


def test_the_prediction_column_is_visible_to_a_later_operator(sample, dataset) -> None:
    # An appended column the plan does not know about is invisible above the UDF, so a
    # downstream filter on it would fail to resolve. This is that regression.
    model = _logistic(sample)
    scored = dataset.ml.predict(model, features=["a", "b", "c"])
    assert scored.filter(bt.col("prediction") == 1).count() > 0


def test_an_empty_batch_keeps_the_output_schema(sample) -> None:
    model = _logistic(sample)
    ds = bt.from_pydict({"a": [1.0], "b": [1.0], "c": [1.0]}).filter(bt.col("a") > 100)
    out = ds.ml.predict(model, features=["a", "b", "c"]).to_pydict()
    assert out["prediction"] == []


def test_nulls_are_filled_with_the_declared_sentinel(sample) -> None:
    model = _logistic(sample)
    ds = bt.from_pydict({"a": [None], "b": [0.0], "c": [0.0]})
    out = ds.ml.predict(model, features=["a", "b", "c"], missing=0.0).to_pydict()
    assert out["prediction"] == model.predict([[0.0, 0.0, 0.0]]).tolist()


# --- guards ------------------------------------------------------------------------


def test_a_reordered_feature_list_raises_at_plan_time(sample) -> None:
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    features, labels = sample
    frame = pd.DataFrame(features, columns=["a", "b", "c"])
    model = LogisticRegression().fit(frame, labels)
    ds = bt.from_pydict({"a": [0.0], "b": [0.0], "c": [0.0]})
    with pytest.raises(PlanError, match="re-ordering"):
        ds.ml.predict(model, features=["b", "a", "c"])


def test_a_short_feature_list_raises_at_plan_time(sample) -> None:
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    features, labels = sample
    frame = pd.DataFrame(features, columns=["a", "b", "c"])
    model = LogisticRegression().fit(frame, labels)
    ds = bt.from_pydict({"a": [0.0], "b": [0.0], "c": [0.0]})
    with pytest.raises(PlanError, match="expects 3 features"):
        ds.ml.predict(model, features=["a", "b"])


def test_an_unsupported_method_names_the_supported_ones(sample, dataset) -> None:
    model = _logistic(sample)
    with pytest.raises(PlanError, match="method="):
        dataset.ml.predict(model, features=["a", "b", "c"], method="leaf")


def test_an_unknown_framework_is_rejected(sample, dataset) -> None:
    model = _logistic(sample)
    with pytest.raises(PlanError, match="unknown framework"):
        dataset.ml.predict(model, features=["a", "b", "c"], framework="pytorch")


def test_detect_framework_recognizes_a_duck_typed_estimator() -> None:
    class Estimator:
        def fit(self, x, y):  # pragma: no cover - never called
            return self

        def predict(self, x):
            return [0.0] * len(x)

    assert detect_framework(Estimator()) == "sklearn"


def test_detect_framework_rejects_an_object_with_no_predict() -> None:
    with pytest.raises(PlanError, match="framework="):
        detect_framework(object())


def test_the_predictor_class_is_memoized(sample) -> None:
    # The distributed warm pool keys on the UDF's identity, so two identical calls must
    # hand the engine the same class or the model reloads on every collect().
    model = _logistic(sample)
    assert tabular_predictor(model, ("a", "b", "c")) is tabular_predictor(model, ("a", "b", "c"))


# --- XGBoost -----------------------------------------------------------------------

try:
    import xgboost
except ImportError:  # pragma: no cover - XGBoost is an optional extra
    xgboost = None

# A module-level `importorskip` would skip this whole file, scikit-learn tests included,
# so the optional frameworks are gated per test instead.
needs_xgboost = pytest.mark.skipif(xgboost is None, reason="XGBoost is an optional extra")


@needs_xgboost
def test_xgboost_sklearn_wrapper_matches(sample, dataset) -> None:
    features, labels = sample
    model = xgboost.XGBClassifier(n_estimators=10, max_depth=3).fit(features, labels)
    got = dataset.ml.predict(model, features=["a", "b", "c"]).to_pydict()["prediction"]
    assert got == model.predict(features).tolist()


@needs_xgboost
def test_xgboost_booster_predict_matches(sample, dataset) -> None:
    features, labels = sample
    booster = xgboost.XGBClassifier(n_estimators=10).fit(features, labels).get_booster()
    got = dataset.ml.predict(booster, features=["a", "b", "c"]).to_pydict()["prediction"]
    expected = booster.predict(xgboost.DMatrix(features)).tolist()
    assert got == pytest.approx(expected, rel=1e-6)


@needs_xgboost
def test_xgboost_raw_margin_matches(sample, dataset) -> None:
    features, labels = sample
    booster = xgboost.XGBClassifier(n_estimators=10).fit(features, labels).get_booster()
    got = dataset.ml.predict(booster, features=["a", "b", "c"], method="raw").to_pydict()
    expected = booster.predict(xgboost.DMatrix(features), output_margin=True).tolist()
    assert got["prediction"] == pytest.approx(expected, rel=1e-6)


@needs_xgboost
def test_xgboost_contrib_is_one_column_per_feature_plus_bias(sample, dataset) -> None:
    features, labels = sample
    booster = xgboost.XGBClassifier(n_estimators=5).fit(features, labels).get_booster()
    out = dataset.ml.predict(booster, features=["a", "b", "c"], method="contrib").to_pydict()
    assert [c for c in out if c.startswith("prediction_")] == [f"prediction_{i}" for i in range(4)]


@needs_xgboost
def test_xgboost_multiclass_widens_to_one_column_per_class(sample, dataset) -> None:
    features, _ = sample
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 3, size=len(features))
    booster = xgboost.XGBClassifier(n_estimators=5).fit(features, labels).get_booster()
    out = dataset.ml.predict(booster, features=["a", "b", "c"]).to_pydict()
    appended = [c for c in out if c.startswith("prediction_")]
    assert appended == ["prediction_0", "prediction_1", "prediction_2"]


@needs_xgboost
def test_xgboost_loads_from_a_saved_path(sample, dataset, tmp_path) -> None:
    features, labels = sample
    booster = xgboost.XGBClassifier(n_estimators=5).fit(features, labels).get_booster()
    path = str(tmp_path / "model.ubj")
    booster.save_model(path)
    got = dataset.ml.predict(path, features=["a", "b", "c"]).to_pydict()["prediction"]
    assert got == pytest.approx(booster.predict(xgboost.DMatrix(features)).tolist(), rel=1e-6)


@needs_xgboost
def test_xgboost_iteration_range_truncates_the_ensemble(sample, dataset) -> None:
    features, labels = sample
    booster = xgboost.XGBClassifier(n_estimators=20).fit(features, labels).get_booster()
    out = dataset.ml.predict(
        booster, features=["a", "b", "c"], options={"iteration_range": (0, 5)}
    ).to_pydict()
    expected = booster.predict(xgboost.DMatrix(features), iteration_range=(0, 5)).tolist()
    assert out["prediction"] == pytest.approx(expected, rel=1e-6)


@needs_xgboost
def test_predict_proba_on_a_bare_booster_is_an_actionable_error(sample, dataset) -> None:
    features, labels = sample
    booster = xgboost.XGBClassifier(n_estimators=5).fit(features, labels).get_booster()
    with pytest.raises(PlanError, match="XGBClassifier"):
        dataset.ml.predict(
            booster, features=["a", "b", "c"], method="predict_proba", output_columns=["p"]
        ).to_pydict()


def test_sklearn_defaults_to_float64_so_predictions_match_exactly(sample, dataset) -> None:
    # float32 features shift the last digits of a float64 estimator's output. The boosters
    # compute in float32 and do not care; scikit-learn does, so it gets float64 by default.
    model = _logistic(sample)
    got = dataset.ml.predict(model, features=["a", "b", "c"], method="raw").to_pydict()
    assert got["prediction"] == model.decision_function(sample[0]).tolist()


def test_an_all_null_feature_column_is_treated_as_missing(sample) -> None:
    # A batch where one feature is entirely null types as Arrow `null`, which is not a
    # numeric type but is also not a modelling error.
    model = _logistic(sample)
    ds = bt.from_pydict({"a": [None, None], "b": [0.0, 1.0], "c": [1.0, 0.0]})
    out = ds.ml.predict(model, features=["a", "b", "c"], missing=0.0).to_pydict()
    assert out["prediction"] == model.predict([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).tolist()


# --- LightGBM ----------------------------------------------------------------------


def test_lightgbm_sklearn_wrapper_matches(sample, dataset) -> None:
    lgb = pytest.importorskip("lightgbm", reason="LightGBM is an optional extra")
    features, labels = sample
    model = lgb.LGBMClassifier(n_estimators=10, verbose=-1).fit(features, labels)
    got = dataset.ml.predict(model, features=["a", "b", "c"]).to_pydict()["prediction"]
    assert got == model.predict(features).tolist()


def test_lightgbm_booster_predict_matches(sample, dataset) -> None:
    lgb = pytest.importorskip("lightgbm", reason="LightGBM is an optional extra")
    features, labels = sample
    booster = lgb.LGBMClassifier(n_estimators=10, verbose=-1).fit(features, labels).booster_
    got = dataset.ml.predict(booster, features=["a", "b", "c"]).to_pydict()["prediction"]
    assert got == pytest.approx(booster.predict(features).tolist(), rel=1e-9)
