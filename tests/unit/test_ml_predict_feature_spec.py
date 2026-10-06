"""`ds.ml.predict(features=FeatureSpec, on_null=...)` and the class order of probabilities.

AP-381/414: a model trained on a bare array records no feature names, so nothing could
catch a reordered frame. A `FeatureSpec` pins the order (and dtypes) once. AP-412: the
``prediction_i`` columns of ``predict_proba`` follow ``model.classes_``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import DataQualityError, PlanError
from batcher.ml import FeatureSpec

pytestmark = pytest.mark.unit

sklearn_linear = pytest.importorskip("sklearn.linear_model")


def _model():
    # Trained on an ndarray: no feature_names_in_, so only the spec can pin the order.
    x = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    return sklearn_linear.LinearRegression().fit(x, [10.0, 1.0, 20.0, 2.0])


def _spec():
    return FeatureSpec(["a", "b"], {"a": "double", "b": "double"})


def test_a_spec_scores_a_reordered_frame_identically_to_an_aligned_one():
    model = _model()
    aligned = bt.from_pydict({"a": [3.0], "b": [1.0], "id": [7]})
    reordered = bt.from_pydict({"id": [7], "b": [1.0], "a": [3.0]})
    want = aligned.ml.predict(model, features=["a", "b"]).to_pydict()["prediction"]
    got = reordered.ml.predict(model, features=_spec()).to_pydict()["prediction"]
    assert got == pytest.approx(want)
    assert got == pytest.approx([31.0])


def test_a_retyped_feature_is_refused_when_the_query_is_built():
    retyped = bt.from_pydict({"a": [3], "b": [1.0]})  # a is int64, pinned double
    with pytest.raises(PlanError, match="dtype mismatch"):
        retyped.ml.predict(_model(), features=_spec())


def test_a_missing_spec_feature_is_refused():
    with pytest.raises(PlanError, match="'b'"):
        bt.from_pydict({"a": [3.0]}).ml.predict(_model(), features=_spec())


def test_on_null_fill_is_the_default_and_passes_nulls_as_missing():
    from sklearn.ensemble import HistGradientBoostingRegressor

    x = np.array([[1.0], [2.0], [3.0], [4.0]] * 5)
    model = HistGradientBoostingRegressor(max_iter=5).fit(x, x[:, 0])
    out = bt.from_pydict({"a": [1.0, None]}).ml.predict(model, features=["a"]).to_pydict()
    assert all(not math.isnan(v) for v in out["prediction"])


@pytest.mark.parametrize("value", [None, float("nan")])
def test_on_null_error_refuses_a_null_or_nan_naming_the_column(value):
    ds = bt.from_pydict({"a": [1.0, value, 3.0], "b": [0.0, 0.0, 0.0]})
    with pytest.raises(DataQualityError, match="'a': 1"):
        ds.ml.predict(_model(), features=["a", "b"], on_null="error").collect()


def test_on_null_error_is_silent_on_complete_features():
    ds = bt.from_pydict({"a": [1.0], "b": [0.0]})
    out = ds.ml.predict(_model(), features=["a", "b"], on_null="error").to_pydict()
    assert out["prediction"] == pytest.approx([10.0])


def test_an_unknown_on_null_policy_is_refused():
    with pytest.raises(PlanError, match="on_null must be one of"):
        bt.from_pydict({"a": [1.0], "b": [0.0]}).ml.predict(
            _model(),
            features=["a", "b"],
            on_null="drop",  # type: ignore[arg-type]
        )


# --- AP-412: which class is prediction_i? --------------------------------------------


def test_predict_proba_columns_follow_classes_order():
    x = np.array([[0.0], [1.0], [2.0]] * 10)
    y = np.array(["emu", "cat", "dog"] * 10)
    model = sklearn_linear.LogisticRegression().fit(x, y)
    assert list(model.classes_) == ["cat", "dog", "emu"]
    ds = bt.from_pydict({"x": [0.0, 1.0, 2.0]})
    out = ds.ml.predict(model, features=["x"], method="predict_proba").to_pydict()
    want = model.predict_proba(np.array([[0.0], [1.0], [2.0]]))
    for i in range(3):
        assert out[f"prediction_{i}"] == pytest.approx(want[:, i].tolist())

    named = ds.ml.predict(
        model,
        features=["x"],
        method="predict_proba",
        output_columns=[f"p_{c}" for c in model.classes_],
    ).to_pydict()
    assert named["p_emu"] == pytest.approx(want[:, 2].tolist())


def test_output_columns_of_the_wrong_length_raise_rather_than_mislabel():
    x = np.array([[0.0], [1.0], [2.0]] * 10)
    model = sklearn_linear.LogisticRegression().fit(x, np.array(["a", "b", "c"] * 10))
    ds = bt.from_pydict({"x": [0.0]})
    with pytest.raises(PlanError, match="names 2 column"):
        ds.ml.predict(
            model, features=["x"], method="predict_proba", output_columns=["p0", "p1"]
        ).collect()
