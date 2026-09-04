"""Scoring a model referenced by an MLflow URI, end to end against a real tracking store.

A team that trains with MLflow refers to a model as `models:/name/version`, and that
reference is the point of the registry: it survives retraining and carries the alias the
rest of the platform is configured with. This exercises the whole path — log a model to a
local tracking store, then score a Batcher dataset by URI — against a real MLflow rather
than a stub, because the parts that break are the resolution and the signature, which a
stub removes.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.tabular.registry import detect_framework, get_adapter, load_model

mlflow = pytest.importorskip("mlflow", reason="scoring by URI needs mlflow")
pytest.importorskip("sklearn", reason="the logged model is a scikit-learn one")

pytestmark = pytest.mark.integration


@pytest.fixture()
def logged_model(tmp_path):
    """A scikit-learn regressor logged to a local tracking store, and its `models:/` URI."""
    from sklearn.linear_model import LinearRegression

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_registry_uri(f"sqlite:///{tmp_path}/mlflow.db")

    rng = np.random.default_rng(11)
    features = rng.normal(size=(200, 2))
    target = 3.0 * features[:, 0] - 2.0 * features[:, 1] + 0.5
    estimator = LinearRegression().fit(features, target)

    # Logged from a bare array on purpose: that produces a *tensor* signature, which is the
    # shape that used to be misread as a single feature named "0".
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            estimator,
            name="model",
            input_example=features[:5],
            registered_model_name="reg",
        )
    return estimator, info.model_uri


@pytest.fixture()
def named_model(tmp_path):
    """The same regressor logged with a DataFrame, so its signature carries column names."""
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/named.db")
    mlflow.set_registry_uri(f"sqlite:///{tmp_path}/named.db")

    rng = np.random.default_rng(12)
    frame = pd.DataFrame(rng.normal(size=(120, 2)), columns=["f0", "f1"])
    target = 3.0 * frame["f0"] - 2.0 * frame["f1"]
    estimator = LinearRegression().fit(frame, target)

    with mlflow.start_run():
        info = mlflow.sklearn.log_model(estimator, name="model", input_example=frame.head())
    return estimator, info.model_uri


def test_an_mlflow_uri_is_recognized_without_an_extension():
    """`models:/churn/3` has no file extension, so suffix detection cannot see it."""
    assert detect_framework("models:/churn/3") == "mlflow"
    assert detect_framework("runs:/abc123def/model") == "mlflow"
    # The control: a plain path still resolves by suffix to its own framework.
    assert detect_framework("/tmp/model.onnx") == "onnx"


def test_the_uri_reaches_the_loader_untouched(monkeypatch):
    """A registry reference must not be copied to a temp file on the way to the loader.

    There is no single file at the end of `models:/churn/3`, and copying would strip
    exactly the indirection that makes the reference worth using.
    """
    seen: list[str] = []

    class _Spy:
        name = "mlflow"
        handles_uri = True

        def load(self, path: str) -> str:
            seen.append(path)
            return "loaded"

    monkeypatch.setattr("batcher.ml.tabular.registry.get_adapter", lambda _name: _Spy())
    assert load_model("models:/churn/3", "mlflow") == "loaded"
    assert seen == ["models:/churn/3"], "the URI was rewritten before the loader saw it"


def test_scoring_by_uri_matches_the_estimator(logged_model):
    """The whole path: a dataset scored by `models:/` URI equals the in-process estimator."""
    estimator, uri = logged_model
    rng = np.random.default_rng(5)
    features = rng.normal(size=(64, 2))
    ds = bt.from_pydict({"f0": features[:, 0].tolist(), "f1": features[:, 1].tolist()})

    scored = ds.ml.predict(model=uri, features=["f0", "f1"], output_column="yhat").to_pydict()

    np.testing.assert_allclose(
        np.asarray(scored["yhat"], dtype=float),
        estimator.predict(features),
        rtol=1e-9,
        atol=1e-9,
    )


def test_a_named_signature_is_surfaced_as_feature_names(named_model):
    """A model logged with a DataFrame records real column names, and they are used.

    A signature is the closest thing MLflow has to the feature names an XGBoost booster
    carries, and checking against it is what catches a pipeline feeding columns in a
    different order than training did — which produces confident, wrong numbers.
    """
    _, uri = named_model
    adapter = get_adapter("mlflow")
    names = adapter.feature_names(adapter.load(uri))

    assert names == ["f0", "f1"]


def test_a_tensor_signature_reports_no_feature_names(logged_model):
    """A model logged from a bare array has shape and dtype, and no column names.

    MLflow answers `input_names()` for such a schema with the positional index as a string,
    so reading it as feature names refused every correct call with "the model expects 1
    features but features= names 2". Unknown names must be None.
    """
    _, uri = logged_model
    adapter = get_adapter("mlflow")

    assert adapter.feature_names(adapter.load(uri)) is None


def test_a_method_pyfunc_cannot_offer_is_refused(logged_model):
    """pyfunc exposes only `predict`; asking for `predict_proba` must say so, not guess."""
    _, uri = logged_model
    ds = bt.from_pydict({"f0": [0.1], "f1": [0.2]})

    with pytest.raises(PlanError, match="method"):
        ds.ml.predict(model=uri, features=["f0", "f1"], method="predict_proba", output_column="p")


def test_the_output_width_comes_from_the_logged_signature(logged_model):
    """The plan needs the output width before the first batch, and the signature has it.

    A regressor's logged output is a tensor of shape ``(-1,)`` — one value per row. Reading
    it is what lets `output_column="yhat"` work without the caller also declaring a width.
    """
    _, uri = logged_model
    adapter = get_adapter("mlflow")

    assert adapter.output_width(adapter.load(uri), "predict", 2) == 1


def test_an_unsigned_model_asks_for_the_width_rather_than_guessing(tmp_path):
    """Without a signature the width is unknowable, and a guess would fail at execution."""
    from sklearn.linear_model import LinearRegression

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/unsigned.db")
    rng = np.random.default_rng(3)
    features = rng.normal(size=(40, 2))
    estimator = LinearRegression().fit(features, features[:, 0])

    # No input_example, so MLflow logs no signature at all.
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(estimator, name="model")

    adapter = get_adapter("mlflow")
    model = adapter.load(info.model_uri)

    assert adapter.output_width(model, "predict", 2) is None
    assert adapter.feature_names(model) is None
