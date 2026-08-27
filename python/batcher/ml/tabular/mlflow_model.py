"""Scoring a model out of an MLflow registry, by URI, without leaving the pipeline.

A team that trains with MLflow refers to a model the way MLflow does — `models:/churn/3`,
`models:/churn@champion`, `runs:/<run-id>/model` — and that reference is the whole point of
having a registry: it survives retraining, it carries a stage or an alias, and it is what
the rest of the platform is already configured with. Without this adapter a Batcher
pipeline had to resolve the reference itself, download the artifact, work out which
framework was inside, and load it with that framework's own loader — reimplementing the
registry client in every project.

# Why it is an adapter rather than a special case

`ds.ml.predict` already dispatches through `registry.FRAMEWORKS`, and the predictor
constructs its model **once per worker** (`predictor._TabularModel.__init__`). Registering
here means an MLflow model is loaded on the worker that scores with it, against that
machine's own MLflow credentials, exactly as an XGBoost booster is. Nothing about the
distributed path needs to know this framework exists.

That also settles where the artifact download happens. `registry.load_model` copies a
remote path to a local file before handing it to a loader, which is right for a `.json`
booster in S3 and wrong here: there is no single file at the end of `models:/churn/3`, and
copying would strip the indirection that makes the reference worth using. The adapter
declares `handles_uri`, and the loader passes the URI through untouched.

# The pyfunc flavor, and what it costs

Loading goes through `mlflow.pyfunc`, which is the flavor every logged model has, so one
adapter covers scikit-learn, XGBoost, LightGBM, PyTorch and a custom `PythonModel` alike.
The price is that `pyfunc.predict` takes a DataFrame or an array and returns whatever the
flavor returns, so this adapter cannot offer `predict_proba` or `contrib` — those are
framework-specific entry points that pyfunc does not expose. A caller who needs one should
name the framework directly and point at the artifact. That is a real limitation and it is
stated in the error rather than worked around, because guessing which flavor is underneath
and reaching into it is how a scoring path starts disagreeing with the training code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.optional import require
from batcher.ml.tabular.registry import BaseAdapter, register

if TYPE_CHECKING:
    import numpy as np

__all__ = ["MlflowAdapter"]


class MlflowAdapter(BaseAdapter):
    """Loads and scores a model referenced by an MLflow URI."""

    name = "mlflow"
    methods = ("predict",)
    # The three references MLflow itself resolves. `mlflow-artifacts:` is included because
    # it carries `://`, which `registry._localize` would otherwise treat as a remote file
    # to copy.
    uri_schemes = ("models:/", "runs:/", "mlflow-artifacts:")
    handles_uri = True
    #: A pyfunc model is a Python object, and NumPy's default precision is what the flavors
    #: underneath were trained against. Narrowing to float32 here would shift the last
    #: digits of every prediction away from what the training code produced.
    default_dtype = "float64"
    modules = ("mlflow",)

    def owns(self, model: Any) -> bool:
        """True for a loaded pyfunc model, whichever flavor it wraps."""
        return type(model).__module__.split(".")[0] == "mlflow"

    def load(self, path: str) -> Any:
        """Resolve `path` through MLflow and return the loaded pyfunc model.

        The tracking URI is not set here. It comes from the worker's own environment
        (``MLFLOW_TRACKING_URI``) or from whatever the host process configured, so a worker
        authenticates as itself rather than inheriting a driver's session.
        """
        pyfunc = require(
            "mlflow.pyfunc",
            feature="scoring an MLflow model",
            provides="mlflow",
            extra="mlflow",
        )
        return pyfunc.load_model(path)

    def predict(self, model: Any, matrix: np.ndarray, method: str, options: dict[str, Any]) -> Any:
        """Score `matrix` through the pyfunc interface.

        `method` is always ``"predict"``; the predictor validates it against `methods`
        before reaching here, and pyfunc exposes nothing else.
        """
        _ = method, options
        return model.predict(matrix)

    def feature_names(self, model: Any) -> list[str] | None:
        """The input column names from the model's logged signature, when it has one.

        A logged signature is the closest thing MLflow has to the feature names an XGBoost
        booster carries, and checking against it is what catches a pipeline feeding columns
        in a different order than training did — which produces confident, wrong numbers
        rather than an error.
        """
        try:
            signature = model.metadata.get_input_schema()
        except Exception:  # pragma: no cover - a model logged without a signature
            return None
        if signature is None:
            return None
        # A model logged from a bare array gets a *tensor* schema, which describes shape and
        # dtype and carries no column names at all. `input_names()` answers it with the
        # positional index as a string — `["0"]` for a 2-column array — so taking that as
        # feature names refuses every correct call with "the model expects 1 features but
        # features= names 2". A tensor schema means "unknown names", which is None.
        try:
            if signature.is_tensor_spec():
                return None
            names = [name for name in signature.input_names() if name is not None]
        except Exception:  # pragma: no cover - an unfamiliar schema shape
            return None
        return [str(name) for name in names] or None

    def output_width(self, model: Any, method: str, n_features: int) -> int | None:
        """Values per row, read from the model's logged *output* signature.

        The plan needs the output schema before the first batch runs, and a pyfunc model
        does not otherwise describe its shape. A logged signature does, and MLflow records
        one whenever a model was logged with an `input_example` — which is the documented
        way to log one and what the autologging integrations do.

        A tensor output of shape ``(-1,)`` is one value per row, ``(-1, k)`` is `k`. A
        column schema is one value per named column. Anything else, or a model logged
        without a signature, returns None and the caller names the columns with
        `output_columns=` or collects them with `as_list=True`. That is deliberate: a
        guess would build the plan's schema on an assumption and fail at execution with a
        width mismatch, which is a far worse error than being asked for the width.
        """
        _ = method, n_features
        try:
            schema = model.metadata.get_output_schema()
        except Exception:  # pragma: no cover - a model logged without a signature
            return None
        if schema is None:
            return None
        try:
            if not schema.is_tensor_spec():
                return len(schema.inputs) or None
            specs = list(schema.inputs)
            if len(specs) != 1:
                return None
            shape = tuple(specs[0].shape)
        except Exception:  # pragma: no cover - an unfamiliar schema shape
            return None
        if len(shape) == 1:
            return 1
        if len(shape) == 2 and isinstance(shape[1], int) and shape[1] > 0:
            return int(shape[1])
        return None

    def configure_threads(self, model: Any, threads: int) -> None:
        """No thread pool to cap: the flavor underneath owns its own threading."""
        _ = model, threads


# Registered as an instance, matching the module-level-singleton convention the other
# adapter modules follow: the registry stores the object the predictor calls, not the class.
register(MlflowAdapter())
