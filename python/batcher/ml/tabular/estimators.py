"""scikit-learn and ONNX Runtime adapters.

Two very different shapes behind one interface. A scikit-learn estimator is any object
with ``fit``/``predict``, which makes it both the widest surface here (linear models,
random forests, SVMs, pipelines, calibrated classifiers) and the one with the least
metadata to inspect. An ONNX graph is the opposite: a frozen computation with a declared
input signature, no Python object model, and its own device placement.

The scikit-learn adapter deliberately accepts **any** duck-typed estimator, including a
full ``sklearn.pipeline.Pipeline`` — scoring a fitted pipeline is how most tabular
production models are actually shipped, and refusing anything not literally under
``sklearn.`` would exclude every third-party estimator that follows the same contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher._internal.optional import require
from batcher.ml.tabular.registry import BaseAdapter, register

if TYPE_CHECKING:
    import numpy as np

__all__ = ["ONNXAdapter", "SklearnAdapter"]


class SklearnAdapter(BaseAdapter):
    """Scores any fitted scikit-learn-style estimator or ``Pipeline``."""

    name = "sklearn"
    modules = ("sklearn", "imblearn")
    methods = ("predict", "predict_proba", "raw", "transform")
    suffixes = ("pkl", "pickle", "joblib")
    default_dtype = "float64"

    def owns(self, model: Any) -> bool:
        """True for a scikit-learn class, or any object exposing the ``fit``/``predict`` pair."""
        if super().owns(model):
            return True
        return hasattr(model, "predict") and hasattr(model, "fit")

    def load(self, path: str) -> Any:
        """Unpickle a saved estimator (``joblib`` when available, else ``pickle``).

        Loading a pickle executes arbitrary code from the file, so this path is only ever
        reached for a model path the caller supplied.
        """
        try:
            joblib = require(
                "joblib", feature="scikit-learn model loading", provides="joblib", extra="sklearn"
            )
        except Exception:
            import pickle

            with open(path, "rb") as handle:
                return pickle.load(handle)
        return joblib.load(path)

    def feature_names(self, model: Any) -> list[str] | None:
        """The estimator's ``feature_names_in_``, recorded when it was fitted on a DataFrame."""
        names = getattr(model, "feature_names_in_", None)
        return [str(n) for n in names] if names is not None and len(names) else None

    def configure_threads(self, model: Any, threads: int) -> None:
        """Cap the estimator's ``n_jobs`` to `threads` where it has one."""
        try:
            if hasattr(model, "set_params") and "n_jobs" in getattr(model, "get_params", dict)():
                model.set_params(n_jobs=threads)
        except Exception:  # pragma: no cover - an estimator without n_jobs is fine
            return

    def predict(self, model: Any, matrix: np.ndarray, method: str, options: dict[str, Any]) -> Any:
        """Call the estimator's own ``predict`` / ``predict_proba`` / ``decision_function``."""
        _ = options
        self._check_method(method)
        if method == "predict_proba":
            if not hasattr(model, "predict_proba"):
                raise PlanError(
                    f"{type(model).__name__} has no predict_proba (it is not a probabilistic "
                    "classifier). Use method='predict', or wrap it in "
                    "sklearn.calibration.CalibratedClassifierCV."
                )
            return model.predict_proba(matrix)
        if method == "raw":
            if not hasattr(model, "decision_function"):
                raise PlanError(
                    f"{type(model).__name__} has no decision_function, so there is no raw "
                    "score to return. Use method='predict' or 'predict_proba'."
                )
            return model.decision_function(matrix)
        if method == "transform":
            if not hasattr(model, "transform"):
                raise PlanError(f"{type(model).__name__} has no transform. Use method='predict'.")
            return model.transform(matrix)
        return model.predict(matrix)


class ONNXAdapter(BaseAdapter):
    """Scores an ONNX graph through ONNX Runtime (CPU or CUDA)."""

    name = "onnx"
    modules = ("onnxruntime", "onnx")
    methods = ("predict", "predict_proba", "raw")
    suffixes = ("onnx",)

    def owns(self, model: Any) -> bool:
        """True for an ``onnxruntime.InferenceSession`` (or the module's other model types)."""
        return super().owns(model)

    def load(self, path: str) -> Any:
        """Open an ONNX Runtime ``InferenceSession`` over the graph at `path`.

        Providers are chosen at load: CUDA first when this worker has a visible GPU and the
        GPU build of ONNX Runtime is installed, then CPU. Getting that wrong is the classic
        silent 20x slowdown, because a CPU-only session on a GPU actor still returns correct
        answers.
        """
        ort = require(
            "onnxruntime",
            feature="ONNX batch inference",
            provides="onnxruntime",
            extra="onnx",
        )
        from batcher.ml.runtimes.providers import onnx_providers

        # The provider choice is the same question the deep-model path answers, so it is
        # answered in the same place: `onnx_providers` covers ROCm, MIGraphX, DirectML and
        # CoreML as well as CUDA, and — unlike the two-name list this replaced — it declines
        # to name an accelerated provider on a host with no visible accelerator, which is
        # where a GPU-build wheel otherwise fails at session creation.
        resolved = onnx_providers(None, list(ort.get_available_providers()))
        return ort.InferenceSession(path, providers=resolved or None)

    def feature_names(self, model: Any) -> list[str] | None:
        """ONNX inputs are tensors, not named features, so there is nothing to compare."""
        _ = model
        return None

    def configure_threads(self, model: Any, threads: int) -> None:
        """ONNX Runtime fixes its thread pools at session creation, so this is a no-op."""
        _ = model, threads

    def predict(self, model: Any, matrix: np.ndarray, method: str, options: dict[str, Any]) -> Any:
        """Run the graph over `matrix`, returning the output the `method` names.

        ``predict`` takes the first output, which is the label/value for every converter
        (``skl2onnx``, ``onnxmltools``) in common use. ``predict_proba`` takes the second,
        which is where those converters put the probability tensor — as a list of
        ``{class: probability}`` maps for a ZipMap-enabled graph, which is flattened here
        back into a dense matrix.

        The feature matrix is cast to **the dtype the graph declares**, not to float32
        whenever the declaration mentions "float". That substring test read ``tensor(float16)``
        as float32 — so every half-precision export was fed the wrong width and rejected — and
        read ``tensor(double)`` as *not* float, so a float64 graph got whatever the matrix
        happened to be.
        """
        self._check_method(method)
        spec = model.get_inputs()[0]
        input_name = options.get("input_name") or spec.name
        feed = {input_name: _as_graph_dtype(matrix, spec.type)}
        outputs = model.run(None, feed)
        if method == "predict":
            return outputs[0]
        if len(outputs) < 2:
            raise PlanError(
                f"method={method!r} needs a second graph output (the probability tensor), but "
                f"this ONNX model has {len(outputs)}. Use method='predict'."
            )
        return _dense_probabilities(outputs[1])


def _as_graph_dtype(matrix: np.ndarray, declared: str) -> np.ndarray:
    """The feature matrix in the element type the graph's input declares.

    An ONNX graph does not coerce: it rejects a feed whose dtype is not the one it was
    exported with, and reports that as a message about the tensor's *shape*. `ONNX_TO_NUMPY`
    is the single mapping both this path and the deep-model runtime resolve against, so an
    element type either is handled in both or is left alone in both.
    """
    from batcher.ml.runtimes.onnx import ONNX_TO_NUMPY

    target = ONNX_TO_NUMPY.get(declared)
    if target is None:
        return matrix  # a string/sequence input; leave it for the runtime to judge
    return matrix.astype(target, copy=False)


def _dense_probabilities(output: Any) -> Any:
    """A ZipMap-style ``list[dict[class, prob]]`` output as a dense 2-D array.

    ``skl2onnx`` emits classifier probabilities as a sequence of maps by default. Left as
    is, each row would become a Python dict and the append step would build an object
    column, so this flattens it once, in class-id order, exactly as the map is keyed.
    """
    import numpy as np

    if isinstance(output, list) and output and isinstance(output[0], dict):
        classes = list(output[0].keys())
        return np.asarray([[row[c] for c in classes] for row in output], dtype="float64")
    return np.asarray(output)


register(SklearnAdapter())
register(ONNXAdapter())
