"""The tabular-framework registry — detect, load, and score a model uniformly.

Every supported framework (XGBoost, LightGBM, CatBoost, scikit-learn, ONNX Runtime) gets
one `TabularAdapter`. The adapter answers four questions the predictor UDF needs and
nothing else: *is this object mine*, *how do I load one from a path*, *what does it call a
prediction*, and *what are its feature names*. Keeping that behind one small protocol is
what lets `ds.ml.predict` accept any of them with the same keywords.

Detection is by the model object's defining module (never by ``isinstance``, which would
force an import of every framework to check any of them), or by file extension for a path.
An explicit ``framework=`` always wins, so an unusual wrapper is never a dead end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

__all__ = [
    "FRAMEWORKS",
    "BaseAdapter",
    "TabularAdapter",
    "check_feature_names",
    "detect_framework",
    "get_adapter",
    "load_model",
    "n_classes",
    "register",
    "resolve_threads",
]


class TabularAdapter(Protocol):
    """What the predictor UDF needs from one tabular ML framework.

    Implementations are module-level singletons registered in `FRAMEWORKS`; they hold no
    state, so one instance serves every worker.
    """

    name: str
    #: Prediction methods this framework understands, in the order they are documented.
    methods: tuple[str, ...]
    #: File suffixes that identify a saved model of this framework.
    suffixes: tuple[str, ...]
    #: The feature-matrix precision this framework should be fed by default.
    default_dtype: str

    def owns(self, model: Any) -> bool:
        """Whether `model` is an instance of this framework's model type."""
        ...

    def load(self, path: str) -> Any:
        """Load a saved model from a local filesystem `path`."""
        ...

    def predict(self, model: Any, matrix: np.ndarray, method: str, options: dict[str, Any]) -> Any:
        """Score `matrix` with `model`, returning a NumPy-convertible array."""
        ...

    def feature_names(self, model: Any) -> list[str] | None:
        """The feature names the model was trained with, when it records them."""
        ...

    def configure_threads(self, model: Any, threads: int) -> None:
        """Cap the model's own thread pool to `threads` (best effort, in place)."""
        ...

    def output_width(self, model: Any, method: str, n_features: int) -> int | None:
        """Values per row this `method` produces, or None when only the data can say."""
        ...


class BaseAdapter:
    """Shared adapter behavior: module-name detection and no-op thread/feature hooks."""

    name = ""
    methods: tuple[str, ...] = ("predict",)
    suffixes: tuple[str, ...] = ()
    #: Top-level module names whose classes belong to this framework.
    modules: tuple[str, ...] = ()
    # float32 is the boosters' own internal precision, so building float64 doubles the
    # per-batch copy and changes nothing. A scikit-learn estimator computes in float64,
    # and feeding it float32 shifts the last few digits of every prediction against what
    # the same estimator returns in-process — a difference that shows up as a failing
    # parity test, not as an error. Each adapter therefore names its own default.
    default_dtype: str = "float32"

    def owns(self, model: Any) -> bool:
        """True when `model`'s defining module is one of this framework's `modules`."""
        root = type(model).__module__.split(".")[0]
        return root in self.modules

    def feature_names(self, model: Any) -> list[str] | None:
        """No recorded feature names by default; frameworks that keep them override this."""
        _ = model
        return None

    def configure_threads(self, model: Any, threads: int) -> None:
        """No thread pool to cap by default."""
        _ = model, threads

    def output_width(self, model: Any, method: str, n_features: int) -> int | None:
        """How many values per row this `method` produces, or None when it is unknowable.

        The plan needs the output schema *before* the first batch runs, and only the model
        knows how wide its output is. Everything derivable from the fitted estimator's own
        attributes is derived here; a genuinely unknowable width returns None and the
        caller asks for `output_columns=` or `as_list=True` rather than guessing.
        """
        if method == "contrib":
            return n_features + 1
        classes = n_classes(model)
        if method == "predict_proba":
            return classes
        if method == "raw":
            return 1 if classes in (None, 2) else classes
        if method == "predict":
            outputs = getattr(model, "n_outputs_", None)
            return int(outputs) if isinstance(outputs, int) and outputs > 0 else 1
        return None

    def _check_method(self, method: str) -> None:
        if method not in self.methods:
            from batcher._internal.errors import suggestion

            hint = suggestion(method, self.methods)
            tail = f" {hint}" if hint else ""
            raise PlanError(
                f"{self.name} supports method= {sorted(self.methods)}, got {method!r}.{tail}"
            )


def n_classes(model: Any) -> int | None:
    """The number of classes a fitted classifier predicts, or None when it is not one.

    Args:
        model: A fitted model object.

    Returns:
        The class count, or None for a regressor or a model that records none.

    Examples:
        .. doctest::

            >>> from sklearn.linear_model import LogisticRegression
            >>> from batcher.ml.tabular.registry import n_classes
            >>> n_classes(LogisticRegression().fit([[0.0], [1.0]], [0, 1]))
            2
    """
    classes = getattr(model, "classes_", None)
    if classes is not None:
        try:
            return len(classes)
        except TypeError:  # pragma: no cover - a scalar `classes_` is not a class list
            return None
    count = getattr(model, "n_classes_", None)
    return int(count) if isinstance(count, int) and count > 0 else None


#: Every registered adapter, keyed by its `framework=` name. Populated by `boosters` and
#: `estimators` at import; `get_adapter` is the only supported way to read it.
FRAMEWORKS: dict[str, TabularAdapter] = {}


def register(adapter: TabularAdapter) -> TabularAdapter:
    """Add `adapter` to the framework registry and return it (used at module scope)."""
    FRAMEWORKS[adapter.name] = adapter
    return adapter


def _load_adapters() -> None:
    """Import the adapter modules so `FRAMEWORKS` is populated (idempotent)."""
    if FRAMEWORKS:
        return
    from batcher.ml.tabular import boosters, estimators  # noqa: F401  (registration import)


def get_adapter(framework: str) -> TabularAdapter:
    """The adapter registered under `framework`.

    Args:
        framework: The framework name (``"xgboost"``, ``"lightgbm"``, ``"catboost"``,
            ``"sklearn"``, ``"onnx"``).

    Returns:
        The registered `TabularAdapter`.

    Raises:
        PlanError: If no adapter is registered under that name.

    Examples:
        .. doctest::

            >>> from batcher.ml.tabular.registry import get_adapter
            >>> get_adapter("sklearn").name
            'sklearn'
    """
    _load_adapters()
    try:
        return FRAMEWORKS[framework]
    except KeyError:
        from batcher._internal.errors import suggestion

        hint = suggestion(framework, sorted(FRAMEWORKS))
        tail = f" {hint}" if hint else ""
        raise PlanError(
            f"unknown framework {framework!r}; expected one of {sorted(FRAMEWORKS)}.{tail}"
        ) from None


def detect_framework(model: Any) -> str:
    """Identify which framework `model` (an object or a path) belongs to.

    Args:
        model: A loaded model object, or a path/URI to a saved model.

    Returns:
        The framework name.

    Raises:
        PlanError: If the framework cannot be identified, naming the fix
            (pass ``framework=``).

    Examples:
        .. doctest::

            >>> from sklearn.linear_model import LinearRegression
            >>> from batcher.ml.tabular.registry import detect_framework
            >>> detect_framework(LinearRegression())
            'sklearn'
    """
    _load_adapters()
    if isinstance(model, str):
        suffix = model.rsplit(".", 1)[-1].lower() if "." in model else ""
        for adapter in FRAMEWORKS.values():
            if suffix in adapter.suffixes:
                return adapter.name
        raise PlanError(
            f"cannot tell which framework the model file {model!r} belongs to from its "
            f"extension. Pass framework= explicitly (one of {sorted(FRAMEWORKS)})."
        )
    for adapter in FRAMEWORKS.values():
        if adapter.owns(model):
            return adapter.name
    # A duck-typed estimator (a custom class with `predict`) is the scikit-learn contract,
    # so accept it rather than refusing a model that would work.
    if hasattr(model, "predict"):
        return "sklearn"
    raise PlanError(
        f"cannot tell which ML framework {type(model).__name__} belongs to, and it has no "
        f"predict() method. Pass framework= explicitly (one of {sorted(FRAMEWORKS)})."
    )


def load_model(source: Any, framework: str) -> Any:
    """Return the model itself, loading it from `source` when `source` is a path or URI.

    A remote URI (``s3://``, ``gs://``, ``https://``) is fetched to a temporary local file
    through the project's filesystem façade first, because every framework's loader takes a
    local path. A model object passes straight through.

    Args:
        source: A model object, or a path/URI to a saved model.
        framework: The framework whose loader to use.

    Returns:
        The loaded model object.

    Raises:
        PlanError: If the file cannot be read or the loader rejects it.
    """
    if not isinstance(source, str):
        return source
    adapter = get_adapter(framework)
    path = _localize(source)
    try:
        return adapter.load(path)
    except PlanError:
        raise
    except Exception as exc:
        raise PlanError(
            f"failed to load the {framework} model at {source!r}: {type(exc).__name__}: {exc}"
        ) from exc


def _localize(source: str) -> str:
    """A local path for `source`, downloading it once per worker when it is remote."""
    if "://" not in source or source.startswith("file://"):
        return source.removeprefix("file://")
    import os
    import tempfile

    from batcher.io.filesystem import resolve_filesystem

    fs = resolve_filesystem(source)
    suffix = os.path.splitext(source)[1]
    fd, local = tempfile.mkstemp(prefix="batcher-model-", suffix=suffix)
    with fs.open(source, "rb") as remote, os.fdopen(fd, "wb") as out:
        out.write(remote.read())
    return local


def resolve_threads(threads: int | None) -> int:
    """The thread count a tabular model should use inside one worker.

    Mirrors the CPU-inference thread cap on the transformers path: a booster's default is
    the *host* core count, so several co-located actors each grab every core and thrash.
    An explicit value wins; otherwise the container's usable core count.

    Args:
        threads: The caller's explicit thread count, or None for the automatic cap.

    Returns:
        A positive thread count.
    """
    if threads is not None and threads > 0:
        return int(threads)
    import os

    omp = os.environ.get("OMP_NUM_THREADS", "").strip()
    if omp.isdigit() and int(omp) > 0:
        return int(omp)
    from batcher._internal.hardware import available_cpu_count

    return max(1, available_cpu_count())


def check_feature_names(adapter: TabularAdapter, model: Any, features: Sequence[str]) -> None:
    """Raise when the batch's feature order disagrees with the model's recorded one.

    A tabular model scores by *position*. Passing the right columns in the wrong order is
    not an error anywhere else in the stack — it silently produces confident nonsense — so
    where the model records its training feature names, this is the one place that can
    catch it.

    Only two disagreements are real, and both are checked. A **count** mismatch is always
    wrong. A **permutation** (same names, different order) is always wrong. Names that
    simply differ are not: a booster trained from a bare NumPy matrix records generic
    ``f0…fN``, which matches no real column name and means nothing.

    Args:
        adapter: The framework adapter, which knows where the names live.
        model: The loaded model.
        features: The feature order this predictor will build each matrix in.

    Raises:
        PlanError: On a feature-count mismatch, or a same-set-different-order mismatch.
    """
    trained = adapter.feature_names(model)
    if not trained:
        return
    wanted = list(features)
    if len(trained) != len(wanted):
        raise PlanError(
            f"the model expects {len(trained)} features but features= names {len(wanted)}. "
            f"Model features: {list(trained)}."
        )
    if list(trained) == wanted:
        return
    if set(trained) == set(wanted):
        raise PlanError(
            f"features= is a re-ordering of the model's own feature names. A tabular model "
            f"scores by position, so this would silently change every prediction. Pass "
            f"features={list(trained)}."
        )
