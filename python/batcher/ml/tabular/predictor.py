"""`tabular_predictor` — the load-once class UDF that scores a tabular model.

This is the whole tabular-inference path in one object: a class whose ``__init__`` loads
the model once per worker and whose ``__call__`` turns one Arrow batch into a matrix, one
prediction, and one set of appended columns. It is a *class* rather than a function
because the distributed warm pool keys on the UDF's identity — a plain function would
reload a 400 MB booster on every batch, which is the single most expensive mistake
available on this path.

The factory is memoized on its arguments, so ``ds.ml.predict(<same model>)`` twice hands
the engine the same class object and the second `collect()` reuses the warm pool instead
of rebuilding it.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.tabular.features import (
    append_columns,
    feature_matrix,
    prediction_columns,
)
from batcher.ml.tabular.registry import (
    check_feature_names,
    detect_framework,
    get_adapter,
    load_model,
    resolve_threads,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyarrow as pa

__all__ = ["predicted_column_names", "tabular_predictor"]


def predicted_column_names(
    model: Any,
    *,
    framework: str,
    method: str,
    features: Sequence[str],
    output_column: str,
    output_columns: Sequence[str] | None,
    as_list: bool,
) -> list[str]:
    """The column names a predictor will append, resolved before the query runs.

    The plan needs the output schema up front (an undeclared appended column is invisible
    to every operator above it), but only the model knows how wide its output is. So the
    width is derived from the fitted model here, on the driver — and when it genuinely
    cannot be derived, the caller is told to name the columns rather than left with a plan
    whose schema is a guess.

    This is also where the feature list is checked against the model's own recorded feature
    names, so a wrong or re-ordered `features=` raises when the query is *built* rather than
    after the first batch has been scored on a remote worker.

    Args:
        model: The fitted model, or a path to a saved one (which is opened to measure).
        framework: The resolved framework name.
        method: The prediction method being run.
        features: The feature columns, in the order the model will see them.
        output_column: The base name for the appended column(s).
        output_columns: Explicit per-output names, when the caller gave them.
        as_list: Whether the outputs collapse into a single list column.

    Returns:
        The appended column names, in output order.

    Raises:
        PlanError: If `method` is unknown to the framework, the feature list contradicts
            the model's own, or the width cannot be derived and no columns were named.

    Examples:
        .. doctest::

            >>> from sklearn.linear_model import LogisticRegression
            >>> from batcher.ml.tabular import predicted_column_names
            >>> model = LogisticRegression().fit([[0.0], [1.0]], [0, 1])
            >>> predicted_column_names(
            ...     model,
            ...     framework="sklearn",
            ...     method="predict_proba",
            ...     features=["x"],
            ...     output_column="p",
            ...     output_columns=None,
            ...     as_list=False,
            ... )
            ['p_0', 'p_1']
    """
    adapter = get_adapter(framework)
    if method not in adapter.methods:
        from batcher._internal.errors import suggestion

        hint = suggestion(method, adapter.methods)
        tail = f" {hint}" if hint else ""
        raise PlanError(
            f"{framework} supports method= {sorted(adapter.methods)}, got {method!r}.{tail}"
        )
    # A model given as a path has to be opened to be measured. That load is cached per
    # path, so naming the same saved model in several queries pays for it once, and the
    # alternative — assuming a single output — would silently truncate a multi-class model.
    inspected = _inspect_saved(model, framework) if isinstance(model, str) else model
    check_feature_names(adapter, inspected, features)
    if output_columns is not None:
        return list(output_columns)
    if as_list:
        return [output_column]
    width = adapter.output_width(inspected, method, len(features))
    if width is None:
        raise PlanError(
            f"cannot tell how many values method={method!r} produces per row for this "
            f"{framework} model, and the plan needs the output schema before it runs. "
            "Pass output_columns=[...] to name them, or as_list=True to collect them into "
            "one list column."
        )
    if width == 1:
        return [output_column]
    return [f"{output_column}_{i}" for i in range(width)]


def _null_feature_error(exc: Exception, matrix: Any, features: Sequence[str]) -> PlanError | None:
    """Translate a framework's NaN rejection into the feature columns that caused it.

    scikit-learn answers a null feature with ``Input X contains NaN``, which is accurate and
    unhelpful here for two reasons. The NaN is not in the caller's data — it is what a *null*
    became on the way into the matrix, because `missing` defaults to NaN, which is XGBoost's
    and LightGBM's convention rather than scikit-learn's. And the message names neither the
    column nor either way out, so a single null in one row of one column fails the whole batch
    (and, distributed, the whole task) with a message pointing at the solver.

    Returns `None` for a `ValueError` that is not about missing values, so an unrelated failure
    keeps its own message and traceback.

    Args:
        exc: The exception the framework raised.
        matrix: The feature matrix that was scored, used to find the offending columns.
        features: The feature column names, positionally aligned with `matrix`.

    Returns:
        A `PlanError` naming the columns and the remedies, or `None` if `exc` is unrelated.
    """
    text = str(exc)
    if not any(token in text for token in ("NaN", "infinity", "missing values")):
        return None
    import numpy as np

    bad = [
        name
        for index, name in enumerate(features)
        if index < matrix.shape[1] and not np.isfinite(matrix[:, index]).all()
    ]
    named = ", ".join(repr(name) for name in bad) if bad else "the feature columns"
    return PlanError(
        f"this model cannot score a null feature, and {named} "
        f"{'contains' if len(bad) == 1 else 'contain'} one. A null becomes NaN in the feature "
        "matrix, which XGBoost and LightGBM read as missing but scikit-learn rejects. Either "
        "fill the nulls before scoring (batcher.ml.SimpleImputer, or an sklearn.impute step "
        "inside the pipeline), or pass missing=<value> to ds.ml.predict to substitute a "
        f"constant. The framework said: {text.splitlines()[0]}"
    )


@functools.cache
def _inspect_saved(path: str, framework: str) -> Any:
    """Load a saved model on the driver so its output width can be read (cached per path)."""
    return load_model(path, framework)


@functools.cache
def tabular_predictor(
    model: Any,
    features: tuple[str, ...],
    *,
    framework: str | None = None,
    method: str = "predict",
    output_column: str = "prediction",
    output_columns: tuple[str, ...] | None = None,
    as_list: bool = False,
    missing: float = float("nan"),
    dtype: str | None = None,
    threads: int | None = None,
    options: tuple[tuple[str, Any], ...] = (),
) -> type:
    """Build a load-once class UDF that scores `model` over each batch's `features`.

    The returned class is what `map_batches` wants: constructed once per worker (loading
    the model), then called with each `pyarrow.RecordBatch`. Every argument is hashable so
    the factory can be memoized, which is what keeps the distributed warm-pool key stable
    across `collect()` calls.

    Args:
        model: A fitted model object, or a path/URI to a saved one.
        features: The feature columns, in the exact order the model expects them.
        framework: ``"xgboost"``/``"lightgbm"``/``"catboost"``/``"sklearn"``/``"onnx"``;
            detected from the model when omitted.
        method: What to compute — ``"predict"``, ``"predict_proba"``, ``"raw"``,
            ``"leaf"``, ``"contrib"``, or ``"transform"`` (framework-dependent).
        output_column: The base name for the appended prediction column(s).
        output_columns: Explicit names for a multi-output model's columns.
        as_list: Emit one `List<Float64>` column instead of one column per output.
        missing: The value a null feature becomes (NaN, the boosters' own convention).
        dtype: The feature-matrix dtype, ``"float32"`` or ``"float64"``; the framework's
            own precision when omitted (float32 for the boosters, float64 for sklearn).
        threads: The model's thread-pool size inside one worker; auto-capped when unset.
        options: Extra framework keywords as a hashable tuple of items, such as
            ``(("iteration_range", (0, 50)),)`` for XGBoost.

    Returns:
        A class whose instances score one Arrow batch each.

    Raises:
        PlanError: If `features` is empty, or `framework`/`method` is unknown.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from sklearn.linear_model import LinearRegression
            >>> from batcher.ml.tabular import tabular_predictor
            >>> model = LinearRegression().fit([[0.0], [1.0]], [0.0, 2.0])
            >>> udf = tabular_predictor(model, ("x",))()
            >>> scored = udf(pa.RecordBatch.from_pydict({"x": [2.0]}))
            >>> round(scored.to_pydict()["prediction"][0], 6)
            4.0
    """
    if not features:
        raise PlanError("a tabular predictor needs at least one feature column")
    resolved_framework = framework or detect_framework(model)
    adapter = get_adapter(resolved_framework)
    if method not in adapter.methods:
        from batcher._internal.errors import suggestion

        hint = suggestion(method, adapter.methods)
        tail = f" {hint}" if hint else ""
        raise PlanError(
            f"{resolved_framework} supports method= {sorted(adapter.methods)}, "
            f"got {method!r}.{tail}"
        )
    feature_list = list(features)
    names = list(output_columns) if output_columns is not None else None
    opts = dict(options)
    matrix_dtype = dtype or adapter.default_dtype

    class _TabularModel:
        """Scores one Arrow batch with a model loaded once per worker."""

        def __init__(self) -> None:
            self._adapter = get_adapter(resolved_framework)
            self._model = load_model(model, resolved_framework)
            self._threads = resolve_threads(threads)
            self._adapter.configure_threads(self._model, self._threads)
            check_feature_names(self._adapter, self._model, feature_list)

        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            if batch.num_rows == 0:
                return append_columns(batch, _empty_outputs(output_column, names, as_list))
            matrix = feature_matrix(batch, feature_list, dtype=matrix_dtype, missing=missing)
            call_opts = {"missing": missing, "threads": self._threads, **opts}
            try:
                raw = self._adapter.predict(self._model, matrix, method, call_opts)
            except ValueError as exc:
                translated = _null_feature_error(exc, matrix, feature_list)
                if translated is None:
                    raise
                raise translated from exc
            columns = prediction_columns(
                raw, output_column=output_column, output_columns=names, as_list=as_list
            )
            return append_columns(batch, columns)

    _TabularModel.__name__ = f"{resolved_framework.capitalize()}Predictor"
    _TabularModel.__qualname__ = _TabularModel.__name__
    return _TabularModel


def _empty_outputs(
    output_column: str, output_columns: list[str] | None, as_list: bool
) -> dict[str, pa.Array]:
    """The appended columns for a zero-row batch, without calling the model.

    Every framework here raises or returns an ill-shaped array for an empty matrix, and a
    zero-row batch is routine (an empty partition, a filter that matched nothing). The
    schema still has to match the non-empty case, so the declared columns are emitted
    empty. With an unknown output width the single-column shape is the only one available,
    which is also the only shape a zero-row batch can be asked for without `output_columns`.
    """
    import pyarrow as pa

    names = output_columns or [output_column]
    kind = pa.list_(pa.float64()) if as_list else pa.float64()
    return {name: pa.array([], type=kind) for name in names}
