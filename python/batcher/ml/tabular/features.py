"""Arrow batch → dense feature matrix, and model output → Arrow columns.

Tree models (XGBoost, LightGBM, CatBoost), scikit-learn estimators, and ONNX graphs all
want one dense row-major ``(n_rows, n_features)`` array. Arrow gives us columns, so the
transpose has to happen somewhere; doing it here — once per batch, with NumPy — keeps it
out of per-row Python and off the engine's hot path.

Two details this module exists to get right:

- **Nulls are not zeros.** A missing feature that silently becomes ``0.0`` changes the
  prediction without any error. Every framework here treats NaN as "missing" (XGBoost and
  LightGBM learn a default direction for it), so nulls become NaN by default and the caller
  may override with an explicit sentinel the model was trained with.
- **Feature order is the contract.** A model scores by *position*, not by name, so a batch
  whose columns arrive in a different order than training silently produces garbage. The
  feature list is fixed once at build time and every batch is projected through it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

__all__ = [
    "FEATURE_DTYPES",
    "append_columns",
    "feature_matrix",
    "prediction_columns",
    "resolve_features",
]

#: The dtypes a feature matrix may be built in. ``float32`` is the default because every
#: booster casts to it internally anyway, so building float64 doubles the copy for nothing.
FEATURE_DTYPES = ("float32", "float64")

# Arrow types that carry a numeric value a model can consume once cast.
_NUMERIC = (pa.types.is_integer, pa.types.is_floating, pa.types.is_boolean, pa.types.is_decimal)


def resolve_features(names: Sequence[str] | None, available: Sequence[str]) -> list[str]:
    """The ordered feature list to score, defaulting to every available column.

    Args:
        names: The explicit feature columns, in model order, or None for "all of them".
        available: The dataset's column names, in schema order.

    Returns:
        The feature names in the exact order the model will see them.

    Raises:
        PlanError: If `names` is empty, holds a duplicate, or names a missing column.

    Examples:
        .. doctest::

            >>> from batcher.ml.tabular.features import resolve_features
            >>> resolve_features(["b", "a"], ["a", "b", "c"])
            ['b', 'a']
            >>> resolve_features(None, ["a", "b"])
            ['a', 'b']
    """
    if names is None:
        feats = list(available)
        if not feats:
            raise PlanError("a tabular predictor needs at least one feature column")
        return feats
    feats = [names] if isinstance(names, str) else list(names)
    if not feats:
        raise PlanError("features= must name at least one column")
    seen: set[str] = set()
    for name in feats:
        if not isinstance(name, str):
            raise PlanError(f"features= must be column names (strings), got {name!r}")
        if name in seen:
            raise PlanError(
                f"features= lists {name!r} twice; a model scores by position, so a repeated "
                "feature shifts every later column onto the wrong slot"
            )
        seen.add(name)
    missing = [f for f in feats if f not in set(available)]
    if missing:
        from batcher._internal.errors import ColumnNotFoundError, unknown_message

        raise ColumnNotFoundError(
            unknown_message("column", missing[0], list(available), hint="Pass a feature column.")
        )
    return feats


def _column_values(batch: pa.RecordBatch, name: str, missing: float) -> Any:
    """One feature column as a 1-D float NumPy array, with nulls filled by `missing`."""
    import numpy as np

    column = batch.column(name)
    dtype = column.type
    if pa.types.is_null(dtype):
        # An all-null column in this batch types as `null`, which is not a numeric type but
        # is also not a modelling error — every value is simply missing. Refusing it would
        # fail a query on a batch boundary rather than on the data.
        return np.full(len(column), missing, dtype="float64")
    if pa.types.is_dictionary(dtype):
        # A dictionary-encoded categorical decodes to its value type; a *string* dictionary
        # is still not a number, so it lands in the type error below with a real type name.
        column = column.cast(dtype.value_type)
        dtype = column.type
    if not any(check(dtype) for check in _NUMERIC):
        raise PlanError(
            f"feature column {name!r} has type {dtype}, which a tabular model cannot score. "
            "Encode it first (OrdinalEncoder / OneHotEncoder / TargetEncoder), or drop it "
            "from features=."
        )
    if pa.types.is_decimal(dtype) or pa.types.is_boolean(dtype):
        column = column.cast(pa.float64())
    if column.null_count:
        # `fill_null` before `to_numpy` so the null mask never reaches NumPy, which would
        # otherwise force `zero_copy_only=False` into an object array on some types.
        column = column.fill_null(missing)
    return np.asarray(column.to_numpy(zero_copy_only=False))


def feature_matrix(
    batch: pa.RecordBatch,
    features: Sequence[str],
    *,
    dtype: str = "float32",
    missing: float = float("nan"),
) -> np.ndarray:
    """Assemble `features` from `batch` into a dense row-major matrix.

    Args:
        batch: The Arrow batch to read the feature columns from.
        features: The feature column names, in the order the model expects them.
        dtype: The matrix dtype, ``"float32"`` (default) or ``"float64"``.
        missing: The value a null feature becomes; NaN by default, which is what
            XGBoost and LightGBM treat as missing.

    Returns:
        A C-contiguous ``(batch.num_rows, len(features))`` array.

    Raises:
        PlanError: On an unknown `dtype`, or a feature column a model cannot score.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.ml.tabular.features import feature_matrix
            >>> batch = pa.RecordBatch.from_pydict({"a": [1, 2], "b": [0.5, None]})
            >>> feature_matrix(batch, ["a", "b"], missing=0.0).tolist()
            [[1.0, 0.5], [2.0, 0.0]]
    """
    import numpy as np

    if dtype not in FEATURE_DTYPES:
        raise PlanError(f"dtype must be one of {FEATURE_DTYPES}, got {dtype!r}")
    out = np.empty((batch.num_rows, len(features)), dtype=dtype, order="C")
    for i, name in enumerate(features):
        out[:, i] = _column_values(batch, name, missing)
    return out


def _as_2d(values: Any) -> np.ndarray:
    """A model output as a 2-D ``(n_rows, n_outputs)`` array, whatever shape it arrived in."""
    import numpy as np

    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def prediction_columns(
    values: Any,
    *,
    output_column: str,
    output_columns: Sequence[str] | None = None,
    as_list: bool = False,
) -> dict[str, pa.Array]:
    """Turn a model's raw output into the Arrow columns to append to the batch.

    A single-output model (regression, binary ``predict``) becomes one column named
    `output_column`. A multi-output model (``predict_proba``, multi-class margin, SHAP
    contributions) becomes either one list column — the default when `as_list` — or one
    column per output, named ``{output_column}_0 …`` unless `output_columns` names them.

    Args:
        values: The model output: a scalar, 1-D, 2-D, or higher array.
        output_column: The base name for the appended column(s).
        output_columns: Explicit names for a multi-output model's columns.
        as_list: Emit one `List<Float64>` column instead of one column per output.

    Returns:
        A ``{name: pyarrow.Array}`` mapping, in output order.

    Raises:
        PlanError: If `output_columns` does not match the model's output width.

    Examples:
        .. doctest::

            >>> from batcher.ml.tabular.features import prediction_columns
            >>> cols = prediction_columns([[0.1, 0.9]], output_column="p")
            >>> {k: v.to_pylist() for k, v in cols.items()}
            {'p_0': [0.1], 'p_1': [0.9]}
    """
    arr = _as_2d(values)
    width = arr.shape[1]
    if width == 1 and output_columns is None and not as_list:
        return {output_column: pa.array(arr[:, 0])}
    if as_list:
        return {output_column: pa.array(arr.tolist(), type=pa.list_(pa.float64()))}
    if output_columns is not None:
        names = list(output_columns)
        if len(names) != width:
            raise PlanError(
                f"output_columns names {len(names)} column(s) but the model returned {width} "
                f"output(s) per row. Pass {width} names, or leave output_columns unset."
            )
    else:
        names = [f"{output_column}_{i}" for i in range(width)]
    return {name: pa.array(arr[:, i]) for i, name in enumerate(names)}


def append_columns(batch: pa.RecordBatch, columns: dict[str, pa.Array]) -> pa.RecordBatch:
    """Append (or replace) `columns` on `batch`, preserving the input schema order.

    Args:
        batch: The batch to extend.
        columns: The ``{name: array}`` columns to add; an existing name is replaced
            in place rather than duplicated.

    Returns:
        A new `pyarrow.RecordBatch` carrying the added columns.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.ml.tabular.features import append_columns
            >>> batch = pa.RecordBatch.from_pydict({"a": [1]})
            >>> append_columns(batch, {"p": pa.array([0.5])}).schema.names
            ['a', 'p']
    """
    out = batch
    for name, array in columns.items():
        if name in out.schema.names:
            out = out.set_column(out.schema.get_field_index(name), name, array)
        else:
            out = out.append_column(name, array)
    return out
