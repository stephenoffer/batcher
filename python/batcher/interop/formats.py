"""`batch_format` conversion for `map_batches` — Arrow ↔ numpy / pandas / torch.

`map_batches` always crosses the engine boundary as Arrow (zero-copy). `batch_format`
lets the user's `fn` instead *receive and return* NumPy / pandas / PyTorch — the Ray
Data parity surface — by converting **only around the per-batch call**; the data plane
stays Arrow. The NumPy/torch directions reuse `interop.arrays` so the Arrow↔framework
tensor logic has a single home.

It sits in `interop` rather than in `ml` because its callers are the *executor*:
`core.udf.{apply,call,processes}` convert around every user function, and `core` is a
subsystem forbidden from importing the user-facing `ml` package. `ml.batch_format` re-exports
these names, so the public import path is unchanged.

A non-Arrow `fn` result reduces to something `core.udf.call._coerce_udf_result` already
turns into batches (a column dict via ``from_pydict``, or a `RecordBatch`), so the
result path stays one normalizer, not four.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher.interop.arrays import arrays_to_torch, to_numpy_batches

__all__ = ["FORMATS", "result_to_arrowable", "to_format"]

#: The batch formats a `map_batches` `fn` may speak. ``polars`` is Arrow-native (near
#: zero-copy) and ``jax`` reuses the numpy path — the two most-requested Ray Data / Daft
#: parity formats beyond the originals.
FORMATS = ("pyarrow", "numpy", "pandas", "torch", "polars", "jax")


def to_format(batch: pa.RecordBatch, fmt: str) -> Any:
    """Convert one Arrow `RecordBatch` to the object `fn` should receive.

    ``pyarrow`` returns the batch unchanged; ``numpy`` a ``{col: ndarray}`` dict;
    ``pandas`` a ``DataFrame``; ``torch`` a ``{col: tensor}`` dict (numeric columns only);
    ``polars`` a ``polars.DataFrame`` (Arrow-native, near zero-copy); ``jax`` a
    ``{col: jax.Array}`` dict (numeric columns only). Requires the matching library.

    **Tensor columns and `polars`.** The `pyarrow`, `numpy`, and `pandas` paths all hand a
    tensor column over with its per-row shape intact. Polars has no dtype for one, so
    `pl.from_arrow` reads the canonical extension type as its flat storage: a ``(3, 3)`` image
    arrives as a 9-element list and comes back as one. The values are unharmed and the shape
    is not; use ``"numpy"`` or ``"pandas"`` for a `fn` that needs it.

    Raises:
        ValueError: if `fmt` is not one of `FORMATS`.
    """
    if fmt == "pyarrow":
        return batch
    if fmt == "numpy":
        return next(to_numpy_batches([batch]))
    if fmt == "pandas":
        return _to_pandas(batch)
    if fmt == "torch":
        arrays = next(to_numpy_batches([batch]))
        _warn_dropped(arrays, fmt)
        return arrays_to_torch(arrays)
    if fmt == "polars":
        pl = _require_polars()
        return pl.from_arrow(pa.Table.from_batches([batch]))
    if fmt == "jax":
        jnp = _require_jax()
        arrays = next(to_numpy_batches([batch]))
        _warn_dropped(arrays, fmt)
        return {n: jnp.asarray(a) for n, a in arrays.items() if a.dtype.kind in "biufc"}
    raise ValueError(f"unknown batch_format {fmt!r}; expected one of {FORMATS}")


def _to_pandas(batch: pa.RecordBatch) -> Any:
    """`batch` as a `DataFrame`, with tensor columns kept shaped.

    ``RecordBatch.to_pandas()`` on a fixed-shape tensor column hands back an object column of
    **flat** arrays: a ``(3, 3)`` image arrives as a 9-element vector, and the shape is gone
    before the user's `fn` is called. A `fn` that reshapes by hand is guessing, and one that
    does not silently computes on a vector. The result path then infers a plain list column
    from those flat arrays, so the shape never comes back either.

    Overwriting those columns with their real per-row arrays fixes both ends at once: the
    `fn` sees ``(3, 3)``, and returning them unchanged rebuilds the tensor column.
    """
    from batcher.interop.arrays import _column_to_numpy
    from batcher.io.formats.ml.ragged import is_ragged_tensor_column
    from batcher.io.formats.ml.tensor import is_tensor_column

    frame = batch.to_pandas()
    # Resolved once per batch, not once per column: this runs on every `batch_format="pandas"`
    # call, and most batches have no tensor column at all.
    for name in batch.schema.names:
        column = batch.column(name)
        if is_tensor_column(column) or is_ragged_tensor_column(column):
            frame[name] = list(_column_to_numpy(column))
    return frame


#: Column sets already warned about, so a dropped column is reported once per stage rather
#: than once per batch (a 10,000-batch scan would otherwise emit 10,000 identical warnings).
_WARNED_DROPS: set[tuple[str, tuple[str, ...]]] = set()


def _warn_dropped(arrays: dict[str, Any], fmt: str) -> None:
    """Warn once that a tensor `batch_format` is dropping this batch's non-numeric columns.

    ``batch_format="torch"`` (and ``"jax"``) can only hand a `fn` numeric columns, so a
    string ``id``/``label``/``caption`` alongside the features vanished from the dict with no
    signal at all — the `fn` then either raised a `KeyError` far from its cause, or, worse,
    ran fine and wrote a result with the identifying column silently gone. The loader path
    has warned about exactly this since it was written (`converters._warn_dropped_non_numeric`);
    the `map_batches` path did not, which is the more common way to meet it.
    """
    dropped = tuple(sorted(n for n, a in arrays.items() if a.dtype.kind not in "biufc"))
    if not dropped or (fmt, dropped) in _WARNED_DROPS:
        return
    _WARNED_DROPS.add((fmt, dropped))
    import warnings

    warnings.warn(
        f"batch_format={fmt!r} cannot represent {_why_dropped(arrays, dropped)} "
        f"{list(dropped)}, so the function will not receive them and they will be missing "
        f"from its output. Keep them by using batch_format='pyarrow' (or 'pandas'), or select "
        f"the numeric columns explicitly with `input_columns=` so the drop is intentional.",
        UserWarning,
        stacklevel=4,
    )


def _why_dropped(arrays: dict[str, Any], dropped: tuple[str, ...]) -> str:
    """Name the *reason* a column is being dropped, not just the fact.

    "non-numeric" is accurate for a string id and misleading for a variable-shape tensor
    column, which is numeric in every sense the user cares about and is dropped because rows
    of differing shape have no single tensor to become. A reader told "non-numeric" about an
    image column goes looking for the wrong thing.
    """
    import numpy as np

    def holds_arrays(name: str) -> bool:
        column = arrays[name]
        return column.dtype == object and any(isinstance(v, np.ndarray) for v in column[:1])

    ragged = [name for name in dropped if holds_arrays(name)]
    if ragged and len(ragged) == len(dropped):
        return "variable-shape tensor column(s) — rows of differing shape have no one tensor —"
    return "non-numeric column(s)"


def result_to_arrowable(result: Any, fmt: str) -> Any:
    """Normalize a `fn` result to a value `core.udf.call._coerce_udf_result` accepts.

    Reduces a NumPy/torch/jax column dict, a pandas frame, or a polars frame to a
    `RecordBatch`/`Table`/column-dict; ``pyarrow`` results pass through untouched.
    """
    if fmt == "pyarrow":
        return result
    if fmt == "numpy":
        return result  # a {col: ndarray} dict — from_pydict handles ndarrays
    if fmt == "pandas":
        return _pandas_result(result)
    if fmt == "torch":
        return _tensors_to_numpy(result)
    if fmt == "polars":
        return _polars_result(result)
    if fmt == "jax":
        return _tensors_to_numpy(result)
    raise ValueError(f"unknown batch_format {fmt!r}; expected one of {FORMATS}")


def _pandas_result(result: Any) -> Any:
    """A ``batch_format="pandas"`` result reduced to something the Arrow normalizer accepts.

    A frame is the documented return, but the natural one-column transform is
    ``df["x"] * 2`` — a `Series`. That used to reach `RecordBatch.from_pandas`, which asks
    for ``.columns`` and fails with ``'Series' object has no attribute 'columns'``: an error
    that names neither `map_batches` nor pandas nor the fix. A named `Series` becomes its own
    one-column frame; an Arrow object or a plain dict passes straight through, so a `fn` that
    finds it easier to build the result some other way is not forced back into pandas.
    """
    if isinstance(result, pa.RecordBatch | pa.Table | dict):
        return result
    if type(result).__name__ == "Series" and type(result).__module__.startswith("pandas"):
        if result.name is None:
            raise ValueError(
                "map_batches with batch_format='pandas' returned an unnamed Series; Arrow needs "
                "a column name. Return a DataFrame, or name it with `series.rename('col')`."
            )
        return {str(result.name): result.to_numpy()}
    if _holds_arrays(result):
        # A tensor column has no pandas dtype, so it arrives as an object column of ndarrays
        # and `from_pandas` infers a plain list column from it — losing the shape, and (via
        # the boundary's widening of narrow numerics) turning a uint8 image into int64, eight
        # bytes a pixel. Handing back a column dict routes it through the same tensorization
        # the `pyarrow` and `numpy` paths use, which rebuilds the tensor column instead.
        return {str(name): result[name].to_numpy() for name in result.columns}
    return pa.RecordBatch.from_pandas(result, preserve_index=False)


def _holds_arrays(frame: Any) -> bool:
    """Whether any column of a pandas `frame` holds NumPy arrays rather than scalars.

    Checked on the first value of each object column only: it is `O(columns)` per batch, and a
    frame whose column changes from scalars to arrays partway down is not the case this is for.
    """
    import numpy as np

    for name in frame.columns:
        column = frame[name]
        if column.dtype != object or not len(column):
            continue
        if isinstance(column.iloc[0], np.ndarray):
            return True
    return False


def _polars_result(result: Any) -> Any:
    """A ``batch_format="polars"`` result reduced the same way `_pandas_result` reduces pandas.

    A polars `Series` is the one-column analog of the pandas case and always carries a name,
    so it converts unconditionally.
    """
    if isinstance(result, pa.RecordBatch | pa.Table | dict):
        return result
    if type(result).__name__ == "Series" and type(result).__module__.startswith("polars"):
        return result.to_frame().to_arrow()
    return result.to_arrow() if hasattr(result, "to_arrow") else result


def _tensors_to_numpy(result: Any) -> Any:
    """A ``{col: tensor}`` result reduced to ``{col: ndarray}``, tolerant of stray values.

    A torch/jax UDF may return a dict mixing tensors with a plain ndarray or a Python
    scalar, or a bare tensor. The old ``tensor.detach()`` on every value crashed on the
    first non-tensor; this coerces per value (tensor → host ndarray, everything else
    unchanged) and wraps a bare tensor, so a valid-but-unusual return no longer aborts far
    from the user's `fn`.
    """
    import numpy as np

    def to_np(value: Any) -> Any:
        if hasattr(value, "detach"):  # torch tensor (possibly on device / requiring grad)
            return value.detach().cpu().numpy()
        if type(value).__module__.startswith("jax"):  # jax.Array
            return np.asarray(value)
        return value

    if isinstance(result, dict):
        return {name: to_np(value) for name, value in result.items()}
    return to_np(result)


def _require_polars() -> Any:
    from batcher._internal.optional import require

    return require("polars", feature="batch_format='polars'", provides="polars", extra="polars")


def _require_jax() -> Any:
    from batcher._internal.optional import require

    return require("jax.numpy", feature="batch_format='jax'", provides="jax", extra="jax")
