"""`batch_format` conversion for `map_batches` — Arrow ↔ numpy / pandas / torch.

`map_batches` always crosses the engine boundary as Arrow (zero-copy). `batch_format`
lets the user's `fn` instead *receive and return* NumPy / pandas / PyTorch — the Ray
Data parity surface — by converting **only around the per-batch call**; the data plane
stays Arrow. The NumPy/torch directions reuse `ml.converters` so the Arrow↔framework
tensor logic has a single home.

A non-Arrow `fn` result reduces to something `core.udf.call._coerce_udf_result` already
turns into batches (a column dict via ``from_pydict``, or a `RecordBatch`), so the
result path stays one normalizer, not four.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher.ml.converters import arrays_to_torch, to_numpy_batches

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

    Raises:
        ValueError: if `fmt` is not one of `FORMATS`.
    """
    if fmt == "pyarrow":
        return batch
    if fmt == "numpy":
        return next(to_numpy_batches([batch]))
    if fmt == "pandas":
        return batch.to_pandas()
    if fmt == "torch":
        return arrays_to_torch(next(to_numpy_batches([batch])))
    if fmt == "polars":
        pl = _require_polars()
        return pl.from_arrow(pa.Table.from_batches([batch]))
    if fmt == "jax":
        jnp = _require_jax()
        return {name: jnp.asarray(arr) for name, arr in next(to_numpy_batches([batch])).items()}
    raise ValueError(f"unknown batch_format {fmt!r}; expected one of {FORMATS}")


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
        return pa.RecordBatch.from_pandas(result, preserve_index=False)
    if fmt == "torch":
        return _tensors_to_numpy(result)
    if fmt == "polars":
        return result.to_arrow() if hasattr(result, "to_arrow") else result
    if fmt == "jax":
        return _tensors_to_numpy(result)
    raise ValueError(f"unknown batch_format {fmt!r}; expected one of {FORMATS}")


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
