"""`batch_format` conversion for `map_batches` — Arrow ↔ numpy / pandas / torch.

`map_batches` always crosses the engine boundary as Arrow (zero-copy). `batch_format`
lets the user's `fn` instead *receive and return* NumPy / pandas / PyTorch — the Ray
Data parity surface — by converting **only around the per-batch call**; the data plane
stays Arrow. The NumPy/torch directions reuse `ml.converters` so the Arrow↔framework
tensor logic has a single home.

A non-Arrow `fn` result reduces to something `core.udf._coerce_udf_result` already
turns into batches (a column dict via ``from_pydict``, or a `RecordBatch`), so the
result path stays one normalizer, not four.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher.ml.converters import arrays_to_torch, to_numpy_batches

__all__ = ["FORMATS", "result_to_arrowable", "to_format"]

#: The batch formats a `map_batches` `fn` may speak.
FORMATS = ("pyarrow", "numpy", "pandas", "torch")


def to_format(batch: pa.RecordBatch, fmt: str) -> Any:
    """Convert one Arrow `RecordBatch` to the object `fn` should receive.

    ``pyarrow`` returns the batch unchanged; ``numpy`` a ``{col: ndarray}`` dict;
    ``pandas`` a ``DataFrame``; ``torch`` a ``{col: tensor}`` dict (numeric columns
    only). Requires `pandas`/`torch` for those formats.

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
    raise ValueError(f"unknown batch_format {fmt!r}; expected one of {FORMATS}")


def result_to_arrowable(result: Any, fmt: str) -> Any:
    """Normalize a `fn` result to a value `core.udf._coerce_udf_result` accepts.

    Reduces a NumPy/torch column dict or a pandas frame to a `RecordBatch`/`Table`/
    column-dict; ``pyarrow`` results pass through untouched.
    """
    if fmt == "pyarrow":
        return result
    if fmt == "numpy":
        return result  # a {col: ndarray} dict — from_pydict handles ndarrays
    if fmt == "pandas":
        return pa.RecordBatch.from_pandas(result, preserve_index=False)
    if fmt == "torch":
        return {name: tensor.detach().cpu().numpy() for name, tensor in result.items()}
    raise ValueError(f"unknown batch_format {fmt!r}; expected one of {FORMATS}")
