"""Framework converters — hand Arrow batches to NumPy / PyTorch training loops.

These bridge the engine's Arrow output to ML frameworks without a per-row Python
loop: each whole `RecordBatch` becomes a dict of column arrays (zero-copy for
non-null numeric columns), so a training loop can consume the engine's output
directly. This is the `to_dataloader`/`to_torch_dataset` parity surface, built over
the public batch iterator rather than the `Dataset` internals.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pyarrow as pa

__all__ = ["arrays_to_torch", "to_numpy_batches", "to_tf_dataset", "to_torch_iterable"]


def arrays_to_torch(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Convert a `{column: np.ndarray}` dict to `{column: torch.Tensor}`.

    Only numeric columns (``bool``/``int``/``uint``/``float``/``complex``) convert;
    others are dropped (move text/ids through the engine, not the trainer hot path).
    Each tensor owns a **writable** copy — a training loop mutates batches in place
    and the Arrow-backed buffer is read-only (torch's "undefined behavior" warning).

    Args:
        arrays: a column-name → NumPy array dict (e.g. one `to_numpy_batches` item).

    Returns:
        A `{column: torch.Tensor}` dict over the numeric columns. Requires `torch`.
    """
    import torch

    return {
        name: torch.from_numpy(array.copy())
        for name, array in arrays.items()
        if array.dtype.kind in "biufc"
    }


def to_numpy_batches(
    batches: Iterable[pa.RecordBatch],
    *,
    columns: Sequence[str] | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    """Convert each Arrow batch to a `{column: np.ndarray}` dict.

    Numeric, non-null columns convert zero-copy; nullable/string columns copy.

    Args:
        batches: an iterable of `pyarrow.RecordBatch`.
        columns: optional subset of column names to keep (default: all).

    Yields:
        One dict per input batch, mapping column name to its NumPy array.
    """
    for batch in batches:
        names = list(batch.schema.names) if columns is None else list(columns)
        yield {name: batch.column(name).to_numpy(zero_copy_only=False) for name in names}


def to_torch_iterable(
    batches: Iterable[pa.RecordBatch],
    *,
    columns: Sequence[str] | None = None,
) -> Any:
    """Wrap Arrow batches as a `torch.utils.data.IterableDataset` of tensor dicts.

    Each yielded item is a `{column: torch.Tensor}` dict for one batch; non-numeric
    columns (e.g. strings) are skipped. Requires `torch`. The dataset is single-pass
    over `batches` unless `batches` is itself re-iterable.

    Raises:
        ImportError: if `torch` is not installed.
    """
    from torch.utils.data import IterableDataset

    source = batches
    select = columns

    class _ArrowIterable(IterableDataset):  # type: ignore[misc]
        def __iter__(self) -> Iterator[dict[str, Any]]:
            for arrays in to_numpy_batches(source, columns=select):
                yield arrays_to_torch(arrays)

    return _ArrowIterable()


def to_tf_dataset(
    batches: Iterable[pa.RecordBatch],
    *,
    columns: Sequence[str] | None = None,
) -> Any:
    """Wrap Arrow batches as a ``tf.data.Dataset`` of `{column: tensor}` dicts.

    Each element is one batch's columns as TensorFlow tensors; non-numeric columns
    (e.g. strings) are skipped. Requires `tensorflow`. Re-iterable iff `batches` is.

    Raises:
        ImportError: if `tensorflow` is not installed.
    """
    import tensorflow as tf

    source = batches
    select = columns

    def _gen() -> Iterator[dict[str, Any]]:
        for arrays in to_numpy_batches(source, columns=select):
            yield {n: a for n, a in arrays.items() if a.dtype.kind in "biuf"}

    # Probe the first batch to derive the output signature (column names + dtypes).
    first = next(_gen(), None)
    if first is None:
        return tf.data.Dataset.from_tensor_slices({})
    sig = {
        name: tf.TensorSpec(shape=(None,), dtype=tf.dtypes.as_dtype(arr.dtype))
        for name, arr in first.items()
    }
    return tf.data.Dataset.from_generator(_gen, output_signature=sig)
