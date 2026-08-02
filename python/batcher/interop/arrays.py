"""Arrow columns as NumPy / PyTorch arrays — the primitives every framework bridge shares.

The lowest layer of the Arrow-to-framework conversion: one `RecordBatch` becomes a dict of
column arrays, zero-copy where the column allows it, with a fixed-shape-tensor or
fixed-size-list column restored to its full `(n, width...)` array rather than left as an
opaque per-row object array.

It lives here, below every subsystem, because three unrelated callers need it and one of them
could not reach the others. `ml.converters` uses it for the training-loop bridge (`to_torch`,
`to_tf`), `interop.formats` for `map_batches(batch_format=...)`, and `core.udf` for the
per-batch conversion around a user function — and `core` is a subsystem that must not import
the user-facing `ml` package, which is where all of this used to live. That import was one of
eighteen upward edges the layered-architecture contract found.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

from batcher._internal.optional import require

if TYPE_CHECKING:
    import numpy as np
    import pyarrow as pa

__all__ = ["arrays_to_torch", "to_numpy_batches"]


def _require_torch() -> Any:
    """Import ``torch`` or raise a `MissingDependencyError` naming the install command."""
    return require("torch", feature="PyTorch conversion", provides="torch", extra="torch")


def _worker_stride() -> tuple[int, int]:
    """This DataLoader worker's ``(offset, stride)`` over a batch sequence.

    ``(0, 1)`` outside a worker process — a plain loop, or ``num_workers=0``.

    An `IterableDataset` is *replicated* into every DataLoader worker process, and each one
    runs ``__iter__`` in full. So a dataset that ignores `get_worker_info` yields its entire
    sequence once per worker: ``DataLoader(ds, num_workers=4)`` — the ordinary thing to write —
    silently trains on every sample **four times per epoch**, with no error and no warning. The
    loss curve just quietly means something else.

    Striding by ``(id, num_workers)`` partitions the batches instead: the union across workers
    is exactly the rank's sequence, each batch produced by exactly one worker, order preserved.
    Batches are strided (not rows) so a worker skips the shard read and the tensorize for the
    batches it does not own, which is what makes `num_workers` buy anything.
    """
    try:
        from torch.utils.data import get_worker_info
    except ImportError:  # torch is an optional extra; a non-torch caller is single-stream.
        return 0, 1
    info = get_worker_info()
    if info is None or info.num_workers <= 1:
        return 0, 1
    return int(info.id), int(info.num_workers)


def _warn_dropped_non_numeric(arrays: dict[str, np.ndarray], kinds: str = "biufc") -> None:
    """Warn about non-numeric columns a tensor conversion silently drops.

    The loader warns about exactly this (`loader.tensors.warn_dropped_columns`); the
    converters used to drop a string ``label``/``id`` with no signal, so a training loop
    read a `KeyError` or, worse, trained on the features with the target column gone.
    """
    dropped = sorted(n for n, a in arrays.items() if a.dtype.kind not in kinds)
    if dropped:
        import warnings

        warnings.warn(
            f"converter is dropping non-numeric column(s) {dropped}: they cannot become "
            "tensors and will be missing from every batch. Pass `columns=` to select the "
            "numeric columns you want.",
            UserWarning,
            stacklevel=3,
        )


def arrays_to_torch(arrays: dict[str, np.ndarray], *, zero_copy: bool = False) -> dict[str, Any]:
    """Convert a `{column: np.ndarray}` dict to `{column: torch.Tensor}`.

    Only numeric columns (``bool``/``int``/``uint``/``float``/``complex``) convert;
    others are dropped (move text/ids through the engine, not the trainer hot path).

    By default each tensor owns a **writable** copy — a training loop mutates batches
    in place and the Arrow-backed buffer is read-only (torch's "undefined behavior"
    warning). Set `zero_copy=True` for **read-only inference** to skip that copy: the
    tensor is a DLPack view sharing the Arrow buffer (one fewer CPU copy before a
    ``.to(device)``), so do not mutate it. Falls back to a copy for buffers DLPack
    can't view (non-contiguous, unsupported dtype).

    Args:
        arrays: a column-name → NumPy array dict (e.g. one `to_numpy_batches` item).
        zero_copy: hand the Arrow/NumPy buffer to torch via DLPack (read-only).

    Returns:
        A `{column: torch.Tensor}` dict over the numeric columns. Requires `torch`.
    """
    torch = _require_torch()

    def _convert(array: np.ndarray) -> Any:
        if zero_copy:
            try:
                return torch.from_dlpack(array)  # zero-copy view (read-only)
            except (TypeError, RuntimeError, BufferError, ValueError):
                pass  # non-contiguous / unsupported → fall through to a copy
        # Own writable memory decoupled from any read-only Arrow buffer — but avoid a
        # redundant copy when the array is ALREADY an owned, writable, contiguous buffer
        # (e.g. the result of a shuffle's fancy-index gather, or a decoded batch): a training
        # loop may mutate it in place safely, and skipping the copy roughly halves the
        # tensorize cost of a wide feature column.
        if array.flags.owndata and array.flags.writeable and array.flags.c_contiguous:
            return torch.from_numpy(array)
        import numpy as np

        return torch.from_numpy(np.ascontiguousarray(array).copy())

    return {name: _convert(array) for name, array in arrays.items() if array.dtype.kind in "biufc"}


def to_numpy_batches(
    batches: Iterable[pa.RecordBatch],
    *,
    columns: Sequence[str] | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    """Convert each Arrow batch to a `{column: np.ndarray}` dict.

    Numeric, non-null columns convert zero-copy; nullable/string columns copy.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.ml import to_numpy_batches
            >>> batch = pa.record_batch({"x": [1, 2], "y": [3.0, 4.0]})
            >>> next(to_numpy_batches([batch], columns=["x"]))
            {'x': array([1, 2])}

    Args:
        batches: an iterable of `pyarrow.RecordBatch`.
        columns: optional subset of column names to keep (default: all).

    Yields:
        One dict per input batch, mapping column name to its NumPy array.
    """
    for batch in batches:
        names = list(batch.schema.names) if columns is None else list(columns)
        yield {name: _column_to_numpy(batch.column(name)) for name in names}


def _column_to_numpy(column: pa.Array) -> np.ndarray:
    """One Arrow column → NumPy, restoring a fixed-shape-tensor **or** fixed-size-list
    numeric column to its full ``(n, width...)`` array (image/embedding/feature-vector
    columns) rather than an opaque per-row object array — so a feature/embedding column
    feeds a training loop as a real 2-D tensor instead of being silently dropped. Plain
    columns convert as before (zero-copy where possible)."""
    import pyarrow as pa

    from batcher.io.formats.ml.tensor import is_tensor_column

    arr = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if is_tensor_column(arr):
        return arr.to_numpy_ndarray()  # (n, *shape), shape from the tensor type
    if pa.types.is_fixed_size_list(arr.type) and pa.types.is_primitive(arr.type.value_type):
        w = arr.type.list_size
        n = len(arr)
        # Slice the child buffer by (offset, length) rather than `flatten()`, which *drops*
        # null rows — that would return fewer than `n` rows and silently misalign this
        # column against its siblings (a feature row falling out from under its label). The
        # offset-aware slice keeps every row; a null row surfaces as NaN via `to_numpy`.
        child = arr.values.slice(arr.offset * w, n * w).to_numpy(zero_copy_only=False)
        if child.dtype.kind in "biufc":  # numeric child → (n, W); else fall through
            return child.reshape(n, w)
    return arr.to_numpy(zero_copy_only=False)
