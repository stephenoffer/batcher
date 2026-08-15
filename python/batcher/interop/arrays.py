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

    from batcher.io.formats.ml.ragged import is_ragged_tensor_column, ragged_to_numpy
    from batcher.io.formats.ml.tensor import is_tensor_column

    arr = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if is_tensor_column(arr):
        return arr.to_numpy_ndarray()  # (n, *shape), shape from the tensor type
    if is_ragged_tensor_column(arr):
        # Rows of differing shape have no stacked form, so this is an object array of real
        # per-row `ndarray`s. Without it the caller received the raw `{data, shape}` dicts
        # and had to reassemble the arrays the engine had just taken apart.
        return ragged_to_numpy(arr)
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
    matrix = uniform_list_to_matrix(arr)
    if matrix is not None:
        return matrix
    return _flat_to_numpy(arr)


def _flat_to_numpy(arr: pa.Array) -> np.ndarray:
    """A plain (non-nested) Arrow column as NumPy, keeping a nullable numeric column numeric.

    Arrow has no null in a NumPy integer or boolean array, so `to_numpy` widens a nullable
    integer column to `float64` with NaN — and for a nullable **boolean** column it gives up
    and returns an `object` array instead. Every tensor conversion above this drops a
    non-numeric column, so a boolean column converted cleanly right up until the first batch
    that happened to contain a null, and then **vanished from the batch dict mid-epoch**: the
    training loop reads a `KeyError` on a key it had been reading for an hour, or quietly
    trains on a batch with one feature missing.

    Whether a column can become a tensor is a property of its *type*, not of which rows
    happened to land in this batch. So a nullable boolean is widened the same way a nullable
    integer already is — to `float64` with NaN — and the key set is the same for every batch.
    The widening is announced, because a NaN in a label column is a silent trainer of nothing.
    """
    import pyarrow as pa

    out = arr.to_numpy(zero_copy_only=False)
    if out.dtype.kind in "biufc" or not (
        pa.types.is_boolean(arr.type) or pa.types.is_integer(arr.type)
    ):
        if arr.null_count and pa.types.is_integer(arr.type):
            _warn_null_widening(arr.type)
        return out
    _warn_null_widening(arr.type)
    return arr.cast(pa.float64()).to_numpy(zero_copy_only=False)


def _warn_null_widening(arrow_type: pa.DataType) -> None:
    """Announce a nullable integer/boolean column widening to float with NaN."""
    import warnings

    warnings.warn(
        f"a nullable {arrow_type} column has no NumPy/tensor equivalent, so it converts to "
        "float64 with NaN for the nulls. Cast or fill it first "
        "(`col(...).fill_null(0)`, or `ds.drop_null()`) if a NaN would be trained on.",
        UserWarning,
        stacklevel=4,
    )


def uniform_list_to_matrix(array: pa.Array) -> np.ndarray | None:
    """An `(n, W)` array when every row of a numeric list column is `W` wide, else `None`.

    The same feature matrix `FixedSizeList` carries, stored as a plain `List<T>` — which is
    what `from_pydict`, a Parquet or JSON read, and `collect_list` all produce. Only the
    fixed-size spelling was recognized, so an embedding column built any of those ordinary
    ways became an object array of per-row lists: dropped from every torch batch, with no
    error and no warning, so a training loop simply never saw its features.

    A list column *can* hold a rectangle and usually does — an embedding is 384 or 1536 wide
    on every row, it just carries a type that does not say so. The widths come off the
    offsets, so proving rectangularity is one vectorized subtraction over `n + 1` integers
    rather than a pass over the data.

    A genuinely ragged column, or one with nulls, returns `None` and is handled exactly as
    before. Nulls are excluded deliberately: a null row's span is empty, so nothing here
    could place it in the matrix without sliding every later row up under the wrong label —
    the misalignment the fixed-size path's offset slice exists to prevent.

    Args:
        array: The Arrow column to reshape.

    Returns:
        The `(n, W)` array, or `None` when the column is not a numeric uniform-width list.
    """
    import numpy as np
    import pyarrow as pa

    dtype = array.type
    if not (pa.types.is_list(dtype) or pa.types.is_large_list(dtype)):
        return None
    if not pa.types.is_primitive(dtype.value_type) or array.null_count:
        return None
    offsets = np.asarray(array.offsets, dtype=np.int64)
    if offsets.size < 2:
        return None  # an empty column has no width to agree on
    widths = np.diff(offsets)
    width = int(widths[0])
    if width <= 0 or not bool((widths == width).all()):
        return None  # ragged: no rectangular form exists
    rows = len(array)
    child = array.values.slice(int(offsets[0]), rows * width).to_numpy(zero_copy_only=False)
    if child.dtype.kind not in "biufc":
        return None
    return child.reshape(rows, width)
