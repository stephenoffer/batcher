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

from batcher._internal.optional import require

if TYPE_CHECKING:
    import numpy as np
    import pyarrow as pa

__all__ = [
    "arrays_to_torch",
    "to_numpy_batches",
    "to_tf",
    "to_tf_dataset",
    "to_torch",
    "to_torch_dataloader",
    "to_torch_iterable",
]


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


def to_torch_iterable(
    batches: Iterable[pa.RecordBatch],
    *,
    columns: Sequence[str] | None = None,
) -> Any:
    """Wrap Arrow batches as a `torch.utils.data.IterableDataset` of tensor dicts.

    Each yielded item is a `{column: torch.Tensor}` dict for one batch; non-numeric
    columns (e.g. strings) are skipped. Requires `torch`. The dataset is single-pass
    over `batches` unless `batches` is itself re-iterable.

    Examples:
        .. doctest::

            >>> import batcher as bt  # doctest: +SKIP
            >>> from batcher.ml import to_torch_iterable  # doctest: +SKIP
            >>> ds = bt.from_pydict({"x": [1, 2, 3]})  # doctest: +SKIP
            >>> loader = to_torch_iterable(ds.iter_batches())  # doctest: +SKIP

    Args:
        batches: an iterable of `pyarrow.RecordBatch`.
        columns: optional subset of column names to keep (default: all).

    Returns:
        A `torch.utils.data.IterableDataset` yielding one tensor dict per batch.

    Raises:
        MissingDependencyError: if `torch` is not installed (naming ``pip install
            'batcher-engine[torch]'``).
    """
    IterableDataset = require(
        "torch.utils.data",
        "IterableDataset",
        feature="PyTorch conversion",
        provides="torch",
        extra="torch",
    )

    source = batches
    select = columns

    class _ArrowIterable(IterableDataset):  # type: ignore[misc]
        def __iter__(self) -> Iterator[dict[str, Any]]:
            offset, stride = _worker_stride()
            warned = False
            for i, arrays in enumerate(to_numpy_batches(source, columns=select)):
                if i % stride == offset:
                    if not warned:  # warn once, on the first batch this worker yields
                        _warn_dropped_non_numeric(arrays)
                        warned = True
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

    Examples:
        .. doctest::

            >>> import batcher as bt  # doctest: +SKIP
            >>> from batcher.ml import to_tf_dataset  # doctest: +SKIP
            >>> ds = bt.from_pydict({"x": [1, 2, 3]})  # doctest: +SKIP
            >>> tf_ds = to_tf_dataset(ds.iter_batches())  # doctest: +SKIP

    Args:
        batches: an iterable of `pyarrow.RecordBatch`.
        columns: optional subset of column names to keep (default: all).

    Returns:
        A ``tf.data.Dataset`` yielding one `{column: tensor}` dict per batch.

    Raises:
        MissingDependencyError: if `tensorflow` is not installed (naming ``pip install
            'batcher-engine[tensorflow]'``).
    """
    tf = require(
        "tensorflow", feature="TensorFlow conversion", provides="tensorflow", extra="tensorflow"
    )

    def _numeric(arrays: dict[str, Any]) -> dict[str, Any]:
        return {n: a for n, a in arrays.items() if a.dtype.kind in "biuf"}

    # ONE iterator, advanced exactly once for the signature probe and then resumed by the
    # generator. Probing with a second `to_numpy_batches(...)` pass consumed batch 0 out of
    # the one-shot source (`ds.iter_batches()`) and `from_generator` re-entered the
    # already-advanced iterator, so every TF training run silently lost its first batch.
    remaining = iter(to_numpy_batches(batches, columns=columns))
    first_arrays = next(remaining, None)
    if first_arrays is None:
        return tf.data.Dataset.from_tensor_slices({})
    _warn_dropped_non_numeric(first_arrays, kinds="biuf")
    first = _numeric(first_arrays)

    def _gen() -> Iterator[dict[str, Any]]:
        # `first` is replayed only on the first pass; the source is one-shot, so a second
        # pass (tf.data re-entering for another epoch) correctly yields nothing.
        if not consumed:
            consumed.append(True)
            yield first
            for arrays in remaining:
                yield _numeric(arrays)

    consumed: list[bool] = []
    # The row axis is dynamic (``None``); every trailing axis is fixed by the column's
    # per-row shape. A feature/embedding column (`(n, W)`) or an image/tensor column
    # (`(n, H, W, C)`) must keep those inner dims, or `from_generator` rejects the yielded
    # element with "shape (n, W) where an element of shape (None,) was expected" — a hard
    # failure on exactly the multi-dimensional columns tf.data is used for.
    sig = {
        name: tf.TensorSpec(shape=(None, *arr.shape[1:]), dtype=tf.dtypes.as_dtype(arr.dtype))
        for name, arr in first.items()
    }
    return tf.data.Dataset.from_generator(_gen, output_signature=sig)


def to_torch_dataloader(
    batches: Iterable[pa.RecordBatch],
    *,
    columns: Sequence[str] | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    **dataloader_kwargs: Any,
) -> Any:
    """Wrap Arrow batches as a ready-to-iterate ``torch.utils.data.DataLoader``.

    The batching is done here (one Arrow batch is one training batch), so the returned
    loader is built with ``batch_size=None`` and simply streams the ready
    ``{column: tensor}`` dicts. ``num_workers`` and ``pin_memory`` are the standard
    PyTorch knobs and pass straight through; with ``num_workers > 0`` the Arrow shards
    are split across worker processes (each yields a disjoint slice, so an epoch sees
    every batch exactly once). Requires `torch`.

    Args:
        batches: an iterable of `pyarrow.RecordBatch`.
        columns: optional subset of column names to keep (default: all).
        num_workers: PyTorch DataLoader worker processes (0 loads in the main process).
        pin_memory: page-lock host buffers for faster host-to-device copies.
        **dataloader_kwargs: further ``DataLoader`` keyword arguments (e.g.
            ``persistent_workers``), forwarded unchanged.

    Returns:
        A ``torch.utils.data.DataLoader`` yielding one ``{column: torch.Tensor}`` batch
        per Arrow batch.

    Examples:
        .. doctest::

            >>> import batcher as bt  # doctest: +SKIP
            >>> from batcher.ml import to_torch_dataloader  # doctest: +SKIP
            >>> ds = bt.from_pydict({"x": [1, 2, 3]})  # doctest: +SKIP
            >>> loader = to_torch_dataloader(ds.iter_batches(), num_workers=2)  # doctest: +SKIP
    """
    from torch.utils.data import DataLoader

    iterable = to_torch_iterable(batches, columns=columns)
    return DataLoader(
        iterable,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **dataloader_kwargs,
    )


def to_torch(
    batches: Iterable[pa.RecordBatch],
    *,
    columns: Sequence[str] | None = None,
) -> Any:
    """Alias of `to_torch_iterable` under the shorter ``to_torch`` name.

    Args:
        batches: an iterable of `pyarrow.RecordBatch`.
        columns: optional subset of column names to keep (default: all).

    Returns:
        A `torch.utils.data.IterableDataset` yielding one tensor dict per batch.

    Examples:
        .. doctest::

            >>> import batcher as bt  # doctest: +SKIP
            >>> from batcher.ml import to_torch  # doctest: +SKIP
            >>> loader = to_torch(bt.from_pydict({"x": [1]}).iter_batches())  # doctest: +SKIP
    """
    return to_torch_iterable(batches, columns=columns)


def to_tf(
    batches: Iterable[pa.RecordBatch],
    *,
    columns: Sequence[str] | None = None,
) -> Any:
    """Alias of `to_tf_dataset` under the shorter ``to_tf`` name.

    Args:
        batches: an iterable of `pyarrow.RecordBatch`.
        columns: optional subset of column names to keep (default: all).

    Returns:
        A ``tf.data.Dataset`` yielding one `{column: tensor}` dict per batch.

    Examples:
        .. doctest::

            >>> import batcher as bt  # doctest: +SKIP
            >>> from batcher.ml import to_tf  # doctest: +SKIP
            >>> tf_ds = to_tf(bt.from_pydict({"x": [1]}).iter_batches())  # doctest: +SKIP
    """
    return to_tf_dataset(batches, columns=columns)
