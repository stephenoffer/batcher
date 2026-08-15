"""Framework converters — hand Arrow batches to NumPy / PyTorch training loops.

The array-level primitives moved to `interop.arrays`, which sits below every subsystem: the
executor and the distributed map path need the same conversion, and neither may import this
package. `arrays_to_torch` and `to_numpy_batches` are re-exported here, so the documented
import path is unchanged.

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

# Re-exported, not merely imported: `batcher.ml.converters._column_to_numpy` and its siblings
# are existing import paths (tests and `ml.serving` reach for them), so the move down into
# `interop` has to leave them resolvable here. `__all__` covers the public two; the private
# ones are named in `_REEXPORTED` so a linter cannot decide they are unused and delete them.
from batcher.interop.arrays import (
    _column_to_numpy,
    _require_torch,
    _warn_dropped_non_numeric,
    _worker_stride,
    arrays_to_torch,
    to_numpy_batches,
)

if TYPE_CHECKING:
    import pyarrow as pa

#: Names this module re-exports from `interop.arrays` purely to preserve import paths. Held in
#: a tuple so `F401` sees them used; they are not part of the public surface.
_REEXPORTED = (_column_to_numpy, _require_torch, _warn_dropped_non_numeric, _worker_stride)

__all__ = [
    "arrays_to_torch",
    "tf_dataset_from_arrays",
    "to_numpy_batches",
    "to_tf",
    "to_tf_dataset",
    "to_torch",
    "to_torch_dataloader",
    "to_torch_iterable",
]


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
    return tf_dataset_from_arrays(to_numpy_batches(batches, columns=columns))


def tf_dataset_from_arrays(arrays: Iterable[dict[str, Any]]) -> Any:
    """Wrap a stream of ``{column: ndarray}`` batches as a ``tf.data.Dataset``.

    Split out from `to_tf_dataset` so the TensorFlow path can consume the *same* prepared
    stream the PyTorch one does (`batcher.ml.loader.lazy.numpy_batch_stream`) — shuffled,
    sized, and tailed by shared code — instead of being the one loader in the package with
    none of those options.

    Args:
        arrays: an iterable of ``{column: numpy.ndarray}`` batches.

    Returns:
        A ``tf.data.Dataset`` yielding one `{column: tensor}` dict per batch.

    Raises:
        MissingDependencyError: if `tensorflow` is not installed (naming ``pip install
            'batcher-engine[tensorflow]'``).
    """
    tf = require(
        "tensorflow", feature="TensorFlow conversion", provides="tensorflow", extra="tensorflow"
    )

    def _numeric(batch: dict[str, Any]) -> dict[str, Any]:
        return {n: a for n, a in batch.items() if a.dtype.kind in "biuf"}

    # ONE iterator, advanced exactly once for the signature probe and then resumed by the
    # generator. Probing with a second pass consumed batch 0 out of the one-shot source
    # (`ds.iter_batches()`) and `from_generator` re-entered the already-advanced iterator,
    # so every TF training run silently lost its first batch.
    remaining = iter(arrays)
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
            for batch in remaining:
                yield _numeric(batch)

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
