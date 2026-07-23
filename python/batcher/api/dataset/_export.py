"""Framework-export helpers behind `Dataset.to_torch` / `to_tf` / `to_torch_dataloader`.

These bridge a `Dataset`'s output batches to PyTorch / TensorFlow training loops
via the `Dataset`-free converters in `batcher.ml.converters`. The batch source is
**re-iterable** — each pass re-runs the query — so a multi-epoch loader streams in
bounded memory rather than materializing the whole dataset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from batcher.api.dataset.frame import Dataset


class _ReiterableBatches:
    """A re-iterable view over a dataset's output batches: each ``iter()`` re-runs
    the query, so a training framework can take multiple passes (epochs)."""

    __slots__ = ("_size", "_source")

    def __init__(self, source: Dataset, size: int | None) -> None:
        self._source = source
        self._size = size

    def __iter__(self) -> Any:
        return self._source.iter_batches(self._size)


def to_torch(ds: Dataset, columns: list[str] | None, batch_size: int | None) -> Any:
    """A re-iterable `torch.utils.data.IterableDataset` of per-batch tensor dicts."""
    from batcher.ml.converters import to_torch_iterable

    return to_torch_iterable(_ReiterableBatches(ds, batch_size), columns=columns)


def to_torch_dataloader(
    ds: Dataset, columns: list[str] | None, batch_size: int | None, **dl_kwargs: Any
) -> Any:
    """A `torch.utils.data.DataLoader` over the engine-batched tensor dicts.

    The engine already produces batches, so the loader uses ``batch_size=None``
    (one engine batch = one training batch); pass `batch_size` to size them.
    """
    from torch.utils.data import DataLoader

    return DataLoader(to_torch(ds, columns, batch_size), batch_size=None, **dl_kwargs)


def to_tf(ds: Dataset, columns: list[str] | None, batch_size: int | None) -> Any:
    """A re-iterable ``tf.data.Dataset`` of per-batch tensor dicts."""
    from batcher.ml.converters import to_tf_dataset

    return to_tf_dataset(_ReiterableBatches(ds, batch_size), columns=columns)


def to_jax(ds: Dataset, columns: list[str] | None) -> dict[str, Any]:
    """The full result as a ``{column: jax.Array}`` dict (needs ``jax``).

    The JAX counterpart of `to_numpy`: materializes each column as a NumPy array (tensor
    columns reshaped to ``(n, *shape)``) and wraps it with ``jax.numpy.asarray``. Raises
    `BackendError` if JAX is not installed.
    """
    try:
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - optional extra
        from batcher._internal.errors import BackendError

        msg = "Dataset.to_jax() needs JAX installed (pip install jax)"
        raise BackendError(msg) from exc

    return {name: jnp.asarray(arr) for name, arr in to_numpy(ds, columns).items()}


def to_numpy(ds: Dataset, columns: list[str] | None) -> dict[str, Any]:
    """The full result as a ``{column: numpy.ndarray}`` dict.

    Streams the output batches and concatenates each column, so a fixed-shape-tensor or
    fixed-size-list column (an image/embedding/feature-vector column) comes back as a real
    ``(n, *shape)`` array — not an opaque per-row object array — feeding NumPy / scikit-learn
    directly. Reuses `batcher.ml.to_numpy_batches` so the per-column conversion (tensor
    reshape, null→NaN, zero-copy where possible) matches the training-loader path exactly.
    """
    import numpy as np

    from batcher.ml.converters import to_numpy_batches

    names = list(ds.columns) if columns is None else list(columns)
    parts: dict[str, list[Any]] = {name: [] for name in names}
    for batch in to_numpy_batches(ds.iter_batches(), columns=names):
        for name in names:
            parts[name].append(batch[name])
    return {
        name: (np.concatenate(chunks) if chunks else np.array([])) for name, chunks in parts.items()
    }
