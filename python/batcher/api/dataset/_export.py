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
