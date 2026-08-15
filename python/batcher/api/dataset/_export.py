"""Framework-export helpers behind `Dataset.to_torch` / `to_tf` / `to_ray_dataset`.

These bridge a `Dataset`'s output batches to PyTorch / TensorFlow training loops
via the `Dataset`-free converters in `batcher.ml.converters`. The batch source is
**re-iterable** — each pass re-runs the query — so a multi-epoch loader streams in
bounded memory rather than materializing the whole dataset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa

from batcher._internal.optional import require
from batcher.plan.types import retained_bytes

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
    jnp = require("jax.numpy", feature="Dataset.to_jax()", provides="JAX", extra="jax")
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


#: Fallback target size for one Ray Data block, used when Ray's own
#: ``DataContext.target_max_block_size`` cannot be read. Matches Ray Data's documented
#: default so a Batcher-produced dataset blocks the same way a `read_parquet` one does.
_RAY_TARGET_BLOCK_BYTES = 128 * 1024 * 1024


def _ray_target_block_bytes() -> int:
    """Ray Data's configured target block size, or the documented default.

    Read from the live `DataContext` rather than hard-coded, so a cluster that tuned
    ``target_max_block_size`` gets blocks that match the rest of its pipeline. Ray moves
    this attribute between releases, so an unreadable context falls back rather than
    failing an export over a tuning knob.

    Anyscale's field guidance puts the usable band at **1 MiB to 128 MiB per block**, and
    Ray's own default sits at the top of it. That is the range a caller overriding
    `block_size_bytes` should stay inside: below it, per-block task overhead dominates and
    the dataset spends its time coalescing; above it, a single block stops fitting the
    object store's slack and spills.
    """
    try:
        from ray.data import DataContext

        size = getattr(DataContext.get_current(), "target_max_block_size", None)
        return int(size) if size else _RAY_TARGET_BLOCK_BYTES
    except Exception:  # pragma: no cover - Ray internals move between releases
        return _RAY_TARGET_BLOCK_BYTES


def _ray_blocks(
    ds: Dataset, batch_size: int | None, block_bytes: int, distributed: bool | str
) -> Any:
    """Yield the query's output as Arrow tables sized near `block_bytes`.

    Engine morsels (16,384 rows) are far smaller than a Ray Data block, and a Ray Dataset
    of tens of thousands of tiny blocks schedules badly — per-block task overhead dominates
    and Ray spends the job coalescing. Consecutive morsels are therefore accumulated until
    they reach the target, which is the same shape `read_parquet` produces. The driver holds
    at most one block at a time, so this stays bounded-memory no matter how large the result.
    """
    pending: list[pa.RecordBatch] = []
    pending_bytes = 0
    for batch in ds.iter_batches(batch_size, distributed=distributed):
        pending.append(batch)
        pending_bytes += retained_bytes(batch)
        if pending_bytes >= block_bytes:
            yield pa.Table.from_batches(pending)
            pending = []
            pending_bytes = 0
    if pending:
        yield pa.Table.from_batches(pending)


def to_ray_dataset(
    ds: Dataset,
    *,
    batch_size: int | None = None,
    block_size_bytes: int | None = None,
    distributed: bool | str = False,
) -> Any:
    """Hand the query's result to Ray Data as a `ray.data.Dataset`.

    The return leg of `bt.from_ray_dataset`: a Batcher result becomes the input of a Ray
    Train / Tune / Serve stage without a round trip through storage. Output batches are
    coalesced into Ray-sized Arrow blocks and put into the object store one block at a
    time, so the driver's footprint is one block rather than the whole result.
    """
    ray = require("ray", feature="Dataset.to_ray_dataset()", provides="Ray", extra="ray")
    require("ray.data", feature="Dataset.to_ray_dataset()", provides="Ray", extra="ray")

    target = int(block_size_bytes) if block_size_bytes else _ray_target_block_bytes()
    refs = [ray.put(block) for block in _ray_blocks(ds, batch_size, target, distributed)]
    if not refs:
        # A zero-batch result still has a schema, and a Ray Dataset built from no blocks
        # has none — `.schema()` returns None and every downstream op fails on a column
        # the user can see in `ds.schema`. One empty block carries it across.
        schema = ds.schema
        empty = pa.Table.from_arrays([pa.array([], type=f.type) for f in schema], schema=schema)
        refs = [ray.put(empty)]
    return ray.data.from_arrow_refs(refs)
