"""The indexed loaders: a deterministic, balanced, resumable global sample order per rank.

Both entry points here wrap `streaming_sampler`'s ordering contract in a
``torch.utils.data.IterableDataset``. They differ only in memory regime — `stream_loader`
streams the corpus once and retains this rank's shard of it (see `sharding`);
`shard_stream_loader` never materializes anything, computing its indices one batch at a time
from the keyed permutation. Same sample order either way.

Both stride their batch sequence across DataLoader workers (`_worker_stride`); an
`IterableDataset` that does not is replayed *in full* by every worker, so an epoch silently sees
each sample `num_workers` times.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.converters import _worker_stride
from batcher.ml.loader.sharding import gather_rank_shard
from batcher.ml.loader.tensors import tensorize, warn_dropped_columns
from batcher.ml.streaming_sampler import num_rank_batches, rank_index_batches

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["shard_stream_loader", "stream_loader"]


def _check_batch_size(batch_size: int) -> int:
    """Validate a loader `batch_size`, raising rather than silently coercing.

    The sampler primitives these loaders wrap already reject ``batch_size < 1``; a
    silent ``max(1, batch_size)`` here would instead turn a typo'd ``batch_size=0`` into
    an epoch of one-row batches — correct output, but catastrophically slow, with no
    error to point at.
    """
    if batch_size < 1:
        raise PlanError(f"batch_size must be >= 1, got {batch_size}")
    return batch_size


def stream_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    world_size: int = 1,
    rank: int = 0,
    epoch: int = 0,
    seed: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
    columns: list[str] | None = None,
    global_consumed: int = 0,
    collate_fn: Any = None,
) -> Any:
    """A `torch.utils.data.IterableDataset` of ``{column: tensor}`` batches for `rank`.

    The corpus is read once as a batch stream and only **this rank's shard** is retained, so
    peak driver memory is ``num_rows / world_size`` rows rather than the whole corpus per rank.
    Past that, write the corpus with `batcher.io.formats.ml.write_shards` and use
    `shard_stream_loader`: bounded memory, identical sample order.

    The dataset is read twice — once for `Dataset.count` (answered from metadata where the
    engine can) and once for the rows — so it must be deterministic in row order, which is the
    same assumption the shuffled global order already makes.

    Examples:
        .. doctest::

            >>> from batcher.ml import stream_loader  # doctest: +SKIP
            >>> loader = stream_loader(ds, batch_size=32, world_size=8, rank=0)  # doctest: +SKIP
            >>> for tensors in loader:  # doctest: +SKIP
            ...     step(tensors["features"], tensors["label"])

    Args:
        dataset: a bounded Batcher `Dataset` (streamed once; only this rank's rows are kept).
        batch_size: rows per yielded batch.
        world_size: ranks in the data-parallel group.
        rank: this process's slot in that group.
        epoch: selects this epoch's global order (with `seed`).
        seed: keys the permutation.
        shuffle: iterate in corpus order instead when false.
        drop_last: drop the epoch's tail remainder, else keep it and pad back up to a whole
            number of ranks. Either way every rank yields the same batch count, so no rank
            stalls the DDP barrier.
        columns: subset to tensorize (default: all tensorizable columns).
        global_consumed: samples already processed this epoch (resume point).
        collate_fn: optional ``pyarrow.Table -> Any`` collation applied to each batch instead
            of the default tensorization. The escape hatch for columns that cannot become
            tensors (string labels/ids) and for ragged sequences needing padding and an
            attention mask.

    Returns:
        A `torch.utils.data.IterableDataset` of this rank's ``{column: tensor}`` batches.
    """
    from batcher._internal.optional import require

    IterableDataset = require(
        "torch.utils.data",
        "IterableDataset",
        feature="stream_loader",
        provides="torch",
        extra="torch",
    )

    bs = _check_batch_size(batch_size)
    # Gathered eagerly, here in the parent process, so the DataLoader workers forked below
    # inherit the shard instead of each re-reading the corpus.
    table, order = gather_rank_shard(
        dataset,
        num_rows=dataset.count(),
        world_size=world_size,
        rank=rank,
        epoch=epoch,
        seed=seed,
        shuffle=shuffle,
        drop_last=drop_last,
        global_consumed=global_consumed,
    )
    keep = list(columns) if columns is not None else list(dataset.columns)
    if table.num_rows:
        warn_dropped_columns(table.slice(0, 1), keep, collate_fn)

    class _StreamLoader(IterableDataset):  # type: ignore[misc]
        def __len__(self) -> int:
            # The rank's total across all DataLoader workers, not one worker's share — which
            # is what a training loop wants for its step count.
            return len(order) // bs if drop_last else (len(order) + bs - 1) // bs

        def __iter__(self):
            import pyarrow as pa

            n = len(order)
            limit = (n // bs) * bs if drop_last else n
            # Stride before `take`, so a worker pays no gather for a batch it does not own.
            offset, stride = _worker_stride()
            for i, start in enumerate(range(0, limit, bs)):
                if i % stride == offset:
                    rows = pa.array(order[start : start + bs])
                    yield tensorize(table.take(rows), keep, collate_fn)

    return _StreamLoader()


def shard_stream_loader(
    directory: str,
    *,
    batch_size: int,
    world_size: int = 1,
    rank: int = 0,
    epoch: int = 0,
    seed: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
    columns: list[str] | None = None,
    global_consumed: int = 0,
    cache_size: int = 4,
    collate_fn: Any = None,
) -> Any:
    """A bounded-memory `IterableDataset` streaming a **shard directory** written by Batcher.

    Same sample-order contract as `stream_loader`, but the corpus is never materialized: the
    row count comes from the shard index (written by `batcher.io.formats.ml.write_shards`) and
    each batch is gathered through a `ShardReader` whose LRU cache keeps at most `cache_size`
    shards resident — so it scales past RAM, the case `stream_loader` can't handle.

    Examples:
        .. doctest::

            >>> from batcher.ml import shard_stream_loader  # doctest: +SKIP
            >>> loader = shard_stream_loader("/data/shards", batch_size=32)  # doctest: +SKIP
            >>> len(loader)  # batches this rank yields  # doctest: +SKIP

    Args:
        directory: the shard directory to stream (holds the shards and their index).
        batch_size: rows per yielded batch.
        world_size: ranks in the data-parallel group.
        rank: this process's slot in that group.
        epoch: selects this epoch's global order (with `seed`).
        seed: keys the permutation.
        shuffle: iterate in corpus order instead when false.
        drop_last: drop the epoch's tail remainder and any partial final batch.
        columns: subset to tensorize (default: all tensorizable columns).
        global_consumed: samples already processed this epoch (resume point).
        cache_size: shards kept resident in the reader's LRU cache.
        collate_fn: optional ``pyarrow.Table -> Any`` collation applied to each batch instead
            of the default tensorization, for columns that cannot become tensors and for
            ragged sequences needing padding and an attention mask.

    Returns:
        A `torch.utils.data.IterableDataset` of this rank's ``{column: tensor}`` batches.
    """
    from batcher._internal.optional import require

    IterableDataset = require(
        "torch.utils.data",
        "IterableDataset",
        feature="stream_loader",
        provides="torch",
        extra="torch",
    )

    from batcher.io.formats.ml.shards import ShardReader

    reader = ShardReader(directory, cache_size=cache_size)
    probe = reader.take([0])
    keep = list(columns) if columns is not None else list(probe.column_names)
    warn_dropped_columns(probe, keep, collate_fn)
    bs = _check_batch_size(batch_size)
    # The larger-than-RAM path, so the *index* sequence must not be materialized either: a
    # shuffled list of every index costs ~28 bytes/sample (280 GB at 10^10 samples).
    # `rank_index_batches` computes this rank's indices one batch at a time from a keyed
    # permutation — identical order, O(batch_size) memory.
    sampler_kwargs: dict[str, Any] = {
        "batch_size": bs,
        "world_size": world_size,
        "rank": rank,
        "epoch": epoch,
        "seed": seed,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "global_consumed": global_consumed,
    }

    class _ShardLoader(IterableDataset):  # type: ignore[misc]
        def __len__(self) -> int:
            return num_rank_batches(
                reader.total_rows,
                batch_size=bs,
                world_size=world_size,
                rank=rank,
                drop_last=drop_last,
                global_consumed=global_consumed,
            )

        def __iter__(self):
            # Stride the *index* generator, so a worker never issues the shard read for a
            # batch another worker owns. `rank_index_batches` is O(1) memory, so advancing
            # past a skipped batch costs arithmetic, not I/O.
            offset, stride = _worker_stride()
            for i, indices in enumerate(rank_index_batches(reader.total_rows, **sampler_kwargs)):
                if i % stride == offset:
                    yield tensorize(reader.take(indices), keep, collate_fn)

    return _ShardLoader()
