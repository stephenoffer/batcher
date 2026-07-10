"""Streaming training-data loader — Batcher feeding PyTorch DDP/FSDP/DeepSpeed.

Wraps the deterministic/elastic/resumable ordering from `streaming_sampler` in a
``torch.utils.data.IterableDataset`` that yields ``{column: tensor}`` batches for one
rank, so it drops straight into a distributed training loop. The guarantees that
matter (and that Ray Train's split iterator lacks):

* **balanced** — every rank yields the same number of batches (the epoch's tail is
  dropped or padded, per ``drop_last``), so no rank finishes early and stalls the
  others at the all-reduce barrier;
* **deterministic / elastic** — same ``(seed, epoch)`` → same global order regardless
  of world size, so a job can resume on a differently-sized cluster;
* **resumable** — pass ``global_consumed`` (the sample count already processed this
  epoch, from a checkpoint) to resume mid-epoch with no repeated or skipped samples;
* **independent ranks** — each rank reads its own index slice with no central
  coordinator, so a slow/idle rank never blocks the others (the Ray ``#42008`` hang).

Tensor collation is zero-copy where Arrow allows: a `FixedShapeTensor` column becomes
a correctly-shaped tensor, numeric columns share their buffer; non-numeric columns
are skipped (move text/ids through the engine, not the trainer hot path).

Two loaders share that contract, one per memory regime. `stream_loader` materializes
the dataset once (`collect()`) and indexes it — simplest, and right whenever the corpus
fits in RAM. `shard_stream_loader` streams shards from disk with a bounded cache, and
draws its indices from `rank_index_batches`, which *computes* the shuffled order from a
keyed permutation instead of materializing it — so neither the samples nor their index
list is ever held, and a corpus of any size loads in constant driver memory.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import partial
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher._internal.prefetch import prefetch
from batcher.ml.streaming_sampler import (
    elastic_shard,
    epoch_order,
    num_rank_batches,
    rank_index_batches,
)

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.api.dataset import Dataset

__all__ = ["column_to_tensor", "iter_torch_batches", "stream_loader", "streaming_split"]

DEFAULT_BATCH_ROWS = 1024


def column_to_tensor(array: pa.Array) -> Any | None:
    """Convert one Arrow column to a torch tensor, or ``None`` if not tensorizable.

    A `FixedShapeTensor` column becomes a shaped tensor; numeric columns convert;
    strings/other types return ``None`` so the caller drops them from the batch.

    The returned tensor owns **writable** memory decoupled from the Arrow buffer: a
    training loop routinely mutates batches in place (augmentation, normalization) and
    the source Arrow buffer is immutable, so handing back a read-only view sharing it
    is "undefined behavior" (torch's own warning) and can corrupt the source data.
    We therefore materialize a writable copy — one batch's worth, negligible next to a
    forward/backward pass.
    """
    import numpy as np
    import pyarrow as pa
    import torch

    # `Table.column(...)` hands back a ChunkedArray; collapse it so extension arrays
    # (e.g. FixedShapeTensor) expose their typed `to_numpy_ndarray`.
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()

    if hasattr(array, "to_numpy_ndarray"):  # FixedShapeTensor extension array
        nd = array.to_numpy_ndarray()
    elif pa.types.is_fixed_size_list(array.type) and pa.types.is_primitive(array.type.value_type):
        # A fixed-width numeric vector column (a feature/embedding vector stored as
        # FixedSizeList<T, W>) → a (n, W) tensor. `to_numpy` on a fixed-size-list yields an
        # array of per-row arrays (object dtype); reshape the flat child buffer instead, so a
        # feature column is a real 2-D tensor rather than being silently dropped.
        w = array.type.list_size
        child = array.flatten().to_numpy(zero_copy_only=False)
        if child.dtype.kind not in "biuf":
            return None
        nd = child.reshape(-1, w)
    else:
        try:
            nd = array.to_numpy(zero_copy_only=False)
        except (ValueError, TypeError):
            return None
        if nd.dtype.kind not in "biuf":  # bool/int/uint/float only
            return None
    # `np.array(..., copy=True)` guarantees an owned, contiguous, writable buffer
    # regardless of whether the source view was read-only (Arrow buffers are).
    return torch.from_numpy(np.array(nd, copy=True))


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
) -> Any:
    """A `torch.utils.data.IterableDataset` of ``{column: tensor}`` batches for `rank`.

    The dataset is **materialized once** (``collect()``) — fine up to RAM. For a
    larger-than-memory corpus, write it with `batcher.io.formats.ml.write_shards` and
    use `shard_stream_loader`, which streams from disk with a bounded shard cache and
    the identical sample-order contract.

    Args:
        dataset: a bounded Batcher `Dataset` (it is materialized once).
        batch_size: rows per yielded batch.
        world_size / rank: this process's slot in the data-parallel group.
        epoch / seed / shuffle: control the deterministic global order.
        drop_last: drop the epoch's tail remainder (True, the default) or keep it
            and pad back up to a whole number of ranks (False). Either way every
            rank yields the same batch count, so no rank stalls the DDP barrier.
        columns: subset to tensorize (default: all tensorizable columns).
        global_consumed: samples already processed this epoch (resume point).

    Raises:
        ImportError: if torch is not installed.
    """
    from torch.utils.data import IterableDataset

    table = dataset.collect()
    order = epoch_order(table.num_rows, epoch=epoch, seed=seed, shuffle=shuffle)
    indices = elastic_shard(
        order,
        world_size=world_size,
        rank=rank,
        global_consumed=global_consumed,
        drop_last=drop_last,
    )
    keep = list(columns) if columns is not None else list(table.column_names)
    bs = max(1, batch_size)

    class _StreamLoader(IterableDataset):  # type: ignore[misc]
        def __len__(self) -> int:
            # Whole batches this rank yields (drop_last → equal across ranks).
            full = len(indices) // bs if drop_last else (len(indices) + bs - 1) // bs
            return full

        def __iter__(self):
            n = len(indices)
            limit = (n // bs) * bs if drop_last else n
            for start in range(0, limit, bs):
                yield _tensorize(table.take(indices[start : start + bs]), keep)

    return _StreamLoader()


def _tensorize(batch: Any, keep: list[str]) -> dict[str, Any]:
    """A `{column: tensor}` dict for one (Arrow Table/RecordBatch) batch, dropping
    non-tensorizable columns. Shared by the in-memory and shard loaders."""
    out = {}
    for c in keep:
        t = column_to_tensor(batch.column(c))
        if t is not None:
            out[c] = t
    return out


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
) -> Any:
    """A `torch.utils.data.IterableDataset` streaming from a **shard directory**
    (written by `batcher.io.formats.ml.write_shards`) with **bounded memory**.

    Same deterministic / balanced / elastic / resumable sample order as
    `stream_loader`, but the corpus is never materialized: row count comes from the
    shard index, and each batch is gathered through a `ShardReader` whose LRU cache
    keeps at most `cache_size` shards resident — so this scales to larger-than-RAM
    training sets, the case `stream_loader` (which calls `collect()`) can't handle.

    Args mirror `stream_loader`; `cache_size` bounds resident shards. Requires `torch`.
    """
    from torch.utils.data import IterableDataset

    from batcher.io.formats.ml.shards import ShardReader

    reader = ShardReader(directory, cache_size=cache_size)
    keep = list(columns) if columns is not None else list(reader.take([0]).column_names)
    bs = max(1, batch_size)
    # The larger-than-RAM path, so the *index* sequence must not be materialized either:
    # a shuffled list of every index costs ~28 bytes/sample (280 GB at 10^10 samples).
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
            for indices in rank_index_batches(reader.total_rows, **sampler_kwargs):
                yield _tensorize(reader.take(indices), keep)

    return _ShardLoader()


def iter_torch_batches(
    dataset: Dataset,
    *,
    batch_size: int | None = None,
    columns: list[str] | None = None,
    device: Any = "auto",
    collate_fn: Any = None,
    prefetch_batches: int = 1,
    pin_memory: bool = False,
    zero_copy: bool = False,
    local_shuffle_buffer_size: int | None = None,
    seed: int = 0,
) -> Any:
    """Stream this dataset to a PyTorch training loop, batch by batch (the lazy path).

    Unlike `stream_loader` (which materializes once for a deterministic global order),
    this consumes `dataset.iter_batches()` incrementally in **bounded memory** — the
    Ray Data ``iter_torch_batches`` role — so it scales to larger-than-memory and
    streaming sources. Yields ``{column: tensor}`` dicts (numeric columns; others are
    dropped) unless `collate_fn` is given, in which case it receives the
    ``{column: ndarray}`` batch and its return is yielded.

    Args:
        batch_size: rows per yielded batch (engine default when None).
        columns: subset to convert (default: all numeric columns).
        device: where to move tensors. ``"auto"`` (default) picks the best available
            accelerator (CUDA/ROCm/Intel-XPU/Apple-MPS) or CPU; pass an explicit torch
            device (``"cuda:1"``, ``"cpu"``) to override, or ``None`` to leave on CPU.
            The move overlaps the next batch's host work when `prefetch_batches` > 0.
        collate_fn: optional ``{col: ndarray} -> Any`` to build the batch yourself.
        prefetch_batches: batches to prefetch on a background thread (0 disables).
        pin_memory: page-lock CPU tensors for faster asynchronous host→device copies
            (only meaningful with a non-CPU `device`).
        zero_copy: for **read-only inference**, hand the Arrow buffer to torch via
            DLPack (one fewer CPU copy before the device move). Do not mutate the
            tensors; leave False for training (which mutates batches in place).
        local_shuffle_buffer_size: if set, shuffle within blocks of this many rows
            before batching (a streaming approximation of a global shuffle).
        seed: seed for the local shuffle.

    Raises:
        ImportError: if torch is not installed.
    """
    import torch  # noqa: F401  (fail fast with a clear error if torch is absent)

    from batcher.ml.converters import arrays_to_torch, to_numpy_batches

    if device == "auto":
        from batcher.ml.gpu import torch_device

        device = torch_device()
    arrow_batches = dataset.iter_batches(batch_size)
    if local_shuffle_buffer_size:
        out_rows = batch_size or DEFAULT_BATCH_ROWS
        numpy_stream = _shuffle_to_numpy(
            arrow_batches, local_shuffle_buffer_size, out_rows, seed, columns
        )
    else:
        numpy_stream = to_numpy_batches(arrow_batches, columns=columns)
    move = device if device not in (None, "cpu") else None
    to_torch = partial(arrays_to_torch, zero_copy=zero_copy)
    tensors = (_to_torch_out(a, to_torch, collate_fn, move, pin_memory) for a in numpy_stream)
    yield from prefetch(tensors, prefetch_batches if prefetch_batches else 0)


def streaming_split(
    dataset: Dataset,
    world_size: int,
    *,
    rank: int | None = None,
    queue_depth: int = 2,
    **loader_kwargs: Any,
) -> Any:
    """Split the stream across `world_size` ranks for data-parallel training (lazy).

    Two modes, both yielding `iter_torch_batches`-shaped ``{column: tensor}`` batches:

    * **Whole fleet** (no `rank`) → a list of `world_size` iterators. A single reader
      consumes the dataset **once** and fans batches out round-robin to per-rank
      queues, so the data is read once total (not once per rank). All ranks must be
      consumed **concurrently** (the DDP norm); bounded `queue_depth` applies
      backpressure. Each rank yields the *same* number of batches (a trailing partial
      round is dropped) so no rank stalls the all-reduce barrier.
    * **Single rank** (`rank` given) → one iterator that reads the stream and keeps its
      ``i % world_size == rank`` shard. Like the fleet mode it emits only **complete
      rounds**, dropping a trailing partial one, so every rank yields the same batch
      count. For separate DDP processes over a *bounded* corpus prefer `stream_loader`,
      whose indexed split is exactly balanced, deterministic, and resumable without
      re-reading the others' shards.

    Use either for unbounded/streaming sources that have no global length.
    """
    if world_size < 1:
        raise PlanError("streaming_split requires world_size >= 1")
    if rank is not None:
        if not 0 <= rank < world_size:
            raise PlanError(f"streaming_split: rank {rank} out of range for {world_size}")
        return _rank_shard_stream(iter_torch_batches(dataset, **loader_kwargs), world_size, rank)
    return _round_robin_split(dataset, world_size, queue_depth, loader_kwargs)


def _rank_shard_stream(batches: Iterable[Any], world_size: int, rank: int) -> Iterator[Any]:
    """This rank's batch from each **complete** round of `world_size` batches.

    Keeping only ``i % world_size == rank`` would give rank 0 ``ceil(n / world_size)``
    batches and the last rank ``floor(...)`` — the low ranks then reach the all-reduce
    barrier once more than the high ones and the job hangs. Withholding a batch until
    its round is known complete costs nothing (at most one batch is held) and makes the
    counts equal, matching the fleet path; the trailing partial round is dropped.
    """
    pending = None
    for i, batch in enumerate(batches):
        position = i % world_size
        if position == rank:
            pending = batch
        if position == world_size - 1:  # the round is complete — this rank's batch counts
            yield pending
            pending = None


def _round_robin_split(
    dataset: Dataset, world_size: int, queue_depth: int, loader_kwargs: dict
) -> list:
    """A single reader fans the tensor stream out to `world_size` per-rank queues.

    The dataset is read once; a producer thread distributes complete rounds of
    `world_size` batches (one per rank), dropping any trailing partial round so every
    rank yields an equal count. A producer error surfaces to every rank's consumer."""
    import queue
    import threading

    queues: list[queue.Queue] = [queue.Queue(maxsize=queue_depth) for _ in range(world_size)]
    state: dict = {"error": None}
    done = object()

    def _producer() -> None:
        try:
            round_: list = []
            for batch in iter_torch_batches(dataset, **loader_kwargs):
                round_.append(batch)
                if len(round_) == world_size:
                    for i, item in enumerate(round_):
                        queues[i].put(item)
                    round_ = []
            # A trailing partial round is dropped to keep all ranks balanced.
        except Exception as exc:  # surface to every consumer
            state["error"] = exc
        finally:
            for q in queues:
                q.put(done)

    threading.Thread(target=_producer, daemon=True).start()

    def _rank_iter(q: queue.Queue) -> Any:
        while True:
            item = q.get()
            if item is done:
                if state["error"] is not None:
                    raise state["error"]
                return
            yield item

    return [_rank_iter(q) for q in queues]


def _to_torch_out(
    arrays: dict, arrays_to_torch: Any, collate_fn: Any, device: Any, pin_memory: bool = False
) -> Any:
    """Convert one `{col: ndarray}` batch to the yielded output, optionally moved to
    `device` (non-blocking when `pin_memory` page-locks the source)."""
    out = collate_fn(arrays) if collate_fn is not None else arrays_to_torch(arrays)
    if device is None:
        return out
    # Apple MPS does not support 64-bit tensors; downcast so `device="auto"` works on
    # Apple silicon (the common dev box) instead of crashing on an int64/float64 column.
    is_mps = str(device).startswith("mps")

    def _move(t: Any) -> Any:
        if not hasattr(t, "to"):
            return t
        if is_mps:
            t = _mps_safe_dtype(t)
        if pin_memory and hasattr(t, "pin_memory"):
            t = t.pin_memory()
        return t.to(device, non_blocking=pin_memory)

    if isinstance(out, dict):
        return {k: _move(v) for k, v in out.items()}
    return _move(out)


def _mps_safe_dtype(tensor: Any) -> Any:
    """Downcast a 64-bit tensor to 32-bit (MPS supports no 64-bit dtypes)."""
    import torch

    if tensor.dtype == torch.float64:
        return tensor.to(torch.float32)
    if tensor.dtype == torch.int64:
        return tensor.to(torch.int32)
    return tensor


def _shuffle_to_numpy(
    batches: Any, buffer_rows: int, out_rows: int, seed: int, columns: Any
) -> Any:
    """Local-shuffle a batch stream and yield ``{column: ndarray}`` batches directly.

    A streaming approximation of a global shuffle: fill a block of ~`buffer_rows`, permute
    it once, emit it in `out_rows` chunks, repeat. The shuffle happens in **NumPy space** —
    the block's columns are converted to NumPy once (a feature/embedding vector becomes a
    contiguous ``(n, width)`` array), then each emitted chunk is a single fancy-index gather.
    This avoids the Arrow round-trip a `take`-then-reconvert shuffle pays (build a permuted
    `Table`, re-lay-out nested fixed-size-list offsets, reconvert to NumPy — ~3-4 copies of a
    wide column per block), which made the shuffled loader lose to Ray Data; here it is one
    conversion + one gather. The consumer is `arrays_to_torch`, same as `to_numpy_batches`.
    """
    import numpy as np
    import pyarrow as pa

    from batcher.ml.converters import _column_to_numpy

    rng = np.random.RandomState(seed)

    def _emit(chunks: list) -> Any:
        table = pa.Table.from_batches(chunks)
        names = list(table.column_names) if columns is None else list(columns)
        cols = {name: _column_to_numpy(table.column(name)) for name in names}
        perm = rng.permutation(table.num_rows)
        for start in range(0, table.num_rows, out_rows):
            idx = perm[start : start + out_rows]
            yield {name: arr[idx] for name, arr in cols.items()}

    block: list = []
    rows = 0
    for b in batches:
        block.append(b)
        rows += b.num_rows
        if rows >= buffer_rows:
            yield from _emit(block)
            block, rows = [], 0
    if block:
        yield from _emit(block)
