"""The lazy path: stream a dataset to torch with no global length and no materialization.

Where `indexed` needs a bounded corpus to compute a deterministic global order, this consumes
`dataset.iter_batches()` incrementally — the regime for unbounded and streaming sources. The
price is that the split is round-robin over the arriving stream rather than an exact index
partition, so both entry points here emit only **complete rounds** of `world_size` batches: a
trailing partial round would leave the low ranks one batch ahead at the all-reduce barrier and
hang the job.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import partial
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher._internal.prefetch import prefetch
from batcher.ml.loader.tensors import to_torch_out

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["iter_torch_batches", "streaming_split"]

DEFAULT_BATCH_ROWS = 1024


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

    Unlike `stream_loader` (which materializes once for a deterministic global order), this
    consumes `dataset.iter_batches()` incrementally in **bounded memory** — the Ray Data
    ``iter_torch_batches`` role — so it scales to larger-than-memory and streaming sources. It
    yields ``{column: tensor}`` dicts (numeric columns; others are dropped) unless `collate_fn`
    is given, which receives the ``{column: ndarray}`` batch instead.

    Examples:
        .. doctest::

            >>> from batcher.ml import iter_torch_batches  # doctest: +SKIP
            >>> batches = iter_torch_batches(ds, batch_size=64, device="cuda")  # doctest: +SKIP

    Args:
        dataset: the dataset to stream (consumed incrementally, never materialized).
        batch_size: rows per yielded batch (engine default when None).
        columns: subset to convert (default: all numeric columns).
        device: where to move tensors. ``"auto"`` (default) picks the best available accelerator
            (CUDA/ROCm/Intel-XPU/Apple-MPS) or CPU; pass a torch device (``"cuda:1"``, ``"cpu"``)
            to override, or ``None`` to leave on CPU. The move overlaps the next batch's host
            work when `prefetch_batches` > 0.
        collate_fn: optional ``{col: ndarray} -> Any`` to build the batch yourself.
        prefetch_batches: batches to prefetch on a background thread (0 disables).
        pin_memory: page-lock CPU tensors for faster async host→device copies (only
            meaningful with a non-CPU `device`).
        zero_copy: for **read-only inference**, hand the Arrow buffer to torch via DLPack
            (one fewer CPU copy before the device move). Do not mutate the tensors; leave
            False for training, which mutates batches in place.
        local_shuffle_buffer_size: if set, shuffle within blocks of this many rows before
            batching (a streaming approximation of a global shuffle).
        seed: seed for the local shuffle.

    Yields:
        One ``{column: tensor}`` dict per batch, or `collate_fn`'s return value.
    """
    import torch  # noqa: F401  (fail fast with a clear error if torch is absent)

    from batcher.ml.converters import arrays_to_torch, to_numpy_batches

    if batch_size is not None and batch_size < 1:
        # Guard the edge with a typed error; otherwise a `batch_size=0` surfaces deep in
        # the engine as a bare `ValueError: range() arg 3 must not be zero`.
        raise PlanError(f"batch_size must be >= 1, got {batch_size}")

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
    tensors = (to_torch_out(a, to_torch, collate_fn, move, pin_memory) for a in numpy_stream)
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

    Both modes yield `iter_torch_batches`-shaped ``{column: tensor}`` batches, and emit only
    **complete rounds** of `world_size` batches (a trailing partial round is dropped) so every
    rank yields the same count and none stalls the all-reduce barrier. Without `rank` you get a
    list of `world_size` iterators: one reader consumes the dataset **once** and fans batches
    out round-robin to bounded per-rank queues, so the data is read once total, not once per
    rank — all ranks must then be consumed **concurrently** (the DDP norm). With `rank` you get
    one iterator keeping that rank's ``i % world_size`` shard.

    Use either for unbounded/streaming sources with no global length; for separate DDP processes
    over a *bounded* corpus prefer `stream_loader`, whose indexed split is exactly balanced,
    deterministic, and resumable without re-reading the other ranks' shards.

    Examples:
        .. doctest::

            >>> from batcher.ml import streaming_split  # doctest: +SKIP
            >>> per_rank = streaming_split(ds, world_size=4, batch_size=64)  # doctest: +SKIP

    Args:
        dataset: the dataset to stream (read once, incrementally).
        world_size: ranks in the data-parallel group.
        rank: this process's slot — omit for the whole-fleet mode.
        queue_depth: batches buffered per rank before the reader blocks (fleet mode).
        loader_kwargs: forwarded to `iter_torch_batches` (``batch_size``, ``device``, ...).

    Returns:
        One iterator for `rank`, or a list of `world_size` iterators when `rank` is None.

    Raises:
        PlanError: if `world_size` is below 1 or `rank` is out of range.
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

    Keeping only ``i % world_size == rank`` would give rank 0 ``ceil(n / world_size)`` batches
    and the last rank ``floor(...)``, so the low ranks reach the all-reduce barrier once more
    than the high ones and the job hangs. Withholding a batch until its round is known complete
    holds at most one batch and makes the counts equal, matching the fleet path.
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

    The dataset is read once; a producer thread distributes complete rounds of `world_size`
    batches (one per rank), dropping any trailing partial round so every rank yields an equal
    count. A producer error surfaces to every consumer."""
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


def _shuffle_to_numpy(
    batches: Any, buffer_rows: int, out_rows: int, seed: int, columns: Any
) -> Any:
    """Local-shuffle a batch stream and yield ``{column: ndarray}`` batches directly.

    A streaming approximation of a global shuffle: fill a block of ~`buffer_rows`, permute it
    once, emit it in `out_rows` chunks, repeat. The shuffle happens in **NumPy space** (one
    conversion, then a gather per chunk), avoiding the ~3-4 copies of a wide column a
    `take`-then-reconvert Arrow shuffle pays — which made this loader lose to Ray Data.
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
