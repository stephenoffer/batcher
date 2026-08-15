"""The lazy path: stream a dataset to torch with no global length and no materialization.

Where `indexed` needs a bounded corpus to compute a deterministic global order, this consumes
`dataset.iter_batches()` incrementally — the regime for unbounded and streaming sources. The
price is that the split is round-robin over the arriving stream rather than an exact index
partition, so both entry points here emit only **complete rounds** of `world_size` batches: a
trailing partial round would leave the low ranks one batch ahead at the all-reduce barrier and
hang the job.
"""

from __future__ import annotations

import contextlib
import warnings
import weakref
from collections.abc import Iterable, Iterator
from functools import partial
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher._internal.prefetch import prefetch
from batcher.ml.loader.tensors import DeviceMover
from batcher.ml.permutation import _mix64
from batcher.plan.types import retained_bytes

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["cast_arrays", "iter_torch_batches", "numpy_batch_stream", "streaming_split"]

DEFAULT_BATCH_ROWS = 1024
# Ceiling on the *bytes* a local-shuffle block may hold, independent of the row count the
# caller asked for. `local_shuffle_buffer_size` is a row count, and a row count says nothing
# about memory: 50,000 narrow tabular rows is a few MB, while 50,000 decoded 224x224 images
# is ~30 GB. Ray Data users hit exactly this — the pattern catalog records a 2-5x slowdown
# and OOMs on large rows, with "use 512-2048 for wide rows" as the manual fix. Bounding the
# block by bytes as well as rows makes the knob mean "decorrelate as much as fits", so a
# request that is too large for the row width degrades to a smaller window instead of an
# OOM. Cutting a block early only *narrows* the shuffle window; it never drops or repeats a
# row, so the stream's contents are unchanged.
_SHUFFLE_BLOCK_MAX_BYTES = 256 << 20

#: How long the fan-out producer parks on a full rank queue before re-checking whether every
#: consumer has walked away. Only ever paid by a producer that has already filled its
#: look-ahead, so it delays nothing anyone is waiting for.
_ABANDON_POLL_SECONDS = 0.1


def iter_torch_batches(
    dataset: Dataset,
    *,
    batch_size: int | None = None,
    columns: list[str] | None = None,
    device: Any = "auto",
    dtypes: dict[str, str] | str | None = None,
    collate_fn: Any = None,
    prefetch_batches: int = 2,
    pin_memory: bool = False,
    zero_copy: bool = False,
    local_shuffle_buffer_size: int | None = None,
    seed: int = 0,
    epoch: int = 0,
    drop_last: bool = False,
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
        dtypes: cast the yielded tensors to a torch dtype — a single name (``"float16"``)
            for every column, or a ``{column: dtype}`` mapping for named columns. Names
            accept abbreviations (``"fp16"``, ``"bf16"``). ``None`` leaves the source dtype.
        device: where to move tensors. ``"auto"`` (default) picks the best available accelerator
            (CUDA/ROCm/Intel-XPU/Apple-MPS) or CPU; pass a torch device (``"cuda:1"``, ``"cpu"``)
            to override, or ``None`` to leave on CPU. The move overlaps the next batch's host
            work when `prefetch_batches` > 0.
        collate_fn: optional ``{col: ndarray} -> Any`` to build the batch yourself.
        prefetch_batches: batches to prefetch on a background thread (0 disables). Two to
            four is the useful band: one batch of look-ahead cannot cover a device copy plus
            the next host read, and past four the extra batches only sit in memory that no
            budget accounts for.
        pin_memory: page-lock CPU tensors for faster async host→device copies (only
            meaningful with a non-CPU `device`). The pinned staging buffers stay referenced
            until the copy completes, and on CUDA the copy runs on its own stream.
        zero_copy: for **read-only inference**, hand the Arrow buffer to torch via DLPack
            (one fewer CPU copy before the device move). Do not mutate the tensors; leave
            False for training, which mutates batches in place.
        local_shuffle_buffer_size: if set, shuffle within blocks of this many rows before
            batching (a streaming approximation of a global shuffle). A larger buffer
            decorrelates neighbouring samples better; if the corpus is already written in
            random order, a small buffer (or none) is enough. Past that trade, shuffle
            globally with `stream_loader` instead of buying decorrelation with RAM. The row
            count is a **request, not a reservation**: the block is also **bounded by bytes**,
            so the same number that costs nothing over narrow tabular rows cannot become an
            unbounded allocation over decoded images or long sequences. A request too large
            for the row width silently gets the widest window that fits.
        seed: seed for the local shuffle, combined with `epoch`.
        epoch: reshuffles the local-shuffle order; pass the epoch number so successive passes
            over the same dataset differ. Without it every epoch sees an identical order,
            which quietly costs convergence.
        drop_last: drop a final batch with fewer than `batch_size` rows, so a ragged tail
            never reaches DDP. Requires `batch_size`.

    Yields:
        One ``{column: tensor}`` dict per batch, or `collate_fn`'s return value.

    Raises:
        PlanError: if `batch_size` is below 1, or `drop_last` is set without a `batch_size`.
    """
    from batcher._internal.optional import require

    # Fail fast with an actionable install hint if torch is absent, before any work.
    require("torch", feature="iter_torch_batches", provides="torch", extra="torch")

    from batcher.interop.arrays import _warn_dropped_non_numeric
    from batcher.ml.converters import arrays_to_torch

    if batch_size is not None and batch_size < 1:
        # Guard the edge with a typed error; otherwise a `batch_size=0` surfaces deep in
        # the engine as a bare `ValueError: range() arg 3 must not be zero`.
        raise PlanError(f"batch_size must be >= 1, got {batch_size}")
    if drop_last and batch_size is None:
        # Without a target width there is no such thing as a "short" batch, and silently
        # keeping the ragged tail is exactly the failure `drop_last` was asked to prevent.
        raise PlanError("drop_last requires an explicit batch_size")
    if columns is not None:
        # At the edge, before the query starts. A name that is not a column otherwise
        # surfaced as a bare pyarrow `KeyError` on the first `next()` — past the point where
        # the engine had begun reading, and with no suggestion of what was meant.
        from batcher.ml.stats._shared import require_names

        require_names(list(dataset.columns), *columns, hint="Pass an existing column.")

    if device == "auto":
        from batcher.ml.gpu import torch_device

        device = torch_device()
    numpy_stream = numpy_batch_stream(
        dataset,
        batch_size=batch_size,
        columns=columns,
        local_shuffle_buffer_size=local_shuffle_buffer_size,
        seed=seed,
        epoch=epoch,
        drop_last=drop_last,
    )
    move = device if device not in (None, "cpu") else None
    to_torch = partial(arrays_to_torch, zero_copy=zero_copy)
    cast = _dtype_caster(dtypes)
    depth = prefetch_batches if prefetch_batches else 0
    mover = DeviceMover(move, pin_memory=pin_memory, depth=depth) if move is not None else None

    warn = collate_fn is None  # a collate_fn receives every column, so nothing is dropped

    def _build(arrays: dict) -> Any:
        nonlocal warn
        if warn:
            # Once, on the first batch. Every other conversion path in the package announces
            # the columns it cannot tensorize; this one dropped a string `label`/`id` in
            # silence, so a training loop read a `KeyError` — or trained on what was left.
            _warn_dropped_non_numeric(arrays)
            warn = False
        out = collate_fn(arrays) if collate_fn is not None else cast(to_torch(arrays))
        return out if mover is None else mover(out)

    yield from prefetch((_build(a) for a in numpy_stream), depth)


def numpy_batch_stream(
    dataset: Dataset,
    *,
    batch_size: int | None = None,
    columns: list[str] | None = None,
    local_shuffle_buffer_size: int | None = None,
    seed: int = 0,
    epoch: int = 0,
    drop_last: bool = False,
) -> Iterator[dict]:
    """The engine's output as ``{column: ndarray}`` batches, shuffled and sized to order.

    Everything a framework loader needs before the framework enters the picture: the read,
    the optional local shuffle, the exact batch width, and the ragged tail. Shared rather
    than restated, because the TensorFlow path needs precisely the same four decisions the
    PyTorch one does — and when it did not have them, ``to_tf`` was the only loader in the
    package with no shuffle, no `drop_last`, and no dtype control.

    Args:
        dataset: The dataset to stream, consumed incrementally.
        batch_size: Rows per yielded batch (engine default when None).
        columns: Subset to convert (default: every column).
        local_shuffle_buffer_size: Shuffle within blocks of this many rows before batching.
        seed: Seed for that shuffle, combined with `epoch`.
        epoch: Reshuffles the local-shuffle order, so successive passes differ.
        drop_last: Drop a final batch narrower than `batch_size`.

    Yields:
        One ``{column: numpy.ndarray}`` dict per batch.
    """
    from batcher.ml.converters import to_numpy_batches

    arrow_batches = dataset.iter_batches(batch_size)
    if local_shuffle_buffer_size:
        out_rows = batch_size or DEFAULT_BATCH_ROWS
        stream: Iterator[dict] = _shuffle_to_numpy(
            arrow_batches, local_shuffle_buffer_size, out_rows, _epoch_seed(seed, epoch), columns
        )
    else:
        stream = to_numpy_batches(arrow_batches, columns=columns)
    if drop_last and batch_size:
        stream = _full_batches_only(stream, batch_size)
    return stream


def cast_arrays(stream: Iterator[dict], dtypes: dict[str, str] | str | None) -> Iterator[dict]:
    """Cast a ``{column: ndarray}`` stream to the requested dtypes, before any framework.

    The NumPy-space twin of `_dtype_caster`, and it resolves names through the same
    `batcher.ml.devices.resolve_dtype`, so ``"fp16"`` means the same thing to the
    TensorFlow loader that it means to the PyTorch one. Casting here rather than in the
    framework also halves what crosses into it: a float64 feature column narrowed to
    float32 is half the bytes tf.data has to copy.

    Args:
        stream: The batches to cast.
        dtypes: One dtype name for every column, a ``{column: dtype}`` mapping, or None.

    Yields:
        The same batches, with the named columns cast.

    Raises:
        PlanError: If a requested dtype has no numeric NumPy equivalent (``bfloat16``).
    """
    if dtypes is None:
        yield from stream
        return
    import numpy as np

    from batcher.ml.devices import resolve_dtype

    def _numpy_dtype(name: str) -> Any:
        resolved = resolve_dtype(name)
        try:
            dtype = np.dtype(resolved)
        except TypeError:
            dtype = None
        # Both halves matter, and the second is the one that bites. `bfloat16` is a
        # framework dtype NumPy has no native equivalent for — but with `ml_dtypes`
        # installed (TensorFlow depends on it) `np.dtype("bfloat16")` *succeeds* and yields
        # kind ``"V"``, a void/opaque type. Every numeric check downstream then rejects the
        # column, so asking for bf16 silently emptied the batch and reported every column as
        # "non-numeric, dropped" rather than saying the dtype was the problem.
        if dtype is None or dtype.kind not in "biuf":
            raise PlanError(
                f"dtype {name!r} resolves to {resolved!r}, which has no numeric NumPy "
                "equivalent; leave it unset and cast inside the framework instead"
            )
        return dtype

    if isinstance(dtypes, str):
        target = _numpy_dtype(dtypes)
        for batch in stream:
            yield {k: v.astype(target, copy=False) for k, v in batch.items()}
        return
    resolved = {col: _numpy_dtype(name) for col, name in dtypes.items()}
    for batch in stream:
        yield {
            k: (v.astype(resolved[k], copy=False) if k in resolved else v) for k, v in batch.items()
        }


def _dtype_caster(dtypes: dict[str, str] | str | None) -> Any:
    """Build a ``{col: tensor} -> {col: tensor}`` cast from a `dtypes` request.

    ``None`` is the identity. A single dtype string casts every tensor; a
    ``{column: dtype}`` mapping casts only the named columns. Names go through
    `batcher.ml.devices.resolve_dtype`, so abbreviations (``"fp16"``) are accepted.
    """
    if dtypes is None:
        return lambda batch: batch
    import torch

    from batcher.ml.devices import resolve_dtype

    def _torch_dtype(name: str) -> Any:
        return getattr(torch, resolve_dtype(name))

    if isinstance(dtypes, str):
        target = _torch_dtype(dtypes)
        return lambda batch: {k: v.to(target) for k, v in batch.items()}
    resolved = {col: _torch_dtype(name) for col, name in dtypes.items()}
    return lambda batch: {k: (v.to(resolved[k]) if k in resolved else v) for k, v in batch.items()}


def _epoch_seed(seed: int, epoch: int) -> int:
    """Fold `epoch` into `seed`, so successive passes shuffle differently.

    Keyed exactly as `epoch_permutation` keys the global order, so the two notions of "this
    epoch's order" cannot drift apart. Truncated to 32 bits for `numpy.random.RandomState`.
    """
    return _mix64(seed * 0x9E3779B1 + epoch) & 0xFFFFFFFF


def _full_batches_only(stream: Any, batch_size: int) -> Iterator[dict]:
    """Drop batches narrower than `batch_size` — the ragged tail DDP cannot balance.

    Measured once per batch rather than per column: `all()` over an empty column dict is
    ``True``, so a batch that had been projected down to no columns counted as full and a
    zero-row batch reached the training step.
    """
    for arrays in stream:
        if _row_count(arrays) >= batch_size:
            yield arrays


def streaming_split(
    dataset: Dataset,
    world_size: int,
    *,
    rank: int | None = None,
    queue_depth: int = 2,
    drop_last: bool = True,
    **loader_kwargs: Any,
) -> Any:
    """Split the stream across `world_size` ranks for data-parallel training (lazy).

    Both modes yield `iter_torch_batches`-shaped ``{column: tensor}`` batches in **complete
    rounds** of `world_size` batches, so every rank yields the same count and none stalls the
    all-reduce barrier. `drop_last` decides what happens to a trailing partial round: dropped
    (the default, with a warning naming how many batches went), or completed by repeating
    batches from the front of that round, so no data is lost and the ranks stay balanced.
    With ``world_size=8`` the dropped tail is up to 7 batches **per epoch**, which is why it
    is now announced rather than silent. Without `rank` you get a
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
        drop_last: drop the trailing partial round (warning how many batches that cost)
            rather than completing it by repeating batches from the front of the round.
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
        batches = iter_torch_batches(dataset, **loader_kwargs)
        return _rank_shard_stream(batches, world_size, rank, drop_last)
    return _round_robin_split(dataset, world_size, queue_depth, drop_last, loader_kwargs)


def _warn_dropped_round(count: int) -> None:
    """Announce a dropped trailing round, so the loss is visible rather than inferred."""
    warnings.warn(
        f"streaming_split dropped {count} trailing batch(es): the final round was incomplete "
        f"and every rank must yield the same count. Pass drop_last=False to keep them by "
        f"repeating batches from the front of the round instead.",
        UserWarning,
        stacklevel=3,
    )


def _rank_shard_stream(
    batches: Iterable[Any], world_size: int, rank: int, drop_last: bool = True
) -> Iterator[Any]:
    """This rank's batch from each **complete** round of `world_size` batches.

    Keeping only ``i % world_size == rank`` would give rank 0 ``ceil(n / world_size)`` batches
    and the last rank ``floor(...)``, so the low ranks reach the all-reduce barrier once more
    than the high ones and the job hangs. Withholding a batch until its round is known complete
    holds at most one batch and makes the counts equal, matching the fleet path.

    With `drop_last` (the default) that is all this holds. Without it the current round must be
    buffered, because completing a partial round means handing this rank a batch that belongs
    to another slot — so the memory rises to `world_size` batches, which is what the fleet path
    holds anyway.
    """
    if not drop_last:
        round_: list = []
        for batch in batches:
            round_.append(batch)
            if len(round_) == world_size:
                yield round_[rank]
                round_ = []
        if round_:  # complete the partial round cyclically, exactly as the fleet path does
            yield round_[rank % len(round_)]
        return
    pending = None
    dropped = 0
    for i, batch in enumerate(batches):
        position = i % world_size
        if position == rank:
            pending = batch
        if position == world_size - 1:  # the round is complete — this rank's batch counts
            yield pending
            pending, dropped = None, 0
        else:
            dropped = position + 1
    if dropped:
        _warn_dropped_round(dropped)


def _round_robin_split(
    dataset: Dataset, world_size: int, queue_depth: int, drop_last: bool, loader_kwargs: dict
) -> list:
    """A single reader fans the tensor stream out to `world_size` per-rank queues.

    The dataset is read once; a producer thread distributes complete rounds of `world_size`
    batches (one per rank). A trailing partial round is dropped (with a warning) or completed
    by repeating batches from the front of the round, per `drop_last` — either way every rank
    yields an equal count. A producer error surfaces to every consumer.

    Abandonment winds the producer down. A training script that early-stops, raises, or simply
    stops iterating used to strand this thread on a full queue for the life of the process,
    holding an open engine stream and ``queue_depth * world_size`` batches of GPU-bound
    tensors — once per call, and a script that builds a loader per epoch leaks one per epoch.
    The producer now parks with a timeout and re-checks a stop flag, which is the same
    wind-down `batcher._internal.prefetch` performs for the single-stream case.
    """
    import queue
    import threading

    from batcher._internal.concurrency import start_context_thread

    queues: list[queue.Queue] = [queue.Queue(maxsize=queue_depth) for _ in range(world_size)]
    state: dict = {"error": None}
    done = object()
    stop = threading.Event()
    open_ranks = set(range(world_size))
    lock = threading.Lock()

    def _offer(index: int, item: Any) -> bool:
        """Hand one batch to a rank, giving up if every consumer has walked away."""
        while not stop.is_set():
            try:
                queues[index].put(item, timeout=_ABANDON_POLL_SECONDS)
            except queue.Full:
                continue
            return True
        return False

    def _producer() -> None:
        batches = iter_torch_batches(dataset, **loader_kwargs)
        try:
            round_: list = []
            for batch in batches:
                round_.append(batch)
                if len(round_) == world_size:
                    for i, item in enumerate(round_):
                        if not _offer(i, item):
                            return
                    round_ = []
            if round_ and drop_last:
                _warn_dropped_round(len(round_))
            elif round_:  # pad the round cyclically; every rank still gets exactly one
                for i in range(world_size):
                    if not _offer(i, round_[i % len(round_)]):
                        return
        except Exception as exc:  # surface to every consumer
            state["error"] = exc
        finally:
            # Close the engine stream we were pulling; on the abandonment path nothing else
            # runs its `finally`, so the source stays open until the process exits.
            close = getattr(batches, "close", None)
            if close is not None:
                with contextlib.suppress(Exception):
                    close()
            for i in range(world_size):
                _offer(i, done)

    # The producer runs a Batcher pipeline (`iter_torch_batches`), so it has to see the same
    # `Config` the caller does. A bare thread reads every context variable at its default,
    # which silently dropped a `config_context` around the loader.
    start_context_thread(_producer, name="batcher-loader-producer", daemon=True)

    def _release(index: int) -> None:
        """Note that rank `index` has no consumer left, and stop the producer at the last."""
        with lock:
            open_ranks.discard(index)
            if not open_ranks:
                stop.set()

    def _rank_iter(index: int) -> Any:
        q = queues[index]
        try:
            while True:
                item = q.get()
                if item is done:
                    if state["error"] is not None:
                        raise state["error"]
                    return
                yield item
        finally:
            _release(index)

    iterators = [_rank_iter(i) for i in range(world_size)]
    # Registered against the generator *object*, not its body: a rank the caller never
    # started has no `finally` to run — closing an unstarted generator does not execute it —
    # so a fleet abandoned before the first `next()` on some rank would never release, and
    # the producer would park forever holding the engine stream open. This fires whichever
    # way the iterator goes away.
    for i, iterator in enumerate(iterators):
        weakref.finalize(iterator, _release, i)
    return iterators


def _shuffle_to_numpy(
    batches: Any, buffer_rows: int, out_rows: int, seed: int, columns: Any
) -> Iterator[dict]:
    """Local-shuffle a batch stream and yield ``{column: ndarray}`` batches of `out_rows`.

    A streaming approximation of a global shuffle: fill a block of ~`buffer_rows`, permute it
    once, emit it in `out_rows` chunks, repeat. The shuffle happens in **NumPy space** (one
    conversion, then a gather per chunk), avoiding the ~3-4 copies of a wide column a
    `take`-then-reconvert Arrow shuffle pays — which made this loader lose to Ray Data.

    The block is bounded by `_SHUFFLE_BLOCK_MAX_BYTES` as well as by `buffer_rows`, so the
    caller's row count cannot turn into an unbounded allocation when the rows turn out to be
    decoded images or long sequences. Whichever bound binds first cuts the block.

    Block boundaries do **not** reach the caller: a block's trailing remainder is carried into
    the next one, so every batch but the last holds exactly `out_rows` rows. Emitting the
    remainder as its own short batch made the batch width jump around mid-epoch — with
    ``buffer_rows=50000`` and ``batch_size=1024`` every 49th batch was 848 rows — and, far
    worse, `drop_last` then discarded that batch rather than a ragged tail, silently dropping
    ~2% of every epoch throughout the stream instead of a few rows at the end.
    """
    return _rebatch(_shuffled_blocks(batches, buffer_rows, seed, columns), out_rows)


def _shuffled_blocks(batches: Any, buffer_rows: int, seed: int, columns: Any) -> Iterator[dict]:
    """Yield each accumulated block once, as whole permuted ``{column: ndarray}`` columns."""
    import numpy as np
    import pyarrow as pa

    from batcher.ml.converters import _column_to_numpy

    rng = np.random.RandomState(seed)

    def _permute(chunks: list) -> dict:
        table = pa.Table.from_batches(chunks)
        names = list(table.column_names) if columns is None else list(columns)
        perm = rng.permutation(table.num_rows)
        return {name: _column_to_numpy(table.column(name))[perm] for name in names}

    block: list = []
    rows = 0
    nbytes = 0
    for b in batches:
        block.append(b)
        rows += b.num_rows
        nbytes += retained_bytes(b)
        if rows >= buffer_rows or nbytes >= _SHUFFLE_BLOCK_MAX_BYTES:
            yield _permute(block)
            block, rows, nbytes = [], 0, 0
    if block:
        yield _permute(block)


def _rebatch(blocks: Iterator[dict], out_rows: int) -> Iterator[dict]:
    """Cut a stream of ``{column: ndarray}`` blocks into batches of exactly `out_rows`.

    The trailing rows of a block are held over and concatenated onto the next one, so only
    the final batch of the whole stream may be short.
    """
    import numpy as np

    carry: dict | None = None
    for block in blocks:
        if carry is not None:
            block = {name: np.concatenate([carry[name], arr]) for name, arr in block.items()}
            carry = None
        total = _row_count(block)
        start = 0
        while total - start >= out_rows:
            yield {name: arr[start : start + out_rows] for name, arr in block.items()}
            start += out_rows
        if start < total:
            carry = {name: arr[start:] for name, arr in block.items()}
    if carry is not None and _row_count(carry):
        yield carry


def _row_count(arrays: dict) -> int:
    """Rows in a ``{column: ndarray}`` batch — 0 when it carries no columns."""
    for arr in arrays.values():
        return len(arr)
    return 0
