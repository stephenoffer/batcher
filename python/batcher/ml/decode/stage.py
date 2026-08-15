"""Shared scaffolding every decode stage is built on.

A decode stage is always the same shape: pull a column out of an Arrow batch, do some
bounded-concurrency work per row, and append the result as a new column. The three
pieces of that shape live here so `transfer`, `media`, and `video` share one
implementation rather than three — they sit in the same package precisely because
copy-pasting a thread pool or an append-or-replace into each of them is how the three
drift apart.
"""

from __future__ import annotations

import threading
from typing import Any

from batcher._internal.errors import PlanError
from batcher.ml.stats._shared import require_columns

__all__: list[str] = []

_POOLS: dict[int, Any] = {}
_POOL_LOCK = threading.Lock()


def _with_column(batch: Any, name: str, col: Any) -> Any:
    """Append `col` as `name`, or replace the column of that name if it already exists."""
    if name in batch.schema.names:
        return batch.set_column(batch.schema.get_field_index(name), name, col)
    return batch.append_column(name, col)


def _shared_pool(max_workers: int) -> Any:
    """A per-process thread pool for `max_workers`, reused across batches.

    Building a `ThreadPoolExecutor` per batch spawns and tears down a fresh set of OS
    threads for every batch, which on an ingest of many small batches costs more than
    the transfers it is there to overlap.
    """
    from concurrent.futures import ThreadPoolExecutor

    with _POOL_LOCK:
        pool = _POOLS.get(max_workers)
        if pool is None:
            pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="bt-media")
            _POOLS[max_workers] = pool
        return pool


def _bounded_map(
    fn: Any, items: Any, workers: int, *, timeout: float | None = None, on_timeout: Any = None
) -> Any:
    """Map `fn` over `items` in order, keeping at most `workers` calls in flight.

    `Executor.map` cannot be used: it drains its input iterable and submits every call up
    front, which would pull every clip's bytes into memory at once — exactly the
    whole-batch materialization a lazy source iterator exists to avoid. Bounding
    submission also makes `timeout` meaningful per call rather than per batch, because a
    call is submitted only once a worker is free to start it, so its submit time is its
    start time. `on_timeout` supplies the value to yield instead of raising.

    A `timeout` is served by one guarded daemon thread per call rather than by the shared
    pool, because a timed-out call is abandoned and keeps running. Parked in the shared
    fixed-size pool it held its worker forever, and `_POOLS` is module-level — so once
    `workers` fetches had hung, *every* later download in the process timed out on a full
    queue however healthy its URL was, and the stage returned nulls and timeout errors for
    good data. Silent, permanent, and process-wide. `call_with_timeout` keeps a hung call's
    cost to the call that earned it; the untimed path still shares the pool.
    """
    import functools
    import time
    from collections import deque

    from batcher._internal.concurrency.timeout import start_call

    pending: deque[tuple[float, Any]] = deque()
    # Both branches start the call at submit time, so `workers` of them are in flight at
    # once and a deadline measured from submit time measures the call itself.
    if timeout is None:
        pool = _shared_pool(workers)
        submit: Any = lambda item: pool.submit(fn, item)  # noqa: E731
    else:
        submit = lambda item: start_call(  # noqa: E731
            functools.partial(fn, item), name="bt-media"
        )

    def _take() -> Any:
        submitted, handle = pending.popleft()
        if timeout is None:
            return handle.result()
        try:
            return handle.result(max(timeout - (time.monotonic() - submitted), 0.0))
        except TimeoutError:
            if on_timeout is None:
                raise
            return on_timeout()

    for item in items:
        pending.append((time.monotonic(), submit(item)))
        if len(pending) >= workers:
            yield _take()
    while pending:
        yield _take()


def _require_source_column(ds: Any, column: str, *, who: str, param: str) -> None:
    """Reject a source column that is not in `ds`, before the decode stage runs.

    The image and audio stages get this for free — their output is a projection, whose own
    validation names the missing column — but the video, download, and upload stages read
    the column inside a `map_batches` callback, so a typo reached Arrow and came back as
    ``KeyError: 'Field "nope" does not exist in schema'``: no mention of the function, the
    argument, or the columns that do exist. The same slip therefore reported two different
    ways depending on which modality you were decoding.
    """
    require_columns(ds, column, hint=f"{who} reads {param}.")


def _require_frames(num_frames: int, who: str) -> int:
    """Reject a non-positive frame count, which the tensor type cannot express.

    Zero surfaced as ``ArrowInvalid: list_size needs to be a strict positive integer`` and a
    negative as NumPy's ``negative dimensions are not allowed`` — two internal messages for
    one argument, neither naming it.
    """
    if num_frames < 1:
        raise PlanError(f"{who} num_frames must be >= 1, got {num_frames}")
    return num_frames


def _require_size(size: tuple[int, int] | None, who: str) -> tuple[int, int]:
    if size is None:
        raise PlanError(
            f"{who} requires size=(height, width), e.g. size=(224, 224). The native decode "
            f"emits a fixed-shape tensor column, which needs one shape for every row. To keep "
            f"each image at its own resolution, decode in a `map_batches` and return the "
            f"arrays — the engine carries mixed shapes as a variable-shape tensor column."
        )
    height, width = size
    if height <= 0 or width <= 0:
        raise PlanError(f"{who} size must be positive, got {size}")
    return height, width
