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
    """
    import time
    from collections import deque
    from concurrent.futures import TimeoutError as FuturesTimeout

    pool = _shared_pool(workers)
    pending: deque[tuple[float, Any]] = deque()

    def _take() -> Any:
        submitted, future = pending.popleft()
        if timeout is None:
            return future.result()
        try:
            return future.result(timeout=max(timeout - (time.monotonic() - submitted), 0.0))
        except FuturesTimeout:
            if on_timeout is None:
                raise
            return on_timeout()

    for item in items:
        pending.append((time.monotonic(), pool.submit(fn, item)))
        if len(pending) >= workers:
            yield _take()
    while pending:
        yield _take()


def _require_size(size: tuple[int, int] | None, who: str) -> tuple[int, int]:
    if size is None:
        raise PlanError(f"{who} requires size=(height, width), e.g. size=(224, 224)")
    height, width = size
    if height <= 0 or width <= 0:
        raise PlanError(f"{who} size must be positive, got {size}")
    return height, width
