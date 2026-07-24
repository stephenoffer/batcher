"""Run an async (`async def`) `map_batches` fn: overlap I/O-bound calls across batches.

An LLM-inference / API-enrichment `fn` is I/O-bound — it spends its time awaiting a remote
service, not burning CPU. The thread path works but couples concurrency to a thread per
in-flight batch; an `async def` `fn` runs many batches' awaits concurrently on ONE event loop,
bounded by a semaphore. That is how you issue thousands of concurrent LLM requests without
thousands of threads, and it is the concurrency model Ray Data's async `map_batches` uses.

Detection is from the callable itself — a coroutine function, or a class instance whose
`__call__` is one — so a user opts in just by writing `async def`; the synchronous path is
untouched. Two properties the sync path cannot offer:

- **Real timeout.** `asyncio.wait_for` *cancels* a pending coroutine at its next await point,
  so a timed-out remote call is actually abandoned. The thread-based timeout in `resilience.py`
  can only stop waiting on a call it cannot preempt.
- **Ordered, bounded overlap.** Results are returned in input-batch order regardless of which
  coroutine finishes first, and no more than `max_concurrency` run at once, so a slow tail
  cannot unbounded-buffer the whole input.

This module owns only the async *mechanism*; `apply.apply_udf` decides when to route here,
and `resilience`/`call` supply the retry policy and result coercion it reuses.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

import pyarrow as pa

from batcher.core.udf.call import _coerce_udf_result, _record_dropped_row
from batcher.plan.logical import MapBatches

__all__ = ["is_async_udf", "run_async_batches", "run_coroutine_blocking"]

# In-flight coroutine cap for an async `fn` with no explicit `max_concurrency`. An I/O-bound
# `fn` wants many concurrent awaits (a thread pool would size to cores); 32 is a conservative
# default that overlaps latency well without risking a remote service's rate limit by default.
_DEFAULT_ASYNC_CONCURRENCY = 32


def is_async_udf(fn: object) -> bool:
    """Whether `fn`'s per-batch call is a coroutine (an `async def`, directly or via `__call__`).

    `fn` here is the already-resolved per-batch callable (a class factory is built first), so a
    class UDF is detected by its instance's `__call__`, matching how the sync path calls it.
    """
    if inspect.iscoroutinefunction(fn):
        return True
    # `getattr_static` reads `__call__` without binding/executing a descriptor, and works for a
    # class object (its own `__call__`) and an instance (its class's `__call__`) alike.
    return inspect.iscoroutinefunction(inspect.getattr_static(fn, "__call__", None))


def run_async_batches(
    call: Callable[[pa.RecordBatch], Awaitable[Any]], batches: list[pa.RecordBatch], op: MapBatches
) -> list[pa.RecordBatch]:
    """Apply an async `call` to each batch concurrently on one event loop, preserving order.

    At most `max_concurrency` batches are in flight at once; each call is retried with `op`'s
    backoff policy and bounded by `op.timeout_s` (a real cancel via `asyncio.wait_for`). The
    coerced per-batch results are concatenated in input-batch order. Runs the loop to completion
    and returns a materialized list — the caller owns any streaming.
    """
    if not batches:
        return []
    limit = op.max_concurrency if op.max_concurrency > 0 else _DEFAULT_ASYNC_CONCURRENCY
    results = run_coroutine_blocking(lambda: _gather_batches(call, batches, op, limit))
    out: list[pa.RecordBatch] = []
    for coerced in results:
        out.extend(coerced)
    return out


def run_coroutine_blocking(coro_factory: Callable[[], Awaitable[Any]]) -> Any:
    """Drive an async run to completion, safe to call inside an existing event loop.

    `asyncio.run` cannot be called from a thread that already has a running loop — exactly the
    case when a Batcher pipeline is driven from a Jupyter notebook or an async web app (the
    primary ML environments). When a loop is already running, the coroutine is run on a *fresh*
    loop in a short-lived worker thread, so it never touches the caller's loop; otherwise
    `asyncio.run` is used directly. `coro_factory` (not a coroutine) so the coroutine is created
    on whichever thread ends up running it.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())  # no running loop — the common single-node case
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro_factory())).result()


async def _gather_batches(
    call: Callable[[pa.RecordBatch], Awaitable[Any]],
    batches: list[pa.RecordBatch],
    op: MapBatches,
    limit: int,
) -> list[list[pa.RecordBatch]]:
    """Await every batch's call under a concurrency semaphore, returning per-batch coerced lists
    in input order."""
    sem = asyncio.Semaphore(max(1, limit))

    budget = _error_budget(op) if op.max_errored_rows > 0 else None

    async def _one(batch: pa.RecordBatch) -> list[pa.RecordBatch]:
        async with sem:
            if budget is not None:
                return await _resilient_batch(call, batch, op, budget)
            return _coerce_udf_result(await _call_resilient(call, batch, op))

    # `gather` preserves the order of its arguments in its results regardless of completion
    # order, so the output stays aligned to the input batches.
    return await asyncio.gather(*(_one(b) for b in batches))


def _error_budget(op: MapBatches) -> list[int]:
    """The shared per-worker `[remaining, dropped]` error budget for `op` (see `strategy`)."""
    from batcher.core.udf.strategy import error_budget

    return error_budget(op)


async def _resilient_batch(
    call: Callable[[pa.RecordBatch], Awaitable[Any]],
    batch: pa.RecordBatch,
    op: MapBatches,
    budget: list[int],
) -> list[pa.RecordBatch]:
    """The async twin of `call._resilient_call`: retry each call, then bisect a still-failing
    batch to isolate and drop the offending row(s) against `budget`.

    Gives an async `fn` the same `max_errored_rows` dirty-data tolerance as a synchronous one —
    a corrupt row is retried (via `_call_resilient`), then row-isolated and dropped up to the
    budget, rather than crashing the whole query. Runs on the event loop, so the halves are
    awaited concurrently.
    """
    try:
        return _coerce_udf_result(await _call_resilient(call, batch, op))
    except Exception as exc:
        if batch.num_rows <= 1:
            if budget[0] <= 0:
                raise  # the error budget is spent — a real bug on clean data still fails
            budget[0] -= 1
            _record_dropped_row(budget, exc)
            return []  # drop the one corrupt row and carry on
        mid = batch.num_rows // 2
        left, right = await asyncio.gather(
            _resilient_batch(call, batch.slice(0, mid), op, budget),
            _resilient_batch(call, batch.slice(mid), op, budget),
        )
        return left + right


async def _call_resilient(
    call: Callable[[pa.RecordBatch], Awaitable[Any]], batch: pa.RecordBatch, op: MapBatches
) -> Any:
    """One async call with `op`'s retry + timeout policy (the async twin of `resilience`).

    A retryable error (see `op.retry_on`; any `Exception` when empty) is retried up to
    `op.max_retries` times with exponential backoff; a call exceeding `op.timeout_s` is
    cancelled via `asyncio.wait_for` and surfaced as `TimeoutError`, itself retryable.
    """
    retry_on: tuple[type[BaseException], ...] = op.retry_on or (Exception,)
    timeout = op.timeout_s if op.timeout_s > 0.0 else None
    if timeout is not None and not any(issubclass(TimeoutError, t) for t in retry_on):
        retry_on = (*retry_on, TimeoutError)
    base = max(0.0, op.retry_backoff_s)
    max_retries = max(0, op.max_retries)
    attempt = 0
    while True:
        try:
            if timeout is not None:
                try:
                    return await asyncio.wait_for(call(batch), timeout=timeout)
                except (asyncio.TimeoutError, TimeoutError) as exc:
                    # Normalize a cancelled call to a clear `TimeoutError`; caught below and
                    # retried like any transient (it was added to `retry_on` above).
                    raise TimeoutError(
                        f"async map_batches fn exceeded timeout of {timeout:g}s"
                    ) from exc
            return await call(batch)
        except retry_on:
            if attempt >= max_retries:
                raise
        delay = min(30.0, base * (2.0**attempt))
        if delay > 0.0:
            await asyncio.sleep(delay)
        attempt += 1
