"""Retry and timeout policy wrapping a per-batch `map_batches` call (Core, layer 3).

`call.py` owns the per-batch *mechanism* (format conversion, failure bisection, dirty-row
tolerance); this module owns the *transient-failure policy* that sits just outside it:
retrying a batch whose `fn` raised a retryable error with exponential backoff, and bounding
a batch whose `fn` hangs with a wall-clock timeout. It is the resilience layer for the
external/flaky UDF — an LLM API call, a vector-DB upsert, a model that occasionally OOMs —
which is the ML-inference workload Batcher targets.

The layering, outermost last:

    fn  →  retry/timeout (here)  →  failure bisection + error budget (`call._resilient_call`)

A transient error is retried here first; only a failure that survives every retry falls
through to the budget, where it is isolated and charged against `max_errored_rows`. The two
are independent knobs: `max_retries` alone retries transients but still fails fast on a real
bug (budget 0), and `max_errored_rows` alone drops dirty rows without retrying. Because the
wrapper preserves the ``batch -> result`` signature, every scheduling path (sequential,
threads, the streaming stages, the GPU autobatch pool) gets the same behavior by wrapping
its per-batch callable once, before it is scheduled.

Timeout has a hard limit worth stating plainly: Python cannot preempt a running call, so a
timed-out call's worker thread is *abandoned* (it finishes in the background and its result
is discarded), not killed. Use `timeout_s` as a guard so one hung external call cannot stall
the whole query — not as a resource kill. The multiprocessing path (reserved for CPU-bound
pure-Python `fn`s, which do not make flaky external calls) is left unwrapped; retries there
would re-run a deterministic bug.

That abandonment is why the call is bounded by `_internal.concurrency.call_with_timeout`
rather than a `ThreadPoolExecutor`: an executor's atexit hook joins every worker thread it
ever started, so an abandoned call wedged the process at exit — see that module for the full
account, and for why the guarded call also has to carry the caller's context.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pyarrow as pa

from batcher._internal.concurrency.timeout import call_with_timeout
from batcher._internal.logging import note_suppressed
from batcher.plan.logical import MapBatches

__all__ = ["wants_resilience", "wrap_resilient"]

# Cap the exponential backoff so a large `max_retries` with a modest base cannot sleep for
# minutes between attempts (``0.5 * 2**12`` is already 34 minutes). A retry is meant to ride
# out a brief transient — a rate-limit window, a connection reset — not to wait out an outage.
_MAX_BACKOFF_S = 30.0


def _backoff(base: float, attempt: int) -> float:
    """Capped exponential backoff for `attempt`, with equal jitter.

    The workload this exists for is an LLM or vector-DB API, and the failure it exists for
    is that service rate-limiting. Both mean every worker fails at the same instant, so an
    undithered ``base * 2**attempt`` marches all of them back to the service together, on
    exactly the schedule most likely to trip the limit again. Equal jitter — half the delay
    fixed, half uniform — keeps the growth curve while decorrelating the retries.

    Timing only: a retry's schedule cannot change what a `fn` computes, so this does not
    touch the determinism the interpreter-as-oracle property rests on.
    """
    import random

    delay = min(_MAX_BACKOFF_S, base * (2.0**attempt))
    return delay / 2.0 + random.random() * (delay / 2.0)


def wants_resilience(op: MapBatches) -> bool:
    """Whether `op` asked for any retry/timeout handling (else the raw call is used unchanged)."""
    return op.max_retries > 0 or op.timeout_s > 0.0


def wrap_resilient(call: Callable[[pa.RecordBatch], Any], op: MapBatches) -> Callable:
    """Wrap a per-batch `call` with `op`'s retry + timeout policy (identity when neither is set).

    Returns a callable with the same ``batch -> result`` contract, so a caller wraps once and
    schedules the result exactly as it scheduled the raw call. A retryable error (see
    `op.retry_on`; any `Exception` when empty) is retried up to `op.max_retries` times with
    exponential backoff; a call exceeding `op.timeout_s` raises `TimeoutError`, itself
    retryable. Every retry and timeout is announced on the observability bus.
    """
    if not wants_resilience(op):
        return call
    retry_on: tuple[type[BaseException], ...] = op.retry_on or (Exception,)
    timeout = op.timeout_s if op.timeout_s > 0.0 else None
    # A timeout the user opted into is always worth retrying, even when `retry_on` restricts the
    # set to specific transient types — so add `TimeoutError` unless the set already covers it.
    if timeout is not None and not any(issubclass(TimeoutError, t) for t in retry_on):
        retry_on = (*retry_on, TimeoutError)
    base = max(0.0, op.retry_backoff_s)
    max_retries = max(0, op.max_retries)
    fn_name = _fn_label(op.fn)

    def _run(batch: pa.RecordBatch) -> Any:
        if timeout is None:
            return call(batch)
        # A fresh guarded call per attempt, never a shared one: reusing a worker would queue
        # the retry behind the hung call and defeat the timeout.
        return call_with_timeout(
            lambda: call(batch),
            timeout=timeout,
            name=f"batcher-udf-{fn_name}",
            on_timeout=lambda: (
                f"map_batches fn {fn_name!r} exceeded timeout of {timeout:g}s on a "
                f"{batch.num_rows}-row batch"
            ),
        )

    def _resilient(batch: pa.RecordBatch) -> Any:
        attempt = 0
        while True:
            try:
                return _run(batch)
            except retry_on as exc:
                if attempt >= max_retries:
                    raise
                delay = _backoff(base, attempt)
                _record_retry(fn_name, attempt + 1, max_retries, delay, exc)
                if delay > 0.0:
                    time.sleep(delay)
                attempt += 1

    return _resilient


def _fn_label(fn: object) -> str:
    """A short, stable name for a `fn` for retry/timeout messages and events."""
    return getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or type(fn).__name__


def _record_retry(
    fn_name: str, attempt: int, max_retries: int, delay: float, exc: BaseException
) -> None:
    """Announce a retry on the event bus (type + message only, never the batch's values).

    Best-effort: the observability publish must never turn a retried transient into a hard
    failure, so any error assembling or emitting the event is swallowed.
    """
    try:
        from batcher._internal.events import LOG, publish

        publish(
            LOG,
            name="map_batches",
            retry_attempt=attempt,
            max_retries=max_retries,
            backoff_s=round(delay, 3),
            fn=fn_name,
            error=f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:  # pragma: no cover - observability must never break a query
        note_suppressed("core", "record a UDF retry", exc)
        return
