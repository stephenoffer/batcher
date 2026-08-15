"""Bounding a call that can hang, without leaving the process unable to exit.

The engine makes calls it does not control and cannot preempt: a user `map_batches` `fn`
that reaches an LLM API, a media fetch against an unresponsive host, a model that wedges.
Each is guarded by a timeout, and every one of those guards has the same two obligations.

**The abandoned call must not hold the interpreter open.** Python cannot cancel a running
call, so a timed-out one is *abandoned* — it keeps running and its result is discarded.
That makes `concurrent.futures.ThreadPoolExecutor` the wrong tool, for a reason that is
easy to miss: `concurrent.futures.thread` registers an atexit hook that **joins every
worker thread it ever started**, and `shutdown(wait=False)` returns immediately without
detaching the thread from it. So an executor-backed timeout keeps its promise to the query
and then wedges the process — the query raises `TimeoutError` on schedule, results are
computed and written, and the interpreter hangs forever at exit waiting on precisely the
call the timeout existed to escape. A job that never terminates is a worse failure than the
one being guarded against, and no assertion about the query's *result* can see it.

**An abandoned call must not consume capacity it will never give back.** A hung call parked
in a fixed-size pool holds its worker forever. Share that pool across batches and the
failure compounds into a wrong answer: once `max_workers` calls have hung, every later call
times out on a full queue no matter how healthy it is, so a download stage returns nulls
for URLs that were fine. Giving each guarded call its own daemon thread is what keeps a
timeout local to the call that earned it.

`start_context_thread` supplies both properties (daemon, and the caller's context carried
across the boundary — see `context.py` for why the context matters). This module is the one
place that pairs it with a join deadline, so `core`'s UDF resilience and `ml`'s media
fetches share an implementation rather than each growing their own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from batcher._internal.concurrency.context import start_context_thread

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["AbandonableCall", "call_with_timeout", "start_call"]

T = TypeVar("T")


class AbandonableCall:
    """A call already running on a daemon thread, awaited later with a deadline.

    The two-step form of `call_with_timeout`, for a caller that needs several calls in
    flight at once: start each one as it gets a slot, then await them in order. Awaiting
    one call cannot delay another, and abandoning one cannot cost another its capacity,
    because every call owns its thread.
    """

    __slots__ = ("_outcome", "_thread")

    def __init__(self, fn: Callable[[], Any], /, *, name: str = "batcher-guarded-call") -> None:
        # One slot, appended exactly once by the thread body before it returns. Reading it
        # after a successful join is therefore race-free: a dead thread has already appended.
        self._outcome: list[tuple[bool, Any]] = []

        def _body() -> None:
            try:
                self._outcome.append((True, fn()))
            except BaseException as exc:
                self._outcome.append((False, exc))

        self._thread = start_context_thread(_body, name=name, daemon=True)

    def result(self, timeout: float, /, *, on_timeout: Callable[[], str] | None = None) -> Any:
        """Wait up to `timeout` seconds for the call, then abandon it.

        Args:
            timeout: Seconds to wait before abandoning the call.
            on_timeout: Builds the `TimeoutError` message; a default is used when omitted.

        Returns:
            Whatever the call returned.

        Raises:
            TimeoutError: If the call had not returned within `timeout` seconds.
        """
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError(
                on_timeout() if on_timeout is not None else f"call exceeded timeout of {timeout:g}s"
            )
        succeeded, payload = self._outcome[0]
        if succeeded:
            return payload
        raise payload


def start_call(fn: Callable[[], Any], /, *, name: str = "batcher-guarded-call") -> AbandonableCall:
    """Start `fn()` on a daemon thread and return a handle to await it.

    Args:
        fn: The zero-argument call to start. Bind arguments with `functools.partial`.
        name: Thread name, for stack traces and `ps`.

    Returns:
        An `AbandonableCall` whose `result` awaits the call with a deadline.

    Examples:
        .. doctest::

            >>> from batcher._internal.concurrency.timeout import start_call
            >>> handle = start_call(lambda: 6 * 7)
            >>> handle.result(5.0)
            42
    """
    return AbandonableCall(fn, name=name)


def call_with_timeout(
    fn: Callable[[], T],
    /,
    *,
    timeout: float,
    name: str = "batcher-guarded-call",
    on_timeout: Callable[[], str] | None = None,
) -> T:
    """Run `fn()` on a daemon thread, returning its result or raising `TimeoutError`.

    Anything `fn` raises is re-raised on the calling thread with its type intact, so a
    caller classifying errors (a `retry_on` tuple, an `except ConnectionError`) sees
    exactly what it would have seen from an unguarded call.

    On expiry the call is abandoned rather than cancelled: it runs to completion in the
    background and its result is discarded. Use this as a guard so one wedged call cannot
    stall a query, never as a resource kill — the work does not stop.

    Args:
        fn: The zero-argument call to bound. Bind arguments with `functools.partial`.
        timeout: Seconds to wait before abandoning the call. Must be positive.
        name: Thread name, for stack traces and `ps`.
        on_timeout: Builds the `TimeoutError` message, so a caller can name the operation
            and its inputs. Called only on expiry; a default message is used when omitted.

    Returns:
        Whatever `fn()` returned.

    Raises:
        TimeoutError: If `fn` had not returned within `timeout` seconds.

    Examples:
        .. doctest::

            >>> from batcher._internal.concurrency.timeout import call_with_timeout
            >>> call_with_timeout(lambda: 2 + 2, timeout=5.0)
            4

            >>> import time
            >>> try:
            ...     call_with_timeout(lambda: time.sleep(30), timeout=0.05)
            ... except TimeoutError as exc:
            ...     print("timed out")
            timed out
    """
    result: T = start_call(fn, name=name).result(timeout, on_timeout=on_timeout)
    return result
