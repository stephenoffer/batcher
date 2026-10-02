"""One background thread that runs submitted calls in order — off the caller's critical path.

For work a caller must *finish* but need not *wait for*: the per-query event-log document is
the case that motivated it, where assembling, encoding and writing it after every query sat
between the engine returning and the caller getting its rows — 0.9-1.35 ms of a warm TPC-H
sf1 query's latency — while nothing in the process read the file.

Three properties make that safe rather than merely fast:

* **Order.** One thread, FIFO: calls run in submission order, so a sequence number or a
  retention sweep inside them behaves exactly as it did inline.
* **Bounded.** The queue holds at most `capacity` pending calls; `submit` returns False when it
  is full and the caller runs the call itself. A worker that falls behind slows its producers
  down rather than growing without bound.
* **Nothing lost.** `flush` waits for everything submitted so far, and an `atexit` hook drains
  what is pending when the interpreter shuts down, so a short script's last record still lands.

Each call runs in a copy of the submitter's `contextvars` context (`bound_to_context`), so a
`config_context` in force at submission is the one the call sees.
"""

from __future__ import annotations

import atexit
import queue
import threading
from collections.abc import Callable
from typing import Any

from batcher._internal.concurrency.context import bound_to_context
from batcher._internal.logging import note_suppressed

__all__ = ["SerialWorker"]


class SerialWorker:
    """A lazily started daemon thread that runs submitted calls one at a time, in order.

    Examples:
        .. doctest::

            >>> from batcher._internal.concurrency.serial import SerialWorker
            >>> out = []
            >>> worker = SerialWorker("doc-example", capacity=8)
            >>> worker.submit(out.append, 1) and worker.submit(out.append, 2)
            True
            >>> worker.flush()
            True
            >>> out
            [1, 2]

    Args:
        name: The thread's name, for debuggers and `threading.enumerate()`.
        capacity: The most calls that may be pending at once.
        exit_timeout_s: How long the exit drain waits for pending calls.
    """

    def __init__(self, name: str, capacity: int = 1024, exit_timeout_s: float = 5.0) -> None:
        self._name = name
        self._queue: queue.Queue[Callable[[], Any]] = queue.Queue(maxsize=max(1, capacity))
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._exit_timeout_s = exit_timeout_s

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> bool:
        """Queue `fn(*args, **kwargs)`; False when the queue is full and nothing was queued.

        A False return means the caller should run the call itself — which is what keeps the
        worker bounded. An exception raised by `fn` is the call's own business: catch it
        inside `fn`, since nothing is waiting to receive it.

        Args:
            fn: The call to make.
            *args: Its positional arguments.
            **kwargs: Its keyword arguments.

        Returns:
            Whether the call was queued.
        """
        self._ensure_started()
        try:
            self._queue.put_nowait(bound_to_context(fn, *args, **kwargs))
        except queue.Full:
            return False
        return True

    def flush(self, timeout_s: float | None = None) -> bool:
        """Wait until every call submitted so far has run.

        Args:
            timeout_s: The most to wait, or None to wait as long as it takes.

        Returns:
            Whether everything had run when this returned.
        """
        if self._thread is None:
            return True
        done = threading.Event()
        if not self.submit(done.set):
            self._queue.join()  # full: waiting for the queue to empty is the same answer
            return True
        return done.wait(timeout_s)

    def _ensure_started(self) -> None:
        if self._thread is not None:
            return
        with self._lock:
            if self._thread is None:
                thread = threading.Thread(target=self._run, name=self._name, daemon=True)
                thread.start()
                self._thread = thread
                atexit.register(self.flush, self._exit_timeout_s)

    def _run(self) -> None:
        while True:
            call = self._queue.get()
            try:
                call()
            except Exception as exc:  # a call should catch its own; never let one kill the thread
                note_suppressed("internal", f"run a call on {self._name}", exc)
            finally:
                self._queue.task_done()
