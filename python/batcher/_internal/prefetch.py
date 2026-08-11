"""Overlap a producer generator with its consumer on a background thread.

A neutral utility: a bounded look-ahead over any iterator so the producer keeps
working while the consumer processes the previous item — the pattern that pipelines
a streaming read→transform stage with the write (or H2D copy) that drains it, so
neither stage idles while the other runs. An exception from the source is re-raised
in the consumer, never swallowed (swallowing would silently truncate the stream — a
correctness bug).

**Abandonment is the normal case, not the exceptional one.** A consumer stops early on a
`limit`, on a `stop()`, on any error upstream, and on every plain `break` — and a streaming
inference chain wraps the read *and every stage* in one of these, so an abandoned pipeline
is several at once. The producer thread therefore winds down when the consumer walks away,
and closes the source it was pulling: that source's own `finally` is what releases a broker
connection or a file handle.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Iterator
from typing import TypeVar

from batcher._internal.concurrency import start_context_thread

__all__ = ["prefetch"]

T = TypeVar("T")

#: How long a blocked producer waits before re-checking whether the consumer is gone.
#: Only ever paid by a producer parked on a full queue, which by definition has already
#: filled its look-ahead, so the latency it adds is to work nobody is waiting for.
_STOP_POLL_SECONDS = 0.1


def prefetch(gen: Iterator[T], depth: int = 2) -> Iterator[T]:
    """Yield from `gen`, pulling it on a background thread into a bounded queue.

    `depth` bounds the look-ahead (the queue size) so a fast producer cannot outrun a
    slow consumer without limit. `depth <= 0` disables the overlap and yields `gen`
    directly. Any exception raised while pulling `gen` is re-raised here, in order.

    A consumer that stops early is not a leak. The producer parks on a full queue, so
    without a stop signal it would sit there for the life of the process — one thread and
    `depth` buffered items per abandoned stream, and the source generator never closed, so
    whatever its own `finally` releases stayed open too. Leaving this generator signals the
    producer, which winds down and closes `gen`.

    Args:
        gen: The producer to pull on a background thread.
        depth: Items of look-ahead; ``<= 0`` yields `gen` directly with no thread.

    Yields:
        `gen`'s items, in order.
    """
    if depth <= 0:
        yield from gen
        return

    q: queue.Queue = queue.Queue(maxsize=depth)
    done = object()
    stop = threading.Event()

    def _offer(payload: tuple) -> bool:
        """Hand one item to the consumer, giving up if it has walked away."""
        while not stop.is_set():
            try:
                q.put(payload, timeout=_STOP_POLL_SECONDS)
            except queue.Full:
                continue
            return True
        return False

    def _worker() -> None:
        try:
            for item in gen:
                if not _offer((None, item)):
                    return
        except BaseException as exc:
            # `BaseException`, not `Exception`: a `GeneratorExit` or a `KeyboardInterrupt`
            # in the producer used to skip this branch and fall straight to the `done`
            # marker below, so the consumer saw a clean end-of-stream and the stream was
            # silently *truncated* — the one failure this module's docstring promises not
            # to have.
            _offer((exc, None))
        finally:
            # Close the source we were pulling. Its `finally` is what releases a broker
            # consumer or a file handle, and on the abandonment path nothing else runs it.
            close = getattr(gen, "close", None)
            if close is not None:
                with contextlib.suppress(Exception):
                    close()
            _offer((None, done))

    # Under the caller's context, not a fresh one. `gen` is the *consumer's* work moved to
    # another thread — an `iter_batches` over a source, a decode stage — so it must read the
    # same `Config` the consumer does. A bare thread reads every context variable at its
    # default, which quietly reverted the morsel size, the memory cap and the credential
    # scope of everything pulled through here.
    start_context_thread(_worker, name="batcher-prefetch", daemon=True)
    try:
        while True:
            error, item = q.get()
            if error is not None:
                raise error
            if item is done:
                return
            yield item
    finally:
        stop.set()
        # Free one slot so a producer parked on a full queue notices the flag on its next
        # attempt rather than after its timeout.
        with contextlib.suppress(queue.Empty):
            q.get_nowait()
