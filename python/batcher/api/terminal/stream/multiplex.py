"""Readiness-based merging of several streams, so an idle one cannot stall the rest.

A union of streams and a stream-stream join both consume more than one source from one
driver. They used to pull from each in turn, and a source's read decides when it returns,
so a branch parked on an idle topic held every other branch behind that read for as long as
it waited: a busy stream beside a quiet one stopped emitting, which on an endless source is
indistinguishable from a hang.

`multiplex` gives each input its own reader thread and hands the consumer whichever batch is
ready first. Memory stays bounded: a reader parks once it has `capacity` batches waiting,
so the driver holds at most ``capacity + 1`` batches per input. Each reader thread owns its
iterator for its whole life, including closing it, because a generator cannot be advanced
or closed by two threads. The threads run under the caller's context snapshot so a
`config_context` around the query governs the reads too.

This changes *when* a batch reaches the consumer, never *which* batches do or their order
within one input. That is all the two callers need: UNION ALL is a multiset union, and the
interval join evicts one side's rows by the other side's watermark, which only that side's
own batches advance.

Layer: api (the stream terminal's driver). Per-batch, never per-row.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterable, Iterator
from typing import Any

from batcher._internal.concurrency.context import start_context_thread

__all__ = ["multiplex"]

_ITEM, _ERROR, _END = 0, 1, 2
#: How often a parked reader looks at the stop flag while it waits for room.
_POLL_SECONDS = 0.05


def multiplex(streams: Iterable[Iterator[Any]], *, capacity: int = 1) -> Iterator[tuple[int, Any]]:
    """Yield ``(index, item)`` from `streams` as each item becomes available.

    Args:
        streams: The input iterators. Each is consumed on its own thread.
        capacity: How many items an input may have waiting before its reader parks.

    Returns:
        A generator of ``(input index, item)`` pairs, ending once every input has ended.
        An exception raised by any input is re-raised here, and closing the generator
        tells every reader to stop at its next item.
    """
    inputs = list(streams)

    def gen() -> Iterator[tuple[int, Any]]:
        ready: queue.Queue = queue.Queue()
        room = [threading.Semaphore(capacity) for _ in inputs]
        stop = threading.Event()
        for index, stream in enumerate(inputs):
            start_context_thread(
                _read,
                index,
                stream,
                ready,
                room[index],
                stop,
                name=f"batcher-stream-input-{index}",
            )
        live = len(inputs)
        try:
            while live:
                index, kind, payload = ready.get()
                if kind == _END:
                    live -= 1
                elif kind == _ERROR:
                    raise payload
                else:
                    room[index].release()
                    yield index, payload
        finally:
            stop.set()
            # Unpark every reader waiting for room, so it can see the flag and exit.
            for sem in room:
                sem.release()

    return gen()


def _read(
    index: int,
    stream: Iterator[Any],
    ready: queue.Queue,
    room: threading.Semaphore,
    stop: threading.Event,
) -> None:
    """Drain one input into `ready`, parking while `capacity` of its items wait."""
    try:
        for item in stream:
            while not room.acquire(timeout=_POLL_SECONDS):
                if stop.is_set():
                    return
            if stop.is_set():
                return
            ready.put((index, _ITEM, item))
    except BaseException as exc:
        ready.put((index, _ERROR, exc))
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            try:
                close()
            except Exception as exc:  # a failed cleanup is still a failed input
                ready.put((index, _ERROR, exc))
        ready.put((index, _END, None))
