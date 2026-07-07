"""Overlap a producer generator with its consumer on a background thread.

A neutral utility: a bounded look-ahead over any iterator so the producer keeps
working while the consumer processes the previous item — the pattern that pipelines
a streaming read→transform stage with the write (or H2D copy) that drains it, so
neither stage idles while the other runs. An exception from the source is re-raised
in the consumer, never swallowed (swallowing would silently truncate the stream — a
correctness bug).
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from typing import TypeVar

__all__ = ["prefetch"]

T = TypeVar("T")


def prefetch(gen: Iterator[T], depth: int = 2) -> Iterator[T]:
    """Yield from `gen`, pulling it on a background thread into a bounded queue.

    `depth` bounds the look-ahead (the queue size) so a fast producer cannot outrun a
    slow consumer without limit. `depth <= 0` disables the overlap and yields `gen`
    directly. Any exception raised while pulling `gen` is re-raised here, in order.
    """
    if depth <= 0:
        yield from gen
        return

    q: queue.Queue = queue.Queue(maxsize=depth)
    done = object()

    def _worker() -> None:
        try:
            for item in gen:
                q.put((None, item))
        except Exception as exc:  # surface it to the consumer instead of truncating
            q.put((exc, None))
        finally:
            q.put((None, done))

    threading.Thread(target=_worker, daemon=True).start()
    while True:
        error, item = q.get()
        if error is not None:
            raise error
        if item is done:
            return
        yield item
