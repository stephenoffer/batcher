"""Order-preserving, **byte-bounded** read-ahead over a sequence of files.

`FileSource.iter_batches` wants two things that pull against each other: overlap the
decode of several files (a serial read, not compute, is the ceiling on object storage),
and never hold more than a bounded amount of data. The obvious implementation —
submit `list(iter_file(f))` per file to a pool — gets the overlap and loses the bound:
it materializes each file *whole*, so peak memory is `depth x decoded-file-size`,
independent of the batch size. On a 1 GB shard at depth 16 that is ~16 GB per worker,
and for a multimodal corpus (one row can be a 200 MB video) it is unbounded in practice.

This module keeps the overlap and restores the bound by streaming *batches* rather than
files: each in-flight file is decoded by its own worker into its own small queue, and a
producer blocks once its queue holds `per_file_bytes` worth of undelivered data. The
consumer drains the head file's queue in order, so the output order is byte-identical to
a serial read.

**Why a per-file budget rather than one global one.** A single shared byte budget
deadlocks. Producers for files 1..n-1 run ahead and can fill the whole budget, then the
file-0 producer blocks acquiring credit while the consumer — which only ever releases
credit by consuming *file 0* — waits for a batch that will never come. Giving each
in-flight file its own share removes the cycle: the head file's producer is throttled
only by the consumer that is actively draining it. The total bound is preserved
(`depth x per_file_bytes`), and it is the reason this is a module rather than four lines
inline.
"""

from __future__ import annotations

import itertools
import queue
import threading
from collections.abc import Callable, Iterable, Iterator

import pyarrow as pa

__all__ = ["ordered_readahead"]

# A queue slot count high enough that a producer is never woken per batch, low enough
# that tiny batches cannot make the count (rather than the bytes) the binding limit.
_SLOTS_PER_FILE = 64


class _Done:
    """End-of-file marker placed on a per-file queue."""

    __slots__ = ()


_DONE = _Done()


class _FileStream:
    """One in-flight file: a worker thread decoding into a byte-bounded queue."""

    __slots__ = ("_budget", "_cond", "_outstanding", "_q", "_stop", "_thread")

    def __init__(
        self,
        path: str,
        iter_file: Callable[[str], Iterator[pa.RecordBatch]],
        per_file_bytes: int,
    ) -> None:
        self._q: queue.Queue[pa.RecordBatch | _Done | BaseException] = queue.Queue(
            maxsize=_SLOTS_PER_FILE
        )
        self._budget = per_file_bytes
        self._outstanding = 0
        self._stop = False
        self._cond = threading.Condition()
        self._thread = threading.Thread(
            target=self._run, args=(path, iter_file), name="batcher-readahead"
        )
        self._thread.start()

    def _acquire(self, nbytes: int) -> bool:
        """Reserve credit for one batch. False once the consumer has abandoned this file."""
        with self._cond:
            # `self._outstanding == 0` lets a single batch larger than the whole budget
            # through rather than blocking forever — one oversized row group (or one
            # video blob) must still be deliverable, and holding exactly one of them is
            # the smallest bound this can honour.
            while (
                not self._stop
                and self._outstanding > 0
                and self._outstanding + nbytes > self._budget
            ):
                self._cond.wait()
            if self._stop:
                return False
            self._outstanding += nbytes
            return True

    def release(self, nbytes: int) -> None:
        with self._cond:
            self._outstanding -= nbytes
            self._cond.notify_all()

    def _run(self, path: str, iter_file: Callable[[str], Iterator[pa.RecordBatch]]) -> None:
        try:
            for batch in iter_file(path):
                if not self._acquire(batch.nbytes):
                    return
                # A bounded queue can also park a producer, so the same abandonment check
                # has to cover the hand-off itself, not only the credit.
                while True:
                    try:
                        self._q.put(batch, timeout=0.1)
                        break
                    except queue.Full:
                        if self._stop:
                            return
        except BaseException as exc:
            self._q.put(exc)
        else:
            self._q.put(_DONE)

    def close(self) -> None:
        """Abandon this file and join its producer.

        Threads are non-daemon on purpose: a daemon producer parked inside a pyarrow
        decode is still holding Arrow buffers when the interpreter tears down, which
        aborts the process (`terminate called without an active exception`). Joining is
        cheap because the stop flag is checked at both places a producer can block.
        """
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        # Unblock a producer parked on a full queue by making room for it.
        while self._thread.is_alive():
            try:
                self._q.get_nowait()
            except queue.Empty:
                self._thread.join(timeout=0.05)
        self._thread.join()

    def drain(self) -> Iterator[pa.RecordBatch]:
        """Yield this file's batches in order, releasing each one's credit after it is consumed."""
        while True:
            item = self._q.get()
            if isinstance(item, _Done):
                return
            if isinstance(item, BaseException):
                raise item
            try:
                yield item
            finally:
                # Released *after* the consumer is done with the batch, so the bound
                # covers data the pipeline still holds, not merely data in the queue.
                self.release(item.nbytes)


def ordered_readahead(
    files: Iterable[str],
    iter_file: Callable[[str], Iterator[pa.RecordBatch]],
    *,
    depth: int,
    max_bytes: int,
) -> Iterator[pa.RecordBatch]:
    """Stream every file's batches in file order, decoding `depth` files concurrently.

    Peak undelivered data is bounded by `max_bytes` regardless of how large the files
    are, so a 1 GB shard and a 4 KB one cost the same memory.

    Args:
        files: The paths to read, in the order their batches must be yielded.
        iter_file: Opens one path and yields its batches. Called on a worker thread.
        depth: How many files to decode concurrently.
        max_bytes: Ceiling on the total undelivered decoded bytes held across all
            in-flight files.

    Returns:
        An iterator over every file's batches, in file order.
    """
    depth = max(1, depth)
    per_file_bytes = max(1, max_bytes // depth)
    remaining = iter(files)
    in_flight: list[_FileStream] = [
        _FileStream(f, iter_file, per_file_bytes) for f in itertools.islice(remaining, depth)
    ]
    head = 0
    try:
        while head < len(in_flight):
            stream = in_flight[head]
            yield from stream.drain()
            stream.close()
            in_flight[head] = None  # type: ignore[call-overload]  # let the queue be collected
            head += 1
            nxt = next(remaining, None)
            if nxt is not None:
                in_flight.append(_FileStream(nxt, iter_file, per_file_bytes))
    finally:
        # A consumer that stops early (a `LIMIT`, an exception, a `close()` on the
        # generator) leaves the remaining producers parked on their byte credit or on a
        # full queue. Closing each one wakes it, lets it observe that its file has been
        # abandoned, and joins it — so no thread survives holding Arrow buffers into
        # interpreter shutdown.
        for stream in in_flight[head:]:
            if stream is not None:
                stream.close()
