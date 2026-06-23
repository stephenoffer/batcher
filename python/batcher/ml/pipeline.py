"""Multi-stage streaming pipeline with credit-based backpressure (the GPU-feeding moat).

Ray Data's #1 bottleneck is GPU starvation: a slow read/preprocess stage leaves the
GPU inference stage idle. The fix is to run each stage concurrently and overlap them
— while the GPU stage processes batch *k*, the CPU readers prepare *k+1* — bounded by
**credits** so a slow stage throttles its upstream (no unbounded buffering, no
object-store spill). This module is that pipeline, single-node and in-process: a
chain of `Stage`s, each on its own thread, connected by bounded queues (1 credit = 1
in-flight batch). A slow consumer fills its input queue, which blocks the producer —
backpressure all the way to the source, so peak memory is `sum(stage.credits)`
batches, not the whole stream.

The result is exactly the sequentially-composed stages (each stage preserves order;
the queues are FIFO), so this is a faster *scheduling* of the same computation — the
seq == pipelined contract the rest of the engine also holds. Each `Stage` carries a
`num_gpus` hint the distributed scheduler uses to place CPU stages on CPU workers and
GPU stages on GPU actors; the multi-node placement + Arrow-Flight hand-off layer on
top of this same shape, and is exercised on cluster/GPU hardware.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["Stage", "run_pipeline"]

# Poll interval for stop-aware blocking queue ops: a thread blocked on a full/empty
# queue must still notice `stop` (set on error or completion) and exit, or it would
# leak after a downstream stage dies. Small enough to be responsive, large enough to
# add no measurable overhead to the hot path.
_POLL_S = 0.05

# A stage transforms one whole batch (preprocess, decode, model forward, ...).
StageWorker = Callable[["pa.RecordBatch"], "pa.RecordBatch"]
# Built once per stage thread, so a model/tokenizer/GPU context loads a single time.
StageFactory = Callable[[], StageWorker]


@dataclass(frozen=True, slots=True)
class Stage:
    """One pipeline stage: a worker built once, and the credit window to its output.

    `credits` is the max number of finished batches that may sit between this stage
    and the next before this stage blocks — the backpressure knob (and the prefetch
    depth). `num_gpus` is a placement hint for the distributed scheduler (0 = CPU);
    it does not affect single-node execution.
    """

    factory: StageFactory
    credits: int = 2
    num_gpus: float = 0.0
    name: str = "stage"


# Sentinel pushed through every queue to signal end-of-stream in order.
_DONE = object()


def run_pipeline(
    batches: Iterable[pa.RecordBatch], stages: list[Stage]
) -> Iterator[pa.RecordBatch]:
    """Stream `batches` through `stages`, overlapped and credit-bounded, in order.

    Each stage runs on its own thread (its worker built once there) and reads from a
    bounded queue, so stages run concurrently and a slow stage throttles its upstream.
    Yields the final stage's output batches in input order. Equivalent to applying the
    stages in sequence to each batch — only faster, because the stages overlap.

    Raises the first exception any stage raised (propagated to the consumer), after
    signaling the other threads to stop.
    """
    if not stages:
        yield from batches
        return

    # One bounded queue per stage output; queue i feeds stage i. The producer feeds
    # queue 0. maxsize = credits bounds in-flight batches between stages.
    queues: list[Queue] = [Queue(maxsize=max(1, s.credits)) for s in stages]
    error: list[BaseException] = []
    stop = threading.Event()

    def _put(q: Queue, item: object) -> bool:
        """Put `item`, but abandon (return False) if `stop` is set while blocked on a
        full queue — so a producer never hangs after its consumer has died."""
        while True:
            try:
                q.put(item, timeout=_POLL_S)
                return True
            except Full:
                if stop.is_set():
                    return False

    def _get(q: Queue) -> object:
        """Get the next item, or `_DONE` if `stop` is set while blocked on an empty
        queue — so a consumer never hangs after its producer has died."""
        while True:
            try:
                return q.get(timeout=_POLL_S)
            except Empty:
                if stop.is_set():
                    return _DONE

    def pump(stage: Stage, in_q: Queue, out_q: Queue) -> None:
        try:
            worker = stage.factory()  # built once on this thread
            while True:
                item = _get(in_q)
                if item is _DONE:
                    _put(out_q, _DONE)
                    return
                if not _put(out_q, worker(item)):
                    return  # stop set (downstream died) → exit instead of leaking
        except BaseException as exc:  # propagate to the consumer; unblock the pipeline
            error.append(exc)
            stop.set()
            _put(out_q, _DONE)

    threads: list[threading.Thread] = []
    for i, stage in enumerate(stages):
        in_q = queues[i - 1] if i > 0 else Queue(maxsize=max(1, stages[0].credits))
        if i == 0:
            source_q = in_q  # the producer feeds this
        t = threading.Thread(target=pump, args=(stage, in_q, queues[i]), daemon=True)
        t.start()
        threads.append(t)

    # Feed the source on its own thread; the stop-aware put applies backpressure (it
    # blocks while the pipeline is full) yet abandons cleanly if a stage dies, so the
    # feeder can't leak after a downstream error.
    def feed() -> None:
        try:
            for b in batches:
                if not _put(source_q, b):
                    return  # stop set (a stage died) → abandon feeding
        finally:
            _put(source_q, _DONE)

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()

    # Drain the final stage with the same stop-aware get: if a stage errors while the
    # caller has paused (final queue full → last stage abandoned its put without a
    # `_DONE`), `_get` still returns `_DONE` on stop, so the consumer can't deadlock
    # waiting for a sentinel that will never arrive.
    final_q = queues[-1]
    while True:
        item = _get(final_q)
        if item is _DONE:
            break
        yield item

    stop.set()
    feeder.join(timeout=1.0)
    for t in threads:
        t.join(timeout=1.0)
    if error:
        raise error[0]
