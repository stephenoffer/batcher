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
seq == pipelined contract the rest of the engine also holds. The multi-node placement +
Arrow-Flight hand-off layer (`dist/streaming/pipeline.py`) mirrors this same shape.

Shutdown is **deterministic**: no worker thread outlives `run_pipeline`, whether the
consumer drains the iterator, abandons it mid-stream, or a stage raises. The generator
joins every thread in a `finally`, which is reachable on all three paths.
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
    depth).

    `num_gpus` is declared but **not yet consumed**: single-node execution ignores it,
    and the distributed scheduler (`dist/streaming/pipeline.py`) reads its resource
    class from the logical plan, not from this field. It is documented as a placement
    hint in `docs/ml/model-serving-patterns.md`, so treat it as a promise the engine
    still owes rather than as a knob that does anything today.

    Examples:
        .. doctest::

            >>> from batcher.ml import Stage
            >>> stage = Stage(factory=lambda: (lambda batch: batch), name="decode")
            >>> stage.credits  # two batches may sit between this stage and the next
            2
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

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.ml import Stage, run_pipeline
            >>> def double():  # the worker is built once, on the stage's own thread
            ...     return lambda b: pa.record_batch(
            ...         {"x": [v * 2 for v in b.column("x").to_pylist()]}
            ...     )
            >>> batch = pa.record_batch({"x": [1, 2, 3]})
            >>> [b.column("x").to_pylist() for b in run_pipeline([batch], [Stage(double)])]
            [[2, 4, 6]]

    Args:
        batches: an iterable of `pyarrow.RecordBatch` to feed the first stage.
        stages: the stages to run, in order (empty → `batches` passes through).

    Returns:
        An iterator of the final stage's batches, in input order.
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
        t = threading.Thread(
            target=pump,
            args=(stage, in_q, queues[i]),
            name=f"batcher-ml-pipeline-{i}-{stage.name}",
            daemon=True,
        )
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

    feeder = threading.Thread(target=feed, name="batcher-ml-pipeline-feed", daemon=True)
    feeder.start()

    # Drain the final stage with the same stop-aware get: if a stage errors while the
    # caller has paused (final queue full → last stage abandoned its put without a
    # `_DONE`), `_get` still returns `_DONE` on stop, so the consumer can't deadlock
    # waiting for a sentinel that will never arrive.
    #
    # The drain sits in a `try`/`finally` so shutdown is reached on ALL three exits:
    # a full drain, a stage error, and a consumer that abandons the generator (which
    # throws `GeneratorExit` in at the `yield`). Without it, walking away from the
    # iterator left the feeder blocked on a full queue until a poll happened to notice
    # `stop` — which nothing ever set — so the threads outlived the call.
    try:
        final_q = queues[-1]
        while True:
            item = _get(final_q)
            if item is _DONE:
                break
            yield item
    finally:
        # Joins are UNBOUNDED on purpose: the previous `timeout=1.0` let a thread
        # survive the call, which is a leak, not a shutdown. Every blocking queue op
        # here is stop-aware and polls at `_POLL_S`, so once `stop` is set each thread
        # exits within one poll. The only way to block is a stage worker that itself
        # never returns, and surfacing that hang beats leaking a thread that holds a
        # GPU context or a model.
        stop.set()
        feeder.join()
        for t in threads:
            t.join()
    if error:
        raise error[0]
