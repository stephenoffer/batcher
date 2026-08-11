"""The worker pool itself: dynamic batching, OOM survival, and bounded dispatch.

The engine's `map_batches` runs a Python callback per batch, but reloads nothing:
for model inference you want the expensive resource (the model, the tokenizer, the
GPU context) loaded **once per worker** and reused across batches. `InferencePool`
does that — a fixed pool of workers, each built once from a factory, fed
dynamically-sized batches and run concurrently while preserving input order.

This is the control-plane orchestration twin of the native-pipeline primitives in
the Rust `bc-udf` crate (`OpaqueOperator`/`Rebatcher`/`BatchSizeController`): the
same dynamic-batching idea, applied here over whole Arrow batches for the
actor-pool path. Workers receive whole `pyarrow.RecordBatch`es — never per-row
Python — so the control plane never touches a tuple in the hot path.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
from typing import TYPE_CHECKING, Any

from batcher._internal.mathx import clamp
from batcher.config import active_config

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["InferencePool", "Worker", "WorkerFactory"]

Worker = Callable[["pa.RecordBatch"], "pa.RecordBatch"]
"""Transforms one whole batch (e.g. runs a model forward pass over its columns)."""

WorkerFactory = Callable[[], Worker]
"""Builds a `Worker`, called exactly once per pool slot so the model loads once."""


def _is_cuda_oom(exc: BaseException) -> bool:
    """Whether `exc` is an accelerator out-of-memory error.

    Delegates to the one classifier in `_internal.hardware.devices`, because the same question
    is asked by the distributed fault policy and by the OOM ladder, and three copies of a
    substring list is how one of them silently stops recognizing a vendor the others handle
    (this copy did not know HIP's phrasing, so the halving retry was inert on every ROCm host).
    """
    from batcher._internal.hardware.devices import is_device_oom

    return is_device_oom(exc)


def _empty_cuda_cache() -> None:
    """Best-effort release of cached accelerator blocks so a halved retry has room to run.

    Delegates to `release_device_cache`, which collects Python garbage *before* emptying the
    cache. That ordering is what the local version was missing: on this path the failing
    batch's tensors are still referenced by the exception's traceback frames, so the allocator
    cannot return their blocks and the retry re-ran with much of the memory that just
    overflowed."""
    from batcher._internal.hardware.devices import release_device_cache

    with contextlib.suppress(Exception):
        release_device_cache()


def _run_with_oom_retry(
    worker: Worker, batch: pa.RecordBatch, on_oom: Callable[[int], None] | None = None
) -> tuple[pa.RecordBatch, float]:
    """Run `worker(batch)`, surviving a device OOM by releasing memory and shrinking.

    A transient VRAM spike (a fragmented allocator, a co-tenant model) can OOM a batch that
    would fit at half the size. Rather than fail the job, free the cache and run the two halves
    independently, concatenating their per-row-independent inference outputs — equivalent to
    the whole batch. Re-raises once a single row still OOMs (a genuine over-allocation, not a
    too-large batch) or for any non-OOM error.

    Two refinements over halving unconditionally, both from `classify_oom`:

    * A **fragmented** allocator is retried once at the *same* size after its cached blocks are
      released. The memory was there all along in pieces too small to serve the request, so
      halving throws away half the throughput to work around a problem that releasing the
      cache just solved. If it fails again the size really is the issue and the split proceeds.
    * An **occupied** device — one whose memory a co-tenant holds — is not split at all. This
      process's batch is not what filled the device, so halving it sixteen times recovers
      nothing and turns one placement mistake into minutes of wasted GPU time. The error is
      re-raised with what was measured, which is what a scheduler needs to act on.

    Args:
        worker: The per-batch transform to run.
        batch: The batch to run it on.
        on_oom: Called with the row count that failed, before the batch is split. This is how
            the batch-size controller learns of a failure it could not predict; without it the
            halved batch succeeds, reports good throughput, and the climb grows straight back
            into the same OOM.

    Returns:
        `(output_batch, latency_ms)`; on a split, latency is the halves' sum.
    """
    start = time.perf_counter()
    try:
        out = worker(batch)
        return out, (time.perf_counter() - start) * 1000.0
    except Exception as exc:
        if not _is_cuda_oom(exc):
            raise
        from batcher._internal.hardware.devices import classify_oom

        verdict = classify_oom(exc)
        if not verdict.should_shrink:
            raise
        if on_oom is not None:
            with contextlib.suppress(Exception):  # telemetry must never fail a recoverable batch
                on_oom(batch.num_rows)
        _empty_cuda_cache()
        if verdict.should_retry_same_size:
            try:
                out = worker(batch)
                return out, (time.perf_counter() - start) * 1000.0
            except Exception as retry_exc:
                if not _is_cuda_oom(retry_exc):
                    raise
        if batch.num_rows <= 1:
            raise
        mid = batch.num_rows // 2
        left, left_ms = _run_with_oom_retry(worker, batch.slice(0, mid), on_oom)
        right, right_ms = _run_with_oom_retry(worker, batch.slice(mid), on_oom)
        import pyarrow as pa

        # Concatenate the halves into a single batch. `concat_batches` keeps every
        # row (and raises a clear error on a genuine >2 GiB offset overflow) — unlike
        # `Table.from_batches(...).combine_chunks().to_batches()[0]`, which splits into
        # multiple batches at the 32-bit offset limit and would then silently DROP all
        # but the first, losing rows for large binary/string/list inference outputs.
        out = pa.concat_batches([left, right])
        return out, left_ms + right_ms


class _DynamicBatcher:
    """Coalesce/split incoming batches to ~`target` rows (whole-batch Arrow ops)."""

    def __init__(self, target: int) -> None:
        import pyarrow as pa

        self._pa = pa
        self._target = max(1, target)
        self._buf: list[pa.RecordBatch] = []
        self._rows = 0

    def set_target(self, target: int) -> None:
        self._target = max(1, target)

    def push(self, batch: pa.RecordBatch) -> list[pa.RecordBatch]:
        if batch.num_rows == 0:
            return []
        self._buf.append(batch)
        self._rows += batch.num_rows
        if self._rows < self._target:
            return []
        return self._drain()

    def _drain(self) -> list[pa.RecordBatch]:
        table = self._pa.Table.from_batches(self._buf)
        self._buf = []
        self._rows = 0
        out: list[pa.RecordBatch] = []
        offset = 0
        while table.num_rows - offset >= self._target:
            # `combine_chunks()` splits into MULTIPLE batches at the 32-bit offset limit,
            # so a target-row slice of large binary/string/list data is not always one
            # batch. Keep every piece: taking `[0]` here silently dropped rows for exactly
            # the wide inference outputs `_run_with_oom_retry` documents at the top of
            # this module.
            out.extend(table.slice(offset, self._target).combine_chunks().to_batches())
            offset += self._target
        remainder = table.slice(offset).combine_chunks().to_batches()
        if remainder:
            self._buf = remainder
            self._rows = sum(b.num_rows for b in remainder)
        return out

    def flush(self) -> list[pa.RecordBatch]:
        """Emit everything still buffered, as one batch per 2 GiB offset span."""
        if self._rows == 0:
            return []
        batches = self._pa.Table.from_batches(self._buf).combine_chunks().to_batches()
        self._buf = []
        self._rows = 0
        return batches


class _LatencyController:
    """PID over relative latency error → target batch rows (port of bc-udf's
    `BatchSizeController`; scale-free, anti-windup, bounds-clamped)."""

    def __init__(self, target_ms: float, min_rows: int, max_rows: int, initial: int) -> None:
        self._target = target_ms
        self._min = max(1, min_rows)
        self._max = max(self._min, max_rows)
        self._cur = float(min(max(initial, self._min), self._max))
        self._pid = active_config().pid  # gains/clamps shared with bc-udf
        self._integral = 0.0
        self._prev = 0.0

    def update(self, observed_ms: float) -> int:
        if observed_ms != observed_ms or observed_ms < 0 or self._target <= 0:  # NaN guard
            return self.current()
        pid = self._pid
        error = (self._target - observed_ms) / self._target
        self._integral = clamp(self._integral + error, -pid.integral_clamp, pid.integral_clamp)
        derivative = error - self._prev
        self._prev = error
        raw = pid.kp * error + pid.ki * self._integral + pid.kd * derivative
        adjustment = clamp(raw, -pid.max_step_fraction, pid.max_step_fraction)
        self._cur = min(float(self._max), max(float(self._min), self._cur * (1.0 + adjustment)))
        return self.current()

    def current(self) -> int:
        return int(min(self._max, max(self._min, round(self._cur))))


class InferencePool:
    """Run a stateful per-batch transform across a fixed worker pool.

    Each worker is built once from `worker_factory` (so a model loads once per
    worker, not once per batch) and reused for every batch it handles. Inputs are
    re-chunked to `target_batch_rows` and dispatched concurrently; results are
    yielded in input order. When `target_latency_ms` is set, the batch size is
    retuned online toward that per-batch latency.

    Examples:
        .. doctest::

            >>> from batcher.ml import InferencePool  # doctest: +SKIP
            >>> def make_worker():  # doctest: +SKIP
            ...     model = load_model()  # once per worker
            ...     return lambda batch: model(batch)
            >>> pool = InferencePool(make_worker, num_workers=4)  # doctest: +SKIP
            >>> for out in pool.run(ds.iter_batches()):  # doctest: +SKIP
            ...     ...

    Args:
        worker_factory: zero-arg callable returning a `Worker`; invoked exactly
            `num_workers` times.
        num_workers: pool size (clamped to >= 1).
        target_batch_rows: rows per dispatched batch.
        target_latency_ms: if set, dynamically retune the batch size toward this
            per-batch latency.
        min_batch_rows / max_batch_rows: bounds for the dynamic size.
        max_inflight: cap on submitted-but-unyielded batches, which bounds resident
            memory to the pool rather than the dataset. Defaults to ``num_workers * 4``.
        learned_hub / learned_signature: opt into the persistent batch-size warm-start
            (throughput objective) — the pool records its plateau under `learned_signature`
            in the `MetadataHub` so the next run of the same job starts near the tuned size
            instead of cold-climbing again. Both default to None (the pure hill-climb).
    """

    def __init__(
        self,
        worker_factory: WorkerFactory,
        *,
        num_workers: int = 4,
        target_batch_rows: int = 1024,
        target_latency_ms: float | None = None,
        objective: str = "latency",
        vram_sampler: Callable[[], float | None] | None = None,
        min_batch_rows: int = 1,
        max_batch_rows: int = 65_536,
        max_inflight: int | None = None,
        learned_hub: Any = None,
        learned_signature: str | None = None,
    ) -> None:
        self._factory = worker_factory
        self._num_workers = max(1, num_workers)
        # Bound the submitted-but-unyielded batches. Without a bound, `run` submits the
        # ENTIRE input before the first slow consumer pull returns (the non-blocking drain
        # only pops an already-done head), so every input batch and every result batch is
        # resident at once — an OOM proportional to the dataset, not to the pool. Four
        # batches per worker is Ray's guidance for keeping a pool fed: enough to hide the
        # dispatch/gather round-trip, small enough that memory scales with the pool.
        self._max_inflight = (
            max(1, max_inflight) if max_inflight is not None else self._num_workers * 4
        )
        self._target_rows = max(1, target_batch_rows)
        self._batcher = _DynamicBatcher(self._target_rows)
        # Two adaptive objectives (see ml/autobatch). "latency" drives a PID toward
        # `target_latency_ms` (online serving); "throughput" hill-climbs batch size
        # toward max rows/sec under a VRAM cap (offline batch — the Ray Data bulk).
        self._latency_ctl = (
            _LatencyController(target_latency_ms, min_batch_rows, max_batch_rows, self._target_rows)
            if target_latency_ms is not None and objective == "latency"
            else None
        )
        self._throughput_ctl = None
        # Default the VRAM sampler so the throughput autobatcher's predictive cap is
        # actually fed live data (it is otherwise inert — no caller wires one). The
        # default returns None on a GPU-less host, so the guard stays a no-op there.
        if vram_sampler is None and objective == "throughput":
            from batcher.ml.gpu import sample_gpu_vram_fraction

            vram_sampler = sample_gpu_vram_fraction
        self._vram_sampler = vram_sampler
        if objective == "throughput":
            from batcher.ml.autobatch import ThroughputController

            # Thread the learned-stats hub + signature through so a recurring inference job
            # warm-starts the batch-size climb from its last plateau instead of paying the
            # full cold-start ramp every run. Both default to None → the pure hill-climb.
            self._throughput_ctl = ThroughputController(
                min_rows=min_batch_rows,
                max_rows=max_batch_rows,
                initial=self._target_rows,
                hub=learned_hub,
                signature=learned_signature,
            )
        elif objective != "latency":
            raise ValueError(f"objective must be 'latency' or 'throughput', got {objective!r}")
        # The controllers are read-modify-written from two threads now: `_next_target` from
        # the consumer as it drains results, and `_note_oom` from whichever worker thread hit
        # the failure. Neither is a hot path — one call per batch — so a plain lock is the
        # right cost, and without it an OOM's ceiling could be overwritten by a concurrent
        # `update` that had already read the pre-failure size.
        self._ctl_lock = threading.Lock()

    def _note_oom(self, rows: int) -> None:
        """Lower the batch-size target after a batch ran out of device memory.

        The retry recovers the *rows*; this is what stops the same failure recurring on the
        next batch. Both controllers need it and neither could infer it: the throughput
        controller is fed rows-per-second, and a batch that OOMs and is retried in halves
        produces a perfectly respectable one, so nothing in the measurement says a failure
        happened at all.

        Applied to the dispatch batcher too, so the *next* batch is actually built smaller
        rather than merely being re-split by the retry after failing again.
        """
        target: int | None = None
        with self._ctl_lock:
            target = self._oom_target(rows)
        if target is not None:
            self._batcher.set_target(target)

    def _oom_target(self, rows: int) -> int | None:
        """The post-OOM batch-size target from whichever controller is engaged."""
        if self._throughput_ctl is not None:
            return self._throughput_ctl.note_oom(rows=rows)
        if self._latency_ctl is not None:
            # The latency PID has no failure input, and adding one would mean giving it a
            # second, incommensurable objective. Halving the batcher's target directly is the
            # equivalent action: the PID keeps steering from latency and simply does so from a
            # size that fits, converging back up if the OOM really was transient.
            return max(1, rows // 2)
        return None

    def _next_target(self, out: pa.RecordBatch, latency_ms: float) -> int | None:
        """The next batch-size target from the active controller, or None if neither
        is engaged (a fixed batch size)."""
        with self._ctl_lock:
            return self._observe(out, latency_ms)

    def _observe(self, out: pa.RecordBatch, latency_ms: float) -> int | None:
        """`_next_target`'s body, called with the controller lock already held."""
        if self._latency_ctl is not None:
            return self._latency_ctl.update(latency_ms)
        if self._throughput_ctl is not None:
            # A non-positive latency is a degenerate measurement (clock granularity,
            # an empty batch): skip it rather than feed an infinite throughput, which
            # would poison `best_throughput` so nothing ever "improves" again and the
            # controller freezes. Keep the current target until a real reading lands.
            if latency_ms <= 0:
                return self._throughput_ctl.current()
            throughput = out.num_rows / (latency_ms / 1000.0)
            vram = self._vram_sampler() if self._vram_sampler is not None else None
            return self._throughput_ctl.update(throughput, vram)
        return None

    def _publish_batch(self, rows: int, latency_ms: float, blocked_ms: float, pending: int) -> None:
        """Report one finished micro-batch and the pool's depth on the event bus.

        The pool already measures every one of these for its own controller and then threw
        them away, so a multi-hour batch-inference job — the workload with the longest gap
        between "started" and "finished" of anything the engine runs — reported no progress
        at all. `observe.InferenceProgress` and the `inference` metrics section were both
        written against these events and neither had a publisher.

        `blocked_ms` is the signal worth having: it is the time the *consumer* spent waiting
        on this pool, so a large value means the pool is the bottleneck and a near-zero one
        means the pool is starved by whatever feeds it. The two look identical in rows/sec
        and want opposite fixes — more workers, or a faster source.

        A no-op when nothing is listening, which is the default; this runs once per
        micro-batch, so it must cost a tuple check on the common path.
        """
        from batcher._internal import events

        if not events.listening():
            return
        events.publish(
            events.INFER,
            name="inference",
            rows=rows,
            latency_ms=latency_ms,
            blocked_ms=blocked_ms,
        )
        events.publish(events.POOL, name="inference", size=self._num_workers, pending=pending)

    def run(self, batches: Iterable[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        """Stream `batches` through the pool, yielding result batches in order.

        Examples:
            .. doctest::

                >>> pool = InferencePool(make_worker, num_workers=4)  # doctest: +SKIP
                >>> outputs = list(pool.run(ds.iter_batches()))  # doctest: +SKIP

        Args:
            batches: an iterable of `pyarrow.RecordBatch` to run through the workers.

        Returns:
            An iterator of the workers' output batches, in input order.
        """
        workers: Queue[Worker] = Queue()
        built: list[Worker] = []
        # A factory that raises partway through — the second of eight models finding the device
        # already full is the ordinary way this happens — must not leave the models it already
        # built resident. Each one can hold a CUDA context and a set of weights, so leaking them
        # turns a recoverable "size the pool smaller" into an OOM that outlives the failure and
        # takes the next attempt with it.
        try:
            for _ in range(self._num_workers):
                worker = self._factory()
                built.append(worker)
                workers.put(worker)
        except BaseException:
            _close_workers(built)
            raise

        def dispatch(batch: pa.RecordBatch) -> tuple[pa.RecordBatch, float]:
            worker = workers.get()
            try:
                return _run_with_oom_retry(worker, batch, self._note_oom)
            finally:
                workers.put(worker)

        pending: deque[Future[tuple[pa.RecordBatch, float]]] = deque()
        try:
            with ThreadPoolExecutor(max_workers=self._num_workers) as pool:

                def pop_head() -> pa.RecordBatch:
                    blocked = time.perf_counter()
                    head = pending.popleft()
                    done = head.done()
                    out, latency_ms = head.result()  # blocks until the head is done
                    # Time the *consumer* spent waiting on this pool, which is what
                    # distinguishes a saturated pool from a starved one — and is zero
                    # whenever the head had already finished before we asked.
                    blocked_ms = 0.0 if done else (time.perf_counter() - blocked) * 1000.0
                    self._publish_batch(out.num_rows, latency_ms, blocked_ms, len(pending))
                    target = self._next_target(out, latency_ms)
                    if target is not None:
                        self._batcher.set_target(target)
                    return out

                def drain(block: bool) -> Iterator[pa.RecordBatch]:
                    while pending and (block or pending[0].done()):
                        yield pop_head()

                def submit(rebatched: pa.RecordBatch) -> Iterator[pa.RecordBatch]:
                    """Submit one batch, first yielding results down to the in-flight bound.

                    Blocking on the head here is what applies backpressure all the way to the
                    source iterator: `run` stops pulling input while the pool is saturated."""
                    while len(pending) >= self._max_inflight:
                        yield pop_head()
                    pending.append(pool.submit(dispatch, rebatched))

                try:
                    for batch in batches:
                        for rebatched in self._batcher.push(batch):
                            yield from submit(rebatched)
                            yield from drain(block=False)
                    for tail in self._batcher.flush():
                        yield from submit(tail)
                    yield from drain(block=True)
                finally:
                    # A consumer that stops early — a `limit`, a `break`, an exception — leaves
                    # up to `max_inflight` batches submitted. The executor's shutdown waits for
                    # every one of them, so without this a query that read ten rows of a
                    # streamed inference still paid for the whole in-flight window of forward
                    # passes before returning. Cancelling only affects batches that have not
                    # started; the ones already on a device still finish, as they must.
                    for future in pending:
                        future.cancel()
        finally:
            _close_workers(built)


def _close_workers(workers: list[Worker]) -> None:
    """Call each worker's optional ``close()`` once the pool is done with it.

    `num_workers` models are the point of this pool, and each one can hold a CUDA context,
    an HTTP session, or a database handle. Without this they were released only whenever the
    garbage collector happened to reach them — so a script running two pools back to back
    could hold both generations of models in VRAM at once and OOM on the second. This mirrors
    `core.udf.lifecycle.teardown_udf`, including its best-effort contract: the results are
    already produced, so a failing `close` must not fail the run.

    Deduplicated by identity, because a factory is allowed to hand every slot the *same*
    object (`apply._apply_udf_autobatch` does exactly that to share one loaded model across
    the dispatch slots), and closing one model `num_workers` times is not a teardown.
    """
    seen: set[int] = set()
    for worker in workers:
        if id(worker) in seen:
            continue
        seen.add(id(worker))
        close = getattr(worker, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()
