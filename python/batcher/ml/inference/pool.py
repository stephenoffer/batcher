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
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
from typing import TYPE_CHECKING

from batcher.config import active_config

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["InferencePool", "Worker", "WorkerFactory"]

Worker = Callable[["pa.RecordBatch"], "pa.RecordBatch"]
"""Transforms one whole batch (e.g. runs a model forward pass over its columns)."""

WorkerFactory = Callable[[], Worker]
"""Builds a `Worker`, called exactly once per pool slot so the model loads once."""


def _is_cuda_oom(exc: BaseException) -> bool:
    """Whether `exc` is a CUDA out-of-memory error, checked structurally so torch is
    not a hard import (the name covers `torch.cuda.OutOfMemoryError`; the message
    covers the older `RuntimeError: CUDA out of memory`)."""
    if type(exc).__name__ in ("OutOfMemoryError", "ResourceExhaustedError"):
        return True
    # XLA (TPU) reports exhaustion as "RESOURCE_EXHAUSTED" rather than "out of memory", so
    # matching only the CUDA phrasing left the halving retry disabled on a TPU: the batch
    # simply failed where a CUDA host would have recovered.
    message = str(exc).lower()
    return isinstance(exc, RuntimeError) and (
        "out of memory" in message or "resource_exhausted" in message
    )


def _empty_cuda_cache() -> None:
    """Best-effort release of cached accelerator blocks so a halved retry has room to run.

    Vendor-agnostic: NVIDIA/AMD share ``torch.cuda.empty_cache`` (ROCm shims the CUDA API),
    Intel is ``torch.xpu``, Apple ``torch.mps`` — so the OOM-halving safety net works on any
    accelerator, not just CUDA. A no-op where the backend or method is absent."""
    try:
        import torch
    except Exception:
        return
    for name in ("cuda", "xpu", "mps"):
        backend = getattr(torch, name, None)
        empty = getattr(backend, "empty_cache", None)
        if empty is None:
            continue
        with contextlib.suppress(Exception):
            avail = getattr(backend, "is_available", None)
            if name == "mps" or avail is None or avail():
                empty()
    # XLA (TPU) is not a `torch.<backend>` module and has no `empty_cache`; its allocator
    # is freed by stepping the execution graph. Without this the halved retry re-ran with
    # exactly the memory that just overflowed, so the safety net could not help a TPU.
    with contextlib.suppress(Exception):
        import importlib.util

        if importlib.util.find_spec("torch_xla") is not None:
            import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]

            xm.mark_step()


def _run_with_oom_retry(worker: Worker, batch: pa.RecordBatch) -> tuple[pa.RecordBatch, float]:
    """Run `worker(batch)`, surviving a CUDA OOM by halving and retrying.

    A transient VRAM spike (a fragmented allocator, a co-tenant model) can OOM a batch
    that would fit at half the size. Rather than fail the job, free the cache and run
    the two halves independently, concatenating their per-row-independent inference
    outputs — equivalent to the whole batch. Re-raises once a single row still OOMs (a
    genuine over-allocation, not a too-large batch) or for any non-OOM error. Returns
    `(output_batch, latency_ms)`; on a split, latency is the halves' sum.
    """
    start = time.perf_counter()
    try:
        out = worker(batch)
        return out, (time.perf_counter() - start) * 1000.0
    except Exception as exc:
        if not _is_cuda_oom(exc) or batch.num_rows <= 1:
            raise
        _empty_cuda_cache()
        mid = batch.num_rows // 2
        left, left_ms = _run_with_oom_retry(worker, batch.slice(0, mid))
        right, right_ms = _run_with_oom_retry(worker, batch.slice(mid))
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
        self._integral = max(-pid.integral_clamp, min(pid.integral_clamp, self._integral + error))
        derivative = error - self._prev
        self._prev = error
        raw = pid.kp * error + pid.ki * self._integral + pid.kd * derivative
        adjustment = max(-pid.max_step_fraction, min(pid.max_step_fraction, raw))
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

            self._throughput_ctl = ThroughputController(
                min_rows=min_batch_rows, max_rows=max_batch_rows, initial=self._target_rows
            )
        elif objective != "latency":
            raise ValueError(f"objective must be 'latency' or 'throughput', got {objective!r}")

    def _next_target(self, out: pa.RecordBatch, latency_ms: float) -> int | None:
        """The next batch-size target from the active controller, or None if neither
        is engaged (a fixed batch size)."""
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
        for _ in range(self._num_workers):
            workers.put(self._factory())

        def dispatch(batch: pa.RecordBatch) -> tuple[pa.RecordBatch, float]:
            worker = workers.get()
            try:
                return _run_with_oom_retry(worker, batch)
            finally:
                workers.put(worker)

        pending: deque[Future[tuple[pa.RecordBatch, float]]] = deque()
        with ThreadPoolExecutor(max_workers=self._num_workers) as pool:

            def pop_head() -> pa.RecordBatch:
                out, latency_ms = pending.popleft().result()  # blocks until the head is done
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

            for batch in batches:
                for rebatched in self._batcher.push(batch):
                    yield from submit(rebatched)
                    yield from drain(block=False)
            for tail in self._batcher.flush():
                yield from submit(tail)
            yield from drain(block=True)
