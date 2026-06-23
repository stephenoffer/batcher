"""Actor-pool batch inference — the ML data plane's orchestration layer.

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

# A worker transforms one whole batch (e.g. runs a model forward pass on its columns).
Worker = Callable[["pa.RecordBatch"], "pa.RecordBatch"]
# Builds a worker; called exactly once per pool slot so the model loads once.
WorkerFactory = Callable[[], Worker]


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
            chunk = table.slice(offset, self._target).combine_chunks().to_batches()
            out.append(chunk[0])
            offset += self._target
        remainder = table.slice(offset).combine_chunks().to_batches()
        if remainder:
            self._buf = remainder
            self._rows = remainder[0].num_rows
        return out

    def flush(self) -> pa.RecordBatch | None:
        if self._rows == 0:
            return None
        batches = self._pa.Table.from_batches(self._buf).combine_chunks().to_batches()
        self._buf = []
        self._rows = 0
        return batches[0] if batches else None


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

    Example:
        >>> def make_worker():
        ...     model = load_model()            # once per worker
        ...     return lambda batch: model(batch)
        >>> pool = InferencePool(make_worker, num_workers=4, target_batch_rows=2048)
        >>> for out in pool.run(ds.iter_batches(batch_format="arrow")):
        ...     ...

    Args:
        worker_factory: zero-arg callable returning a `Worker`; invoked exactly
            `num_workers` times.
        num_workers: pool size (clamped to >= 1).
        target_batch_rows: rows per dispatched batch.
        target_latency_ms: if set, dynamically retune the batch size toward this
            per-batch latency.
        min_batch_rows / max_batch_rows: bounds for the dynamic size.
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
    ) -> None:
        self._factory = worker_factory
        self._num_workers = max(1, num_workers)
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
        """Stream `batches` through the pool, yielding result batches in order."""
        workers: Queue[Worker] = Queue()
        for _ in range(self._num_workers):
            workers.put(self._factory())

        def dispatch(batch: pa.RecordBatch) -> tuple[pa.RecordBatch, float]:
            worker = workers.get()
            try:
                start = time.perf_counter()
                out = worker(batch)
                return out, (time.perf_counter() - start) * 1000.0
            finally:
                workers.put(worker)

        pending: deque[Future[tuple[pa.RecordBatch, float]]] = deque()
        with ThreadPoolExecutor(max_workers=self._num_workers) as pool:

            def drain(block: bool) -> Iterator[pa.RecordBatch]:
                while pending and (block or pending[0].done()):
                    out, latency_ms = pending.popleft().result()
                    target = self._next_target(out, latency_ms)
                    if target is not None:
                        self._batcher.set_target(target)
                    yield out

            for batch in batches:
                for rebatched in self._batcher.push(batch):
                    pending.append(pool.submit(dispatch, rebatched))
                    yield from drain(block=False)
            tail = self._batcher.flush()
            if tail is not None:
                pending.append(pool.submit(dispatch, tail))
            yield from drain(block=True)
