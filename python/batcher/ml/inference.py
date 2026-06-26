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
from typing import TYPE_CHECKING, Any

from batcher.config import active_config

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["InferencePool", "Worker", "WorkerFactory", "transformers_pipeline_encoder"]

# A worker transforms one whole batch (e.g. runs a model forward pass on its columns).
Worker = Callable[["pa.RecordBatch"], "pa.RecordBatch"]
# Builds a worker; called exactly once per pool slot so the model loads once.
WorkerFactory = Callable[[], Worker]


def _is_cuda_oom(exc: BaseException) -> bool:
    """Whether `exc` is a CUDA out-of-memory error, checked structurally so torch is
    not a hard import (the name covers `torch.cuda.OutOfMemoryError`; the message
    covers the older `RuntimeError: CUDA out of memory`)."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _empty_cuda_cache() -> None:
    """Best-effort release of cached CUDA blocks so a halved retry has room to run."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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

        merged = pa.Table.from_batches([left, right]).combine_chunks().to_batches()
        out = merged[0] if merged else left
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
        """Stream `batches` through the pool, yielding result batches in order."""
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


def transformers_pipeline_encoder(
    model: str, column: str, *, output_column: str = "prediction", task: str | None = None
) -> type:
    """A load-once class UDF that runs a HuggingFace ``transformers`` pipeline.

    The model-identifier path of `ds.ml.infer`: builds ``transformers.pipeline(task,
    model=model)`` once per worker (the load-once GPU-inference pattern) and runs it
    over each batch's `column`, appending the pipeline's primary output per row as
    `output_column`. For a classification pipeline that is the predicted ``label``;
    for text generation the ``generated_text``; otherwise the raw scalar output. Needs
    ``transformers`` (``batcher-engine[transformers]``).
    """

    class _PipelineModel:
        def __init__(self) -> None:
            try:
                from transformers import pipeline
            except ImportError as exc:  # pragma: no cover - optional extra
                from batcher._internal.errors import BackendError

                msg = "ds.ml.infer(<model id>) needs: pip install 'batcher-engine[transformers]'"
                raise BackendError(msg) from exc
            self._pipe = pipeline(task, model=model)

        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            import pyarrow as pa

            results = self._pipe(batch.column(column).to_pylist())
            out_col = pa.array([_primary_output(r) for r in results])
            if output_column in batch.schema.names:
                idx = batch.schema.get_field_index(output_column)
                return batch.set_column(idx, output_column, out_col)
            return batch.append_column(output_column, out_col)

    return _PipelineModel


def _primary_output(result: Any) -> Any:
    """The single salient value of one pipeline result row (label / text / scalar)."""
    if isinstance(result, list):  # token/aggregated pipelines nest a list per row
        result = result[0] if result else None
    if isinstance(result, dict):
        for key in ("label", "generated_text", "summary_text", "translation_text", "answer"):
            if key in result:
                return result[key]
    return result
