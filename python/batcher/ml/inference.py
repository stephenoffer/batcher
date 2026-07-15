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

import contextlib
import functools
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

Worker = Callable[["pa.RecordBatch"], "pa.RecordBatch"]
"""Transforms one whole batch (e.g. runs a model forward pass over its columns)."""

WorkerFactory = Callable[[], Worker]
"""Builds a `Worker`, called exactly once per pool slot so the model loads once."""


def _is_cuda_oom(exc: BaseException) -> bool:
    """Whether `exc` is a CUDA out-of-memory error, checked structurally so torch is
    not a hard import (the name covers `torch.cuda.OutOfMemoryError`; the message
    covers the older `RuntimeError: CUDA out of memory`)."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


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


def _pipeline_accel_kwargs() -> dict[str, Any]:
    """Zero-config accelerator placement + precision for a ``transformers.pipeline``.

    On a GPU worker: put the model on the detected device and, when the GPU has fast
    half-precision tensor cores, run it in BF16/FP16 (`recommend_inference_dtype`) —
    ~2x throughput on a compute-bound model at negligible inference quality loss. A
    no-op (CPU, FP32) where half gives no speedup or torch is absent, so correctness is
    never traded for the fast path. This is the per-GPU default Ray Data users set by hand.
    """
    kwargs: dict[str, Any] = {}
    try:
        import torch

        from batcher.ml.gpu import detect_backend, recommend_inference_dtype, torch_device

        # Vendor-agnostic device placement. `detect_backend` only names a backend that is
        # actually available, so NVIDIA/AMD (cuda), Intel (xpu), Apple (mps), and TPU (xla)
        # all get placed on the accelerator — not only CUDA. transformers wants an int for
        # the primary CUDA device and a device string for the rest.
        backend = detect_backend()
        if backend != "cpu":
            dev = torch_device(backend)
            kwargs["device"] = 0 if dev == "cuda" else dev
        else:
            # First-class CPU inference: cap torch's MKL/oneDNN/BLIS/OpenBLAS thread pool to the
            # cores this actor may use, so co-located CPU actors don't each grab every host core.
            _configure_cpu_inference_threads()
        dtype = recommend_inference_dtype(backend)
        if dtype is not None:
            kwargs["torch_dtype"] = getattr(torch, dtype)
    except Exception:
        return {}
    return kwargs


def _cpu_inference_thread_target() -> int:
    """The intra-op thread count a CPU-inference process should use.

    An explicit `OMP_NUM_THREADS` (Ray sets it to the actor's `num_cpus`, and a user may pin it)
    wins, so the per-actor allocation is honored; otherwise the container's usable core count
    (cgroup/affinity-aware, via `available_cpu_count`), never the host count.
    """
    import os

    from batcher._internal.hardware import available_cpu_count

    omp = os.environ.get("OMP_NUM_THREADS", "").strip()
    if omp.isdigit() and int(omp) > 0:
        return int(omp)
    return available_cpu_count()


def _configure_cpu_inference_threads() -> int | None:
    """Cap torch's CPU math-library thread pool to the cores this process may actually use.

    torch — and the Intel MKL / oneDNN, AMD BLIS/AOCL, or OpenBLAS backend it dispatches to —
    sizes its intra-op pool to the *host* core count, so a CPU-inference actor over-subscribes
    under a cgroup quota and thrashes *catastrophically* when co-located actors each grab every
    core. Sets the pool to `_cpu_inference_thread_target()`, only ever *lowering* it (never raising
    above torch's or a caller's explicit choice). Returns the applied count, or `None` when torch
    is absent. Best-effort and idempotent; result-invariant (a thread count, not a result)."""
    try:
        import torch
    except Exception:
        return None
    target = _cpu_inference_thread_target()
    try:
        if torch.get_num_threads() > target:
            torch.set_num_threads(target)
    except Exception:
        return None
    return target


def _maybe_compile_pipeline(pipe: Any) -> None:
    """`channels_last` + `torch.compile` a **vision (CNN)** model in place for ~2x GPU inference.

    Applied ONLY to convolutional models (a `Conv2d` present) — fixed input shapes and
    compute-heavy, where `torch.compile`'s kernel fusion + graph capture is a measured ~1.9x at
    inference-identical results (unchanged predicted labels, logits within fp16 tolerance). It is
    deliberately NOT applied to text transformers: their dynamic sequence lengths trigger
    per-shape recompiles and the work is tokenization-bound, so `torch.compile` there measured
    ~0.9x (a regression). Compiled once per worker; the warm pool amortizes the one-time compile
    over every batch. A no-op on CPU, when `distributed.torch_compile` is off, or on any failure
    — eager is the safe fallback, so a perf optimization never breaks or slows inference."""
    if not active_config().distributed.torch_compile:
        return
    try:
        import torch

        model = getattr(pipe, "model", None)
        if not torch.cuda.is_available() or model is None:
            return
        if not any(isinstance(m, torch.nn.Conv2d) for m in model.modules()):
            return  # not a CNN — compile regresses dynamic-shape text models, so skip
        pipe.model = torch.compile(model.to(memory_format=torch.channels_last))
    except Exception:
        pass  # eager fallback — never break inference for a perf optimization


@functools.cache
def transformers_pipeline_encoder(
    model: str, column: str, *, output_column: str = "prediction", task: str | None = None
) -> type:
    """A load-once class UDF that runs a HuggingFace ``transformers`` pipeline.

    The model-identifier path of `ds.ml.infer`: builds ``transformers.pipeline(task,
    model=model)`` once per worker (the load-once GPU-inference pattern) and runs it
    over each batch's `column`, appending the pipeline's primary output per row as
    `output_column`. For a classification pipeline that is the predicted ``label``;
    for text generation the ``generated_text``; otherwise the raw scalar output. On a
    GPU worker the model is placed on the device and auto-cast to BF16/FP16 where the
    hardware benefits (see `_pipeline_accel_kwargs`). Needs ``transformers``
    (``batcher-engine[transformers]``).

    The generated class is **memoized** per ``(model, column, output_column, task)`` so
    repeated ``ds.ml.infer(<same model>)`` calls return the *same* class object. The
    distributed warm-pool key is the UDF's identity (`dist…map._pipeline_signature`), so a
    stable class is what lets a session-warm inference pool be reused across `collect()`s
    instead of reloading the model every call — the whole point of the warm pool on the
    convenience path.
    """

    class _PipelineModel:
        def __init__(self) -> None:
            try:
                from transformers import pipeline
            except ImportError as exc:  # pragma: no cover - optional extra
                from batcher._internal.errors import BackendError

                msg = "ds.ml.infer(<model id>) needs: pip install 'batcher-engine[transformers]'"
                raise BackendError(msg) from exc
            self._pipe = pipeline(task, model=model, **_pipeline_accel_kwargs())
            _maybe_compile_pipeline(self._pipe)

        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            import pyarrow as pa

            # A HF pipeline defaults to batch_size=1 — it runs the model once PER ROW,
            # starving the GPU (the classic pipeline footgun). Feed the whole Arrow batch
            # as one GPU batch (`batch_size=num_rows`) so the forward pass is actually
            # batched; the map_batches `batch_size` already bounds `num_rows`, and the
            # actor-pool OOM-halving is the safety net if a batch is too large for VRAM.
            inputs = batch.column(column).to_pylist()
            results = self._pipe(inputs, batch_size=max(1, len(inputs)))
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
