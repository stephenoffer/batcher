"""Streaming, stage-overlapped execution of a linear `map_batches` chain.

The multi-stage inference shape — ``Scan → map → … → map`` with at least one GPU stage
— runs here instead of materializing each stage in full. Every stage runs on its own
thread behind a bounded prefetch queue, so a CPU stage (decode / resize / tokenize)
prepares morsel *k+1* while the next stage (a GPU forward pass) consumes morsel *k*: the
device stays fed instead of idling through the whole decode. The result is identical to
the materializing path (`core.udf._execute_node`) — same rows, same per-batch contract;
only the scheduling overlaps.

This module owns the *scheduling* of that overlap and the GPU-adaptive sub-batching; the
per-batch application primitives it calls (`build_udf_callable`, `_formatted`,
`_coerce_udf_result`, `_resilient_call`) live in `core.udf`, which routes a linear chain
here from `execute_with_udfs`.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa

from batcher.config import active_config
from batcher.core.udf import (
    _coerce_udf_result,
    _formatted,
    _resilient_call,
    build_udf_callable,
)
from batcher.plan.logical import LogicalPlan, MapBatches, Scan

__all__ = ["linear_map_chain", "stream_eligible", "stream_linear_chain"]

# Bounded look-ahead between pipelined map stages: a stage may run this many morsels
# ahead of its consumer (so a CPU stage overlaps the GPU stage draining it) while keeping
# resident memory to ~`depth` morsels per stage. Env-overridable.
_STREAM_PREFETCH_DEPTH = max(0, int(os.environ.get("BATCHER_STREAM_PREFETCH_DEPTH", "2")))

# Adaptive GPU-inference batch when a GPU stage has no explicit `batch_size` (the truly
# zero-config `ds.map_batches(Model, num_gpus=1)` call). `_GPU_STREAM_BATCH_ROWS` is the
# row cap (large enough to fill the device; the guides' image range is 32-128, 256 suits
# most vision/embedding models); `_GPU_STREAM_BATCH_BYTES` is a per-batch input-byte budget
# so the row count SHRINKS on wide rows (a decoded frame, a float embedding tensor) that
# would otherwise OOM the GPU at the row cap, and stays at the cap for narrow rows. Floored
# so the batch always fills the SMs. An explicit `batch_size` always wins; env-overridable.
_GPU_STREAM_BATCH_ROWS = max(1, int(os.environ.get("BATCHER_GPU_STREAM_BATCH_ROWS", "256")))
_GPU_STREAM_BATCH_BYTES = max(
    1 << 20, int(os.environ.get("BATCHER_GPU_STREAM_BATCH_BYTES", str(64 << 20)))
)
_GPU_STREAM_BATCH_MIN = max(1, int(os.environ.get("BATCHER_GPU_STREAM_BATCH_MIN", "8")))


def linear_map_chain(plan: LogicalPlan) -> tuple[Scan, list[MapBatches]] | None:
    """`(scan, [stage1, …, stageN])` if `plan` is a linear `Scan → map → … → map` chain
    (bottom-up), else `None`. Any non-map / branching node makes it ineligible."""
    stages: list[MapBatches] = []
    node: LogicalPlan = plan
    while isinstance(node, MapBatches):
        stages.append(node)
        node = node.input
    if not stages or not isinstance(node, Scan):
        return None
    stages.reverse()
    return node, stages


def _gpu_batch_rows(batch: pa.RecordBatch) -> int:
    """Adaptive GPU sub-batch row count for a morsel, from its per-row byte width.

    ``min(row_cap, byte_budget / per_row_bytes)`` floored at `_GPU_STREAM_BATCH_MIN`: narrow
    rows batch up to the row cap (fill the device); wide rows (large images/tensors) batch
    fewer rows to stay under the VRAM budget — data-width-adaptive, so the same zero-config
    call is safe for a 150 KB image and a 3 MB frame alike (OOM-halving covers the rest)."""
    if batch.num_rows <= 0:
        return _GPU_STREAM_BATCH_ROWS
    per_row = max(1, batch.nbytes // batch.num_rows)
    by_bytes = _GPU_STREAM_BATCH_BYTES // per_row
    return max(_GPU_STREAM_BATCH_MIN, min(_GPU_STREAM_BATCH_ROWS, by_bytes))


def stream_eligible(stages: list[MapBatches]) -> bool:
    """Whether a linear chain should run on the streaming, stage-overlapped path.

    Targets the **multi-stage inference shape** — two or more stages, at least one on a
    GPU — where a CPU stage (decode/resize/tokenize) feeding a GPU stage overlaps to keep
    the device fed (the whole point). This includes the zero-config `num_gpus>0` stage with
    no `batch_size`: streaming with a VRAM-safe default + OOM-halving beats materializing
    the whole partition (the GPU would idle through the decode), so the overlap wins over
    the autobatch path's dynamic sizing when there is an upstream CPU stage to overlap. A
    single stage has nothing to overlap with (its intra-stage `num_workers` threading, the
    auto-process path, and single-stage GPU autobatch stay on the materializing route,
    unchanged). An explicit `multiprocessing` stage runs across processes — it stays too.
    """
    if len(stages) < 2 or not any(op.num_gpus > 0 for op in stages):
        return False
    return not any(getattr(op, "multiprocessing", False) for op in stages)


def stream_linear_chain(
    scan: Scan, stages: list[MapBatches], sources: list
) -> Iterator[pa.RecordBatch]:
    """Yield the chain's output morsels, each stage pipelined on its own thread.

    The scan streams morsels; each map stage is wrapped in `prefetch`, so stage *i* runs
    on a background thread feeding a bounded queue that stage *i+1* drains — CPU and GPU
    stages overlap. Order is preserved (FIFO prefetch + in-order per-stage application),
    so the concatenated output equals the materializing path's result.
    """
    from batcher._internal.prefetch import prefetch

    gen: Iterator[pa.RecordBatch] = iter(sources[scan.source_id].read())
    for op in stages:
        gen = prefetch(_apply_udf_stream(gen, op), depth=_STREAM_PREFETCH_DEPTH)
    yield from gen


def _apply_udf_stream(gen: Iterator[pa.RecordBatch], op: MapBatches) -> Iterator[pa.RecordBatch]:
    """Apply one `map_batches` stage to a morsel stream, preserving order and semantics.

    The stage callable (a class UDF's model) is built **once** here — load-once is
    preserved. Each incoming morsel is re-chunked to the stage's `batch_size` (or the
    morsel bound); a stage with `num_workers > 1` runs its morsel's sub-batches across a
    persistent thread pool (`pool.map` keeps order — only helps a GIL-releasing `fn`, the
    intended Arrow/NumPy/torch case), else sequentially. Cross-stage overlap comes from
    the caller wrapping this generator in `prefetch`.
    """
    fn = build_udf_callable(op.fn)
    call = fn if op.batch_format == "pyarrow" else _formatted(fn, op.batch_format)
    morsel = max(1, active_config().execution.morsel_rows)
    is_gpu = op.num_gpus > 0
    # An explicit batch_size always wins. A CPU stage keeps the morsel bound. A GPU stage
    # WITHOUT a batch_size gets a size that ADAPTS to the data's row width (`target=None` →
    # `_gpu_batch_rows` per morsel): a fixed row count OOMs the device on wide rows (a large
    # image / float tensor) and under-fills it on narrow ones, so the size is derived from a
    # byte budget instead — capped at the row default and floored so it always fills the SMs.
    # OOM-halving remains the safety net if a model's activations still overflow.
    target: int | None = op.batch_size or (None if is_gpu else morsel)
    workers = op.num_workers if isinstance(op.num_workers, int) and op.num_workers > 1 else 1

    def _subs(batch: pa.RecordBatch) -> list[pa.RecordBatch]:
        t = target if target is not None else _gpu_batch_rows(batch)
        if batch.num_rows > t:
            return pa.Table.from_batches([batch]).to_batches(max_chunksize=t)
        return [batch]

    def _parallel_units(batch: pa.RecordBatch) -> list[pa.RecordBatch]:
        """Sub-batches to run in parallel for a CPU stage: the `batch_size` chunks if that
        already yields ``>= workers`` of them, else the morsel split into `workers` even
        slices — so a decode/preprocess stage with no `batch_size` still uses every spare
        core to stay ahead of a fast GPU stage (the guides' CPU:GPU-ratio feeding)."""
        subs = _subs(batch)
        if len(subs) >= workers or batch.num_rows < workers:
            return subs
        step = -(-batch.num_rows // workers)  # ceil, so exactly <= workers slices
        return [
            batch.slice(i, min(step, batch.num_rows - i)) for i in range(0, batch.num_rows, step)
        ]

    # Resilient per-call handling is needed for a GPU stage (survive a transient CUDA OOM by
    # halving) or when the user allowed skipping corrupt rows (`max_errored_rows`); a plain CPU
    # stage with no error budget calls directly so a real bug still fails fast.
    budget = [op.max_errored_rows]
    resilient = is_gpu or op.max_errored_rows > 0

    def _emit(sub: pa.RecordBatch):
        return (
            _resilient_call(call, sub, budget, is_gpu)
            if resilient
            else _coerce_udf_result(call(sub))
        )

    # A GPU stage runs one CUDA context (num_workers=1) and survives a transient VRAM spike
    # by halving the batch; a CPU stage fans its morsel across a persistent pool of cores.
    if is_gpu or workers <= 1:
        for batch in gen:
            if batch.num_rows:
                for sub in _subs(batch):
                    yield from _emit(sub)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for batch in gen:
            if not batch.num_rows:
                continue
            for out in pool.map(_emit, _parallel_units(batch)):
                yield from out
