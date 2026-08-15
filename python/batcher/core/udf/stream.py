"""Streaming, stage-overlapped execution of a linear `map_batches` chain.

The multi-stage inference shape — ``Scan → map → … → map`` with at least one GPU stage
— runs here instead of materializing each stage in full. Every stage runs on its own
thread behind a bounded prefetch queue, so a CPU stage (decode / resize / tokenize)
prepares morsel *k+1* while the next stage (a GPU forward pass) consumes morsel *k*: the
device stays fed instead of idling through the whole decode. The result is identical to
the materializing path (`core.udf._execute_node`) — same rows, same per-batch contract;
only the scheduling overlaps.

This module owns the *scheduling* of that overlap and the GPU-adaptive sub-batching; the
per-batch call boundary it drives (`_formatted`, `_coerce_udf_result`, `_resilient_call`)
lives in `core.udf.call`, and `execute_with_udfs` routes a linear chain here.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa

from batcher.config import active_config
from batcher.core.udf import strategy as strat
from batcher.core.udf.async_udf import is_async_udf
from batcher.core.udf.call import _coerce_udf_result, _formatted, _resilient_call
from batcher.core.udf.lifecycle import build_udf_callable, teardown_udf
from batcher.core.udf.resilience import wrap_resilient
from batcher.core.udf.sizing import (
    _GPU_BATCH_NS,
    _GPU_STREAM_BATCH_ROWS,
    _STREAM_PREFETCH_DEPTH,
    cpu_batch_rows,
    fold_ema,
    gpu_batch_rows,
    learned_gpu_cap,
    learned_read_depth,
    stage_sig,
    timed_source,
)
from batcher.io.schema.evolution import normalize_batch, unify_schemas
from batcher.plan.logical import LogicalPlan, MapBatches, Scan
from batcher.plan.profile import StageRecorder, metered, stage_kind
from batcher.plan.types import total_logical_bytes

__all__ = [
    "linear_map_chain",
    "reconcile_stream",
    "stream_eligible",
    "stream_linear_chain",
]

# GPU-stage in-flight forwards (see `_pipelined_emit`): >1 overlaps a batch's host-side
# tensorize/copy with the previous batch's device forward, lifting a single-stage inference's
# utilization. Default 1 (serial) — a chain with an upstream CPU stage already feeds the GPU,
# so `stream_linear_chain` raises it only for a lone GPU stage that would otherwise idle.
_GPU_PIPELINE_DEPTH = max(1, int(os.environ.get("BATCHER_GPU_PIPELINE_DEPTH", "1")))
# In-flight forwards for a LONE GPU stage (no upstream CPU stage feeding it): default 2 so a
# scan->GPU inference overlaps read/tensorize with the forward instead of idling the device.
# Public because the MATERIALIZING path needs the same overlap: `apply._apply_udf_autobatch`
# sizes its dispatch pool from it, so a solo GPU stage gets the same two in-flight forwards
# whichever path the plan shape routes it to. One definition, not two that can drift.
_GPU_SOLO_PIPELINE_DEPTH = max(1, int(os.environ.get("BATCHER_GPU_SOLO_PIPELINE_DEPTH", "2")))
# The per-batch sizing constants and the learned refinements that narrow them live in
# `core.udf.sizing`; this module owns the *scheduling* of the overlap, not the sizing.


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


def stream_eligible(stages: list[MapBatches]) -> bool:
    """Whether a linear chain should run on the streaming, stage-overlapped path.

    Targets the **GPU inference shape**, where the streaming path keeps the device fed by
    running the scan/read and each stage on their own prefetch threads:

    * **Multi-stage** (>= 2 stages, at least one GPU): a CPU stage (decode/resize/tokenize)
      feeding a GPU stage overlaps so the GPU never idles through the CPU work — including
      the zero-config `num_gpus>0` stage with no `batch_size` (a VRAM-safe adaptive size +
      OOM-halving beats materializing the whole partition and idling the GPU through decode).
    * **Single GPU stage with an explicit `batch_size`**: there is still something to overlap
      — the **input read/transfer**. `stream_linear_chain` prefetches the scan, so reading
      partition-chunk *k+1* (from storage / the shuffle) overlaps the forward pass of chunk
      *k*, lifting utilization on a scan->GPU inference (measured: a storage read starved the
      device to ~60% util otherwise). A single *zero-config* GPU stage stays on the
      throughput-hill-climbing autobatch path (its dynamic sizing is the zero-config win, and
      it re-chunks internally); a single CPU stage keeps the materializing thread/process
      strategy — neither is streaming-eligible.

    An explicit `multiprocessing` stage runs across processes — it stays on the materializing
    path regardless.
    """
    if not any(op.num_gpus > 0 for op in stages):
        return False
    if any(getattr(op, "multiprocessing", False) for op in stages):
        return False
    # An async (`async def`) stage runs on its own event loop (`apply._apply_udf_async`); the
    # synchronous stage-overlap path here would hand it an un-awaited coroutine. Keep any async
    # stage on the materializing path, which routes it correctly.
    if any(is_async_udf(op.fn) for op in stages):
        return False
    if len(stages) >= 2:
        return True
    return stages[0].num_gpus > 0 and stages[0].batch_size is not None


def stream_linear_chain(
    scan: Scan,
    stages: list[MapBatches],
    sources: list,
    projections: dict[int, list[str]] | None = None,
    recorder: StageRecorder | None = None,
    op_ids: dict[int, int] | None = None,
) -> Iterator[pa.RecordBatch]:
    """Yield the chain's output morsels, each stage pipelined on its own thread.

    The scan streams morsels; the scan read and each map stage are each wrapped in
    `prefetch`, so they run on their own background threads feeding bounded queues the next
    stage drains — the input read/transfer, CPU stages, and GPU stages all overlap. Order is
    preserved (FIFO prefetch + in-order per-stage application), so the concatenated output
    equals the materializing path's result.

    Iterating the source **lazily** (`iter_batches`, not the eager `read`) and prefetching it
    is what lets reading chunk *k+1* from storage / the shuffle overlap the compute on chunk
    *k* — without it the whole partition materializes before the first stage runs and the GPU
    idles through the read.

    With a `recorder`, each stage is metered around its own `fn` call — not around its output
    generator, which would charge it for time spent waiting on a slower upstream (see
    `_apply_udf_stream`). The read is metered at the generator, because there the wait *is*
    the work. Together they give a streaming inference pipeline a measured `stats()` tree
    without changing any scheduling.
    """
    from batcher._internal.prefetch import prefetch

    # A lone GPU stage has no upstream CPU stage to keep it fed, so it pipelines its own
    # forwards (host prep of the next batch overlaps the current forward); a GPU stage in a
    # multi-stage chain is already fed by the CPU stage ahead of it, so it stays serial (no
    # extra in-flight VRAM). A CPU stage's own `num_workers` threading is unchanged.
    solo_gpu = len(stages) == 1
    # A source measured as slow to read on a prior run gets a deeper look-ahead so more chunks
    # overlap compute (`learned_read_depth`); a fast source keeps the base depth. `timed_source`
    # measures this run's read throughput to refine that next time. Both are pure scheduling —
    # prefetch order and result are unchanged at any depth.
    src = sources[scan.source_id]
    read_depth = learned_read_depth(src)
    # Stream only the columns the plan needs (None = every column, for an undeclared UDF).
    projection = (projections or {}).get(scan.source_id)
    read: Iterator[pa.RecordBatch] = timed_source(src, iter(src.iter_batches(projection)))
    if recorder is not None and op_ids is not None and id(scan) in op_ids:
        # Meter the read too, not only the stages: the scan's row count is what the first
        # stage's `rows_in` is read from, and without it that stage reports consuming nothing.
        read = metered(read, recorder, op_ids[id(scan)], "Scan")
    gen: Iterator[pa.RecordBatch] = prefetch(read, depth=read_depth)
    for op in stages:
        depth = _GPU_SOLO_PIPELINE_DEPTH if solo_gpu else _GPU_PIPELINE_DEPTH
        hook = _stage_recorder_hook(recorder, op_ids, op)
        gen = prefetch(_apply_udf_stream(gen, op, depth, hook), depth=_STREAM_PREFETCH_DEPTH)
    yield from gen


def _stage_recorder_hook(
    recorder: StageRecorder | None, op_ids: dict[int, int] | None, op: MapBatches
) -> Callable[[int, pa.RecordBatch, list], None] | None:
    """The per-sub-batch profiling callback for `op`, or `None` when not profiling."""
    if recorder is None or op_ids is None or id(op) not in op_ids:
        return None
    op_id = op_ids[id(op)]
    backend = "gpu" if op.num_gpus > 0 else ""
    kind = stage_kind(op.fn)

    def _record(elapsed_ns: int, sub: pa.RecordBatch, out: list) -> None:
        recorder.record(
            op_id,
            kind=kind,
            rows_in=sub.num_rows,
            rows_out=sum(b.num_rows for b in out),
            elapsed_ns=elapsed_ns,
            result_bytes=total_logical_bytes(out),
            backend=backend,
        )

    return _record


def _apply_udf_stream(
    gen: Iterator[pa.RecordBatch],
    op: MapBatches,
    gpu_depth: int = 1,
    record: Callable[[int, pa.RecordBatch, list], None] | None = None,
) -> Iterator[pa.RecordBatch]:
    """Apply one `map_batches` stage to a morsel stream, preserving order and semantics.

    The stage callable (a class UDF's model) is built **once** here — load-once is
    preserved. Each incoming morsel is re-chunked to the stage's `batch_size` (or the
    morsel bound); a stage with `num_workers > 1` runs its morsel's sub-batches across a
    persistent thread pool (`pool.map` keeps order — only helps a GIL-releasing `fn`, the
    intended Arrow/NumPy/torch case), else sequentially. Cross-stage overlap comes from
    the caller wrapping this generator in `prefetch`.

    `record` is the profiling hook, called per sub-batch with the time **this stage's own
    call** took and the rows it consumed and produced. It is placed around the `fn` call
    rather than around the output generator deliberately. Timing the generator would
    measure *residency* — a stage blocked on an empty input queue would be charged for its
    upstream's slowness — and that distortion is unbounded: a stage fed by something 10x
    slower reads 10x too high. It would also invert `gpu-starved`, whose whole job is
    comparing a GPU stage against the CPU stages feeding it. Here the only overstatement is
    the bounded one from `gpu_depth` concurrent in-flight forwards on a GPU stage.
    """
    fn = build_udf_callable(op.fn)
    call = fn if op.batch_format == "pyarrow" else _formatted(fn, op.batch_format)
    morsel = max(1, active_config().execution.morsel_rows)
    is_gpu = op.num_gpus > 0
    if is_gpu:
        # Autograd off, then autocast (tensor-core half precision) — both no-ops when
        # disabled or unavailable. Brings the managed path's FP16 win, and the activation
        # memory autograd would otherwise pin, to a raw `map_batches(model)`.
        from batcher.ml.gpu import autocast_call, inference_mode_call

        call = inference_mode_call(call)
        call = autocast_call(call)
    # Retry a transient failure / bound a hung call around the raw `fn`, before the GPU
    # OOM-halving and error-budget bisection in `_emit` — so a flaky external stage streams
    # with the same resilience as the materializing path.
    call = wrap_resilient(call, op)
    # An explicit batch_size always wins. Without one, the chunk ADAPTS to the data's row width:
    # a GPU stage sizes from a VRAM byte budget capped at the model's learned safe size
    # (`learned_gpu_cap`), and a CPU stage sizes from a (larger) byte budget capped at the
    # morsel (`cpu_batch_rows`) so a post-decode multimodal stage shrinks its transient output
    # on wide rows. A fixed row count would OOM the device / balloon memory on wide rows and
    # under-fill on narrow ones. OOM-halving remains the safety net if activations still overflow.
    explicit: int | None = op.batch_size or None
    gpu_cap = learned_gpu_cap(op) if is_gpu else _GPU_STREAM_BATCH_ROWS
    workers = op.num_workers if isinstance(op.num_workers, int) and op.num_workers > 1 else 1
    # What each morsel's *data width* permits, sized against the config cap rather than the
    # learned one. Folding the size actually applied would make the EMA its own input: the
    # applied size is `min(learned_cap, by_bytes) <= learned_cap`, so every run could only
    # ever fold a value at or below the prior EMA. That is a one-way ratchet — a single
    # wide-row run permanently shrank a model's GPU batch on every later run, including over
    # narrow data, drifting toward the floor with no path back up. Measuring against the
    # config cap keeps the signal a property of the data, so it can recover.
    observed_gpu: list[int] = []

    def _subs(batch: pa.RecordBatch) -> list[pa.RecordBatch]:
        if explicit is not None:
            t = explicit
        elif is_gpu:
            t = gpu_batch_rows(batch, gpu_cap)
            observed_gpu.append(gpu_batch_rows(batch, _GPU_STREAM_BATCH_ROWS))
        else:
            t = cpu_batch_rows(batch, morsel)
        if batch.num_rows > t:
            return pa.Table.from_batches([batch]).to_batches(max_chunksize=t)
        return [batch]

    def _parallel_units(batch: pa.RecordBatch) -> list[pa.RecordBatch]:
        """Sub-batches to run in parallel for a CPU stage.

        With no explicit `batch_size`, a stage whose adaptive chunk is the whole morsel would
        hand the pool a single unit and idle every other core; splitting the morsel into
        `workers` even slices keeps a decode/preprocess stage ahead of a fast GPU stage (the
        guides' CPU:GPU-ratio feeding). An **explicit** `batch_size` still wins — the same
        invariant `_subs` holds — so it is parallelized over its own chunks and never re-sliced
        to fill the pool, which would silently change the batch boundaries the `fn` sees."""
        subs = _subs(batch)
        if explicit is not None or len(subs) >= workers or batch.num_rows < workers:
            return subs
        step = -(-batch.num_rows // workers)  # ceil, so exactly <= workers slices
        return [
            batch.slice(i, min(step, batch.num_rows - i)) for i in range(0, batch.num_rows, step)
        ]

    # Resilient per-call handling is needed for a GPU stage (survive a transient CUDA OOM by
    # halving) or when the user allowed skipping corrupt rows (`max_errored_rows`); a plain CPU
    # stage with no error budget calls directly so a real bug still fails fast.
    # The allowance is shared per worker process (`strategy.error_budget`), not rebuilt here:
    # this module and `apply.apply_udf` used to build one each, so a query routed through
    # both paths — or a streaming query calling in once per window — got several full budgets.
    budget = strat.error_budget(op)
    resilient = is_gpu or op.max_errored_rows > 0

    def _emit(sub: pa.RecordBatch):
        started = time.perf_counter_ns() if record is not None else 0
        out = (
            _resilient_call(call, sub, budget, is_gpu)
            if resilient
            else _coerce_udf_result(call(sub))
        )
        if record is not None:
            record(time.perf_counter_ns() - started, sub, out)
        return out

    # A GPU stage keeps `_GPU_PIPELINE_DEPTH` forwards in flight on a small thread pool: torch
    # releases the GIL during the device work, so batch k+1's CPU-side prep (the NumPy->torch
    # tensorize + host->device copy) overlaps batch k's forward — the device stops idling
    # through the per-call CPU/transfer overhead that caps a single-stage inference's
    # utilization. One CUDA context is shared (safe for inference); a transient VRAM spike from
    # the extra in-flight batch is caught by the same OOM-halving in `_emit`. Order preserved.
    try:
        if is_gpu:
            yield from _pipelined_emit(gen, _subs, _emit, gpu_depth)
            return
        if workers <= 1:
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
    finally:
        # Persist the SMALLEST size this run's data width permitted (the widest morsel's
        # byte-budget size) so the next run seeds its cap from it instead of the default.
        # Measured against the config cap, never the learned one — see `observed_gpu`, or the
        # EMA becomes a ratchet that can only fall. Purely a starting-size hint; the size never
        # changes what the model computes, and OOM-halving stays the in-run safety net.
        if is_gpu and explicit is None and observed_gpu:
            fold_ema(_GPU_BATCH_NS, stage_sig(op), float(min(observed_gpu)))
        # Release a class model this stage owns (a plain class `fn` built here, not a prebuilt
        # instance the streaming loop owns) — deterministic teardown at stage end.
        teardown_udf(fn, op)


def _pipelined_emit(gen, subs_fn, emit_fn, depth: int) -> Iterator[pa.RecordBatch]:
    """Yield `emit_fn` over every sub-batch, keeping up to `depth` calls in flight (in order).

    A bounded window of futures on a `depth`-thread pool: while the consumer drains result k,
    calls k+1 .. k+depth-1 are already running, so a GPU forward overlaps the next batch's
    host-side tensorize/copy. `depth <= 1` collapses to the plain serial path.
    """
    from collections import deque

    def _subs_stream():
        for batch in gen:
            if batch.num_rows:
                yield from subs_fn(batch)

    if depth <= 1:
        for sub in _subs_stream():
            yield from emit_fn(sub)
        return
    with ThreadPoolExecutor(max_workers=depth) as pool:
        inflight: deque = deque()
        for sub in _subs_stream():
            inflight.append(pool.submit(emit_fn, sub))
            if len(inflight) >= depth:
                yield from inflight.popleft().result()
        while inflight:
            yield from inflight.popleft().result()


def reconcile_stream(gen: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
    """Yield `gen`'s batches, each normalized to the union of the schemas seen *so far*.

    The incremental counterpart of `reconcile_batches`, keeping a drifting-schema UDF (LLM
    structured outputs that gain a field) concatenable downstream with one batch resident
    instead of the whole output. Deliberately a weaker contract than the list form: a batch
    already yielded cannot be widened retroactively, so an early batch keeps the narrower
    schema. A consumer needing ONE schema over the entire result must use `execute_with_udfs`
    and pay the materialization — that guarantee is what the memory bound is traded for.
    """
    target: pa.Schema | None = None
    for batch in gen:
        if target is None:
            target = batch.schema
        elif not batch.schema.equals(target):
            target = unify_schemas([target, batch.schema], mode="union")
        yield batch if batch.schema.equals(target) else normalize_batch(batch, target)
