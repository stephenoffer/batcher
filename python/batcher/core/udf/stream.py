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

import contextlib
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa

from batcher.config import active_config
from batcher.core.udf.call import _coerce_udf_result, _formatted, _resilient_call
from batcher.core.udf.execute import build_udf_callable
from batcher.plan.logical import LogicalPlan, MapBatches, Scan

__all__ = ["linear_map_chain", "stream_eligible", "stream_linear_chain"]

# Bounded look-ahead between pipelined map stages: a stage may run this many morsels
# ahead of its consumer (so a CPU stage overlaps the GPU stage draining it) while keeping
# resident memory to ~`depth` morsels per stage. Env-overridable.
_STREAM_PREFETCH_DEPTH = max(0, int(os.environ.get("BATCHER_STREAM_PREFETCH_DEPTH", "2")))
# GPU-stage in-flight forwards (see `_pipelined_emit`): >1 overlaps a batch's host-side
# tensorize/copy with the previous batch's device forward, lifting a single-stage inference's
# utilization. Default 1 (serial) — a chain with an upstream CPU stage already feeds the GPU,
# so `stream_linear_chain` raises it only for a lone GPU stage that would otherwise idle.
_GPU_PIPELINE_DEPTH = max(1, int(os.environ.get("BATCHER_GPU_PIPELINE_DEPTH", "1")))
# In-flight forwards for a LONE GPU stage (no upstream CPU stage feeding it): default 2 so a
# scan->GPU inference overlaps read/tensorize with the forward instead of idling the device.
_GPU_SOLO_PIPELINE_DEPTH = max(1, int(os.environ.get("BATCHER_GPU_SOLO_PIPELINE_DEPTH", "2")))

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

# Per-batch input-byte budget for a CPU (decode/preprocess) stage with no explicit `batch_size`:
# like the GPU budget, this SHRINKS the chunk below the morsel when a stage's rows are huge
# (a decoded frame, a raw tensor) so a transient per-thread output stays bounded, and keeps the
# full morsel for narrow rows. Result-invariant — the chunk only shards. Env-overridable.
_CPU_STREAM_BATCH_BYTES = max(
    1 << 20, int(os.environ.get("BATCHER_CPU_STREAM_BATCH_BYTES", str(128 << 20)))
)
# The deepest source-read look-ahead the learned readahead may request (a slow source hides more
# of its latency behind compute); bounds resident memory to ~this many morsels for the read.
_STREAM_MAX_PREFETCH_DEPTH = max(
    _STREAM_PREFETCH_DEPTH, int(os.environ.get("BATCHER_STREAM_MAX_PREFETCH_DEPTH", "8"))
)

# Hub namespaces for the streaming path's learned sizing, keyed by a stable per-stage / per-source
# signature. None of these can change a UDF's result — a batch/chunk size only shards rows, a
# prefetch depth only reorders when a chunk is read — so a warm start is byte-identical to cold.
_GPU_BATCH_NS = "udf_gpu_batch"  # learned VRAM-safe GPU batch rows per model signature
_SCAN_TPUT_NS = "udf_scan_tput"  # learned source read throughput (rows/sec) per source identity


def _stream_hub():
    """The process-wide MetadataHub, or `None` if unreachable — learned reads are best-effort."""
    try:
        from batcher.core.runtime import default_hub

        return default_hub()
    except Exception:  # pragma: no cover - learning must never break a query
        return None


def _stage_sig(op: MapBatches) -> str | None:
    """A stable per-stage signature for `op` (its UDF's ``module.qualname``), or `None`."""
    fn = op.fn
    mod = getattr(fn, "__module__", None)
    qual = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
    return f"{mod}.{qual}" if mod and qual else None


def _fold_ema(namespace: str, key: str | None, value: float) -> None:
    """Fold one observation into a per-signature EMA bucket ``{ema, n}`` in the hub. Best-effort.

    (A compact local copy of the dist learner's fold — `core` cannot import the `dist` layer.)"""
    if key is None or value != value or value <= 0.0:
        return
    hub = _stream_hub()
    if hub is None:
        return
    try:
        from batcher.config import active_config

        s = hub.get_keyed_param(namespace, key) or {}
        a = float(active_config().optimizer.learning_smoothing_alpha)
        prior = s.get("ema")
        ema = value if prior is None else a * value + (1.0 - a) * float(prior)
        hub.put_keyed_param(namespace, key, {"ema": ema, "n": int(s.get("n", 0)) + 1})
    except Exception:  # pragma: no cover - learning must never break a query
        return


def _read_ema(namespace: str, key: str | None) -> float | None:
    """The learned EMA for a signature (best-effort), or `None` when cold/unreachable."""
    if key is None:
        return None
    hub = _stream_hub()
    if hub is None:
        return None
    try:
        s = hub.get_keyed_param(namespace, key) or {}
    except Exception:  # pragma: no cover
        return None
    return float(s["ema"]) if "ema" in s else None


def _learned_gpu_cap(op: MapBatches) -> int:
    """The GPU batch-row cap for a model, seeded from its learned VRAM-safe size when known.

    A prior run's settled adaptive batch size (persisted per model signature) caps the byte-budget
    sizing so the next run starts at the learned safe size instead of rediscovering it from the
    default row cap — the automatic form of "we already found this model fits N rows". Never above
    the config cap; a cold model keeps the cap. OOM-halving remains the in-run safety net."""
    learned = _read_ema(_GPU_BATCH_NS, _stage_sig(op))
    cap = _GPU_STREAM_BATCH_ROWS
    if learned is not None and learned >= 1.0:
        cap = min(cap, int(learned))
    return max(_GPU_STREAM_BATCH_MIN, cap)


def _cpu_batch_rows(batch: pa.RecordBatch, morsel: int) -> int:
    """Byte-adaptive chunk row count for a CPU stage with no explicit `batch_size`.

    ``min(morsel, byte_budget / per_row_bytes)`` floored at 1: narrow rows keep the full morsel;
    wide (post-decode multimodal) rows chunk fewer so a per-thread output stays bounded. The chunk
    only shards the morsel, so the concatenated output is identical to the plain morsel path."""
    if batch.num_rows <= 0:
        return morsel
    per_row = max(1, batch.nbytes // batch.num_rows)
    by_bytes = _CPU_STREAM_BATCH_BYTES // per_row
    return max(1, min(morsel, by_bytes))


def _learned_read_depth(source) -> int:
    """The source-read prefetch depth, deepened for a source measured as slow to read.

    Reads the source's learned throughput (rows/sec, persisted per identity): a slow source (remote
    object storage, a throttled connector) gets a deeper look-ahead so more chunks overlap compute,
    while a fast local source keeps the base depth (deeper prefetch would only add resident memory).
    Clamped to ``[_STREAM_PREFETCH_DEPTH, _STREAM_MAX_PREFETCH_DEPTH]``. Prefetch only reorders when
    a chunk is read, never which rows it holds, so the result is identical at any depth."""
    try:
        ident = source.identity()
    except Exception:  # pragma: no cover
        return _STREAM_PREFETCH_DEPTH
    tput = _read_ema(_SCAN_TPUT_NS, ident)
    if tput is None:
        return _STREAM_PREFETCH_DEPTH
    import math

    fast = 5_000_000.0  # rows/sec: a fast local scan needs no extra readahead
    if tput >= fast:
        return _STREAM_PREFETCH_DEPTH
    extra = int(math.log2(fast / max(tput, 1.0)))
    deep = _STREAM_PREFETCH_DEPTH + extra
    return max(_STREAM_PREFETCH_DEPTH, min(_STREAM_MAX_PREFETCH_DEPTH, deep))


def _timed_source(source, gen: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
    """Yield the source's morsels, recording its measured read throughput (rows/sec) on exhaustion.

    Times only the source iteration (rows / elapsed) and folds it into the learned readahead
    signal, so the next run can deepen the prefetch for a slow source. Timing is a driver-side
    counter — it touches no row and cannot change what is yielded."""
    import time

    ident = None
    with contextlib.suppress(Exception):
        ident = source.identity()
    rows = 0
    t0 = time.perf_counter()
    try:
        for batch in gen:
            rows += batch.num_rows
            yield batch
    finally:
        elapsed = time.perf_counter() - t0
        if ident is not None and rows > 0 and elapsed > 0.0:
            _fold_ema(_SCAN_TPUT_NS, ident, rows / elapsed)


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


def _gpu_batch_rows(batch: pa.RecordBatch, row_cap: int = _GPU_STREAM_BATCH_ROWS) -> int:
    """Adaptive GPU sub-batch row count for a morsel, from its per-row byte width.

    ``min(row_cap, byte_budget / per_row_bytes)`` floored at `_GPU_STREAM_BATCH_MIN`: narrow
    rows batch up to the row cap (fill the device); wide rows (large images/tensors) batch
    fewer rows to stay under the VRAM budget — data-width-adaptive, so the same zero-config
    call is safe for a 150 KB image and a 3 MB frame alike (OOM-halving covers the rest). The
    `row_cap` is the model's learned VRAM-safe size when known (see `_learned_gpu_cap`), else
    the config default."""
    if batch.num_rows <= 0:
        return row_cap
    per_row = max(1, batch.nbytes // batch.num_rows)
    by_bytes = _GPU_STREAM_BATCH_BYTES // per_row
    return max(_GPU_STREAM_BATCH_MIN, min(row_cap, by_bytes))


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
    if len(stages) >= 2:
        return True
    return stages[0].num_gpus > 0 and stages[0].batch_size is not None


def stream_linear_chain(
    scan: Scan,
    stages: list[MapBatches],
    sources: list,
    projections: dict[int, list[str]] | None = None,
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
    """
    from batcher._internal.prefetch import prefetch

    # A lone GPU stage has no upstream CPU stage to keep it fed, so it pipelines its own
    # forwards (host prep of the next batch overlaps the current forward); a GPU stage in a
    # multi-stage chain is already fed by the CPU stage ahead of it, so it stays serial (no
    # extra in-flight VRAM). A CPU stage's own `num_workers` threading is unchanged.
    solo_gpu = len(stages) == 1
    # A source measured as slow to read on a prior run gets a deeper look-ahead so more chunks
    # overlap compute (`_learned_read_depth`); a fast source keeps the base depth. `_timed_source`
    # measures this run's read throughput to refine that next time. Both are pure scheduling —
    # prefetch order and result are unchanged at any depth.
    src = sources[scan.source_id]
    read_depth = _learned_read_depth(src)
    # Stream only the columns the plan needs (None = every column, for an undeclared UDF).
    projection = (projections or {}).get(scan.source_id)
    gen: Iterator[pa.RecordBatch] = prefetch(
        _timed_source(src, iter(src.iter_batches(projection))), depth=read_depth
    )
    for op in stages:
        depth = _GPU_SOLO_PIPELINE_DEPTH if solo_gpu else _GPU_PIPELINE_DEPTH
        gen = prefetch(_apply_udf_stream(gen, op, depth), depth=_STREAM_PREFETCH_DEPTH)
    yield from gen


def _apply_udf_stream(
    gen: Iterator[pa.RecordBatch], op: MapBatches, gpu_depth: int = 1
) -> Iterator[pa.RecordBatch]:
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
    if is_gpu:
        # Run the forward under autocast (tensor-core half precision) — a no-op when disabled
        # or unavailable. Brings the managed path's FP16 win to a raw `map_batches(model)`.
        from batcher.ml.gpu import autocast_call

        call = autocast_call(call)
    # An explicit batch_size always wins. Without one, the chunk ADAPTS to the data's row width:
    # a GPU stage sizes from a VRAM byte budget capped at the model's learned safe size
    # (`_learned_gpu_cap`), and a CPU stage sizes from a (larger) byte budget capped at the morsel
    # (`_cpu_batch_rows`) so a post-decode multimodal stage shrinks its transient output on wide
    # rows. A fixed row count would OOM the device / balloon memory on wide rows and under-fill on
    # narrow ones. OOM-halving remains the safety net if a model's activations still overflow.
    explicit: int | None = op.batch_size or None
    gpu_cap = _learned_gpu_cap(op) if is_gpu else _GPU_STREAM_BATCH_ROWS
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
            t = _gpu_batch_rows(batch, gpu_cap)
            observed_gpu.append(_gpu_batch_rows(batch, _GPU_STREAM_BATCH_ROWS))
        else:
            t = _cpu_batch_rows(batch, morsel)
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
            _fold_ema(_GPU_BATCH_NS, _stage_sig(op), float(min(observed_gpu)))


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
