"""Apply one `map_batches` stage to a set of batches (Core, layer 3).

`execute` walks the plan tree and decides *which* stage runs *where*; this module owns *how* a
single stage's Python `fn` is run over its input batches — the rebatching policy, the
threads/processes/async/GPU-autobatch dispatch, the retry/timeout wrap, the error-budget
bisection, and the load-once build + teardown of the model. Keeping it separate from `execute`
keeps that file the plan orchestrator and this one the per-stage engine, and lets the streaming
path (`stream`) reuse the same building blocks (`lifecycle`, `call`, `resilience`).
"""

from __future__ import annotations

import contextlib
import threading
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa

from batcher._internal.mathx import ceil_div
from batcher.config import active_config
from batcher.core.udf import strategy as strat
from batcher.core.udf.async_udf import is_async_udf, run_async_batches
from batcher.core.udf.call import (
    _check_declared_columns,
    _coerce_udf_result,
    _formatted,
    _resilient_call,
)
from batcher.core.udf.lifecycle import build_udf_callable, teardown_udf
from batcher.core.udf.resilience import wrap_resilient
from batcher.plan.logical import MapBatches

__all__ = ["apply_udf", "rechunk"]

# Idle dispatch pools, keyed by worker count, waiting to be leased again. A stage's per-batch
# calls used to run on a `ThreadPoolExecutor` built and shut down *inside* `_run_sync_udf`,
# which is fine when a stage runs once (a `collect`) and ruinous when it runs repeatedly — and
# it does: `iter_batches` over a `map_batches` chain calls `execute_with_udfs` per *window*, so
# a 16 M-row four-stage chain built 260 pools and spawned 1,677 threads, and
# `ThreadPoolExecutor.__exit__` was 9.8 s of a 9.6 s profile. The same shape, measured in
# isolation, is **4,352 ms against 147 ms** for a reused pool (30x). Streaming micro-batch
# queries pay it per micro-batch, which is worse still.
#
# Leased rather than shared outright: a lease hands out a pool *exclusively*, so a stage still
# gets exactly `num_workers` concurrent calls and two stages running at once get two pools —
# identical concurrency to building one per call, which is what keeps this a scheduling change
# and not a semantic one. Only the *idle* pools are reused.
_IDLE_POOLS: dict[int, list[ThreadPoolExecutor]] = {}
_POOLS_LOCK = threading.Lock()
# Idle pools retained per distinct worker count. Two covers the ordinary nesting depth (a
# stage's pool plus one held by a concurrently-running stage) without keeping a fleet of
# parked threads alive for a query that has moved on; anything beyond it is shut down.
_MAX_IDLE_POOLS = 2
# ...and a bound on the *total*, because "two per width" is unbounded in the number of widths.
# A width is a stage's `num_workers`, so one process normally sees one or two of them and this
# never binds — but a long-lived server handling many differently-configured pipelines would
# otherwise accumulate a parked pool per distinct value and never release one, which is the
# same unbounded-growth shape the reuse is meant to fix, just slower.
_MAX_IDLE_TOTAL = 4


@contextlib.contextmanager
def _leased_pool(workers: int):
    """A `ThreadPoolExecutor` of `workers` threads, held exclusively for the block.

    Reuses an idle pool of the same width when one is parked, else builds one; on exit the
    pool is parked for the next lease (up to `_MAX_IDLE_POOLS`) instead of being torn down.
    See `_IDLE_POOLS` for why. `concurrent.futures` registers its own interpreter-exit hook
    that joins parked worker threads, so a retained pool cannot hang shutdown.
    """
    with _POOLS_LOCK:
        idle = _IDLE_POOLS.get(workers)
        pool = idle.pop() if idle else None
    if pool is None:
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="batcher-map")
    try:
        yield pool
    except BaseException:
        # A lease that ended in an exception may still have calls queued from its own
        # `map`, so it is retired rather than parked: `shutdown(wait=True)` drains them
        # exactly as the `with ThreadPoolExecutor(...)` block it replaces did, and no
        # later lease inherits a pool with work already on it.
        pool.shutdown(wait=True)
        raise
    else:
        with _POOLS_LOCK:
            parked = _IDLE_POOLS.setdefault(workers, [])
            total = sum(len(v) for v in _IDLE_POOLS.values())
            if len(parked) < _MAX_IDLE_POOLS and total < _MAX_IDLE_TOTAL:
                parked.append(pool)
                return
        pool.shutdown(wait=False)


def rechunk(batches: list[pa.RecordBatch], target: int) -> list[pa.RecordBatch]:
    """Coalesce/split `batches` into batches of ~`target` rows, streaming and copy-bounded.

    pyarrow's ``Table.to_batches(max_chunksize=n)`` only *splits* an oversized chunk; it
    never *merges* undersized ones — so re-chunking a finely batched input (the common case:
    a morsel-sized native scan feeding a per-batch Python `fn`) up to a coarse `target`
    silently no-ops, leaving hundreds of tiny batches that make the fixed per-call overhead
    (FFI + framework conversion + schema build) dominate. This walks the input once, merging
    consecutive small batches until they reach `target` (and splitting any single batch above
    it), so the transient concatenation is bounded to ~one `target`-sized batch instead of a
    full-table materialization. A relation-level no-op: same rows in the same order, only the
    batching changes.
    """
    if target <= 0 or not batches:
        return batches
    if len(batches) == 1 and batches[0].num_rows <= target:
        return batches  # nothing to split and nothing to merge
    out: list[pa.RecordBatch] = []
    buf: list[pa.RecordBatch] = []
    buf_rows = 0

    def flush() -> None:
        nonlocal buf, buf_rows
        if not buf:
            return
        out.append(buf[0] if len(buf) == 1 else pa.concat_batches(buf))
        buf, buf_rows = [], 0

    for b in batches:
        if b.num_rows == 0:
            continue
        if b.num_rows >= target:
            flush()  # keep order: emit the pending run before this large batch
            for start in range(0, b.num_rows, target):
                out.append(b.slice(start, target))
            continue
        buf.append(b)
        buf_rows += b.num_rows
        if buf_rows >= target:
            flush()
    flush()
    return out


def apply_udf(current: list[pa.RecordBatch], op: MapBatches) -> list[pa.RecordBatch]:
    """Apply a `map_batches` function, optionally rebatching to `batch_size` and
    running the per-batch calls across `num_workers` threads (order preserved).

    When `op.batch_format` is not ``"pyarrow"``, each Arrow batch is converted to the
    requested framework object (numpy/pandas/torch) for the call and the result
    converted back — the data plane stays Arrow, only the call is reframed.

    Every dispatch route funnels back through here, which is why the `output_columns`
    declaration is checked at this one point rather than inside each of them."""
    out = _dispatch_udf(current, op)
    _check_declared_columns(out, op)
    return out


def _dispatch_udf(current: list[pa.RecordBatch], op: MapBatches) -> list[pa.RecordBatch]:
    """Route one stage to the async / GPU-autobatch / process / thread path and run it."""
    if not current:
        return current
    batches = current
    # Detect async on the *unbuilt* `op.fn` (a class's `async def __call__` is visible without
    # instantiating it) so a class model is not loaded here just to check — the build happens
    # once, inside the branch that actually runs it.
    if is_async_udf(op.fn):
        # An `async def` fn is I/O-bound (LLM/API enrichment): run its batches concurrently on
        # one event loop, not across the thread/process pools. This wins over the GPU-autobatch
        # and multiprocessing routes below — async concurrency is about overlapping awaits, not
        # filling cores or a device.
        return _apply_udf_async(op, current, build_udf_callable(op.fn))
    if op.num_gpus > 0 and op.batch_size is None:
        # Auto batch sizing for a GPU inference stage with no explicit `batch_size`:
        # hill-climb the size online toward the VRAM-capped throughput plateau, so the
        # user never hand-tunes it (a hand-set or unset `batch_size` is Ray Data's #1
        # OOM cause). This is a GPU-VRAM concern only — a CPU `map_batches` (including a
        # load-once class `fn`, the CPU batch-inference pattern) has no VRAM ceiling, so
        # it runs the normal morsel-batched threaded/process path (the 256-row
        # inference-pool micro-batches would cripple a vectorized CPU model instead).
        return _apply_udf_autobatch(op, batches)

    total = sum(b.num_rows for b in current)
    use_processes = strat.wants_processes(op, total, current)
    morsel = max(1, active_config().execution.morsel_rows)
    if op.batch_size is not None:
        batches = rechunk(current, op.batch_size)
    elif use_processes:
        # A process-pool `fn` pays a per-batch pickle/IPC round-trip to the child, so
        # morsel-sized batches make that overhead dominate (measured: an order of
        # magnitude slower than a coarse split). We want a few coarse batches per worker.
        # But re-chunking concatenates the WHOLE input into one table and re-splits it —
        # a full copy of every column (a superlinear cost at scale). The natural batches
        # of a many-file read are already coarse and plentiful, so skip that copy unless
        # the current batching is actually bad: too few to fill the pool, any batch large
        # enough to unbalance a worker / bloat one IPC payload, or any morsel-tiny batch.
        cap = max(morsel, ceil_div(total, max(1, op.num_workers)))  # ~one worker's share
        floor = min(cap, max(morsel, strat.PROC_MIN_BATCH_ROWS))
        sizes = [b.num_rows for b in current]
        already_coarse = (
            len(current) >= op.num_workers and max(sizes) <= cap and min(sizes) >= floor
        )
        if not already_coarse:
            target = max(
                floor, ceil_div(total, max(1, op.num_workers * strat.PROC_BATCHES_PER_WORKER))
            )
            batches = rechunk(current, target)
    else:
        # A threaded CPU `fn` with no explicit `batch_size`: rebatch to a COARSE target,
        # not the morsel. The morsel (16,384) minimizes cache footprint for the vectorized
        # relational kernels, but a per-batch Python call pays a fixed overhead (FFI +
        # framework conversion + schema build) that morsel-sized batches make dominate.
        # The target is one coarse batch per worker (parallel fill) with a floor that
        # amortizes the per-call overhead and a cap that bounds a row-exploding / wide-row
        # `fn` — so the whole partition is never handed over as one unbounded batch (the
        # OOM Ray Data's `batch_size=None` default hits). A relation-level no-op: same
        # rows, per-batch by contract; only the chunking changes.
        target = strat.thread_batch_target(op, total, op.num_workers, morsel, current)
        sizes = [b.num_rows for b in current]
        too_coarse = max(sizes, default=0) > target
        too_fine = len(current) > op.num_workers and min(sizes, default=0) < target
        if too_coarse or too_fine:
            batches = rechunk(current, target)

    strategy = strat.map_strategy(op, len(batches), use_processes)
    if strategy == "processes":
        # Run the per-batch calls across processes so a CPU-bound pure-Python `fn`
        # (which the GIL would serialize across threads) uses multiple cores. Any
        # process failure (an `fn` that turns out not to be process-safe) falls back
        # to threads — never a dropped batch.
        try:
            from batcher.core.udf.processes import run_map_processes

            results = run_map_processes(
                build_udf_callable(op.fn),
                batches,
                op.num_workers,
                op.batch_format,
                budget_key=strat.budget_key(op),
                max_errored_rows=op.max_errored_rows,
            )
        except Exception as exc:
            # A process pool can be unavailable for the whole session — e.g. a script
            # that runs the pipeline at import time is not import-safe, so forkserver/
            # spawn refuse to start a child. Remember that (`_disable_processes`) so the
            # fallback happens once here and every later call goes straight to threads,
            # instead of re-failing (and re-warning) on every batch/window.
            strat.disable_processes(exc)
            strategy = "threads"

    if strategy != "processes":
        return _run_sync_udf(op, batches, strategy)

    out = []
    for result in results:
        out.extend(_coerce_udf_result(result))
    return out


def _run_sync_udf(op: MapBatches, batches: list[pa.RecordBatch], strategy: str) -> list:
    """Run the per-batch calls synchronously (threads or sequential), building the model once
    and tearing it down when this call owns it (see `teardown_udf`).

    Owns the whole non-process lifetime of a `map_batches` stage: format conversion, autocast,
    the retry/timeout wrap, the optional error-budget bisection, and the thread-vs-sequential
    dispatch. The model is built once here (a class `fn` is a load-once factory) and released in
    a `finally`, so a load-once resource is freed deterministically per owned call.
    """
    fn = build_udf_callable(op.fn)
    try:
        call = fn if op.batch_format == "pyarrow" else _formatted(fn, op.batch_format)
        if op.num_gpus > 0:
            from batcher.ml.gpu import autocast_call, inference_mode_call

            # Autograd off first (pure win, no numerics change), then half precision.
            call = inference_mode_call(call)
            call = autocast_call(call)  # tensor-core half precision (no-op when off/CPU)
        # Retry a transient failure / bound a hung call BEFORE the error-budget bisection below,
        # so only a failure that survives every retry is charged against `max_errored_rows`.
        call = wrap_resilient(call, op)
        if op.max_errored_rows > 0:
            # Dirty-data tolerance: isolate and skip corrupt rows (up to the budget) instead
            # of crashing — a single-stage inference / preprocess over messy data survives.
            # Shared per worker process, not per call — see `strategy.error_budget`. Rebuilding
            # it here handed every partition (and the streaming path, independently) its own
            # full allowance, so the effective bound scaled with parallelism.
            budget = strat.error_budget(op)
            is_gpu = op.num_gpus > 0

            def _emit(b: pa.RecordBatch) -> list[pa.RecordBatch]:
                return _resilient_call(call, b, budget, is_gpu)

            if strategy == "threads":
                with _leased_pool(op.num_workers) as pool:
                    chunks = list(pool.map(_emit, batches))
            else:
                chunks = [_emit(b) for b in batches]
            out: list[pa.RecordBatch] = []
            for c in chunks:
                out.extend(c)
            return out
        if strategy == "threads":
            # ThreadPoolExecutor.map keeps input order; concurrency only helps when `fn`
            # releases the GIL (Rust/GPU/NumPy inference), which is the intended use.
            with _leased_pool(op.num_workers) as pool:
                results = list(pool.map(call, batches))
        else:
            results = [call(batch) for batch in batches]
        out = []
        for result in results:
            out.extend(_coerce_udf_result(result))
        return out
    finally:
        teardown_udf(fn, op)


def _apply_udf_async(
    op: MapBatches, current: list[pa.RecordBatch], fn: object
) -> list[pa.RecordBatch]:
    """Run an `async def` `fn` over the batches concurrently on one event loop.

    The I/O-bound inference/enrichment path: batches are optionally re-chunked to `batch_size`
    (a coarse chunk amortizes the per-call overhead the same as the sync path), then dispatched
    to `run_async_batches`, which overlaps up to `max_concurrency` awaits and preserves order.
    `batch_format` conversion happens around the await; retry/timeout is applied inside the
    async runner (with a real `asyncio.wait_for` cancel), so it is not wrapped again here.
    """
    batches = rechunk(current, op.batch_size) if op.batch_size is not None else current
    if op.batch_format == "pyarrow":
        call = fn
    else:
        from batcher.interop.formats import result_to_arrowable, to_format

        async def call(batch: pa.RecordBatch) -> object:  # type: ignore[misc]
            return result_to_arrowable(await fn(to_format(batch, op.batch_format)), op.batch_format)

    try:
        return run_async_batches(call, batches, op)
    finally:
        teardown_udf(fn, op)


def _apply_udf_autobatch(op: MapBatches, batches: list[pa.RecordBatch]) -> list[pa.RecordBatch]:
    """Run the inference `fn` with online batch-size auto-tuning (zero-config inference).

    Routes through a throughput `InferencePool`: it re-chunks the input dynamically and
    hill-climbs the batch size toward the VRAM-capped throughput plateau (`ml.autobatch`),
    surviving a CUDA OOM by halving the batch — so the user never hand-tunes `batch_size`.
    Input order is preserved (a relation-level no-op vs a fixed size), and the model is
    built once and shared across the dispatch slots.

    Two things are taken from the streaming path rather than re-invented, so the same model
    behaves the same way whichever path a plan's shape routes it to. The **seed batch size** is
    the model's learned VRAM-safe size (`sizing.learned_gpu_cap`), not a hardcoded 256
    that cold-started the two paths differently and discarded what the last run measured; the
    hill-climb still moves from there. And the **dispatch width** is at least
    `stream._GPU_SOLO_PIPELINE_DEPTH`: a GPU stage resolves to ``num_workers == 1``, which left
    this pool with one slot and no intra-worker overlap, so the device idled through each
    batch's host-side tensorize and copy while the streaming path overlapped exactly that. The
    slots share one built model and one CUDA context, so this buys overlap, not a second load.
    """
    from batcher.core.udf.sizing import learned_gpu_cap
    from batcher.core.udf.stream import _GPU_SOLO_PIPELINE_DEPTH
    from batcher.ml.gpu import autocast_call, inference_mode_call
    from batcher.ml.inference import InferencePool

    fn = build_udf_callable(op.fn)
    call = fn if op.batch_format == "pyarrow" else _formatted(fn, op.batch_format)
    call = inference_mode_call(call)  # autograd off — frees the activation memory that caps batches
    call = autocast_call(call)  # tensor-core half precision (GPU stage; no-op when off/CPU)
    call = wrap_resilient(call, op)  # transient-retry / timeout, inside the OOM-halving pool

    def worker(batch: pa.RecordBatch) -> pa.RecordBatch:
        coerced = _coerce_udf_result(call(batch))
        if not coerced:
            return batch.slice(0, 0)
        # `concat_batches` keeps every row and raises a clear error on a genuine >2 GiB
        # offset overflow, unlike `Table.from_batches(...).combine_chunks().to_batches()[0]`,
        # which splits at the 32-bit offset limit and then silently drops all but the first
        # batch — losing rows for large binary/string/list inference outputs.
        return coerced[0] if len(coerced) == 1 else pa.concat_batches(coerced)

    pool = InferencePool(
        lambda: worker,
        num_workers=max(op.num_workers, _GPU_SOLO_PIPELINE_DEPTH),
        target_batch_rows=learned_gpu_cap(op),
        objective="throughput",
    )
    try:
        return list(pool.run(iter(batches)))
    finally:
        teardown_udf(fn, op)
