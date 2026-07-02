"""Execution of pipelines containing `map_batches` (opaque Python/ML operators).

The Rust engine cannot call arbitrary Python UDFs, so a pipeline that mixes
relational operators with `map_batches` is orchestrated here. The plan is walked
as a **tree**: each relational operator runs on the native engine over its
already-materialized inputs (children replaced by scans of those batches), and
each `map_batches` applies its Python function (the ML model / preprocessing) to
the Arrow batches at that point. The two compose at *any* operator — including
joins and unions — so `read(a) → infer → join(read(b))` works. Batches flow as
Arrow the whole way (zero-copy from the engine into the UDF and back).
"""

from __future__ import annotations

import dataclasses
import os
import pickle
import warnings
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pyarrow as pa

from batcher.config import active_config
from batcher.plan.logical import LogicalPlan, MapBatches, Scan
from batcher.plan.schema import SchemaRef
from batcher.plan.visitor import children, with_children

__all__ = ["build_udf_callable", "execute_with_udfs", "has_map_batches", "prebuild_factories"]

# When a `map_batches` `fn` runs across processes with no explicit `batch_size`, split
# the input into ~this many coarse batches per worker: enough for load balance, coarse
# enough that the per-batch pickle/IPC to the child is amortized (morsel-sized batches
# make that transfer dominate — measured an order of magnitude slower).
_PROC_BATCHES_PER_WORKER = 3
# The smallest batch worth handing a process worker: below this the pickle/IPC per batch
# outweighs the work, so tinier batches are coalesced up to at least this many rows.
_PROC_MIN_BATCH_ROWS = 65_536


def prebuild_factories(node: LogicalPlan) -> LogicalPlan:
    """Instantiate every class (factory) `map_batches` UDF in a linear plan once,
    returning an equivalent plan whose `fn`s are the built callables.

    The model loads a single time here instead of once per `execute_with_udfs` call —
    so a long-lived caller (a distributed inference actor, a streaming micro-batch loop)
    reuses one loaded model across many batches. A non-class `fn` is already the
    callable and is left as is; `build_udf_callable` is idempotent on a built instance.
    """
    if isinstance(node, MapBatches):
        return dataclasses.replace(
            node, fn=build_udf_callable(node.fn), input=prebuild_factories(node.input)
        )
    child = getattr(node, "input", None)
    if child is not None:
        return dataclasses.replace(node, input=prebuild_factories(child))
    return node


def build_udf_callable(fn: object) -> object:
    """Resolve a `map_batches` `fn` to the per-batch callable.

    A *class* (type) is a stateful factory: it is instantiated once here to load
    the model, and the instance (which must be callable) handles each batch. Any
    other callable is used directly. This is what lets a model load once per worker
    instead of once per batch — the GPU-inference pattern. Called once per worker
    (locally: once; distributed: once per actor).
    """
    return fn() if isinstance(fn, type) else fn


def has_map_batches(plan: LogicalPlan) -> bool:
    """Whether the plan contains any `map_batches` operator."""
    if isinstance(plan, MapBatches):
        return True
    for f in dataclasses.fields(plan):
        v = getattr(plan, f.name)
        if isinstance(v, LogicalPlan) and has_map_batches(v):
            return True
        if isinstance(v, tuple) and any(
            isinstance(x, LogicalPlan) and has_map_batches(x) for x in v
        ):
            return True
    return False


def execute_with_udfs(plan: LogicalPlan, sources: list) -> list[pa.RecordBatch]:
    """Execute a (possibly non-linear) pipeline that contains `map_batches`.

    A linear ``Scan → map_batches → … → map_batches`` chain (the batch-inference /
    multimodal-preprocessing shape) runs through the **streaming, stage-overlapped**
    path (`_stream_linear_chain`): each stage runs on its own thread, pipelined, so a
    CPU stage (decode/resize/tokenize) prepares morsel *k+1* while the next stage (a GPU
    forward pass) runs morsel *k* — the GPU stays fed instead of idling through the whole
    decode. The result is identical to the staged materialization (same rows, per-batch
    contract; only the scheduling overlaps). Non-linear plans (joins/unions between maps)
    and the GPU-autobatch / multiprocessing strategies keep the materializing path.
    """
    chain = _linear_map_chain(plan)
    if chain is not None and _stream_eligible(chain[1]):
        return list(_stream_linear_chain(chain[0], chain[1], sources))
    batches, _schema = _execute_node(plan, sources)
    return batches


# Bounded look-ahead between pipelined map stages: a stage may run this many morsels
# ahead of its consumer (so a CPU stage overlaps the GPU stage draining it) while keeping
# resident memory to ~`depth` morsels per stage. Env-overridable.
_STREAM_PREFETCH_DEPTH = max(0, int(os.environ.get("BATCHER_STREAM_PREFETCH_DEPTH", "2")))


def _linear_map_chain(plan: LogicalPlan) -> tuple[Scan, list[MapBatches]] | None:
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


def _stream_eligible(stages: list[MapBatches]) -> bool:
    """Whether a linear chain should run on the streaming, stage-overlapped path.

    Targets the **multi-stage inference shape** — two or more stages, at least one on a
    GPU — where a CPU stage (decode/resize/tokenize) feeding a GPU stage overlaps to keep
    the device fed (the whole point). A single stage has nothing to overlap with (and its
    intra-stage `num_workers` threading + the auto-process path stay on the materializing
    route, unchanged). A GPU-autobatch stage (`num_gpus > 0`, no `batch_size`) owns its
    own dynamic batching via `InferencePool`, and an explicit `multiprocessing` stage runs
    across processes — both keep the materializing path.
    """
    if len(stages) < 2 or not any(op.num_gpus > 0 for op in stages):
        return False
    for op in stages:
        if op.num_gpus > 0 and op.batch_size is None:
            return False
        if getattr(op, "multiprocessing", False):
            return False
    return True


def _stream_linear_chain(
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
    target = op.batch_size if op.batch_size is not None else morsel
    workers = op.num_workers if isinstance(op.num_workers, int) and op.num_workers > 1 else 1

    def _subs(batch: pa.RecordBatch) -> list[pa.RecordBatch]:
        if batch.num_rows > target:
            return pa.Table.from_batches([batch]).to_batches(max_chunksize=target)
        return [batch]

    if workers <= 1:
        for batch in gen:
            if batch.num_rows:
                for sub in _subs(batch):
                    yield from _coerce_udf_result(call(sub))
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for batch in gen:
            if not batch.num_rows:
                continue
            for res in pool.map(call, _subs(batch)):
                yield from _coerce_udf_result(res)


def _execute_node(node: LogicalPlan, sources: list) -> tuple[list[pa.RecordBatch], pa.Schema]:
    """Materialize `node` to `(batches, schema)`.

    The schema is tracked alongside the batches so an *empty* sub-result (which
    carries no batch to read a schema from) can still be scanned by a parent
    operator — the case that makes joins/unions over filtered-to-empty inputs work.
    """
    if isinstance(node, Scan):
        batches = list(sources[node.source_id].read())
        return batches, (batches[0].schema if batches else node.schema.arrow)
    if isinstance(node, MapBatches):
        inputs, in_schema = _execute_node(node.input, sources)
        out = _apply_udf(inputs, node)
        # On empty input the UDF isn't called; assume a pass-through schema.
        return out, (out[0].schema if out else in_schema)
    # Any other relational operator: materialize each child, then run this single
    # operator on the engine with its children replaced by scans of those batches.
    child_results = [_execute_node(c, sources) for c in children(node)]
    return _run_engine_op(node, child_results)


def _run_engine_op(
    node: LogicalPlan, child_results: list[tuple[list[pa.RecordBatch], pa.Schema]]
) -> tuple[list[pa.RecordBatch], pa.Schema]:
    """Run one relational operator natively over already-materialized child inputs."""
    import batcher._native as nat

    inputs = [batches for batches, _ in child_results]
    scans = [Scan(i, SchemaRef.from_arrow(schema)) for i, (_, schema) in enumerate(child_results)]
    rebuilt = with_children(node, scans)
    out = list(nat.execute_plan(_to_json(rebuilt), inputs, active_config().engine_config_json()))
    # Output schema: the result's own when non-empty; otherwise a best-effort from
    # the first input (exact for schema-preserving/union ops, an approximation only
    # for the rare empty-result-feeds-a-parent case).
    out_schema = out[0].schema if out else (child_results[0][1] if child_results else None)
    return out, out_schema


def _to_json(op: LogicalPlan) -> str:
    import json

    return json.dumps(op.to_ir())


def _apply_udf(current: list[pa.RecordBatch], op: MapBatches) -> list[pa.RecordBatch]:
    """Apply a `map_batches` function, optionally rebatching to `batch_size` and
    running the per-batch calls across `num_workers` threads (order preserved).

    When `op.batch_format` is not ``"pyarrow"``, each Arrow batch is converted to the
    requested framework object (numpy/pandas/torch) for the call and the result
    converted back — the data plane stays Arrow, only the call is reframed."""
    if not current:
        return current
    batches = current
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
    use_processes = _wants_processes(op, total)
    morsel = max(1, active_config().execution.morsel_rows)
    if op.batch_size is not None:
        batches = pa.Table.from_batches(current).to_batches(max_chunksize=op.batch_size)
    elif use_processes:
        # A process-pool `fn` pays a per-batch pickle/IPC round-trip to the child, so
        # morsel-sized batches make that overhead dominate (measured: an order of
        # magnitude slower than a coarse split). We want a few coarse batches per worker.
        # But re-chunking concatenates the WHOLE input into one table and re-splits it —
        # a full copy of every column (a superlinear cost at scale). The natural batches
        # of a many-file read are already coarse and plentiful, so skip that copy unless
        # the current batching is actually bad: too few to fill the pool, any batch large
        # enough to unbalance a worker / bloat one IPC payload, or any morsel-tiny batch.
        cap = max(morsel, -(-total // max(1, op.num_workers)))  # ~one worker's share
        floor = min(cap, max(morsel, _PROC_MIN_BATCH_ROWS))
        sizes = [b.num_rows for b in current]
        already_coarse = (
            len(current) >= op.num_workers and max(sizes) <= cap and min(sizes) >= floor
        )
        if not already_coarse:
            target = max(floor, -(-total // max(1, op.num_workers * _PROC_BATCHES_PER_WORKER)))
            batches = pa.Table.from_batches(current).to_batches(max_chunksize=target)
    else:
        # A plain-function transform with no explicit `batch_size` is bounded to the
        # engine morsel size, so a downloading / row-exploding `fn` never receives the
        # whole partition as one batch (the unbounded-batch OOM — Ray Data's #1 cause;
        # an in-memory or wide-row source can hand a single multi-million-row batch
        # straight to the `fn`). Re-chunk only when an upstream batch actually exceeds
        # the morsel — a relation-level no-op (same rows, smaller chunks; map_batches is
        # per-batch by contract) that leaves an already-morselized pipeline untouched.
        if any(b.num_rows > morsel for b in current):
            batches = pa.Table.from_batches(current).to_batches(max_chunksize=morsel)

    strategy = _map_strategy(op, len(batches), use_processes)
    if strategy == "processes":
        # Run the per-batch calls across processes so a CPU-bound pure-Python `fn`
        # (which the GIL would serialize across threads) uses multiple cores. Any
        # process failure (an `fn` that turns out not to be process-safe) falls back
        # to threads — never a dropped batch.
        try:
            from batcher.core.udf_processes import run_map_processes

            results = run_map_processes(
                build_udf_callable(op.fn), batches, op.num_workers, op.batch_format
            )
        except Exception as exc:
            # A process pool can be unavailable for the whole session — e.g. a script
            # that runs the pipeline at import time is not import-safe, so forkserver/
            # spawn refuse to start a child. Remember that (`_disable_processes`) so the
            # fallback happens once here and every later call goes straight to threads,
            # instead of re-failing (and re-warning) on every batch/window.
            _disable_processes(exc)
            strategy = "threads"

    if strategy != "processes":
        # Build the model once for this whole call (a class `fn` is a load-once factory).
        fn = build_udf_callable(op.fn)
        call = fn if op.batch_format == "pyarrow" else _formatted(fn, op.batch_format)
        if strategy == "threads":
            # ThreadPoolExecutor.map keeps input order; concurrency only helps when `fn`
            # releases the GIL (Rust/GPU/NumPy inference), which is the intended use.
            with ThreadPoolExecutor(max_workers=op.num_workers) as pool:
                results = list(pool.map(call, batches))
        else:
            results = [call(batch) for batch in batches]

    out: list[pa.RecordBatch] = []
    for result in results:
        out.extend(_coerce_udf_result(result))
    return out


def _apply_udf_autobatch(op: MapBatches, batches: list[pa.RecordBatch]) -> list[pa.RecordBatch]:
    """Run the inference `fn` with online batch-size auto-tuning (zero-config inference).

    Routes through a throughput `InferencePool`: it re-chunks the input dynamically and
    hill-climbs the batch size toward the VRAM-capped throughput plateau (`ml.autobatch`),
    surviving a CUDA OOM by halving the batch — so the user never hand-tunes `batch_size`.
    Input order is preserved (a relation-level no-op vs a fixed size), and the model is
    built once and shared across the `num_workers` dispatch slots (a GPU stage resolves to
    one). The seed is conservative (climbs up); a future VRAM-aware seed sharpens the start.
    """
    from batcher.ml.inference import InferencePool

    fn = build_udf_callable(op.fn)
    call = fn if op.batch_format == "pyarrow" else _formatted(fn, op.batch_format)

    def worker(batch: pa.RecordBatch) -> pa.RecordBatch:
        coerced = _coerce_udf_result(call(batch))
        if not coerced:
            return batch.slice(0, 0)
        merged = pa.Table.from_batches(coerced).combine_chunks().to_batches()
        return merged[0] if merged else coerced[0]

    pool = InferencePool(
        lambda: worker, num_workers=op.num_workers, target_batch_rows=256, objective="throughput"
    )
    return list(pool.run(iter(batches)))


# A CPU-bound `map_batches` big enough to amortize the process-pool startup auto-runs
# across processes (the Ray Data default), so a plain `ds.map_batches(fn)` uses every
# core instead of one GIL — threads cap a pure-Python `fn` at one core and a NumPy `fn`
# at the GIL-handoff ceiling. Below this row count the thread path (no pool spawn) wins,
# so small queries keep their low fixed overhead.
_PROC_AUTO_MIN_ROWS = 1_000_000


# Set once if the process pool proves unusable this session (e.g. a non-import-safe
# entrypoint that forkserver/spawn cannot fork a child from). After that every stage
# stays on threads without re-attempting (and re-warning) a doomed pool per batch.
_PROCESSES_DISABLED = False


def _disable_processes(exc: BaseException) -> None:
    """Disable the process path for the rest of the session, warning once."""
    global _PROCESSES_DISABLED
    if not _PROCESSES_DISABLED:
        _PROCESSES_DISABLED = True
        warnings.warn(
            f"map_batches process pool unavailable ({exc!r}); using threads for the rest "
            "of this session (for a CPU-bound pure-Python fn, run under an "
            "`if __name__ == '__main__':` guard so a worker process can start)",
            stacklevel=3,
        )


def _wants_processes(op: MapBatches, total_rows: int) -> bool:
    """Whether to run this `map_batches` across processes (vs threads).

    Processes when the user opted in (`op.multiprocessing`) or — adaptively — when a
    process-capable CPU `fn` is handed enough rows that spreading it across cores beats
    the GIL-bound thread path despite the pool-startup cost. A GPU / class / non-pyarrow
    / unpicklable `fn` never qualifies (see `_process_capable`), nor does anything once
    the pool has proven unusable this session. The runtime still falls back to threads if
    a process actually fails, so this never drops a batch.
    """
    if op.num_workers <= 1 or _PROCESSES_DISABLED:
        return False
    if op.multiprocessing:
        return _process_safe(op)
    return total_rows >= _PROC_AUTO_MIN_ROWS and _process_capable(op)


def _map_strategy(op: MapBatches, n_batches: int, use_processes: bool) -> str:
    """Pick how to run the per-batch calls: ``sequential``, ``threads``, or ``processes``.

    `use_processes` is the pre-computed intent (`_wants_processes`); a single batch or a
    single worker collapses to sequential, everything else to threads.
    """
    if n_batches <= 1 or op.num_workers <= 1:
        return "sequential"
    return "processes" if use_processes else "threads"


def _process_capable(op: MapBatches) -> bool:
    """Whether `op.fn` *can* run in a process pool (a quiet predicate, no warning).

    A factory/class would reload the model per child (and risk OOM); a GPU `fn` wants
    one CUDA context; a lambda/closure `fn` cannot be pickled to a spawned child. Any
    `batch_format` is fine — the numpy/pandas/torch conversion runs in the child from
    the picklable ``(fn, batch, fmt)`` payload (`_proc_call`), no closure required.
    """
    return not isinstance(op.fn, type) and op.num_gpus <= 0 and _is_picklable(op.fn)


def _process_safe(op: MapBatches) -> bool:
    """Whether `op.fn` can run in a process pool; warn-once and reject otherwise.

    The warning variant of `_process_capable`, used when the user *explicitly* asked for
    `multiprocessing=True` — so an ignored request is surfaced, not silently downgraded.
    """
    if isinstance(op.fn, type):
        return _reject("a factory/class fn would reload per process")
    if op.num_gpus > 0:
        return _reject("a GPU fn must keep a single process/CUDA context")
    if not _is_picklable(op.fn):
        return _reject("the fn is not picklable (a lambda or closure)")
    return True


_REJECTED: set[str] = set()


def _reject(reason: str) -> bool:
    """Warn once per distinct reason that processes were declined, then return False."""
    if reason not in _REJECTED:
        _REJECTED.add(reason)
        warnings.warn(
            f"map_batches multiprocessing not used ({reason}); using threads",
            stacklevel=3,
        )
    return False


def _is_picklable(obj: object) -> bool:
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


def _formatted(fn: Any, fmt: str) -> Any:
    """Wrap `fn` so it receives/returns `fmt` batches while the caller stays Arrow."""
    from batcher.ml.batch_format import result_to_arrowable, to_format

    def _call(batch: pa.RecordBatch) -> object:
        return result_to_arrowable(fn(to_format(batch, fmt)), fmt)

    return _call


def _coerce_udf_result(result: object) -> list[pa.RecordBatch]:
    """Normalize a `map_batches` return (RecordBatch / Table / column dict) to batches."""
    if isinstance(result, pa.RecordBatch):
        return [result]
    if isinstance(result, pa.Table):
        return result.to_batches()
    if isinstance(result, dict):
        return [pa.RecordBatch.from_pydict(_tensorize_columns(result))]
    raise TypeError(
        "map_batches function must return a pyarrow RecordBatch, Table, or dict; "
        f"got {type(result).__name__}"
    )


def _tensorize_columns(result: dict) -> dict:
    """Turn any multi-dimensional NumPy value into a fixed-shape-tensor column.

    A `map_batches` `fn` (image decode, embedding, feature-map) commonly returns a
    ``(B, *shape)`` NumPy array per column — the Ray Data tensor-block shape.
    ``from_pydict`` can't build a column from a >1-D array, so multi-dim values are
    converted to the canonical ``arrow.fixed_shape_tensor`` column (`to_tensor_column`),
    which round-trips zero-copy through the FFI with its shape intact. 1-D arrays, lists,
    and Arrow arrays pass through untouched, so scalar/label columns are unchanged. This
    keeps the tensor path identical single-node and distributed, for every modality.
    """
    import numpy as np

    from batcher.io.formats.ml.tensor import to_tensor_column

    converted: dict = {}
    for name, value in result.items():
        if isinstance(value, np.ndarray) and value.ndim >= 2:
            converted[name] = to_tensor_column(value)
        else:
            converted[name] = value
    return converted
