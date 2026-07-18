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
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pyarrow as pa

from batcher._internal.native import engine
from batcher.config import active_config
from batcher.core.udf import strategy as strat
from batcher.io.schema.evolution import reconcile_batches
from batcher.plan.logical import LogicalPlan, MapBatches, Scan
from batcher.plan.schema import SchemaRef
from batcher.plan.visitor import children, with_children

__all__ = ["build_udf_callable", "execute_with_udfs", "has_map_batches", "prebuild_factories"]


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


def execute_with_udfs(
    plan: LogicalPlan,
    sources: list,
    source_projections: dict[int, list[str]] | None = None,
    engine_config: str | None = None,
) -> list[pa.RecordBatch]:
    """Execute a (possibly non-linear) pipeline that contains `map_batches`.

    `engine_config` is the driver's `EngineConfig` JSON, shipped in by a distributed caller.
    A Ray worker's own `active_config()` is that process's default — it never sees the
    driver's session config, and its `parallelism: 0` ("use every core") would give each of
    many concurrent tasks on one node a full-width rayon pool. `None` (the in-process caller)
    uses the ambient config.

    `source_projections` is the per-source column list Kyber decided (Core executes the plan it
    is given; it does not compute projections — see `.claude/rules/architecture.md`). Without it
    the scan beneath a UDF reads **every** column of the source, which is what a `map_batches`
    over one column of a wide table used to do. `None` means "read everything", the old
    behavior, which is what an undeclared (`input_columns=None`) UDF still requires.

    A linear ``Scan → map_batches → … → map_batches`` chain (the batch-inference /
    multimodal-preprocessing shape) runs through the **streaming, stage-overlapped**
    path (`udf_stream.stream_linear_chain`): each stage runs on its own thread, so a
    CPU stage (decode/resize/tokenize) prepares morsel *k+1* while the next stage (a GPU
    forward pass) runs morsel *k* — the GPU stays fed instead of idling through the whole
    decode. The result is identical to the staged materialization (same rows, per-batch
    contract; only the scheduling overlaps). Non-linear plans (joins/unions between maps)
    and the GPU-autobatch / multiprocessing strategies keep the materializing path.
    """
    from batcher.core.udf.stream import linear_map_chain, stream_eligible, stream_linear_chain

    projections = source_projections or {}
    cfg = engine_config or active_config().engine_config_json()
    if not has_map_batches(plan):
        return _run_whole_plan(plan, sources, projections, cfg)
    chain = linear_map_chain(plan)
    if chain is not None and stream_eligible(chain[1]):
        # Reconcile the streamed output to one union schema, exactly as the materializing
        # path does per stage (`_execute_node`). A UDF whose output schema DRIFTS across
        # batches (e.g. LLM structured outputs with varying fields) yields batches of
        # differing schemas; without this the final `Table.from_batches` raises on the
        # first drift, so the streaming path would crash on inputs the staged path handles.
        # The chain's output is already fully listed here, so this adds no extra buffering.
        return reconcile_batches(
            list(stream_linear_chain(chain[0], chain[1], sources, projections))
        )
    batches, _schema = _execute_node(plan, sources, projections, cfg)
    return batches


def _run_whole_plan(
    plan: LogicalPlan, sources: list, projections: dict[int, list[str]], cfg: str
) -> list[pa.RecordBatch]:
    """Run a plan with no `map_batches` in ONE engine call — the whole plan, one pass.

    This entry point exists for pipelines that mix Python UDFs with relational operators, so
    it walks the plan as a tree and runs each operator separately (`_execute_node`). But it is
    also the *distributed map task's* executor (`dist.executors.map._map_udf_task`), and the
    dispatcher routes every breaker-free scan/filter/project over a splittable source there —
    plans that contain no UDF at all. Walking those as a tree costs, per operator, an IR
    serialization, an FFI crossing, and a full Python materialization of the partition
    *between* operators: a `Project(Filter(Scan))` partition was decoded, handed to Rust to
    filter, rebuilt as Python `RecordBatch`es, handed back to Rust to project, and rebuilt
    again — while the engine fuses and morsel-pipelines the whole thing in one pass.

    So when there is no UDF to orchestrate, don't orchestrate: hand the engine the whole plan,
    exactly as the single-node `LocalExecutor` does. Same operators, same engine, same result —
    one crossing instead of N, and no Python-side intermediate.
    """
    nat = engine()
    # `execute_plan` addresses sources positionally by `Scan.source_id`, so the list must keep
    # each source at its own index; only the ones the plan actually scans are read.
    scanned = _scanned_source_ids(plan)
    inputs = [
        list(src.read(projections.get(i))) if i in scanned else [] for i, src in enumerate(sources)
    ]
    return list(nat.execute_plan(_to_json(plan), inputs, cfg))


def _scanned_source_ids(node: LogicalPlan) -> set[int]:
    """The `source_id`s the plan actually reads."""
    if isinstance(node, Scan):
        return {node.source_id}
    ids: set[int] = set()
    for child in children(node):
        ids |= _scanned_source_ids(child)
    return ids


def _resilient_call(
    call, sub: pa.RecordBatch, budget: list[int], is_gpu: bool
) -> list[pa.RecordBatch]:
    """Run a per-batch `call`, isolating failures by bisection — the unified OOM-halving +
    dirty-data-tolerance path.

    On a CUDA OOM (GPU stage) the batch is halved and retried (a too-large batch often fits at
    N/2; the per-row-independent outputs concatenate to the whole result); a single row that
    still OOMs is a genuine over-allocation and re-raises. On any OTHER error the batch is
    bisected to isolate the offending row(s): a failing single row is DROPPED (charged against
    `budget`, the ``max_errored_rows`` allowance) so a corrupt image / malformed record doesn't
    kill a long job — until the budget is exhausted, when it re-raises. With ``budget == 0``
    and a CPU stage this reduces to strict behavior (any error propagates), so a real bug on
    clean data still fails fast."""
    from batcher.ml.inference import _empty_cuda_cache, _is_cuda_oom

    try:
        return _coerce_udf_result(call(sub))
    except Exception as exc:
        oom = is_gpu and _is_cuda_oom(exc)
        if oom:
            _empty_cuda_cache()
        if sub.num_rows <= 1:
            if oom or budget[0] <= 0:
                raise  # genuine single-row over-allocation, or the error budget is spent
            budget[0] -= 1
            return []  # drop the one corrupt row and carry on
        mid = sub.num_rows // 2
        left = _resilient_call(call, sub.slice(0, mid), budget, is_gpu)
        return left + _resilient_call(call, sub.slice(mid), budget, is_gpu)


def _execute_node(
    node: LogicalPlan,
    sources: list,
    projections: dict[int, list[str]] | None = None,
    cfg: str | None = None,
) -> tuple[list[pa.RecordBatch], pa.Schema]:
    """Materialize `node` to `(batches, schema)`.

    The schema is tracked alongside the batches so an *empty* sub-result (which
    carries no batch to read a schema from) can still be scanned by a parent
    operator — the case that makes joins/unions over filtered-to-empty inputs work.
    """
    projections = projections or {}
    cfg = cfg or active_config().engine_config_json()
    if isinstance(node, Scan):
        # Read only the columns the plan needs. Kyber computed them; a `map_batches` that
        # declared no `input_columns` yields None here, so the whole source is read (safe).
        batches = list(sources[node.source_id].read(projections.get(node.source_id)))
        return batches, (batches[0].schema if batches else node.schema.arrow)
    if isinstance(node, MapBatches):
        inputs, in_schema = _execute_node(node.input, sources, projections, cfg)
        # Reconcile a UDF whose output schema drifts across batches (e.g. LLM structured
        # outputs with varying fields) to one union schema, so the stage's batches concat
        # instead of failing — the schema-inference footgun Ray Data hits.
        out = reconcile_batches(_apply_udf(inputs, node))
        # On empty input the UDF isn't called; assume a pass-through schema.
        return out, (out[0].schema if out else in_schema)
    # Any other relational operator: materialize each child, then run this single
    # operator on the engine with its children replaced by scans of those batches.
    # `projections` must reach those children: a Scan under a Filter/Join is still the
    # scan Kyber pruned columns for, and dropping the map here made it read every column.
    child_results = [_execute_node(c, sources, projections, cfg) for c in children(node)]
    return _run_engine_op(node, child_results, cfg)


def _run_engine_op(
    node: LogicalPlan,
    child_results: list[tuple[list[pa.RecordBatch], pa.Schema]],
    cfg: str,
) -> tuple[list[pa.RecordBatch], pa.Schema]:
    """Run one relational operator natively over already-materialized child inputs."""
    nat = engine()
    inputs = [batches for batches, _ in child_results]
    scans = [Scan(i, SchemaRef.from_arrow(schema)) for i, (_, schema) in enumerate(child_results)]
    rebuilt = with_children(node, scans)
    out = list(nat.execute_plan(_to_json(rebuilt), inputs, cfg))
    # Output schema: the result's own when non-empty; otherwise a best-effort from
    # the first input (exact for schema-preserving/union ops, an approximation only
    # for the rare empty-result-feeds-a-parent case).
    out_schema = out[0].schema if out else (child_results[0][1] if child_results else None)
    return out, out_schema


def _to_json(op: LogicalPlan) -> str:
    import json

    return json.dumps(op.to_ir())


def _rechunk(batches: list[pa.RecordBatch], target: int) -> list[pa.RecordBatch]:
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
    use_processes = strat.wants_processes(op, total, current)
    morsel = max(1, active_config().execution.morsel_rows)
    if op.batch_size is not None:
        batches = _rechunk(current, op.batch_size)
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
        floor = min(cap, max(morsel, strat.PROC_MIN_BATCH_ROWS))
        sizes = [b.num_rows for b in current]
        already_coarse = (
            len(current) >= op.num_workers and max(sizes) <= cap and min(sizes) >= floor
        )
        if not already_coarse:
            target = max(floor, -(-total // max(1, op.num_workers * strat.PROC_BATCHES_PER_WORKER)))
            batches = _rechunk(current, target)
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
            batches = _rechunk(current, target)

    strategy = strat.map_strategy(op, len(batches), use_processes)
    if strategy == "processes":
        # Run the per-batch calls across processes so a CPU-bound pure-Python `fn`
        # (which the GIL would serialize across threads) uses multiple cores. Any
        # process failure (an `fn` that turns out not to be process-safe) falls back
        # to threads — never a dropped batch.
        try:
            from batcher.core.udf.processes import run_map_processes

            results = run_map_processes(
                build_udf_callable(op.fn), batches, op.num_workers, op.batch_format
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
        # Build the model once for this whole call (a class `fn` is a load-once factory).
        fn = build_udf_callable(op.fn)
        call = fn if op.batch_format == "pyarrow" else _formatted(fn, op.batch_format)
        if op.num_gpus > 0:
            from batcher.ml.gpu import autocast_call

            call = autocast_call(call)  # tensor-core half precision (no-op when off/CPU)
        if op.max_errored_rows > 0:
            # Dirty-data tolerance: isolate and skip corrupt rows (up to the budget) instead
            # of crashing — a single-stage inference / preprocess over messy data survives.
            budget = [op.max_errored_rows]
            is_gpu = op.num_gpus > 0

            def _emit(b: pa.RecordBatch) -> list[pa.RecordBatch]:
                return _resilient_call(call, b, budget, is_gpu)

            if strategy == "threads":
                with ThreadPoolExecutor(max_workers=op.num_workers) as pool:
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
            with ThreadPoolExecutor(max_workers=op.num_workers) as pool:
                results = list(pool.map(call, batches))
        else:
            results = [call(batch) for batch in batches]

    out = []
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
    from batcher.ml.gpu import autocast_call
    from batcher.ml.inference import InferencePool

    fn = build_udf_callable(op.fn)
    call = fn if op.batch_format == "pyarrow" else _formatted(fn, op.batch_format)
    call = autocast_call(call)  # tensor-core half precision (GPU stage; no-op when off/CPU)

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
        # A 0-row Table yields *no* batches, which would drop the stage's output schema (the
        # parent falls back to the input schema and a downstream ref to a UDF-added column
        # fails). Keep one empty batch so the schema survives, like a 0-row RecordBatch does.
        batches = result.to_batches()
        if batches:
            return batches
        cols = [pa.array([], type=f.type) for f in result.schema]
        return [pa.RecordBatch.from_arrays(cols, schema=result.schema)]
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
