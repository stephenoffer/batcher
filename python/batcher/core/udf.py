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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

import pyarrow as pa

from batcher.config import active_config
from batcher.plan.logical import LogicalPlan, MapBatches, Scan
from batcher.plan.schema import SchemaRef
from batcher.plan.visitor import children, with_children

__all__ = ["build_udf_callable", "execute_with_udfs", "has_map_batches"]


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
    """Execute a (possibly non-linear) pipeline that contains `map_batches`."""
    batches, _schema = _execute_node(plan, sources)
    return batches


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
    if op.batch_size is not None:
        batches = pa.Table.from_batches(current).to_batches(max_chunksize=op.batch_size)

    strategy = _map_strategy(op, len(batches))
    if strategy == "processes":
        # Run the per-batch calls across processes so a CPU-bound pure-Python `fn`
        # (which the GIL would serialize across threads) uses multiple cores. Any
        # process failure (an `fn` that turns out not to be process-safe) falls back
        # to threads — never a dropped batch.
        try:
            results = _apply_udf_processes(op, batches)
        except Exception as exc:
            warnings.warn(
                f"map_batches multiprocessing failed ({exc!r}); falling back to threads",
                stacklevel=2,
            )
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


def _map_strategy(op: MapBatches, n_batches: int) -> str:
    """Pick how to run the per-batch calls: ``sequential``, ``threads``, or ``processes``.

    Threads are the default (they overlap GIL-releasing inference and keep the loaded
    model shared). Processes are opt-in (`op.multiprocessing`) and only for a
    process-safe `fn`; every reject silently falls back to threads.
    """
    if n_batches <= 1 or op.num_workers <= 1:
        return "sequential"
    if op.multiprocessing and _process_safe(op):
        return "processes"
    return "threads"


def _process_safe(op: MapBatches) -> bool:
    """Whether `op.fn` can run in a process pool; warn-once and reject otherwise.

    A factory/class would reload the model per child (and risk OOM); a non-pyarrow
    `batch_format` needs an unpicklable closure; a GPU `fn` wants one CUDA context;
    a lambda/closure `fn` cannot be pickled to a `spawn`ed child.
    """
    if isinstance(op.fn, type):
        return _reject("a factory/class fn would reload per process")
    if op.batch_format != "pyarrow":
        return _reject("batch_format != 'pyarrow' is not supported with processes")
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


# Per-child built callable, set once by the pool initializer so the (possibly heavy)
# `fn` is sent and built once per worker rather than re-pickled per batch.
_WORKER_CALL: object = None


def _udf_worker_init(fn: object) -> None:
    """Pool initializer: build the per-batch callable once in this child process."""
    global _WORKER_CALL
    _WORKER_CALL = build_udf_callable(fn)


def _udf_worker(batch: pa.RecordBatch) -> object:
    """Apply the child's pre-built callable to one batch (pyarrow in, arrowable out)."""
    return _WORKER_CALL(batch)  # type: ignore[operator]


def _apply_udf_processes(op: MapBatches, batches: list[pa.RecordBatch]) -> list[object]:
    """Run the per-batch calls across processes, preserving input order.

    Process count is budgeted against cores — never more processes than batches or
    available CPUs — the single-node analog of a fractional-CPU task share. Batches
    cross to/from children via pyarrow's pickle reducer (its Arrow IPC representation).
    """
    n_procs = max(1, min(op.num_workers, len(batches), os.cpu_count() or 1))
    with ProcessPoolExecutor(
        max_workers=n_procs, initializer=_udf_worker_init, initargs=(op.fn,)
    ) as pool:
        return list(pool.map(_udf_worker, batches))


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
        return [pa.RecordBatch.from_pydict(result)]
    raise TypeError(
        "map_batches function must return a pyarrow RecordBatch, Table, or dict; "
        f"got {type(result).__name__}"
    )
