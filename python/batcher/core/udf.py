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

    # Build the model once for this whole call (a class `fn` is a load-once factory).
    fn = build_udf_callable(op.fn)
    call = fn if op.batch_format == "pyarrow" else _formatted(fn, op.batch_format)
    if op.num_workers > 1 and len(batches) > 1:
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
