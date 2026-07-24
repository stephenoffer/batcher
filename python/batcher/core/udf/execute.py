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
from collections.abc import Iterator

import pyarrow as pa

from batcher._internal.native import engine
from batcher.config import active_config
from batcher.core.udf.apply import apply_udf
from batcher.core.udf.lifecycle import build_udf_callable
from batcher.io.schema.evolution import reconcile_batches
from batcher.plan.logical import LogicalPlan, MapBatches, Scan
from batcher.plan.schema import SchemaRef
from batcher.plan.visitor import children, with_children

__all__ = [
    "build_udf_callable",
    "execute_with_udfs",
    "has_map_batches",
    "prebuild_factories",
    "stream_with_udfs",
]


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

    **This returns a list, so peak memory is the whole output** — the stage overlap is real
    but the bounded-memory half of streaming is not available here. A caller that consumes
    batches incrementally (a distributed map task that writes from the worker) should use
    `stream_with_udfs` instead, which is the same execution with the materialization removed.
    """
    projections = source_projections or {}
    cfg = engine_config or active_config().engine_config_json()
    if not has_map_batches(plan):
        return _run_whole_plan(plan, sources, projections, cfg)
    gen = _linear_stream(plan, sources, projections)
    if gen is not None:
        # Reconcile the streamed output to one union schema, exactly as the materializing
        # path does per stage (`_execute_node`). A UDF whose output schema DRIFTS across
        # batches (e.g. LLM structured outputs with varying fields) yields batches of
        # differing schemas; without this the final `Table.from_batches` raises on the
        # first drift, so the streaming path would crash on inputs the staged path handles.
        # The chain's output is already fully listed here, so this adds no extra buffering.
        return reconcile_batches(list(gen))
    batches, _schema = _execute_node(plan, sources, projections, cfg)
    return batches


def stream_with_udfs(
    plan: LogicalPlan,
    sources: list,
    source_projections: dict[int, list[str]] | None = None,
    engine_config: str | None = None,
) -> Iterator[pa.RecordBatch]:
    """Execute a `map_batches` pipeline, yielding output batches **as they are produced**.

    The incremental form of `execute_with_udfs`, with the same arguments and the same rows in
    the same order. The difference is memory: for a linear ``Scan -> map -> ... -> map`` chain
    on the streaming path, nothing accumulates. Resident memory is the bounded prefetch windows
    between the stages (a few morsels each), not the query's whole output — so a worker can
    read, infer, and write a partition far larger than its RAM. `execute_with_udfs` cannot do
    this by construction: it hands back a `list`.

    A plan the streaming path can't take (a join or union between maps, a multiprocessing
    stage, a CPU-only chain) falls back to `execute_with_udfs` and is yielded from the
    materialized result. That is a scheduling difference only — same rows either way — but the
    memory bound does *not* hold for it, so don't read "iterator" as "bounded" unconditionally.
    """
    projections = source_projections or {}
    if has_map_batches(plan):
        gen = _linear_stream(plan, sources, projections)
        if gen is not None:
            from batcher.core.udf.stream import reconcile_stream

            yield from reconcile_stream(gen)
            return
    yield from execute_with_udfs(plan, sources, source_projections, engine_config)


def _linear_stream(
    plan: LogicalPlan, sources: list, projections: dict[int, list[str]]
) -> Iterator[pa.RecordBatch] | None:
    """The stage-overlapped batch stream for `plan`, or `None` if it isn't eligible.

    The one place the streaming route is decided, so the listing caller and the streaming
    caller can never disagree about which plans stream.
    """
    from batcher.core.udf.stream import linear_map_chain, stream_eligible, stream_linear_chain

    chain = linear_map_chain(plan)
    if chain is None or not stream_eligible(chain[1]):
        return None
    return stream_linear_chain(chain[0], chain[1], sources, projections)


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
        out = reconcile_batches(apply_udf(inputs, node))
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
