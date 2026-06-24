"""Adaptive (intra-query) execution: stage-boundary re-optimization.

A static optimizer plans the whole query once against cardinality *estimates*.
The adaptive executor instead materializes the plan one pipeline breaker at a
time and re-optimizes the remaining plan with that breaker's **exact** output
cardinality fed back as a known-size source. Downstream decisions — notably join
build-side — therefore use *measured* sizes (provenance `exact`) rather than
guesses, even when the estimate would have been badly wrong (e.g. a very
selective filter feeding a join). This is the metadata-driven moat that static
engines (DuckDB) and stage-plan-only adapters can't match.

Mechanism: find the lowest breaker whose inputs are all breaker-free, execute it
through the normal optimize→engine path, replace its subtree with a `Scan` over
an in-memory source holding the result (whose `row_count` is now exact), and
repeat. Each stage is optimized with its inputs already materialized, so a join
over two aggregates picks its build side from the two real sizes.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa

from batcher.io.source import InMemorySource, Source
from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Join,
    Limit,
    LogicalPlan,
    Scan,
    Sort,
    Union,
    Window,
    is_streamable,
)
from batcher.plan.schema import SchemaRef

__all__ = ["AdaptiveResult", "execute_adaptive"]

_BREAKERS = (Aggregate, Sort, Distinct, Window, Limit, Join, Union)


@dataclasses.dataclass(frozen=True, slots=True)
class AdaptiveResult:
    table: pa.Table
    decisions: list  # BuildSideDecision per join, across all re-optimized stages
    stages: int


def execute_adaptive(
    plan: LogicalPlan,
    sources: list[Source],
    hub,
    *,
    distributed: bool = False,
    num_workers: int | None = None,
    transport: str = "auto",
) -> AdaptiveResult:
    """Run a plan with stage-boundary re-optimization.

    When `distributed`, each breaker stage fans out across Ray workers and its
    *exact* output cardinality feeds the next stage's optimizer — so even at scale
    join build-side and broadcast choices use measured sizes, not estimates. This
    is strictly stronger than Spark AQE (which adapts only at stage boundaries on
    coarse stats); the mergeable algebra guarantees the result equals single-node.

    Intermediate distributed stages keep their result *partitioned on disk* (a
    `MaterializedSource`) rather than collecting it to the driver, so a large
    multi-stage query never funnels every breaker's output through driver memory.
    Those intermediates are cleaned up once the whole query finishes.
    """
    srcs = list(sources)
    decisions: list = []
    stages = 0
    intermediates: list = []  # partitioned-on-disk/Flight sources, cleaned up at the end

    try:
        while True:
            target = _lowest_breaker(plan)
            if target is None:
                break
            final = target is plan
            # Intermediate stages may stay partitioned (materialize=False); the final
            # stage must collect a table to return.
            result, decs = _run_stage(
                target, srcs, hub, distributed, num_workers, transport, materialize=final
            )
            decisions.extend(decs)
            stages += 1
            if final:
                return AdaptiveResult(_as_table(result, target), decisions, stages)
            # Splice a Scan over the breaker's result (exact-size) for the rest of the
            # plan. A `MaterializedSource` is scanned in place; a collected table is
            # re-wrapped as an in-memory source (the single-node / fallback path).
            src, schema = _stage_source(result)
            # A partitioned intermediate (disk `MaterializedSource` or
            # `FlightMaterializedSource`) owns resources (files / worker actors) freed
            # after the final result; duck-type on `cleanup` so both are tracked.
            if callable(getattr(src, "cleanup", None)):
                intermediates.append(src)
            sid = len(srcs)
            srcs.append(src)
            plan = _replace(plan, target, Scan(sid, schema))

        result, decs = _run_stage(
            plan, srcs, hub, distributed, num_workers, transport, materialize=True
        )
        decisions.extend(decs)
        return AdaptiveResult(_as_table(result, plan), decisions, stages + 1)
    finally:
        # The final result is a fully in-memory table, independent of the on-disk
        # intermediates, so they can be removed now (best-effort).
        for m in intermediates:
            m.cleanup()


def _run_stage(
    node: LogicalPlan,
    sources: list[Source],
    hub,
    distributed: bool = False,
    num_workers: int | None = None,
    transport: str = "auto",
    *,
    materialize: bool = True,
) -> tuple[pa.Table | Source, list]:
    """Optimize + execute one stage, returning its result and join decisions.

    Each stage runs through the shared `run_relational` orchestrator — the same
    Kyber → Carbonite → Core contract loop the one-shot executors use — so an
    adaptive stage gets the full rule set, resource admission, spill, and the
    metadata feedback loop. Its inputs are already materialized sources with exact
    `row_count`, so the optimizer's estimator reads *measured* sizes for its
    build-side/broadcast/join-order choices, not guesses. With ``materialize=False``
    a distributed stage may return a `MaterializedSource` (result kept on disk).
    """
    from batcher import core
    from batcher.api.orchestration import run_relational

    if core.has_map_batches(node):
        batches = core.execute_with_udfs(node, sources)
        return _table(batches, node), []

    ctx = core.ExecutionContext(
        columns=node.available_columns(),
        hub=hub,
        num_workers=num_workers,
        transport=transport,
    )
    return run_relational(node, sources, ctx, distributed=distributed, materialize=materialize)


def _as_table(result: pa.Table | Source, node: LogicalPlan) -> pa.Table:
    """The stage result as a table — reading a `MaterializedSource` back if needed."""
    if isinstance(result, pa.Table):
        return result
    return _table(list(result.iter_batches()), node)


def _stage_source(result: pa.Table | Source) -> tuple[Source, SchemaRef]:
    """A source + schema to splice in for the next stage's scan over `result`.

    A `MaterializedSource` is passed through (scanned in place, shared-nothing); a
    collected table is wrapped as an `InMemorySource` (its exact `row_count` still
    feeds the optimizer).
    """
    if isinstance(result, pa.Table):
        batches = result.to_batches() or [pa.RecordBatch.from_pylist([], schema=result.schema)]
        return InMemorySource(batches), SchemaRef.from_arrow(result.schema)
    return result, SchemaRef.from_arrow(result.schema())


def _table(batches, node) -> pa.Table:
    if batches:
        return pa.Table.from_batches(batches, schema=batches[0].schema)
    return pa.table({c: [] for c in node.available_columns()})


def _children(node: LogicalPlan) -> list[LogicalPlan]:
    if isinstance(node, Join):
        return [node.left, node.right]
    if isinstance(node, Union):
        return list(node.inputs)
    if hasattr(node, "input"):
        return [node.input]
    return []


def _lowest_breaker(node: LogicalPlan):
    """A breaker whose inputs are all breaker-free (so it can run now)."""
    for child in _children(node):
        found = _lowest_breaker(child)
        if found is not None:
            return found
    if isinstance(node, _BREAKERS) and all(is_streamable(c) for c in _children(node)):
        return node
    return None


def _replace(node: LogicalPlan, target: LogicalPlan, repl: LogicalPlan) -> LogicalPlan:
    if node is target:
        return repl
    if isinstance(node, Join):
        return Join(
            _replace(node.left, target, repl),
            _replace(node.right, target, repl),
            node.left_keys,
            node.right_keys,
            node.join_type,
            node.output,
            node.strategy,
        )
    if isinstance(node, Union):
        return Union(tuple(_replace(i, target, repl) for i in node.inputs), node.distinct)
    if hasattr(node, "input"):
        return dataclasses.replace(node, input=_replace(node.input, target, repl))
    return node
