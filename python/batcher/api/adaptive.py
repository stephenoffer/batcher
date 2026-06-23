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
    """
    srcs = list(sources)
    decisions: list = []
    stages = 0

    while True:
        target = _lowest_breaker(plan)
        if target is None:
            break
        table, decs = _run_stage(target, srcs, hub, distributed, num_workers, transport)
        decisions.extend(decs)
        stages += 1
        if target is plan:
            return AdaptiveResult(table, decisions, stages)
        # Replace the materialized breaker with a scan over its (exact-size) result.
        batches = table.to_batches() or [pa.RecordBatch.from_pylist([], schema=table.schema)]
        sid = len(srcs)
        srcs.append(InMemorySource(batches))
        plan = _replace(plan, target, Scan(sid, SchemaRef.from_arrow(table.schema)))

    table, decs = _run_stage(plan, srcs, hub, distributed, num_workers, transport)
    decisions.extend(decs)
    return AdaptiveResult(table, decisions, stages + 1)


def _run_stage(
    node: LogicalPlan,
    sources: list[Source],
    hub,
    distributed: bool = False,
    num_workers: int | None = None,
    transport: str = "auto",
) -> tuple[pa.Table, list]:
    """Optimize + execute one stage, returning its result and join decisions.

    Each stage runs through the shared `run_relational` orchestrator — the same
    Kyber → Carbonite → Core contract loop the one-shot executors use — so an
    adaptive stage gets the full rule set, resource admission, spill, and the
    metadata feedback loop. Its inputs are already materialized `InMemorySource`s
    with exact `row_count`, so the optimizer's estimator reads *measured* sizes for
    its build-side/broadcast/join-order choices, not guesses.
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
    return run_relational(node, sources, ctx, distributed=distributed)


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
