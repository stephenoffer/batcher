"""Distributed UNION — distribute each branch, then concatenate (and dedup).

A UNION's branches are independent sub-plans, so each is run through the
distributed dispatcher (distributing whatever it can — an aggregate/join/sort
branch shuffles across workers, a plain scan falls back to single-node) and the
results are concatenated. UNION (distinct) then deduplicates the concatenation; the
heavy per-branch work is distributed and the final dedup runs over the already-
reduced union. The result equals single-node.
"""

from __future__ import annotations

import pyarrow as pa

from batcher.dist.executors.partition_io import _apply_above
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan, Union


def _distributed_union(
    above: list[LogicalPlan],
    union: Union,
    sources: list[Source],
    workers: int,
    transport: str,
) -> pa.Table:
    """Run `union` by distributing each branch and concatenating the results."""
    from batcher.dist.executor import execute_distributed

    # Each branch scans its own source ids out of the shared `sources` list.
    tables = [execute_distributed(inp, sources, workers, transport) for inp in union.inputs]
    # Typed, not `pa.table({})`: a column-less table is not this union's relation, so an
    # empty result would disagree with the same query's non-empty run on names and types
    # and fail to concatenate against it.
    result = pa.concat_tables(tables) if tables else _empty_union(union)
    if union.distinct:
        result = _dedup(result)
    return result if not above else _apply_above(above, result)


def _empty_union(union: Union) -> pa.Table:
    """The union's schema with no rows."""
    from batcher.dist.executors.plan_analysis import empty_result_table

    return empty_result_table(union, union.available_columns())


def _dedup(table: pa.Table) -> pa.Table:
    """Deduplicate `table` via the engine's DISTINCT over an in-memory source."""
    if table.num_rows == 0:
        return table
    from batcher.dist.executors.ray_runtime import _single_node
    from batcher.io.source import InMemorySource
    from batcher.plan.logical import Distinct, Scan
    from batcher.plan.schema import SchemaRef

    src = InMemorySource(table.to_batches())
    plan = Distinct(Scan(0, SchemaRef.from_arrow(table.schema)))
    return _single_node(plan, [src])
