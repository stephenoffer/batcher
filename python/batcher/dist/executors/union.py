"""Distributed UNION — one shuffle when the branches allow it, else branch by branch.

A UNION (distinct) over map-only branches is a `Distinct` over UNION ALL, so it goes
through the aggregate shuffle: every branch maps into one bucket space and each reducer
dedups its own bucket, so only the deduped result reaches the driver.

Otherwise the branches are independent sub-plans, so each is run through the distributed
dispatcher (an aggregate/join/sort branch shuffles across workers, a plain scan falls back
to single-node) and the results are concatenated on the driver, with UNION (distinct)
deduplicating the concatenation there. That path holds the whole union on one node, which
is why the shuffle above is preferred wherever it applies. The result equals single-node
either way.
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

    # UNION (distinct) IS a `Distinct` over UNION ALL, and a distinct is a group-by over every
    # column — so when the branches are map-only it rides the aggregate shuffle: each branch
    # maps into the same bucket space and each reducer dedups its own bucket. Only the deduped
    # result comes back. The alternative below concatenates every branch on the driver and
    # dedups there, which is the whole of both inputs through one node and one core.
    if union.distinct:
        deduped = _shuffled_dedup(union, sources, workers)
        if deduped is not None:
            return deduped if not above else _apply_above(above, deduped)

    # Each branch scans its own source ids out of the shared `sources` list.
    tables = [execute_distributed(inp, sources, workers, transport) for inp in union.inputs]
    # Typed, not `pa.table({})`: a column-less table is not this union's relation, so an
    # empty result would disagree with the same query's non-empty run on names and types
    # and fail to concatenate against it.
    result = _concat(tables) if tables else _empty_union(union)
    if union.distinct:
        result = _dedup(result)
    return result if not above else _apply_above(above, result)


def _concat(tables: list[pa.Table]) -> pa.Table:
    """Concatenate the branch results, promoting their types the way a UNION does.

    A union's branches must share column *names*, not column types: `Union.available_schema`
    promotes an `Int64` branch against a `Float64` one, and the single-node engine returns the
    promoted relation. A plain `pa.concat_tables` does not promote, so the same query that
    succeeds single-node raised `ArrowInvalid: Schema at index 1 was different` under
    `distributed=True` — a distributed-only failure on a query with a perfectly good answer.

    `promote_options="permissive"` applies exactly that widening. It is a no-op for the
    overwhelmingly common case where the branches already agree, so the identical-schema path
    is byte-for-byte what it was.
    """
    try:
        return pa.concat_tables(tables, promote_options="permissive")
    except (pa.ArrowInvalid, TypeError):
        # An older pyarrow without `promote_options`, or a genuinely unpromotable pair — the
        # strict concat's error names the mismatch, which is the more useful one to surface.
        return pa.concat_tables(tables)


def _shuffled_dedup(union: Union, sources: list[Source], workers: int) -> pa.Table | None:
    """`union` deduplicated through the aggregate shuffle, or `None` if its shape can't.

    The rewrite is `UNION` → `DISTINCT(UNION ALL)` → the group-by-every-column aggregate
    `Distinct.as_aggregate` already derives for every other dedup path, so no new semantics
    and no distinct-specific state — the reducers run the same `partial → combine → finalize`
    they run for a `GROUP BY`. Returns `None` when the branches are not all map-only over a
    single source each (a branch with its own breaker still needs its own distributed pass
    first), leaving the caller's per-branch path exactly as it was.
    """
    import dataclasses

    from batcher.dist.executor import _require_shared_scratch
    from batcher.dist.executors.aggregate import _distributed_aggregate
    from batcher.dist.executors.plan_analysis import shuffle_branches
    from batcher.plan.logical import Distinct

    union_all = dataclasses.replace(union, distinct=False)
    if shuffle_branches(union_all) is None:
        return None
    # The disk shuffle hands paths between tasks, so it needs scratch every node can read —
    # the same requirement, for the same reason, that the ASOF join carries.
    _require_shared_scratch("union distinct")
    return _distributed_aggregate([], Distinct(union_all).as_aggregate(), sources, workers)


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
