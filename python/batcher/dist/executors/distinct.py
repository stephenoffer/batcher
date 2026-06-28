"""Distributed DISTINCT — deduplicate across workers via the aggregate shuffle.

DISTINCT is an aggregation that groups by *all* columns with no aggregate
functions, so it reuses the mergeable aggregate shuffle verbatim: every row is
hash-shuffled by its full set of column values, identical rows land on the same
reducer and are deduplicated there, and the union of reducers is the global
distinct. The mergeable primitives are reused unchanged, so the result equals
single-node execution.
"""

from __future__ import annotations

from batcher.dist.executors.aggregate import _distributed_aggregate
from batcher.io.source import Source
from batcher.plan.expr_ir import Col
from batcher.plan.logical import Aggregate, Distinct, LogicalPlan, Projection


def _distributed_distinct(
    above: list[LogicalPlan],
    distinct: Distinct,
    sources: list[Source],
    workers: int,
    transport: str = "disk",
    *,
    materialize: bool = True,
):
    """Run `distinct` across `workers` by routing it through the aggregate shuffle.

    Builds the equivalent `Aggregate` (group by every column, no aggregates) and
    reuses the aggregate shuffle — the Flight (Carbonite) path on a multi-node
    cluster, the disk Arrow-IPC path otherwise; `above` (operators after the
    DISTINCT) then runs single-node over the deduplicated result. `materialize=False`
    keeps the disk-path result partitioned (the aggregate shuffle's behavior) for the
    next adaptive stage.
    """
    cols = distinct.input.available_columns()
    group_keys = tuple(Projection(c, Col(c)) for c in cols)
    agg = Aggregate(distinct.input, group_keys, ())
    if transport == "flight":
        from batcher.dist.flight_aggregate import execute_aggregate_flight

        # DISTINCT is a group-by with no aggregates, so it rides the aggregate's
        # `materialize=False` path: with an ambient fleet the deduped result stays on the
        # workers (a `FlightMaterializedSource`) instead of collecting on the head.
        return execute_aggregate_flight(above, agg, sources, workers, materialize=materialize)
    return _distributed_aggregate(above, agg, sources, workers, materialize=materialize)
