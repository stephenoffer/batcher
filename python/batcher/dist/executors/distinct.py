"""Distributed deduplication — whole-row via the aggregate shuffle, keyed via a row shuffle.

A whole-row DISTINCT is an aggregation that groups by *all* columns with no aggregate
functions, so it reuses the mergeable aggregate shuffle verbatim: every row is hash-shuffled
by its full set of column values, identical rows land on the same reducer and are deduplicated
there, and the union of reducers is the global distinct.

A *keyed* dedup cannot ride that shuffle, because it is not a group-by: the surviving row
carries columns the key does not determine, and no combination of per-column aggregates can
pick a whole row. It rides the row shuffle instead — the same one a partitioned window uses —
with one addition that is the whole point of the operator being mergeable: **each mapper
reduces its own partition before it ships anything**. What crosses the network is one row per
key per partition rather than every row, so the shuffle shrinks by the dedup's own selectivity.
The reducer then reduces again, which is valid because `partial` and `combine` are the same
function here (`bc_runtime::agg::distinct_on`).

Either way the mergeable primitives are reused unchanged, so the result equals single-node
execution.
"""

from __future__ import annotations

from batcher.dist.executors.aggregate import _distributed_aggregate
from batcher.io.source import Source
from batcher.plan.logical import Distinct, LogicalPlan


def _distributed_distinct(
    above: list[LogicalPlan],
    distinct: Distinct,
    sources: list[Source],
    workers: int,
    transport: str = "disk",
    *,
    materialize: bool = True,
    hub=None,
    metrics_out=None,
):
    """Run `distinct` across `workers`, by whichever shuffle its form calls for.

    A keyed dedup takes the pre-reducing row shuffle (Flight where available, else the disk
    Arrow-IPC path). A whole-row DISTINCT builds the equivalent `Aggregate` (group by every
    column, no aggregates) and reuses the aggregate shuffle — the Flight (Carbonite) path on a
    multi-node cluster, the disk path otherwise. `above` (operators after the dedup) then runs
    single-node over the deduplicated result. `materialize=False` keeps the disk-path result
    partitioned (the aggregate shuffle's behavior) for the next adaptive stage.
    """
    if distinct.keys:
        return _distributed_distinct_on(
            above, distinct, sources, workers, transport, hub, metrics_out, materialize
        )
    if distinct.limit is not None:
        return _distributed_distinct_limit(above, distinct, sources, workers, hub)
    agg = distinct.as_aggregate()
    if transport == "flight":
        from batcher.dist.flight_aggregate import execute_aggregate_flight

        # DISTINCT is a group-by with no aggregates, so it rides the aggregate's
        # `materialize=False` path: with an ambient fleet the deduped result stays on the
        # workers (a `FlightMaterializedSource`) instead of collecting on the head.
        return execute_aggregate_flight(
            above,
            agg,
            sources,
            workers,
            materialize=materialize,
            hub=hub,
            metrics_out=metrics_out,
        )
    return _distributed_aggregate(
        above, agg, sources, workers, hub, materialize=materialize, metrics_out=metrics_out
    )


def _distributed_distinct_limit(
    above: list[LogicalPlan],
    distinct: Distinct,
    sources: list[Source],
    workers: int,
    hub=None,
):
    """Capped whole-row DISTINCT: each worker's own prefix, then the operator re-applied.

    A `Distinct` carrying a fused `LIMIT k` cannot ride the aggregate shuffle the uncapped
    form uses. The shuffle answers "which rows are distinct", which is order-free and is why
    hash-partitioning is sound for it; the capped form answers "which are the *first* `k`
    distinct rows", and a hash shuffle destroys exactly the order that question is about.

    So it takes the shape a distributed `LIMIT` takes instead. `preserve_order=True` splits the
    source into contiguous source-ordered runs and assembles the results by partition index, so
    the concatenation below is the source's own row order — which is what makes this equal to
    single-node rather than merely the same size. Each worker keeps its own first `k` distinct
    rows, so at most `partitions x k` rows reach the driver however large the source is, and
    the driver re-applies the operator to select the global first `k`.

    Soundness is the mergeable argument on `bc_runtime::agg::DistinctPrefix`: a row among the
    global first `k` distinct rows has at most that many distinct rows before it *within its
    own partition*, so it survives that partition's prefix, and the ordered union therefore
    contains the answer. Pinned in Rust by `combine_finalize_of_partitions_equals_single_node`.
    """
    from batcher.dist.executors.map import _distributed_map
    from batcher.dist.executors.partition_io import _apply_above
    from batcher.dist.executors.ray_runtime import _single_node
    from batcher.io.source import InMemorySource
    from batcher.plan.logical import Scan
    from batcher.plan.schema import SchemaRef

    per_partition = _distributed_map(distinct, sources, workers, hub, preserve_order=True)
    if per_partition.num_rows == 0:
        return per_partition if not above else _apply_above(above, per_partition)
    # Re-apply on the driver. The input here is bounded by `partitions x limit`, so this is a
    # small local dedup rather than a second pass over the source.
    src = InMemorySource(per_partition.to_batches())
    plan = Distinct(
        Scan(0, SchemaRef.from_arrow(per_partition.schema)),
        limit=distinct.limit,
    )
    table = _single_node(plan, [src])
    return table if not above else _apply_above(above, table)


def _distributed_distinct_on(
    above: list[LogicalPlan],
    distinct: Distinct,
    sources: list[Source],
    workers: int,
    transport: str,
    hub=None,
    metrics_out=None,
    materialize: bool = True,
):
    """Keyed dedup across `workers`: reduce locally, shuffle by key, reduce again.

    The map plan is the dedup *itself* over the partition, not the pipeline beneath it. That
    is what makes the shuffle carry one row per key per partition, and it is only sound
    because the reduction is idempotent and associative — applying it to the union of its own
    outputs gives the same rows as applying it to the whole relation.

    `materialize=False` keeps the deduped rows partitioned rather than collecting them here,
    so a following stage scans them in place. That matters more for this operator than for
    the aggregate it sits beside: a dedup's output is rows, not a summary, so what a collect
    costs is a fraction of the whole relation through one node. It is honored on **both**
    transports — the Flight branch used to accept the flag and then drop it on the floor, so
    a keyed `distinct()` on a real multi-node cluster (where `transport="auto"` resolves to
    Flight) collected every deduped row onto the driver while the same query on one node
    did not.
    """
    from batcher.dist.executors.keyed_shuffle import keyed_row_shuffle, scan_rooted_ir
    from batcher.dist.executors.plan_analysis import _relabel_single_source, empty_result_table

    # The map plan carries a breaker, which the Flight mapper's chunked
    # `streaming_map_buckets` normally refuses. It is sound here for the one reason that
    # function names as its exception: this breaker is mergeable and the reduce side below
    # re-applies it, so a per-chunk reduction is a partial rather than a truncation.

    cols = distinct.available_columns()
    # The operator's REAL column types, for the case where every bucket came back empty —
    # a name list alone would make the empty distributed result null-typed where the
    # single-node one is not. See `keyed_shuffle.keyed_row_shuffle`.
    out_schema = empty_result_table(distinct, cols).schema
    key_indices = [cols.index(k) for k in distinct.keys]
    inner, source_id = _relabel_single_source(distinct.input)
    # The mapper runs the dedup over its partition; `_relabel_single_source` rewrote the scan
    # beneath it, so rebuilding the node around the relabeled input keeps the pre-reduction and
    # the reduce side expressing the same operator.
    map_plan = Distinct(inner, distinct.keys, distinct.order)
    reduce_ir = scan_rooted_ir(distinct)
    if transport == "flight":
        from batcher.dist.flight_window import execute_keyed_shuffle_flight

        return execute_keyed_shuffle_flight(
            above,
            map_plan=map_plan,
            reduce_ir=reduce_ir,
            key_names=list(distinct.keys),
            out_schema=out_schema,
            sources=sources,
            source_id=source_id,
            workers=workers,
            hub=hub,
            metrics_out=metrics_out,
            materialize=materialize,
        )
    return keyed_row_shuffle(
        above,
        map_plan=map_plan,
        reduce_ir=reduce_ir,
        key_indices=key_indices,
        out_schema=out_schema,
        source=sources[source_id],
        workers=workers,
        hub=hub,
        tag="dedup",
        metrics_out=metrics_out,
        materialize=materialize,
    )
