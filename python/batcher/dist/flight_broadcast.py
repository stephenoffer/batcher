"""Broadcast (replicated build side) equi-join on the Flight transport — no exchange.

The co-partition shuffle (`flight_join`) moves **both** sides across the network so equal
keys meet on one reducer. When the build side is small enough to replicate, that is the
wrong trade by an order of magnitude: a fact table joined to a dimension pays a full
shuffle of the fact to meet a dimension every worker could simply hold. Kyber already
decides this and marks the join `strategy == "broadcast"`; the disk transport already
honors it. This is the Flight half, and Flight is the transport every genuine multi-node
cluster resolves to (`resolve_transport`), so without it the broadcast strategy was
planned and then discarded exactly where it mattered most.

The shape is map-only: the build side is materialized once on the driver, put in the
object store once, and each worker joins **its own** probe split against the whole of it.
Nothing is shuffled, so nothing is O(probe bytes) on the network — which is also what
makes it scale with workers rather than flatten out.
"""

from __future__ import annotations

import json

import pyarrow as pa

from batcher._internal.native import engine
from batcher.dist.executors.partition_io import partition_descriptors, source_pushdown
from batcher.dist.executors.plan_analysis import empty_result_table
from batcher.dist.executors.ray_runtime import engine_config_json
from batcher.dist.fleet import acquire_fleet, borrows_session_fleet, release_fleet
from batcher.dist.fleet.plan_id import next_result_stage
from batcher.dist.flight_aggregate import _shuffle_credits
from batcher.dist.flight_worker import current_plan_id
from batcher.io.source import Source
from batcher.plan.distribution import BROADCAST_SAFE_JOINS
from batcher.plan.ir_specs import agg_spec_json, task_scan_ir
from batcher.plan.logical import Aggregate, Join, LogicalPlan
from batcher.plan.types import retained_bytes, total_retained_bytes

__all__ = ["broadcast_eligible", "execute_broadcast_join_flight", "stream_probe_join"]

# Probe-chunk byte target: a worker streams its probe split past the resident build side in
# chunks of about this size, so peak memory is one chunk + the build side + that chunk's
# output rather than the whole split. The same figure the disk broadcast path uses, for the
# same reason.
_PROBE_CHUNK_BYTES = 32 << 20


def broadcast_eligible(join: Join) -> bool:
    """Whether this join may take the replicated-build-side path.

    The planner's `strategy` decides *whether it is worth it*; this decides whether it is
    **correct**, and only `BROADCAST_SAFE_JOINS` is. A RIGHT or FULL join must emit build
    rows that matched nothing anywhere, which no single worker holding one probe split can
    determine, so those keep the co-partition shuffle whatever the planner marked.

    Args:
        join: The join being scheduled.

    Returns:
        True when replicating the build side yields the same relation.
    """
    return join.strategy == "broadcast" and join.join_type in BROADCAST_SAFE_JOINS


def stream_probe_join(
    nat,
    probe_ir: str,
    join_ir: str,
    probe_batches,
    build_side: list[pa.RecordBatch],
    engine_config: str,
    gk: str | None = None,
    aj: str | None = None,
) -> list[pa.RecordBatch]:
    """Join a streamed probe side against a resident `build_side`, a chunk at a time.

    Peak memory is the build side plus one chunk plus that chunk's output, so a worker's
    probe split never has to fit in memory at once.

    With `gk`/`aj` each chunk is folded to a **partial** aggregate inside the same native
    call that joins it, so the chunk's join output — by far the largest object here — is
    never mirrored into Python. The partials are returned for the driver to
    `combine_finalize`: a broadcast join co-partitions nothing, so a group legitimately
    spans workers and no worker may finalize.

    A plain function rather than an actor method, so the chunking is testable in process.

    Args:
        nat: The native engine handle.
        probe_ir: The probe side's map-prefix IR, run over each chunk.
        join_ir: The per-chunk join IR (source 0 = probe chunk, source 1 = build side).
        probe_batches: An iterable of the probe split's batches.
        build_side: The whole (replicated) build side.
        engine_config: The driver's engine config JSON.
        gk: Group-key spec of a fused aggregate, or None.
        aj: Aggregate spec of a fused aggregate, or None.

    Returns:
        The joined batches, or the per-chunk partial-aggregate states.
    """
    out: list[pa.RecordBatch] = []
    for chunk in _byte_chunks(probe_batches, _PROBE_CHUNK_BYTES):
        probe_rows = nat.execute_plan(probe_ir, [chunk], engine_config)
        if not any(b.num_rows for b in probe_rows):
            continue
        if gk is not None:
            partial = nat.execute_plan_aggregated(
                join_ir, [probe_rows, build_side], gk, aj, engine_config, False
            )
            if partial is not None:
                out.append(partial)
            continue
        joined = nat.execute_plan(join_ir, [probe_rows, build_side], engine_config)
        out.extend(b for b in joined if b.num_rows)
    return out


def _byte_chunks(batches, target_bytes: int):
    """Group batches into lists of about `target_bytes` of *retained* bytes each.

    Retained, not addressed: a batch sliced from a larger parent pins the parent, so
    `nbytes` would under-report exactly the case the bound exists for.
    """
    chunk: list[pa.RecordBatch] = []
    size = 0
    for b in batches:
        chunk.append(b)
        size += retained_bytes(b)
        if size >= target_bytes:
            yield chunk
            chunk, size = [], 0
    if chunk:
        yield chunk


def execute_broadcast_join_flight(
    above: list[LogicalPlan],
    join: Join,
    sources: list[Source],
    workers: int,
    *,
    fused_agg: Aggregate | None = None,
    materialize: bool = True,
):
    """Replicate the build side to every worker and join each probe split in place.

    Returns `None` when the build side turns out to be empty or larger than the broadcast
    budget, so the caller falls back to the co-partition shuffle. That guard is on the
    **measured** side, not the planner's estimate: an under-estimate that reached the
    replication would put a too-large relation on every worker at once, which is the one
    way this strategy can be worse than the shuffle rather than merely no better.

    `materialize=False` leaves each worker's joined split published on its own Flight
    server and returns a `FlightMaterializedSource`, so an intermediate join in a
    multi-join query never round-trips through the driver.

    Args:
        above: Operators stacked above the join, re-applied to the collected result.
        join: The join to run.
        sources: The bound sources for the whole plan.
        workers: Worker fan-out.
        fused_agg: An aggregate folded into each probe task as a partial.
        materialize: False to leave the result partitioned on the workers.

    Returns:
        The result table, a `FlightMaterializedSource`, or None to fall back.
    """
    import ray

    from batcher.dist.executor import _apply_above, _ensure_ray, _relabel_single_source
    from batcher.dist.flight_join import _project_join_side

    nat = engine()
    _ensure_ray(workers)
    cfg_json = engine_config_json()

    # Carry only the columns the join output needs (plus the keys), as the shuffle path
    # does — here it shrinks what is *replicated*, so a wide dimension does not cost every
    # worker its unused columns.
    probe_need = {o.name for o in join.output if o.side == "left"} | set(join.left_keys)
    build_need = {o.name for o in join.output if o.side == "right"} | set(join.right_keys)
    probe_plan, probe_sid = _relabel_single_source(_project_join_side(join.left, probe_need))
    build_plan, build_sid = _relabel_single_source(_project_join_side(join.right, build_need))
    probe_ir = json.dumps(probe_plan.to_ir())
    join_ir = json.dumps(
        {
            **join.shape_ir(),
            "left": task_scan_ir(),
            "right": task_scan_ir(1),
        }
    )

    # Before the fleet is acquired: a build side that turns out too large falls back to the
    # shuffle, and it should do so without having reserved the cluster first.
    build_side = _materialize_build_side(nat, build_plan, sources[build_sid], cfg_json, workers)
    if build_side is None:
        return None

    gk = aj = None
    if fused_agg is not None:
        gk, aj = agg_spec_json(fused_agg)

    credits = _shuffle_credits()
    # See `fleet.borrows_session_fleet`: asked before the acquire, because that is the only
    # point at which the three acquisition branches can still be told apart.
    borrows_session = borrows_session_fleet()
    actors, pg, fleet_addrs, workers, owns = acquire_fleet(workers, credits, cfg_json)
    publish = materialize is False and not above and fused_agg is None
    keep_actors = False
    try:
        probe_proj, probe_pred = source_pushdown(probe_plan, 0)
        parts = partition_descriptors(
            sources[probe_sid],
            workers,
            projection=probe_proj,
            predicate=probe_pred,
            worker_addrs=fleet_addrs,
        )
        # One object-store copy of the build side for the whole fan-out, so a worker's node
        # pulls it once instead of once per task. Bounded by the broadcast budget — this is
        # small *by the definition of the strategy*, which is what keeps it inside the
        # "Ray moves metadata and bounded input, never shuffle" contract.
        build_ref = ray.put(build_side)
        result_stage = next_result_stage() if publish else 0
        plan_id = current_plan_id()
        refs = [
            actors[i % len(actors)].broadcast_probe_join.remote(
                probe_ir,
                join_ir,
                parts[i],
                build_ref,
                gk,
                aj,
                plan_id,
                publish,
                result_stage,
                i,
            )
            for i in range(len(parts))
        ]
        done = ray.get(refs)

        if publish:
            from batcher.dist.fleet import FlightMaterializedSource

            handles = [(a, t, n) for a, t, n, _s in (h for h in done if h)]
            schemas = [h[3] for h in done if h]
            schema = (
                schemas[0]
                if schemas
                else _empty_output_schema(
                    nat, join, probe_ir, sources[probe_sid], build_side, cfg_json
                )
            )
            keep_actors = True
            src_actors, src_pg = (actors, pg) if owns else (None, None)
            return FlightMaterializedSource(
                handles, schema, src_actors, src_pg, session_lease=borrows_session and not owns
            )
        batches = [b for part in done for b in part if b.num_rows]
    finally:
        if not keep_actors:
            release_fleet(actors, pg, owns)

    if fused_agg is not None:
        # Every worker's every chunk emitted a PARTIAL: a broadcast join co-partitions
        # nothing, so one group's rows are spread over every worker and only the driver can
        # close it. The partials are `workers x chunks x groups` rows of aggregate state,
        # not the join — the whole point is that the join itself never leaves the workers.
        final = nat.combine_finalize(gk, aj, batches) if batches else None
        keys = [k.alias for k in fused_agg.group_keys]
        table = (
            pa.Table.from_batches([final])
            if final is not None
            else empty_result_table(fused_agg, keys + [s.alias for s in fused_agg.aggregates])
        )
        return table if not above else _apply_above(above, table)

    table = (
        pa.Table.from_batches(batches)
        if batches
        else empty_result_table(join, [o.alias for o in join.output])
    )
    return table if not above else _apply_above(above, table)


def _materialize_build_side(nat, build_plan, source: Source, cfg_json: str, workers: int):
    """Read and run the build side on the driver, or None when it must not be replicated.

    None means "fall back to the shuffle", for either of two reasons. An **empty** build
    side is handed back so the caller's shuffle produces the outer-join null-extension
    without a hand-built empty schema here. An **over-budget** one is the guard that keeps
    this strategy safe to attempt on an estimate: the planner's byte figure can be low, and
    the cost of finding out after replicating it to every worker is a cluster-wide OOM
    rather than a slow query.
    """
    from batcher._internal.hardware import l3_cache_bytes
    from batcher.config import active_config
    from batcher.io.source import read_source

    proj, pred = source_pushdown(build_plan, 0)
    rows = nat.execute_plan(
        json.dumps(build_plan.to_ir()), [read_source(source, proj, pred)], cfg_json
    )
    if not rows or not any(b.num_rows for b in rows):
        return None
    # The same number the planner used to mark this join broadcast, asked again against
    # the side that was actually read. Kyber decides on an estimate; this is the measured
    # re-check, so a low estimate costs a fallback rather than a cluster-wide OOM.
    budget = active_config().optimizer.resolved_broadcast_max_bytes(l3_cache_bytes(), workers)
    if total_retained_bytes(rows) > budget:
        return None
    return rows


def _empty_output_schema(nat, join, probe_ir, probe_source, build_side, cfg_json) -> pa.Schema:
    """The join's output schema when every worker's probe split came back empty.

    An empty relation must still advertise the columns the next stage's plan is typed
    against. The build side is already materialized here, so only the probe side needs the
    0-row sub-plan probe `flight_join` uses; `_join_output_schema` then reads both against
    `join.output`, which is the one place that mapping is written down.
    """
    from batcher.dist.flight_join import _join_output_schema

    empty = pa.RecordBatch.from_pylist([], schema=probe_source.schema())
    out = nat.execute_plan(probe_ir, [[empty]], cfg_json)
    return _join_output_schema(join, out[0] if out else empty, build_side[0])
