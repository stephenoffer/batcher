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
from collections.abc import Callable

import pyarrow as pa

from batcher._internal.logging import note_suppressed
from batcher._internal.native import engine
from batcher.dist.executors.partition_io import partition_descriptors, source_pushdown
from batcher.dist.executors.plan_analysis import empty_result_table
from batcher.dist.executors.ray_runtime import engine_config_json
from batcher.dist.executors.ray_runtime.metering import drain_worker_metrics
from batcher.dist.fleet import acquire_fleet, borrows_session_fleet, release_fleet
from batcher.dist.fleet.plan_id import next_result_stage
from batcher.dist.flight_aggregate import _shuffle_credits
from batcher.dist.flight_worker import current_plan_id
from batcher.io.source import Source
from batcher.plan.distribution import BROADCAST_SAFE_JOINS
from batcher.plan.ir_specs import agg_spec_json, task_scan_ir
from batcher.plan.logical import Aggregate, Join, LogicalPlan, Scan
from batcher.plan.types import retained_bytes, total_retained_bytes

__all__ = ["broadcast_eligible", "execute_broadcast_join_flight", "stream_probe_join"]

# Probe-chunk byte target: a worker streams its probe split past the resident build side in
# chunks of about this size, so peak memory is one chunk + the build side + that chunk's
# output rather than the whole split. The same figure the disk broadcast path uses, for the
# same reason.
_PROBE_CHUNK_BYTES = 32 << 20

#: Input bytes the bounded build-side read buffers before running the plan over them.
#: Sets how finely `_bounded_build_side` can stop: the read overshoots the budget by at
#: most one chunk's worth of output. The same figure as the probe chunk above, for the
#: same reason — it is one native call's working set, not a property of either side.
#:
#: It is a *ceiling*, not the chunk size: the read never buffers more than the budget it is
#: checking against, since a chunk larger than the budget can only overshoot it. So a small
#: budget checks proportionally sooner rather than paying a fixed 32 MiB first.
_BUILD_CHUNK_BYTES = 32 << 20

#: Share of a node's RAM one broadcast probe task may accumulate as joined output, used
#: only when the worker was shipped no memory grant of its own.
#:
#: A *fallback*, and deliberately not the primary answer, because a fraction of node RAM is
#: only a bound if you know how many probe tasks share the node — and that is exactly what
#: the fan-out changes. A quarter was right at two workers per node
#: (`dist.executor._numa_sliced`); at the four that function now chooses it is the whole
#: machine, so the bound would have stopped bounding anything as a side effect of a
#: scheduling change made elsewhere. `_output_budget` prefers the worker's *granted* budget,
#: which `dist.executor._size_worker_memory` already divides by the workers packed on a node
#: and `_FlightWorker._reduce_budget` divides again by the calls it runs at once.
_OUTPUT_BUDGET_FRACTION = 0.25


class BroadcastOutputTooLarge(Exception):
    """A probe task's joined output outgrew its node, so the broadcast must not be finished.

    Raised on the worker and caught on the driver, which falls back to the co-partition
    shuffle — the same "measured re-check beats an estimate" contract `_materialize_build_side`
    applies to the *build* side, extended to the side nothing was checking.
    """


def _output_budget(granted: int = 0) -> int:
    """Bytes of joined output one probe task may hold. `0` disables the bound.

    `granted` is what this worker's own memory envelope allows one operation to hold, which
    is the figure that stays correct when the fan-out changes; the node-RAM fraction below
    is the fallback for a worker that was shipped no envelope. See `_OUTPUT_BUDGET_FRACTION`
    for why the fallback cannot be the primary answer.

    Args:
        granted: The worker's per-operation memory grant, or `0` when it has none.

    Returns:
        The bound in bytes, or `0` to disable it.
    """
    if granted > 0:
        return granted
    try:
        import psutil

        return int(psutil.virtual_memory().total * _OUTPUT_BUDGET_FRACTION)
    except Exception as exc:  # pragma: no cover - optional host probe
        note_suppressed("dist", "read node memory for a broadcast output bound", exc)
        return 0


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
    on_metrics: Callable[[str], None] | None = None,
    output_budget: int = 0,
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
        on_metrics: Sink for each chunk's `ExecMetrics` document. The actor passes its own
            buffer's `append`, so the driver drains these after the barrier; in-process
            callers pass nothing and the join runs exactly as before.
        output_budget: Bytes of joined output this task may accumulate before it declines
            the strategy. `0` falls back to a share of node RAM — see `_output_budget`.

    Returns:
        The joined batches, or the per-chunk partial-aggregate states.
    """
    # `Core measures` is a contract, not a single-node convenience, and this path was the
    # hole in it: every chunk ran through the *unmetered* `execute_plan`, so a broadcast
    # join — the shape a fact-to-dimension query takes, and therefore the most common
    # distributed join there is — taught the cost and memory models nothing at all. The
    # co-partition shuffle beside it has been metered throughout, which is why the gap read
    # as "the distributed join learned nothing" only on the fleets that pick broadcast.
    #
    # Only the *join* is metered. The probe map-prefix is deliberately left alone: it is a
    # projection over one chunk, and feeding a per-chunk scan to a per-kind cost calibration
    # teaches it a shape no query runs — the same reason `_run` is kept off the sampling
    # passes.
    from batcher.dist.executors.ray_runtime.metering import execute_metered

    # The *input* is streamed a chunk at a time; the un-fused branch's **output** is not —
    # `out` holds every joined batch this split produces. So the docstring's "one chunk plus
    # that chunk's output" describes the fused branch only, and the plain join's peak is the
    # whole joined split. Measured on TPC-H q5 at sf100: two probe tasks on one node reached
    # 87.7 GB and 85.9 GB, and Ray OOM-killed the node at 179 of 184 GB.
    #
    # Bounded rather than restructured: publishing incrementally would change the worker
    # protocol, and the useful property here is the one `_materialize_build_side` already
    # states — an over-budget broadcast should cost a fallback to the shuffle, never a
    # cluster-wide OOM. Going over raises, the driver catches it, and the co-partition path
    # (which streams both sides) answers the query instead.
    budget = _output_budget(output_budget)
    held = 0
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
                held = _charge(held, budget, [partial])
            continue
        if on_metrics is None:
            joined = nat.execute_plan(join_ir, [probe_rows, build_side], engine_config)
        else:
            joined, metrics_json = execute_metered(join_ir, [probe_rows, build_side], engine_config)
            if metrics_json:
                on_metrics(metrics_json)
        kept = [b for b in joined if b.num_rows]
        out.extend(kept)
        held = _charge(held, budget, kept)
    return out


def _charge(held: int, budget: int, batches) -> int:
    """Add `batches` to the running held-bytes total, raising once it passes `budget`.

    Both branches accumulate — the un-fused one the joined rows, the fused one a partial per
    chunk — so both are charged. A fused partial is usually small (it is one aggregate's state
    for one chunk), but "usually" is what a bound is for: a high-cardinality group key makes it
    proportional to the chunk's distinct keys, not to the aggregate's output.
    """
    if not budget:
        return held
    held += sum(retained_bytes(b) for b in batches)
    if held > budget:
        raise BroadcastOutputTooLarge(
            f"broadcast probe output reached {held / (1 << 30):.1f} GiB on this node, over the "
            f"{budget / (1 << 30):.1f} GiB bound; falling back to the co-partition shuffle"
        )
    return held


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
    hub=None,
    metrics_out: list | None = None,
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
        hub: `FeedbackSink` the workers' measurements are recorded into, or None.
        metrics_out: When given, each worker's parsed metrics document, for the profile.

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
        try:
            done = ray.get(refs)
        except BroadcastOutputTooLarge:
            # A probe task's joined output outgrew its node. Decline the strategy; the caller
            # runs the co-partition shuffle, which streams both sides instead of holding one
            # split's whole join. The work already done is lost, which is the price of a
            # measured check over an estimated one — and it is bounded by this budget.
            return None
        # Drained here rather than in the caller, because both of this function's exits are
        # inside the `try` — the `publish` branch returns while deliberately *keeping* the
        # actors alive, so a drain placed after the call site would run for one shape and
        # not the other. Best-effort by construction (see `drain_worker_metrics`).
        drain_worker_metrics(actors, hub, metrics_out)

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

    The over-budget answer is reached **incrementally** — see `_bounded_build_side` for why
    reading the whole relation first was costing more than the strategy it guards.
    """
    from batcher._internal.hardware import l3_cache_bytes
    from batcher.config import active_config

    proj, pred = source_pushdown(build_plan, 0)
    # The same number the planner used to mark this join broadcast, asked again against
    # the side that was actually read. Kyber decides on an estimate; this is the measured
    # re-check, so a low estimate costs a fallback rather than a cluster-wide OOM.
    budget = active_config().optimizer.resolved_broadcast_max_bytes(l3_cache_bytes(), workers)
    rows = _bounded_build_side(nat, build_plan, source, proj, pred, cfg_json, budget)
    if rows is None or not any(b.num_rows for b in rows):
        return None
    if total_retained_bytes(rows) > budget:
        return None
    return rows


def _bounded_build_side(nat, build_plan, source: Source, proj, pred, cfg_json: str, budget: int):
    """The build side's batches, or None as soon as they are known to exceed `budget`.

    The measured re-check is what makes an *estimated* broadcast decision safe to act on,
    and it used to buy that safety by reading the whole build relation on the driver —
    one node, one stream — before it could say no. Measured on TPC-H sf100
    `lineitem ⋈ orders` over 8 workers, where a runtime-filtered build side estimates
    small and materializes at gigabytes: **10.2 s of driver time spent to learn the answer
    and then discard the data**, turning an 11.7 s query into 23.5 s. The guard has to
    stay; what it must not do is cost more than the strategy it is guarding.

    So stop at the budget instead of at the end of the relation. When the build plan is
    row-wise — a chain of scan/filter/project, which is what a build side under a pushed
    predicate or a runtime join filter is — its output over a concatenation of source
    chunks *is* the concatenation of its output over each chunk. So the plan can be run
    chunk by chunk over the source's own streaming split reader, accumulating output bytes
    and giving up the moment they pass the budget. A build side that genuinely fits is
    read exactly as before (one chunk, one `execute_plan`); one that does not costs about
    a budget's worth of reading rather than a relation's.

    Any other shape — an aggregate or a join underneath, a source that will not split —
    falls back to the whole-relation read, which is the behaviour before this existed.

    Bailing early is never a correctness question: the caller runs the co-partition
    shuffle, which produces the same relation.

    Args:
        nat: The native engine handle.
        build_plan: The per-side plan whose output would be replicated.
        source: The relation `build_plan` scans.
        proj: Columns pushed to the scan, or None for all of them.
        pred: Predicate pushed to the scan, or None.
        cfg_json: The driver's engine config JSON.
        budget: The replication budget in bytes.

    Returns:
        The build side's batches, or None once they are known to exceed `budget`.
    """
    from batcher.io.source import read_source

    build_ir = json.dumps(build_plan.to_ir())
    splits = _splittable_build_reads(build_plan, source)
    if splits is None:
        return nat.execute_plan(build_ir, [read_source(source, proj, pred)], cfg_json)

    from batcher.dist.executors.scan_read import _read_split_batches

    out: list[pa.RecordBatch] = []
    held = 0
    chunk: list[pa.RecordBatch] = []
    chunk_bytes = 0
    chunk_target = max(1, min(_BUILD_CHUNK_BYTES, budget))

    def drain() -> bool:
        """Run the plan over the buffered chunk. False once the budget is passed."""
        nonlocal held, chunk, chunk_bytes
        if chunk:
            out.extend(nat.execute_plan(build_ir, [chunk], cfg_json))
            chunk, chunk_bytes = [], 0
            held = total_retained_bytes(out)
        return held <= budget

    for batch in _read_split_batches(splits, proj, pred):
        chunk.append(batch)
        chunk_bytes += retained_bytes(batch)
        if chunk_bytes >= chunk_target and not drain():
            return None
    return out if drain() else None


def _splittable_build_reads(build_plan, source: Source) -> list | None:
    """The source's splits when the build side may be read in bounded chunks, else None.

    Two preconditions, both of them about whether chunking changes the answer rather than
    about whether it is worth it. The plan must be **row-wise**, so that running it per
    chunk and concatenating equals running it once over the whole input — `_is_row_wise`
    is the same predicate the distributed map path uses to decide a plan is partitionable.
    And the source must actually **subdivide**: a source that hands back one whole-source
    split cannot be read incrementally, so chunking it buys nothing and only adds a code
    path.
    """
    from batcher.dist.executors.scan_read import _SPLIT_TARGET_BYTES
    from batcher.plan.logical.transforms import is_partition_independent

    node = build_plan
    while not isinstance(node, Scan):
        if not is_partition_independent(node) or not hasattr(node, "input"):
            return None
        node = node.input
    try:
        splits = source.splits(_SPLIT_TARGET_BYTES)
    except Exception as exc:  # a source that cannot split is read whole, as before
        note_suppressed("dist", "split the broadcast build side for a bounded read", exc)
        return None
    return splits if len(splits) > 1 else None


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
