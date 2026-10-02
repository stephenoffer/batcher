"""Run an `AlignedCut`: one engine call per key-range unit, then combine on the driver.

Each unit is a Ray task that reads its aligned sources' files restricted to the unit's key
range and runs the cut's body through the engine in one call -- joins, filters and any
key-grouped aggregate included -- so the only thing that crosses the network is what the
body returns. When the cut ends in an aggregate that is not grouped by the key, each unit
returns that aggregate's *partial* state and the driver combines them: the same
`partial -> combine -> finalize` every other executor uses, which is what keeps the result
identical to single-node.

The parts of the body that read no aligned source (a filtered dimension, a dimension joined
to another) are the same on every unit, so they are evaluated **once**, before any unit
runs, and their results handed to every task as one object each node fetches once. What a
worker holds is then the filtered result -- 30M customer keys for TPC-H q3 at SF1000, not
the 150M-row table -- and that result, not the source, is what the broadcast budget bounds.
"""

from __future__ import annotations

import dataclasses
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa

from batcher._internal.logging import get_logger
from batcher._internal.native import engine
from batcher.config.env import env_int
from batcher.dist.executors.aligned.analysis import AlignedCut, AlignedPlan, cut_signature
from batcher.dist.executors.aligned.hoist import (
    _NESTED_BROADCAST_BYTES,
    BROADCAST_BYTES,
    DRIVER_READ_BYTES,
    hoist_broadcasts,
)
from batcher.dist.executors.aligned.local import local_broadcast, resolve_local
from batcher.dist.executors.aligned.reduce import (
    prefer_hash_joins,
    reduce_broadcasts,
    sliceable_broadcasts,
)
from batcher.dist.executors.aligned.rewrite import group_broadcast_joins
from batcher.dist.executors.aligned.transfer import (
    RESULT_BYTES,
    gather_units,
    pack_held,
    pull_units,
    read_unit,
    trace_units,
    unpack_held,
)
from batcher.dist.executors.aligned.units import (
    Unit,
    file_bounds,
    plan_units,
    projected_bytes,
    projected_share,
    source_key_bounds,
    splits_by_file,
    splits_by_piece,
)
from batcher.io.source import Source
from batcher.plan.expr_ir import col, lit
from batcher.plan.logical import Filter, LogicalPlan, Scan

__all__ = [
    "BROADCAST_BYTES",
    "DRIVER_READ_BYTES",
    "run_cut",
    "run_plan",
    "run_units_here",
    "unit_plan",
]

# Projected, decoded bytes of the driving source one unit holds. The engine call keeps its
# inputs resident, so this bounds a worker's peak with room for the join and aggregate
# state built over it. Env-overridable for nodes with more or less memory than 64 GB.
_UNIT_BYTES = env_int("BATCHER_ALIGNED_UNIT_BYTES", 6 << 30, floor=64 << 20)
#: Largest unit, in projected bytes, for which an actor runs three calls at once.
_SMALL_UNITS_BYTES = 1 << 30


def _range_predicate(column: str, lo: object, hi: object):
    """`lo <= column < hi`, open at a None end; None when the unit is the whole domain.

    Aligned key columns hold no NULLs (`units.source_key_bounds` declines one that may), so
    an open-ended range loses no row.
    """
    parts = []
    if lo is not None:
        parts.append(col(column) >= lit(lo))
    if hi is not None:
        parts.append(col(column) < lit(hi))
    pred = parts[0] if parts else None
    for p in parts[1:]:
        pred = pred & p
    return pred


def unit_plan(
    body: LogicalPlan, cut: AlignedCut, unit: Unit, sliced: dict[int, str] | None = None
) -> LogicalPlan:
    """`body` with every aligned scan, and every `sliced` broadcast, cut to `unit`'s range."""
    from batcher.plan.visitor import transform_up

    if cut.key.keyless:
        return body  # split by file: each file is in exactly one unit, so nothing to filter
    sliced = sliced or {}

    def restrict(node: LogicalPlan) -> LogicalPlan:
        if isinstance(node, Scan) and (node.source_id in cut.aligned or node.source_id in sliced):
            column = sliced.get(node.source_id) or cut.key.column_of(node.source_id)
            pred = _range_predicate(column, unit.lo, unit.hi)
            return node if pred is None else Filter(input=node, predicate=pred)
        return node

    return transform_up(body, restrict)


# Cores one unit task reserves. Half a 16-core node, so two units share each node: a unit
# alternates between reading its inputs (network-bound, the cores idle) and one engine call
# over them (CPU-bound, the link idle), and a second unit on the same node fills whichever
# half the first leaves. The engine still spreads each call over every core of the node. An
# 8-core node (the head, typically) takes one. Measured before this, one 15-core task per
# node held every node at 7-60% CPU across a whole SF1000 query.
_UNIT_CPUS = env_int("BATCHER_ALIGNED_UNIT_CPUS", 8, floor=1)


def _unit_slots(workers: int) -> tuple[int, int]:
    """Cores per unit task, and how many such tasks the cluster runs at once."""
    try:
        import ray

        per_node = [
            int(n.get("Resources", {}).get("CPU", 0)) for n in ray.nodes() if n.get("Alive")
        ]
        # Never more than one node has: a task asking for more cores than any node holds is
        # never placed, and the gather waits on it forever (a 4-worker test cluster of
        # 2-CPU nodes hung here on 8-CPU units).
        unit_cpus = min(_UNIT_CPUS, max(per_node, default=0))
        if unit_cpus >= 1:
            return unit_cpus, sum(cpus // unit_cpus for cpus in per_node)
    except Exception as exc:
        from batcher._internal.logging import note_suppressed

        note_suppressed("dist", "read cluster cores for aligned units", exc)
    return _UNIT_CPUS, max(1, workers)


def _aligned_units_task(calls: list[tuple], held: dict, empties: dict) -> list[tuple]:
    """`run_units_here` as a Ray task (the lifecycle wraps this name, not that one)."""
    return run_units_here(calls, held, empties)


def run_units_here(calls: list[tuple], held: dict, empties: dict) -> list[tuple]:
    """Run a sequence of units, reading the next unit while the engine runs this one.

    A unit alone alternates between reading (the link busy, the cores idle) and one engine
    call (the reverse). The engine releases the GIL, so a reader thread fetching unit k+1
    while unit k computes keeps both busy. Each result carries the seconds spent waiting on
    a read the prefetch did not hide, the seconds computing, and the bytes read.
    """
    from batcher.dist.executors.ray_runtime import execute_metered

    started = time.time()  # wall clock, comparable with the driver's submit time
    results = []
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        # The first unit's read starts before the broadcasts are ready, so it overlaps
        # unpacking them and evaluating the ones this node reads for itself.
        pending = prefetch.submit(read_unit, calls[0][1], empties)
        held = resolve_local(unpack_held(held))
        setup = time.time() - started  # unpacking the broadcasts, and evaluating node-read ones
        for i, (body_ir, _descs, agg_spec, cfg_json) in enumerate(calls):
            t0 = time.perf_counter()
            inputs, in_bytes = pending.result()
            waited = time.perf_counter() - t0
            if i + 1 < len(calls):
                pending = prefetch.submit(read_unit, calls[i + 1][1], empties)
            for sid, batches in held.items():
                inputs[sid] = batches
            t1 = time.perf_counter()
            rows, metrics_json = execute_metered(body_ir, inputs, cfg_json)
            if agg_spec is not None:
                rows = [engine().partial_aggregate(agg_spec[0], agg_spec[1], rows)]
            del inputs
            timing = (waited, time.perf_counter() - t1, in_bytes, started, setup)
            results.append((rows, metrics_json, timing))
    return results


def _run_units(
    calls: list[tuple], held: dict, empties: dict, unit_cpus: int, slots: int, depth: int = 2
) -> list[tuple] | None:
    """Run every unit on the cluster as pipelined Ray tasks; results come back in unit order.

    None when what the units return outgrows `transfer.RESULT_BYTES`, which declines the cut.

    On a warm fleet each actor pulls the next units as it finishes some (`pull_units`), so
    a slow node takes fewer; as tasks they are dealt round-robin, so each task's units are
    spread across the key domain, which evens out a key range denser than the rest.
    """
    import ray

    from batcher.dist.fleet import (
        borrow_warm_session_fleet,
        release_session_lease,
        yield_session_fleet,
    )

    held_ref = ray.put(pack_held(held, uuid.uuid4().hex))
    # A warm shuffle fleet from an earlier query holds nearly every core of every node, as one
    # engine actor per node. Its actors run the units themselves, two calls each: as plain
    # tasks the units waited ~0.5 s per query for worker leases, and first had to tear the
    # fleet down to find the cores, which the next shuffling query then paid to respawn.
    actors = borrow_warm_session_fleet()
    try:
        if actors:
            get_logger("dist").info("aligned: units on %d warm fleet actors", len(actors))
            try:
                return pull_units(
                    calls,
                    len(actors),
                    depth,
                    lambda w, units: actors[w].aligned_units.remote(units, held_ref, empties),
                    [getattr(a, "_actor_id", i) for i, a in enumerate(actors)],
                )
            except ray.exceptions.RayError as exc:  # a lost actor: the tasks below still run
                get_logger("dist").info("aligned: fleet actors failed (%s); running as tasks", exc)
        # No warm fleet: plain tasks, and a fleet that is not warm enough to use is yielded
        # when it is what stands between the units and the cores.
        yield_session_fleet(float(unit_cpus))
        # One task per slot, the units dealt round-robin: every slot starts at once and runs
        # the same number of units, each task prefetching its next. Grouped three to a task,
        # 77 units made 26 tasks for 17 slots, and the second wave ran nine slots through six
        # units each while eight sat idle: TPC-H q5 at SF1000 spent ~30 s in units for ~16 s.
        return gather_units(
            calls,
            min(len(calls), slots),
            lambda _t, units: _aligned_units_task.options(num_cpus=unit_cpus).remote(
                units, held_ref, empties
            ),
        )
    finally:
        if actors:
            release_session_lease()


def run_cut(
    cut: AlignedCut,
    sources: list[Source],
    workers: int,
    hub=None,
    metrics_out=None,
    oversized: set[int] | None = None,
) -> pa.Table | None:
    """Execute one `cut` over `sources`, or None when the files or broadcasts do not allow it.

    Declines before any unit runs when a source's footers do not bound its key, when its
    files do not follow the key closely enough to cut units from, or when a broadcast result
    is too large to hold on every worker; and after, when the units' results would not fit
    on the driver.
    """
    from batcher.dist.executors.plan_analysis import _empty_agg_table, empty_result_table
    from batcher.dist.executors.ray_runtime import engine_config_json, record_worker_metrics
    from batcher.plan.ir_specs import agg_spec_json
    from batcher.plan.visitor import walk

    bounds, shares = {}, {}
    for sid in cut.aligned:
        found = (
            file_bounds(sources[sid])
            if cut.key.keyless
            else source_key_bounds(sources[sid], cut.key.column_of(sid))
        )
        if found is None:
            get_logger("dist").info("aligned: source %d's footers bound no key; declined", sid)
            return None
        bounds[sid] = found
        shares[sid] = projected_share(sources[sid], cut.pushdown(cut.body, sid)[0])
    unit_cpus, slots = _unit_slots(workers)
    # Several units per slot: a unit's cost follows its files, not a fixed share, so a
    # cluster cut into exactly one wave per slot finishes at the pace of its slowest; with
    # ~6 per slot the tail is a sixth of a unit, and each unit holds less memory.
    # A warm fleet runs up to three calls an actor (`_run_units`), so a small input is cut for
    # that many streams.
    units = plan_units(bounds, shares, _UNIT_BYTES, min_units=6 * slots, streams=slots * 3 // 2)
    if units is None:
        get_logger("dist").info("aligned: files do not follow the key; declined")
        return None

    started = time.perf_counter()
    held_tables: dict[int, pa.Table] = {}
    local: dict[int, LogicalPlan] = {}
    grouped = group_broadcast_joins(cut.body, cut.aligned)
    body = hoist_broadcasts(grouped, cut, sources, held_tables, workers, local, oversized)
    if body is None:
        return None
    reduce_broadcasts(body, held_tables, local)
    body = prefer_hash_joins(body)
    sliced = sliceable_broadcasts(body, cut, held_tables)
    cfg_json = engine_config_json(num_cpus=unit_cpus)
    held: dict[int, object] = {
        sid: t.to_batches() or [pa.RecordBatch.from_pylist([], schema=t.schema)]
        for sid, t in held_tables.items()
    }
    run_key = uuid.uuid4().hex
    schemas = {n.source_id: n.schema.arrow for n in walk(body) if isinstance(n, Scan)}
    for sid, subtree in local.items():
        held[sid] = local_broadcast(
            prefer_hash_joins(subtree), sources, schemas[sid], f"{run_key}:{sid}", cfg_json
        )
    width = len(sources) + len(held)
    # A keyless unit names row groups (`units.file_bounds`); a keyed one, the files it overlaps.
    split = splits_by_piece if cut.key.keyless else splits_by_file
    by_file = {sid: split(sources[sid], *cut.pushdown(body, sid)) for sid in cut.aligned}
    if any(files is None for files in by_file.values()):
        get_logger("dist").info("aligned: a split names no file; declined")
        return None

    agg_spec = agg_spec_json(cut.aggregate) if cut.aggregate is not None else None
    calls = []
    for unit in units:
        plan = unit_plan(body, cut, unit, sliced)
        descs: list[dict | None] = [None] * width
        for sid in cut.aligned:
            proj, pred = cut.pushdown(plan, sid)
            # A multi-file split is listed under each of its files: read it once per unit.
            splits = list(
                {id(s): s for path in unit.files[sid] for s in by_file[sid].get(path, [])}.values()
            )
            descs[sid] = {"splits": splits, "projection": proj, "predicate": pred}
        calls.append((json.dumps(plan.to_ir()), descs, agg_spec, cfg_json))
    hoisted = time.perf_counter()
    from batcher.dist.executors.partition_io import empty_descriptor, read_partition_descriptor

    empties = {
        sid: read_partition_descriptor(empty_descriptor(sources[sid], cut.pushdown(body, sid)[0]))
        for sid in cut.aligned
    }
    # Three calls to an actor rather than two when units are small: a unit's read is bound
    # by its requests' latency, not the link -- TPC-H q19 at SF100 ran its reads at ~150 MB/s
    # a node with the CPUs 17% busy -- and a third keeps more of them in flight.
    depth = 3 if max((u.nbytes for u in units), default=0) <= _SMALL_UNITS_BYTES else 2
    results = _run_units(calls, held, empties, unit_cpus, slots, depth)
    if results is None:
        get_logger("dist").info(
            "aligned: unit results outgrew the driver budget (%d GiB); declined a %s cut over %s",
            RESULT_BYTES >> 30,
            "partial-aggregate" if cut.aggregate is not None else "row-returning",
            type(cut.body).__name__,
        )
        return None
    gathered = time.perf_counter()
    record_worker_metrics(hub, (m for _rows, m, _t in results), metrics_out)
    batches = [b for rows, _m, _t in results for b in rows if b.num_rows]
    trace_units(
        units, [t for _r, _m, t in results], hoisted - started, gathered - hoisted, unit_cpus
    )

    if cut.aggregate is not None and _past_string_offsets(batches):
        # The driver merges every unit's partial groups as one relation, and a string column
        # past 2 GiB cannot be one 32-bit-offset array. TPC-H q16 at SF1000 returns its
        # `COUNT(DISTINCT ps_suppkey)` partials -- distinct (group, supplier) pairs, each with
        # its `p_type` -- at 2.4 GB of `Utf8` and failed the query here. Partials that large
        # are the shuffle route's to merge, on the workers, so decline to it.
        get_logger("dist").info(
            "aligned: partial groups carry a string column past 2 GiB; declined the cut"
        )
        return None
    if cut.aggregate is not None:
        if batches:
            nat = engine()
            running = nat.combine(agg_spec[0], agg_spec[1], batches)
            final = nat.combine_finalize(agg_spec[0], agg_spec[1], [running])
            table = pa.Table.from_batches([final])
        else:
            table = _empty_agg_table(cut.aggregate)
    else:
        table = (
            pa.Table.from_batches(batches)
            if batches
            else empty_result_table(cut.body, cut.body.available_columns())
        )
    get_logger("dist").info(
        "aligned: combine %.2fs (%d rows)", time.perf_counter() - gathered, table.num_rows
    )
    return table


#: The most bytes a 32-bit-offset string or binary column can address once concatenated.
_STRING_OFFSET_LIMIT = (1 << 31) - 1


def _past_string_offsets(batches: list[pa.RecordBatch]) -> bool:
    """Whether some string or binary column, concatenated across `batches`, passes 2 GiB."""
    if not batches:
        return False
    return any(
        (pa.types.is_string(field.type) or pa.types.is_binary(field.type))
        and sum(b.column(i).nbytes for b in batches) > _STRING_OFFSET_LIMIT
        for i, field in enumerate(batches[0].schema)
    )


def run_plan(
    plan: AlignedPlan,
    sources: list[Source],
    workers: int,
    hub=None,
    metrics_out=None,
    oversized: set[int] | None = None,
) -> pa.Table | None:
    """Execute `plan`'s cuts in order, then its residual on the driver; None to decline.

    A cut may read an earlier cut's result -- a blocking aggregate computed first -- so each
    result joins the source list, at its placeholder id, as soon as it is computed. A decline
    because a broadcast overran adds the sources it read to `oversized`.
    """
    from batcher.dist.executors.ray_runtime import _single_node
    from batcher.io.source import InMemorySource
    from batcher.plan.visitor import scanned_source_ids, transform_up

    # Placeholders are numbered from `len(sources)` in execution order, so each result lands
    # at exactly the id the later cuts and the residual read it by.
    bound: list[Source] = list(sources)
    results: dict[int, pa.Table] = {}
    computed: dict[object, pa.Table] = {}
    for cut, sid in zip(plan.cuts, plan.placeholders, strict=True):
        # The same computation over twin scans of one table is one cut: TPC-H q15 reads its
        # `revenue` view twice, and aggregated it twice, a second full pass for one answer.
        signature = cut_signature(cut, bound)
        table = computed.get(signature) if signature is not None else None
        if table is None:
            table = run_cut(cut, bound, workers, hub, metrics_out, oversized)
            if table is None:
                return None
            if signature is not None:
                computed[signature] = table
        results[sid] = table
        empty = [pa.RecordBatch.from_pylist([], schema=table.schema)]
        bound.append(InMemorySource(table.to_batches() or empty))
    residual = plan.residual
    if isinstance(residual, Scan) and residual.source_id in results:
        return results[residual.source_id]
    started = time.perf_counter()
    order = sorted(scanned_source_ids(residual))
    # A residual that reads a large source -- a dimension a cut left out because no node could
    # hold it -- may align itself: TPC-H q18 joins its few thousand large orders to `customer`,
    # and `customer` is stored in `custkey` order, so the join runs per key range instead of
    # reading 5 GB on the driver.
    residual_bytes = sum(projected_bytes(bound[s], None) or 0 for s in order if s < len(sources))
    if residual_bytes > _NESTED_BROADCAST_BYTES:
        from batcher.dist.executors.aligned.route import choose_plan

        inner = choose_plan(residual, bound, strict=False)
        if inner is not None:
            found = run_plan(inner, bound, workers, hub, metrics_out)
            if found is not None:
                get_logger("dist").info(
                    "aligned: residual aligned %.2fs", time.perf_counter() - started
                )
                return found

    def renumber(node: LogicalPlan) -> LogicalPlan:
        if isinstance(node, Scan):
            return dataclasses.replace(node, source_id=order.index(node.source_id))
        return node

    out = _single_node(transform_up(residual, renumber), [bound[s] for s in order])
    get_logger("dist").info("aligned: residual %.2fs", time.perf_counter() - started)
    return out
