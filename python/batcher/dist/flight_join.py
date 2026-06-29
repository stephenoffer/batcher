"""Distributed hash join over an Arrow Flight shuffle (object store bypassed).

Co-partitions both join sides by key over two shuffle stages, then joins each
co-located bucket. Like the aggregate Flight path, only `(addr, ticket)` strings
(and the small results) transit Ray — the heavy data moves node→node over
credit-bounded Flight, never through the object store. The per-worker Flight
endpoint and the credit window come from the shared `_FlightWorker` /
`_shuffle_credits` in `flight_aggregate`.
"""

from __future__ import annotations

import contextlib
import json

import pyarrow as pa

from batcher.dist.executor import _apply_above, _ensure_ray, _relabel_single_source
from batcher.dist.executors.partition_io import partition_descriptors, source_pushdown
from batcher.dist.executors.ray_runtime import engine_config_json, release_placement
from batcher.dist.fleet import acquire_fleet
from batcher.dist.flight_aggregate import _shuffle_credits
from batcher.io.source import Source
from batcher.plan.logical import Aggregate, Join, LogicalPlan

__all__ = ["execute_join_flight"]


def execute_join_flight(
    above: list[LogicalPlan],
    join: Join,
    sources: list[Source],
    workers: int,
    _fault_inject: set[int] | None = None,
    *,
    fused_agg: Aggregate | None = None,
    combine_partials: bool = False,
) -> pa.Table:
    """Co-partition both join sides over a Flight shuffle and join per bucket.

    `fused_agg` is folded INTO the reduce so only small aggregated/partial buckets reach
    the driver — the full join never materializes on the head (exchange elimination). When
    its group keys ⊇ the join key every group lands in one bucket, so each reducer's
    per-bucket aggregate is complete and the driver concatenates. When they do NOT (set
    `combine_partials`), a group spans buckets: each reducer emits its PARTIAL state and
    the driver does the cross-bucket `combine_finalize` (standard mergeable two-phase),
    so an aggregate over an arbitrary join still runs fully distributed instead of
    collecting the whole join to the driver to aggregate it single-node.

    Left and right mappers publish their key-hashed buckets on their own Flight
    servers (shuffle stages 0 and 1); reducer r fetches bucket r from every mapper
    on both sides and runs the local join. A lost worker's buckets are recomputed
    from its source partitions (still on disk) on a survivor — Spark-style lineage
    recovery, matching the aggregate path. Object store bypassed. `_fault_inject`
    is a test-only hook: worker ids to kill after the map barrier."""
    import ray

    import batcher._native as nat

    _ensure_ray(workers)
    cfg_json = engine_config_json()  # driver config → shipped to worker actors

    # Restrict each side to exactly the columns the join OUTPUT carries (Kyber already
    # pruned `join.output` to what the consumer above needs) plus the join keys, so the
    # shuffle moves only those columns — not the whole wide table. Without this each side
    # ships every source column (the relabeled per-side sub-plan is a bare scan, so its
    # own `source_pushdown` can't see that the join above keeps just a few): TPC-H lineitem
    # has 17 columns but a `…⋈ orders GROUP BY priority` join needs 2, so the un-pruned
    # shuffle moved ~8× the bytes and dominated the join.
    left_need = {o.name for o in join.output if o.side == "left"} | set(join.left_keys)
    right_need = {o.name for o in join.output if o.side == "right"} | set(join.right_keys)
    left_plan, lsid = _relabel_single_source(_project_join_side(join.left, left_need))
    right_plan, rsid = _relabel_single_source(_project_join_side(join.right, right_need))
    left_ir = json.dumps(left_plan.to_ir())
    right_ir = json.dumps(right_plan.to_ir())
    join_ir = json.dumps(
        {
            "op": "hash_join",
            "left": {"op": "scan", "source_id": 0},
            "right": {"op": "scan", "source_id": 1},
            "left_keys": list(join.left_keys),
            "right_keys": list(join.right_keys),
            "join_type": join.join_type,
            "output": [{"side": o.side, "name": o.name, "alias": o.alias} for o in join.output],
        }
    )
    # A fused aggregate's group keys/aggregates (over the join output columns), shipped to
    # each reducer to fold its joined bucket down before it leaves the worker.
    gk = aj = None
    if fused_agg is not None:
        gk = json.dumps([{"expr": k.expr.to_ir(), "alias": k.alias} for k in fused_agg.group_keys])
        aj = json.dumps([s.agg.to_ir(s.alias) for s in fused_agg.aggregates])

    # 0-row schema probes so reducers can type the null-extended side of an outer join.
    def probe(sub_ir, source):
        empty = pa.RecordBatch.from_pylist([], schema=source.schema())
        out = nat.execute_plan(sub_ir, [[empty]], cfg_json)
        return out[0] if out else empty

    credits = _shuffle_credits()

    # Borrow the query-lifetime fleet when the adaptive loop installed one; else spawn
    # one we tear down. Every Flight operator must borrow it — spawning a second
    # placement group would contend with the fleet's held bundles and deadlock.
    actors, pg, addrs, workers, owns = acquire_fleet(workers, credits, cfg_json)
    n_buckets = workers
    try:
        # Each side reads only the columns/rows its map prefix needs (join keys + carried
        # output), not the whole wide table — the biggest scan win on a star-schema join.
        lproj, lpred = source_pushdown(left_plan, lsid)
        rproj, rpred = source_pushdown(right_plan, rsid)
        lparts = partition_descriptors(sources[lsid], workers, projection=lproj, predicate=lpred)
        rparts = partition_descriptors(sources[rsid], workers, projection=rproj, predicate=rpred)

        ray.get(
            [
                actors[i].map_publish_raw.remote(
                    left_ir, list(join.left_keys), lparts[i], n_buckets, 0
                )
                for i in range(workers)
            ]
            + [
                actors[i].map_publish_raw.remote(
                    right_ir, list(join.right_keys), rparts[i], n_buckets, 1
                )
                for i in range(workers)
            ]
        )

        # Simulate worker loss after the map barrier (test hook).
        if _fault_inject:
            for i in _fault_inject:
                ray.kill(actors[i])

        lschema = probe(left_ir, sources[lsid])
        rschema = probe(right_ir, sources[rsid])
        batches = _join_reduce_with_recovery(
            actors,
            list(addrs),
            (lparts, rparts),
            (left_ir, right_ir),
            (list(join.left_keys), list(join.right_keys)),
            join_ir,
            (lschema, rschema),
            n_buckets,
            workers,
            gk,
            aj,
            finalize=not combine_partials,
        )
    finally:
        if owns:
            for a in actors:
                with contextlib.suppress(Exception):
                    ray.kill(a)
            release_placement(pg)

    # Non-fusable fused aggregate: reducers shipped PARTIAL state (one per bucket); the
    # group spans buckets, so do the cross-bucket combine+finalize here on the small
    # partials (workers × groups rows), not on the full join.
    if combine_partials and fused_agg is not None:
        final = nat.combine_finalize(gk, aj, batches) if batches else None
        table = pa.Table.from_batches([final]) if final is not None else _empty_fused(fused_agg)
        return table if not above else _apply_above(above, table)

    if batches:
        table = pa.Table.from_batches(batches)
    elif fused_agg is not None:
        table = _empty_fused(fused_agg)
    else:
        table = pa.table({o.alias: [] for o in join.output})
    return table if not above else _apply_above(above, table)


def _empty_fused(fused_agg: Aggregate) -> pa.Table:
    """The empty result table for a fused post-join aggregate (group keys + aggregates)."""
    keys = [k.alias for k in fused_agg.group_keys]
    return pa.table({c: [] for c in keys + [s.alias for s in fused_agg.aggregates]})


def _project_join_side(side: LogicalPlan, needed: set[str]) -> LogicalPlan:
    """Wrap a join side in a `Project` selecting only `needed` columns, so its scan reads
    and shuffles just those — `source_pushdown` then maps them through any rename/filter
    to the actual scan columns. A no-op when nothing can be pruned (needed ⊇ available),
    so a `SELECT *`-style join whose output Kyber did not prune is unchanged.
    """
    from batcher.plan.expr_ir import Col
    from batcher.plan.logical import Project, Projection

    avail = side.available_columns()
    keep = [c for c in avail if c in needed]  # preserve the side's column order
    if not keep or len(keep) == len(avail):
        return side
    return Project(side, tuple(Projection(alias=c, expr=Col(c)) for c in keep))


def _join_reduce_with_recovery(
    actors,
    addrs,
    parts,
    irs,
    keys,
    join_ir,
    schemas,
    n_buckets,
    workers,
    gk=None,
    aj=None,
    finalize=True,
):
    """Run the join reduce under recompute-on-worker-loss recovery.

    Each reducer is hosted on a live worker; one that reports an unreachable mapper
    (or whose host died) drives a recompute of that worker's *both* sides (the join
    co-partitions left and right) from their on-disk source partitions onto a
    survivor, then a retry. Returns the joined batches.
    """
    import ray

    from batcher._internal.errors import ResourceError
    from batcher.carbonite.resilience import ShuffleRecovery, gather_with_backups
    from batcher.dist.executors.ray_runtime import (
        draining_workers,
        recovery_policy,
        speculation_policy,
    )

    lparts, rparts = parts
    left_ir, right_ir = irs
    left_keys, right_keys = keys
    lschema, rschema = schemas
    dead: set[int] = set()

    def _pick_live(avoid: set[int]) -> int:
        for i in range(workers):
            if i not in dead and i not in avoid:
                return i
        raise ResourceError("no surviving worker to recover the join shuffle on")

    # A bucket join that returns "ok" fetched both its sides completely and is
    # deterministic, so cache it across recovery rounds (keyed by bucket index) and
    # never re-run it. Only pending buckets re-launch, so one lost mapper doesn't
    # re-join every surviving bucket — the amplification that hurt most on a churning
    # spot/autoscaling cluster.
    done: dict[int, object] = {}

    def _host_for(r: int, avoid: set[int]) -> int:
        # `reducer r → actor r`, unless dead/avoided; `avoid` lets a straggler's backup
        # land on a different live worker than the slow original.
        return r if r not in dead and r not in avoid else _pick_live(avoid)

    def attempt():
        failed = set()
        # Launch every *pending* reduce-join concurrently, then collect via
        # `gather_with_backups`: a degraded-but-alive bucket gets a backup on another
        # live worker (deterministic ⇒ byte-identical), and a dead host is classified
        # for recompute — so one slow node cannot stall the join barrier.
        ref_host: dict[object, int] = {}

        def _launch(r: int, avoid: set[int]):
            host = _host_for(r, avoid)
            ref = actors[host].reduce_join.remote(
                join_ir, addrs, r, lschema, rschema, gk, aj, finalize
            )
            ref_host[ref] = host
            return ref

        pending = [r for r in range(n_buckets) if r not in done]
        refs = [_launch(r, set()) for r in pending]

        def _relaunch(idx: int):
            try:
                return _launch(pending[idx], {ref_host[refs[idx]]})
            except ResourceError:
                return _launch(pending[idx], set())

        def _on_failure(_idx: int, ref: object, _exc: Exception):
            return ("__dead__", ref_host.get(ref))

        results = gather_with_backups(refs, _relaunch, speculation_policy(), on_failure=_on_failure)
        for r, (status, payload) in zip(pending, results, strict=True):
            if status == "ok":
                done[r] = payload  # complete + deterministic → cache, never re-run
            elif status == "__dead__":
                if payload is not None:
                    dead.add(payload)  # host died — its mapped sides are lost too
                    failed.add(payload)
            else:
                failed.update(payload)
        return [p for p in done.values() if p], failed

    def recompute(failed):
        for src in failed:
            dead.add(src)  # an unreachable mapper means that worker is gone
            target = _pick_live({src})
            actors[target].map_publish_raw.remote(
                left_ir, left_keys, lparts[src], n_buckets, 0, src
            )
            addrs[src] = ray.get(
                actors[target].map_publish_raw.remote(
                    right_ir, right_keys, rparts[src], n_buckets, 1, src
                )
            )

    # Proactive spot-preemption migration: move a draining worker's mapped sides to a
    # survivor before reclamation (no recovery round, no idle-timeout stall). Best-effort
    # — a failure falls through to the reactive recompute the loop already does.
    proactive = draining_workers(actors, workers)
    if proactive:
        with contextlib.suppress(Exception):
            recompute(proactive)

    finals = ShuffleRecovery(recovery_policy()).run(attempt, recompute)
    out = []
    for res in finals:
        out.extend(b for b in res if b.num_rows > 0)
    return out
