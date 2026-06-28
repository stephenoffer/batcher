"""Distributed window functions over an Arrow Flight shuffle (object store bypassed).

Hash-shuffles the raw input rows by the window's partition keys so every whole
partition lands on one reducer, which runs the ordinary window operator over its
rows — identical to single-node. Like the aggregate/join Flight paths, only
`(addr, ticket)` strings (and the small results) transit Ray; the rows move
node→node over credit-bounded Flight, never through the object store. The
per-worker Flight endpoint, credit window, and `_FlightWorker` actor are the shared
ones from `flight_aggregate`; a lost worker's bucket is recomputed from its source
partition (still on disk) on a survivor — the same Spark-style lineage recovery.
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
from batcher.plan.logical import LogicalPlan, Window

__all__ = ["execute_window_flight"]


def execute_window_flight(
    above: list[LogicalPlan],
    window: Window,
    sources: list[Source],
    workers: int,
    _fault_inject: set[int] | None = None,
) -> pa.Table:
    """Hash-shuffle rows by the window's partition keys over Flight, window per bucket.

    Mappers publish their key-hashed row buckets on their own Flight servers
    (shuffle stage 0); reducer r fetches bucket r from every mapper and runs the
    window operator over the whole partition. `_fault_inject` is a test-only hook:
    worker ids to kill after the map barrier to exercise lineage recovery."""
    import ray

    _ensure_ray(workers)
    cfg_json = engine_config_json()  # driver config → shipped to worker actors

    # Caller guarantees every partition key is a plain column; shuffle by their names.
    key_names = [k.name for k in window.partition_keys]
    map_plan, sid = _relabel_single_source(window.input)
    map_ir = json.dumps(map_plan.to_ir())
    # The reduce runs the window over its bucket as a single in-memory source 0.
    win_ir = window.to_ir()
    win_ir["input"] = {"op": "scan", "source_id": 0}
    win_json = json.dumps(win_ir)
    credits = _shuffle_credits()

    # Borrow the query-lifetime fleet if installed (every Flight operator must, or a
    # second placement group deadlocks against the fleet's bundles); else spawn our own.
    actors, pg, addrs, workers, owns = acquire_fleet(workers, credits, cfg_json)
    n_buckets = workers
    try:
        # Read only the columns/rows the window's map prefix needs (see flight_aggregate).
        projection, predicate = source_pushdown(map_plan, sid)
        parts = partition_descriptors(
            sources[sid], workers, projection=projection, predicate=predicate
        )

        ray.get(
            [
                actors[i].map_publish_raw.remote(map_ir, key_names, parts[i], n_buckets, 0)
                for i in range(workers)
            ]
        )

        if _fault_inject:
            for i in _fault_inject:
                ray.kill(actors[i])

        batches = _window_reduce_with_recovery(
            actors, list(addrs), parts, map_ir, key_names, win_json, n_buckets, workers
        )
    finally:
        if owns:
            for a in actors:
                with contextlib.suppress(Exception):
                    ray.kill(a)
            release_placement(pg)

    table = (
        pa.Table.from_batches(batches)
        if batches
        else pa.table({c: [] for c in window.available_columns()})
    )
    return table if not above else _apply_above(above, table)


def _window_reduce_with_recovery(
    actors, addrs, parts, map_ir, key_names, win_json, n_buckets, workers
):
    """Run the window reduce under recompute-on-worker-loss recovery.

    A reducer that reports an unreachable mapper (or whose host died) drives a
    recompute of that worker's row bucket from its on-disk source partition onto a
    survivor, then a retry — matching the aggregate/join paths. Returns the windowed
    batches.
    """
    import ray

    from batcher._internal.errors import ResourceError
    from batcher.carbonite.resilience import ShuffleRecovery, gather_with_backups
    from batcher.dist.executors.ray_runtime import (
        draining_workers,
        recovery_policy,
        speculation_policy,
    )

    dead: set[int] = set()

    def _pick_live(avoid: set[int]) -> int:
        for i in range(workers):
            if i not in dead and i not in avoid:
                return i
        raise ResourceError("no surviving worker to recover the window shuffle on")

    # A window-bucket reduce that returns "ok" computed its complete partitions
    # deterministically, so cache it across recovery rounds (keyed by bucket index) and
    # never re-run it. Only pending buckets re-launch, so one lost mapper doesn't
    # recompute every surviving bucket — the amplification that hurt most on a churning
    # spot/autoscaling cluster.
    done: dict[int, object] = {}

    def _host_for(r: int, avoid: set[int]) -> int:
        return r if r not in dead and r not in avoid else _pick_live(avoid)

    def attempt():
        failed = set()
        # Launch every *pending* window reduce concurrently, then collect via
        # `gather_with_backups`: a degraded-but-alive bucket gets a backup on another
        # live worker (deterministic ⇒ byte-identical), a dead host is classified for
        # recompute — so one slow node cannot stall the window barrier.
        ref_host: dict[object, int] = {}

        def _launch(r: int, avoid: set[int]):
            host = _host_for(r, avoid)
            ref = actors[host].reduce_window.remote(win_json, addrs, r)
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

        gathered = gather_with_backups(
            refs, _relaunch, speculation_policy(), on_failure=_on_failure
        )
        for r, (status, payload) in zip(pending, gathered, strict=True):
            if status == "ok":
                done[r] = payload  # complete + deterministic → cache, never re-run
            elif status == "__dead__":
                if payload is not None:
                    dead.add(payload)  # its mapped rows are lost too
                    failed.add(payload)
            else:
                failed.update(payload)
        return [p for p in done.values() if p], failed

    def recompute(failed_srcs):
        for src in failed_srcs:
            dead.add(src)  # an unreachable mapper means that worker is gone
            target = _pick_live({src})
            addrs[src] = ray.get(
                actors[target].map_publish_raw.remote(
                    map_ir, key_names, parts[src], n_buckets, 0, src
                )
            )

    # Proactive spot-preemption migration: move a draining worker's window bucket to a
    # survivor before reclamation (no recovery round, no idle-timeout stall). Best-effort
    # — a failure falls through to the reactive recompute the loop already does.
    proactive = draining_workers(actors, workers)
    if proactive:
        with contextlib.suppress(Exception):
            recompute(proactive)

    finals = ShuffleRecovery(recovery_policy()).run(attempt, recompute)
    out: list[pa.RecordBatch] = []
    for res in finals:
        out.extend(b for b in res if b.num_rows > 0)
    return out
