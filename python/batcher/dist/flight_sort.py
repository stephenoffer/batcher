"""Distributed sort over an Arrow Flight shuffle (object store bypassed).

Range-partitions by the leading sort key across workers, sorts each range, and
concatenates the ranges in key order — globally sorted, no final merge. The range
boundaries come from a **sample pass**: each worker samples its OWN split's
leading-key quantile grid (so the input is never read on the driver, unlike the
disk sort), and the driver merges the small grids into `workers-1` boundaries. The
rows then move node→node over credit-bounded Flight, never through the object
store. Reuses the shared `_FlightWorker` and the same Spark-style lineage recovery.

Boundary precision only affects *balance*, never correctness: range-partition →
per-range sort → ordered concat is order-preserving for any boundaries, because the
boundaries are deduplicated and `searchsorted(side="right")` keeps equal keys in one
bucket. Restricted (by the dispatcher) to a leading key that is a plain column over
a breaker-free single source.
"""

from __future__ import annotations

import contextlib
import json

import pyarrow as pa

from batcher.dist.executor import _apply_above, _ensure_ray, _relabel_single_source
from batcher.dist.executors.partition_io import merge_boundaries, partition_descriptors
from batcher.dist.executors.ray_runtime import engine_config_json, release_placement
from batcher.dist.fleet import acquire_fleet
from batcher.dist.flight_aggregate import _shuffle_credits
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan, Sort

__all__ = ["execute_sort_flight"]

# Per-worker CDF sample granularity: a fine grid (33 probe points) so the merged
# boundaries balance the ranges well. Precision affects only balance, not result.
_SAMPLE_PROBS = [i / 32 for i in range(33)]


def execute_sort_flight(
    above: list[LogicalPlan],
    sort: Sort,
    sources: list[Source],
    workers: int,
    _fault_inject: set[int] | None = None,
) -> pa.Table:
    """Range-partition by the leading key over Flight, sort each range, concat in order.

    `_fault_inject` is a test-only hook: worker ids to kill after the map barrier to
    exercise lineage recovery (a lost worker's range bucket is recomputed from its
    source partition on a survivor)."""
    import ray

    _ensure_ray(workers)
    cfg_json = engine_config_json()  # driver config → shipped to worker actors

    key = sort.keys[0]  # caller guarantees a plain-column leading key
    key_name = key.expr.name
    desc, nulls_first = key.descending, key.nulls_first
    map_plan, sid = _relabel_single_source(sort.input)
    map_ir = json.dumps(map_plan.to_ir())
    sort_ir = json.dumps(
        {
            "op": "sort",
            "input": {"op": "scan", "source_id": 0},
            "keys": [
                {"expr": k.expr.to_ir(), "descending": k.descending, "nulls_first": k.nulls_first}
                for k in sort.keys
            ],
            "limit": sort.limit,
        }
    )
    credits = _shuffle_credits()

    # Borrow the query-lifetime fleet if installed (every Flight operator must, or a
    # second placement group deadlocks against the fleet's bundles); else spawn our own.
    actors, pg, addrs, workers, owns = acquire_fleet(workers, credits, cfg_json)
    n_buckets = workers
    try:
        parts = partition_descriptors(sources[sid], workers)

        # SAMPLE: each worker samples its own split's leading-key distribution.
        grids = ray.get(
            [
                actors[i].sample_quantiles.remote(map_ir, key_name, _SAMPLE_PROBS, parts[i])
                for i in range(workers)
            ]
        )
        boundaries = merge_boundaries(grids, workers)

        # MAP: range-partition each split by the boundaries and publish raw rows.
        ray.get(
            [
                actors[i].range_publish.remote(
                    map_ir, key_name, boundaries, n_buckets, nulls_first, desc, parts[i]
                )
                for i in range(workers)
            ]
        )

        if _fault_inject:
            for i in _fault_inject:
                ray.kill(actors[i])

        results = _sort_reduce_with_recovery(
            actors,
            list(addrs),
            parts,
            map_ir,
            key_name,
            boundaries,
            sort_ir,
            nulls_first,
            desc,
            n_buckets,
            workers,
        )
    finally:
        if owns:
            for a in actors:
                with contextlib.suppress(Exception):
                    ray.kill(a)
            release_placement(pg)

    # Concatenate the ranges in leading-key order (reversed for a descending sort) —
    # each bucket is globally ordered relative to the others, so no final merge.
    order = range(workers - 1, -1, -1) if desc else range(workers)
    out: list[pa.RecordBatch] = []
    for r in order:
        out.extend(b for b in results.get(r, []) if b.num_rows > 0)
    table = (
        pa.Table.from_batches(out) if out else pa.table({c: [] for c in sort.available_columns()})
    )
    if sort.limit is not None:
        table = table.slice(0, sort.limit)
    return table if not above else _apply_above(above, table)


def _sort_reduce_with_recovery(
    actors,
    addrs,
    parts,
    map_ir,
    key_name,
    boundaries,
    sort_ir,
    nulls_first,
    desc,
    n_buckets,
    workers,
):
    """Run the sort reduce under recompute-on-worker-loss recovery.

    Returns a `{bucket_id: sorted_batches}` dict so the driver can concatenate the
    ranges in key order regardless of completion order. A reducer reporting an
    unreachable mapper (or whose host died) drives a recompute of that worker's range
    bucket from its on-disk source partition onto a survivor, then a retry.
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
        raise ResourceError("no surviving worker to recover the sort shuffle on")

    # A bucket reduce that returns "ok" sorted its complete range deterministically, so
    # cache it across recovery rounds (keyed by bucket index, which also preserves the
    # final concatenation order) and never re-run it. Only pending buckets re-launch, so
    # one lost mapper doesn't re-sort every surviving bucket — the amplification that
    # hurt most on a churning spot/autoscaling cluster.
    results: dict[int, object] = {}

    def _host_for(r: int, avoid: set[int]) -> int:
        return r if r not in dead and r not in avoid else _pick_live(avoid)

    def attempt():
        failed = set()
        # Launch every *pending* range-bucket reduce concurrently, then collect via
        # `gather_with_backups`: a degraded-but-alive bucket gets a backup on another
        # live worker (deterministic ⇒ byte-identical), a dead host is classified for
        # recompute — so one slow node cannot stall the sort barrier.
        ref_host: dict[object, int] = {}

        def _launch(r: int, avoid: set[int]):
            host = _host_for(r, avoid)
            ref = actors[host].sort_reduce.remote(sort_ir, addrs, r)
            ref_host[ref] = host
            return ref

        pending = [r for r in range(n_buckets) if r not in results]
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
                results[r] = payload or []  # keyed by bucket → final order preserved
            elif status == "__dead__":
                if payload is not None:
                    dead.add(payload)  # its mapped rows are lost too
                    failed.add(payload)
            else:
                failed.update(payload)
        return results, failed

    def recompute(failed_srcs):
        for src in failed_srcs:
            dead.add(src)  # an unreachable mapper means that worker is gone
            target = _pick_live({src})
            addrs[src] = ray.get(
                actors[target].range_publish.remote(
                    map_ir, key_name, boundaries, n_buckets, nulls_first, desc, parts[src], src
                )
            )

    # Proactive spot-preemption migration: move a draining worker's range bucket to a
    # survivor before reclamation (no recovery round, no idle-timeout stall). Best-effort
    # — a failure falls through to the reactive recompute the loop already does.
    proactive = draining_workers(actors, workers)
    if proactive:
        with contextlib.suppress(Exception):
            recompute(proactive)

    return ShuffleRecovery(recovery_policy()).run(attempt, recompute)
