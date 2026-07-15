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

from batcher._internal.native import engine
from batcher.dist.executor import _apply_above, _ensure_ray, _relabel_single_source
from batcher.dist.executors.partition_io import (
    merge_boundaries,
    partition_descriptors,
    source_pushdown,
)
from batcher.dist.executors.ray_runtime import (
    engine_config_json,
    map_barrier,
    shuffle_partitions,
)
from batcher.dist.fleet import acquire_fleet, release_fleet
from batcher.dist.flight_aggregate import _shuffle_credits
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan, Sort

__all__ = ["execute_sort_flight", "execute_topn_flight"]

# Per-worker CDF sample granularity: a fine grid (33 probe points) so the merged
# boundaries balance the ranges well. Precision affects only balance, not result.
_SAMPLE_PROBS = [i / 32 for i in range(33)]


def _sort_ir(keys, limit, input_ir):
    """The sort IR over `input_ir` carrying `keys` and `limit` (None = no limit)."""
    return json.dumps(
        {
            "op": "sort",
            "input": input_ir,
            "keys": [
                {"expr": k.expr.to_ir(), "descending": k.descending, "nulls_first": k.nulls_first}
                for k in keys
            ],
            "limit": limit,
        }
    )


def execute_topn_flight(
    above: list[LogicalPlan],
    sort: Sort,
    sources: list[Source],
    workers: int,
) -> pa.Table:
    """A distributed **top-N** (`ORDER BY ... LIMIT k`) with NO shuffle — mergeable.

    The global top-N is the top-N of the union of each worker's top-N, so every worker
    reads its own split, runs the map prefix + the single-node top-N heap (`sort+limit`),
    and ships only `k` rows; the driver merges the `workers x k` rows with one more
    `sort+limit`. This skips the full range-partition sort entirely (which would shuffle
    every row just to slice the first `k`), the dominant cost for a small `k`.
    """
    import ray

    nat = engine()
    _ensure_ray(workers)
    cfg_json = engine_config_json()
    map_plan, sid = _relabel_single_source(sort.input)
    # Per-worker plan: read the split (scan 0) → map prefix → local top-N heap.
    local_ir = _sort_ir(sort.keys, sort.limit, map_plan.to_ir())
    # Driver merge plan: top-N over the concatenated per-worker top-Ns (scan 0).
    merge_ir = _sort_ir(sort.keys, sort.limit, {"op": "scan", "source_id": 0})

    actors, pg, _addrs, workers, owns = acquire_fleet(workers, _shuffle_credits(), cfg_json)
    try:
        # Partition to the fleet's ACTUAL worker count. `acquire_fleet` may hand back a
        # reused session fleet whose size differs from the requested `workers` (it reassigns
        # `workers` to that size), so `parts` must be built here, after the fleet is known —
        # otherwise parts and actors mismatch: a larger fleet indexes past `parts`, a smaller
        # one silently drops the tail partitions' rows (a wrong result). `execute_sort_flight`
        # already orders it this way.
        # `map_plan`'s scan was relabeled to source 0, so key the analysis on 0, not on the
        # source's original index: a staged plan whose input is an intermediate (source id >
        # 0) missed the lookup and silently read every column.
        projection, predicate = source_pushdown(map_plan, 0)
        parts = partition_descriptors(
            sources[sid], workers, projection=projection, predicate=predicate
        )
        results = ray.get([actors[i].local_topn.remote(local_ir, parts[i]) for i in range(workers)])
    finally:
        release_fleet(actors, pg, owns)

    gathered = [b for r in results for b in r if b.num_rows > 0]
    merged = nat.execute_plan(merge_ir, [gathered], cfg_json) if gathered else []
    if merged:
        table = pa.Table.from_batches(merged)
    else:
        table = pa.table({k.expr.name: [] for k in sort.keys})
    return table if not above else _apply_above(above, table)


def execute_sort_flight(
    above: list[LogicalPlan],
    sort: Sort,
    sources: list[Source],
    workers: int,
    _fault_inject: set[int] | None = None,
    *,
    _fault_inject_map: set[int] | None = None,
) -> pa.Table:
    """Range-partition by the leading key over Flight, sort each range, concat in order.

    Worker loss is survived in every phase: `map_barrier` reprocesses a split whose worker
    dies while sampling or range-partitioning, and `ShuffleRecovery` recomputes a range
    bucket whose worker dies before the reduce fetches it. `_fault_inject` /
    `_fault_inject_map` are test-only hooks: worker ids to kill after / before the map
    barrier."""
    import os as _os0
    import time as _tt0

    import ray

    _profE = _os0.environ.get("BATCHER_SORT_PROFILE")
    _enter = _tt0.perf_counter()
    _ensure_ray(workers)
    if _profE:
        print(f"[sort] _ensure_ray {_tt0.perf_counter() - _enter:.1f}s", flush=True)
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

    import os as _os
    import time as _tt

    _prof0 = _os.environ.get("BATCHER_SORT_PROFILE")
    _ps = _tt.perf_counter()
    # Borrow the query-lifetime fleet if installed (every Flight operator must, or a
    # second placement group deadlocks against the fleet's bundles); else spawn our own.
    actors, pg, _addrs, workers, owns = acquire_fleet(workers, credits, cfg_json)
    n_buckets = shuffle_partitions(workers)
    if _prof0:
        print(f"[sort] acquire_fleet {_tt.perf_counter() - _ps:.1f}s", flush=True)
    try:
        # Push the map prefix's projection + predicate into the read so each worker
        # fetches only the columns/rows it needs (the sort keys + carried output), not
        # the whole wide source — the projection the `map_ir` would otherwise discard
        # after paying to read it (see flight_aggregate).
        # `map_plan`'s scan was relabeled to source 0, so key the analysis on 0, not on the
        # source's original index: a staged plan whose input is an intermediate (source id >
        # 0) missed the lookup and silently read every column.
        projection, predicate = source_pushdown(map_plan, 0)
        parts = partition_descriptors(
            sources[sid], workers, projection=projection, predicate=predicate
        )

        import os
        import time as _t

        _prof = os.environ.get("BATCHER_SORT_PROFILE")

        # Simulate worker loss BEFORE the sample/map barriers (test hook).
        if _fault_inject_map:
            for i in _fault_inject_map:
                ray.kill(actors[i])

        # Both barriers run under worker-loss recovery, sharing one `dead` view of the
        # fleet: a worker preempted while sampling or range-partitioning has its split
        # reprocessed on a survivor rather than failing the whole sort. Sampling only
        # reads (nothing to republish); range-publish carries `src` so the relocated
        # buckets keep the ticket the reducers dial.
        dead: set[int] = set()

        # SAMPLE: each worker samples its own split's leading-key distribution.
        _s = _t.perf_counter()
        grids, dead = map_barrier(
            workers,
            lambda host, src: actors[host].sample_quantiles.remote(
                map_ir, key_name, _SAMPLE_PROBS, parts[src]
            ),
            dead=dead,
        )
        # Cut into exactly `n_buckets` ranges: `shuffle_partitions` can trim the reducer
        # count below `workers` (the `max_shuffle_partitions` cap / learned fan-out), and
        # `merge_boundaries(grids, workers)` would emit up to `workers-1` boundaries — more
        # than `n_buckets-1` — routing rows past the last bucket and panicking the range
        # partitioner. Size the boundaries by the actual bucket count.
        boundaries = merge_boundaries(grids, n_buckets)
        if _prof:
            print(f"[sort] SAMPLE {_t.perf_counter() - _s:.1f}s", flush=True)

        # MAP: range-partition each split by the boundaries and publish raw rows.
        _s = _t.perf_counter()
        mapper_addrs, dead = map_barrier(
            workers,
            lambda host, src: actors[host].range_publish.remote(
                map_ir, key_name, boundaries, n_buckets, nulls_first, desc, parts[src], src
            ),
            dead=dead,
        )
        if _prof:
            print(f"[sort] MAP(range_publish) {_t.perf_counter() - _s:.1f}s", flush=True)

        if _fault_inject:
            for i in _fault_inject:
                ray.kill(actors[i])

        _s = _t.perf_counter()
        results = _sort_reduce_with_recovery(
            actors,
            mapper_addrs,
            parts,
            map_ir,
            key_name,
            boundaries,
            sort_ir,
            nulls_first,
            desc,
            n_buckets,
            workers,
            dead=dead,
        )
        if _prof:
            print(f"[sort] REDUCE(gather+sort) {_t.perf_counter() - _s:.1f}s", flush=True)
    finally:
        release_fleet(actors, pg, owns)

    # Concatenate the ranges in leading-key order (reversed for a descending sort) —
    # each bucket is globally ordered relative to the others, so no final merge.
    _pc = _tt.perf_counter()
    order = range(workers - 1, -1, -1) if desc else range(workers)
    out: list[pa.RecordBatch] = []
    for r in order:
        out.extend(b for b in results.get(r, []) if b.num_rows > 0)
    table = (
        pa.Table.from_batches(out) if out else pa.table({c: [] for c in sort.available_columns()})
    )
    if _prof0:
        print(
            f"[sort] driver_concat {_tt.perf_counter() - _pc:.1f}s ({table.num_rows} rows)",
            flush=True,
        )
    if sort.limit is not None:
        table = table.slice(0, sort.limit)
    if _prof0:
        print(f"[sort] execute_sort_flight TOTAL {_tt.perf_counter() - _enter:.1f}s", flush=True)
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
    dead=None,
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

    dead: set[int] = set(dead or ())

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
