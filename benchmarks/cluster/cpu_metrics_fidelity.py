"""Does a distributed run report what it cost the cluster? Measured, not assumed.

`observe.metrics` publishes `cores_busy`, `time_ms_total` and `execution_ms_total` for every
query, and a family of things reads them: the Prometheus `batcher_cpu_*` series, the
dashboard's CPU panel, and every "is the box holding this query back" finding in
`observe.insights`. On the single-node path those counters are accurate --
`benchmarks/internals/cpu_utilization.py` holds them against `resource.getrusage` and they
agree inside 0.4%.

This is the same check for the distributed path, where the driver's own `getrusage` cannot be
the reference: the work happened on other machines. The reference here is the cluster itself,
sampled per node through `cluster_util.ClusterMonitor` (one `num_cpus=0` actor per node
reading `psutil`), which is the same instrument `vs_ray_daft.py` reports utilization with.

**What it found when it was written (2026-09-01).** A 5-node / 384-core cluster, otherwise
idle, running a 200M-row grouped aggregate off shared storage:

    wall 13.26s   sampled 14.1% of the cluster = 54 of 384 cores = 717.5 CPU-seconds
    engine reported 46 ms of CPU and 0.96 cores busy

Four orders of magnitude.

**The Python aggregation is not the cause, and an earlier revision of this docstring said it
was.** That claim rested on every `record_usage` call site being single-node or spill, which
is true and irrelevant: the distributed path carries usage by a different route --
`stages.py` sets `prof.worker_metrics`, `ProfileCollector.to_profile` folds each worker
document's `query` block in, and `QueryUsage.merged` sums them. Instrumenting the driver
shows that route working exactly as written: both drains fire (`record_worker_metrics` and
`drain_worker_metrics`, 64 documents each), every document carries a `query` block, and
their summed `cpu_ns` reaches `QUERY_END` unchanged -- 51.2 ms in, 51.2 ms out.

The documents themselves are what is small. 128 of them total **0.099 seconds** of worker
CPU, 0.35-5.23 ms apiece, for a run that burned some 300 CPU-seconds across the fleet. So
the gap is upstream of the metrics layer: the great majority of the work runs in tasks that
never produce a metered `ExecMetrics` document, or produce one that accounts for a sliver of
what the task spent. Localising that further means reading the distributed executor, which
is mid-rewrite, or instrumenting the Rust engine.

The consequence is not a cosmetic gap. It is that Batcher's own metrics cannot answer "how
much of this cluster did that query use", which is the first question anyone sizing a cluster
asks, and they answer it *confidently and wrongly* rather than declining.

Run (needs a Ray cluster and a shared source; see `SOURCE`):
    python benchmarks/cluster/cpu_metrics_fidelity.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _ray_env import init_batcher_ray

from envinfo import machine_fingerprint, require_release_build

#: A Parquet source every node can read. Written once with, e.g.
#: ``bt.range(200_000_000).select(...).write.parquet(SOURCE)``. Overridable so the check can
#: run against whatever a fleet already has staged.
SOURCE = os.environ.get("BATCHER_FIDELITY_SOURCE", "/mnt/cluster_storage/batcher_util_probe")

#: How far the engine's reported CPU may fall below the sampled figure before this is a
#: failure. Generous on purpose: the sampler counts whole nodes including the head and any
#: co-tenant, so it legitimately over-counts somewhat, and the reading is a 200 ms cadence.
#: Nothing inside an order of magnitude is being called a defect here. The gap this exists to
#: catch was four orders.
_MAX_UNDER_REPORT = 10.0


def main() -> int:
    # Refuse a dev-profile engine (8-60x slower) and print the machine, because a timing is
    # only reproducible beside the box that produced it. `init_batcher_ray` guards the build
    # too, but it does so *after* attaching to the cluster and it prints no fingerprint --
    # and a check that runs where nobody can see it is how a debug-build number gets
    # published. Same placement and same reasoning as the GPU scripts beside this one.
    require_release_build()
    print(machine_fingerprint())
    init_batcher_ray()

    import ray
    from cluster_util import ClusterMonitor

    import batcher as bt
    from batcher import col
    from batcher._internal import events

    nodes = [n for n in ray.nodes() if n.get("Alive")]
    cluster_cpus = int(sum(n["Resources"].get("CPU", 0) for n in nodes))
    if cluster_cpus <= 0:
        print("no CPU capacity visible on the cluster; nothing to measure")
        return 0

    finished: list[dict] = []
    events.subscribe(
        lambda event: finished.append(event.fields) if event.kind == events.QUERY_END else None
    )

    dataset = (
        bt.read_parquet(SOURCE)
        .group_by(k=col("a") % 200_000)
        .agg(s=col("b").sum(), n=col("a").count())
    )

    monitor = ClusterMonitor(interval_s=0.2)
    monitor.start()
    started = time.monotonic()
    rows = len(dataset.collect(distributed=True))
    wall = time.monotonic() - started
    sampled = monitor.stop()

    usage = (finished[-1].get("usage") if finished else None) or {}
    busy_cores = sampled["mean_busy_pct"] / 100.0 * cluster_cpus
    sampled_cpu_seconds = busy_cores * wall
    reported_cpu_seconds = float(usage.get("cpu_ms", 0.0)) / 1000.0

    print(f"\ncluster: {len(nodes)} nodes / {cluster_cpus} CPUs;  rows {rows};  wall {wall:.2f}s")
    print(
        f"  sampled : {sampled['mean_busy_pct']:.1f}% busy = {busy_cores:.0f} cores"
        f"  ({sampled_cpu_seconds:.1f} CPU-seconds, {sampled['active_nodes']:.0f} active nodes)"
    )
    print(
        f"  engine  : {reported_cpu_seconds:.3f} CPU-seconds,"
        f" cores_busy {float(usage.get('cores_busy', 0.0)):.2f}"
    )

    if sampled_cpu_seconds < 1.0:
        print("\n  the query did not load the cluster enough to judge; use a larger source")
        return 0
    if reported_cpu_seconds <= 0.0:
        print("\n  FAIL: the engine reported no CPU at all for a run that used the cluster")
        return 1
    factor = sampled_cpu_seconds / reported_cpu_seconds
    print(f"  under-report factor: {factor:.0f}x")
    if factor > _MAX_UNDER_REPORT:
        print(
            f"\n  FAIL: distributed CPU accounting is {factor:.0f}x low. The workers' whole-"
            "execution readings are not reaching `QueryUsage` -- see this module's docstring."
        )
        return 1
    print("\n  OK: the engine's CPU accounting is within an order of magnitude of the cluster")
    return 0


if __name__ == "__main__":
    sys.exit(main())
