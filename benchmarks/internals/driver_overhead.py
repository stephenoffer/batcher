"""What one distributed query costs the *driver*, and how that scales with the fleet.

A distributed query's wall clock is dominated by the workers. Its **ceiling on queries per
second, and its load on the cluster's shared control plane, are not** -- those are set by
what the driver does before and between the stages, on one thread, per query. This
benchmark measures that side and nothing else.

Two figures, and the second is the one that matters past a few hundred nodes:

`driver CPU per query`
    Thread CPU time on the driver, so it is immune to how busy the box is and to how long
    the workers took. It is the reciprocal of how many queries one driver can issue.

`O(nodes) control-plane round trips per query`
    `ray.nodes()`, `ray.cluster_resources()` and `available_resources_per_node()` each
    return **one record per node**, so their cost is a property of the cluster, not of the
    call: at a hundred thousand nodes one of them is megabytes of GCS work, network and
    Python deserialization. The *count* is what this benchmark reports, because the count is
    the scale-invariant number -- it can be measured on a laptop and multiplied by whatever
    fleet the reader has. Alongside it, the number of Ray remote-function definitions
    exported per query, which land on the same shared GCS.

Neither figure needs a big cluster to be meaningful, which is the point: a local Ray
instance produces the same counts a ten-thousand-node cluster would, and the counts are what
decide whether the ten-thousand-node cluster works.

Run:
    python benchmarks/internals/driver_overhead.py
    python benchmarks/internals/driver_overhead.py 20        # 20 timed queries

Requires the optional ``ray`` extra; without it the benchmark exits cleanly with a skip.
"""

from __future__ import annotations

import statistics
import sys
import time

import pyarrow as pa

import batcher as bt

#: Rows in the probe table. Small on purpose: the driver's per-query cost is fixed work, and
#: a large table would bury it under worker time without changing it.
PROBE_ROWS = 50_000
#: Distinct group keys, so the query is a genuine shuffle rather than a single reducer.
PROBE_KEYS = 1_000
_WARMUP = 3


def _probe_table() -> pa.Table:
    return pa.table(
        {
            "k": pa.array([i % PROBE_KEYS for i in range(PROBE_ROWS)], type=pa.int64()),
            "v": pa.array(range(PROBE_ROWS), type=pa.int64()),
        }
    )


class _Counters:
    """Call counts for the cluster-state reads whose payload is O(nodes)."""

    def __init__(self) -> None:
        self.nodes = 0
        self.cluster_resources = 0
        self.free_cpus = 0
        self.remote_defs = 0

    @property
    def topology_calls(self) -> int:
        return self.nodes + self.cluster_resources + self.free_cpus

    def reset(self) -> None:
        self.nodes = self.cluster_resources = self.free_cpus = self.remote_defs = 0


def _install_counters(counters: _Counters):
    """Wrap the O(nodes) Ray accessors to count calls; returns an undo callable."""
    import ray
    from ray import remote_function
    from ray._private import state as ray_state

    real_nodes, real_resources = ray.nodes, ray.cluster_resources
    real_free = ray_state.available_resources_per_node
    real_init = remote_function.RemoteFunction.__init__

    def nodes(*a, **k):
        counters.nodes += 1
        return real_nodes(*a, **k)

    def cluster_resources(*a, **k):
        counters.cluster_resources += 1
        return real_resources(*a, **k)

    def free_cpus(*a, **k):
        counters.free_cpus += 1
        return real_free(*a, **k)

    def remote_init(self, *a, **k):
        counters.remote_defs += 1
        return real_init(self, *a, **k)

    ray.nodes = nodes
    ray.cluster_resources = cluster_resources
    ray_state.available_resources_per_node = free_cpus
    remote_function.RemoteFunction.__init__ = remote_init

    def undo() -> None:
        ray.nodes = real_nodes
        ray.cluster_resources = real_resources
        ray_state.available_resources_per_node = real_free
        remote_function.RemoteFunction.__init__ = real_init

    return undo


def _project(per_query: int, label: str) -> str:
    """One line projecting a per-query round-trip count onto real fleet sizes."""
    counts = "  ".join(f"{n:>7,}: {per_query * n:>12,}" for n in (100, 10_000, 100_000))
    return f"{counts}  {label}"


def run(runs: int = 10) -> int:
    """Measure and print the driver's per-query cost. Returns a process exit code."""
    try:
        import ray  # noqa: F401
    except ImportError:
        print("skip: the distributed driver benchmark needs the optional `ray` extra")
        return 0

    ds = bt.from_arrow(_probe_table())

    def query():
        return ds.group_by("k").agg(s=bt.col("v").sum()).collect(distributed=True)

    for _ in range(_WARMUP):
        query()

    counters = _Counters()
    undo = _install_counters(counters)
    try:
        counters.reset()
        wall: list[float] = []
        cpu: list[float] = []
        for _ in range(runs):
            w0, c0 = time.perf_counter(), time.thread_time()
            query()
            wall.append((time.perf_counter() - w0) * 1000.0)
            cpu.append((time.thread_time() - c0) * 1000.0)
    finally:
        undo()

    topology = counters.topology_calls / runs
    print(f"{runs} distributed queries, {PROBE_ROWS:,} rows, {PROBE_KEYS:,} keys\n")
    print(f"  wall per query            {statistics.median(wall):8.1f} ms")
    print(f"  driver CPU per query      {statistics.median(cpu):8.2f} ms")
    per_s = 1000.0 / max(statistics.median(cpu), 1e-9)
    print(f"  -> one driver sustains    {per_s:8.1f} queries/s\n")
    print("  O(nodes) control-plane round trips per query")
    print(f"    ray.nodes()                    {counters.nodes / runs:6.1f}")
    print(f"    ray.cluster_resources()        {counters.cluster_resources / runs:6.1f}")
    print(f"    available_resources_per_node() {counters.free_cpus / runs:6.1f}")
    print(f"    total                          {topology:6.1f}")
    print(f"    remote-function definitions    {counters.remote_defs / runs:6.1f}\n")
    print("  node-records deserialized on the driver, by fleet size")
    print("   " + _project(round(topology), "node records / query"))
    return 0


def main() -> int:
    return run(int(sys.argv[1]) if len(sys.argv) > 1 else 10)


if __name__ == "__main__":
    raise SystemExit(main())
