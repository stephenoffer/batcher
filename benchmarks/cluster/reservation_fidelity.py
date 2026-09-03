"""Does the scheduler reserve what the query actually uses? Predicted against measured.

Every distributed stage tells Ray what it needs before it runs: Kyber predicts a per-operator
CPU share (`ResourceBounds.c_cpu_shares`), Carbonite turns it into a grant
(`SchedulingEnvelope.num_cpus`, `n_tasks`), and `dist` reshapes both against the live cluster
(`_fill_grant`, `_even_cpu_share`, `_accelerator_fill_workers`). The product of the final two
is a **reservation**: cores Ray holds for this query and will not offer to anything else.

Nothing here checks that the reservation resembles the work. It is the one number in the
scheduling path with no feedback loop attached, and it is wrong in both directions for
different reasons. Reserve far more than the query uses and the cluster reads as full while
sitting idle, so no co-tenant can be placed and the autoscaler grows a fleet nobody needs.
Reserve far less and the tasks oversubscribe their nodes and contend.

**The engine's own counters cannot answer this**, which is why this benchmark samples the
cluster instead. `benchmarks/cluster/cpu_metrics_fidelity.py` measured `cores_busy` against
the same sampler on a 200M-row aggregate and found 0.96 reported against 54 measured -- four
orders of magnitude -- because most distributed work runs in tasks that never produce a
metered `ExecMetrics` document. So "actual" here is `cluster_util.ClusterMonitor`: one
`num_cpus=0` actor per node reading whole-node `psutil`, which is independent of anything the
engine reports about itself.

**Whole-node sampling means co-tenants are counted**, and this cluster is shared. Each shape
therefore takes an idle baseline immediately before its run and subtracts it, and prints the
baseline so a reader can see how much of the answer it was. A baseline that is a large
fraction of the signal means the run was contaminated and the row should be rerun, not
believed.

Run: ``python benchmarks/cluster/reservation_fidelity.py [--rows N] [--workers W]``
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import batcher as bt
from envinfo import machine_fingerprint, require_release_build

#: Where the fixture lands. A distributed scan needs storage every worker can reach; a
#: driver-local path makes every partition ship from the driver and measures that instead.
_SHARED = "/mnt/cluster_storage"

#: Sampling cadence and how long the idle baseline is sampled for. The baseline needs enough
#: samples to be a mean rather than a reading, and short enough that the pair of them does not
#: dominate a run.
_INTERVAL_S = 0.25
_BASELINE_S = 3.0

#: How long each shape's sampled window must run for. Well above the sampler's cadence and
#: long enough that fleet setup is a small share of it, so the mean describes the query rather
#: than the pauses around it.
_MIN_WINDOW_S = 12.0

#: Quiet time between one shape's window and the next shape's baseline, so the baseline
#: measures the cluster at rest rather than the previous fleet's teardown.
_SETTLE_S = 5.0


@dataclass(frozen=True)
class Row:
    """One shape's prediction, its measurement, and what the cluster was doing anyway."""

    shape: str
    wall_s: float
    reserved_cores: float
    busy_cores: float
    baseline_cores: float
    workers: int
    per_task_cpus: float
    iterations: int
    peak_cores: float

    @property
    def net_busy(self) -> float:
        """Busy cores attributable to the query: measured minus what the cluster was doing
        anyway. Floored at zero -- a negative here means the baseline was the larger of two
        noise readings, which is a measurement to discard rather than a negative utilization."""
        return max(0.0, self.busy_cores - self.baseline_cores)

    @property
    def measurable(self) -> bool:
        """Whether the query moved the cluster enough to be distinguished from it running
        nothing.

        A shape that finishes far inside the sampler's cadence returns the idle cluster, and
        the ratio computed from it is a reading of the noise floor rather than a small
        utilization. Reported as "below noise" instead of as a number, because a spurious
        `0.00x` in this table would be indistinguishable from a real one and is the more
        damaging of the two.
        """
        return self.net_busy > max(1.0, 0.25 * self.baseline_cores)

    @property
    def ratio(self) -> float:
        """Query-attributable busy cores over cores reserved. 1.0 is an honest reservation."""
        return self.net_busy / self.reserved_cores if self.reserved_cores else float("nan")


def _capture_envelopes():
    """Record every scheduling envelope installed, returning `(installs, restore)`.

    The *final* grant is what Ray reserves, and it is not the one Carbonite returns: the
    distributed executor replaces `n_tasks` with a cluster-shaped fill and then raises
    `num_cpus` to an even share of a node. Reading the envelope Carbonite built would report a
    prediction the run never used.
    """
    from batcher.dist.executors.ray_runtime import scheduling

    installs: list = []
    original = scheduling.set_scheduling_envelope

    def _record(env):
        if env is not None:
            installs.append(env)
        return original(env)

    scheduling.set_scheduling_envelope = _record
    # `dist.executor` imported the name directly, so patching the module alone would record
    # nothing -- the caller holds its own reference.
    from batcher.dist import executor as dist_executor

    executor_original = dist_executor.set_scheduling_envelope
    dist_executor.set_scheduling_envelope = _record

    def _restore() -> None:
        scheduling.set_scheduling_envelope = original
        dist_executor.set_scheduling_envelope = executor_original

    return installs, _restore


def _clear(path: str) -> None:
    """Remove `path`, whichever kind of thing it is."""
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        os.remove(path)


def _write_fixture(path: str, rows: int, *, files: int = 32) -> None:
    """Materialize the scan fixture once, on storage every worker can read.

    One file per chunk under a directory, read back as one relation. A plain file sink has no
    table to append to, and the split shape is what the scan parallelizes over anyway -- a
    single file would measure one worker reading it.
    """
    # Removed whether it is a directory or a file: an earlier failed run can leave a single
    # parquet file at this path, and `isdir` alone then skips the cleanup and `makedirs`
    # fails on the leftover.
    _clear(path)
    os.makedirs(path, exist_ok=True)
    step = max(1, -(-rows // files))
    for idx, start in enumerate(range(0, rows, step)):
        n = min(step, rows - start)
        bt.from_pydict(
            {
                "k": [(start + i) % 5000 for i in range(n)],
                "v": [float(start + i) for i in range(n)],
                "w": [float((start + i) % 997) for i in range(n)],
            }
        ).write.parquet(os.path.join(path, f"part-{idx:04d}.parquet"))


def _shapes(path: str):
    """The spread: breaker-free through shuffle-heavy, which predict very differently."""
    src = lambda: bt.read.parquet(path)  # noqa: E731 - a factory, not a lambda in a hot path
    return {
        "scan+filter+project": lambda w: (
            src()
            .filter(bt.col("v") > 10.0)
            .select("k", "w")
            .collect(distributed=True, num_workers=w)
        ),
        "group_by+agg": lambda w: (
            src()
            .group_by("k")
            .agg(s=bt.col("v").sum(), n=bt.col("w").count())
            .collect(distributed=True, num_workers=w)
        ),
        # A shape that actually shuffles, not `sort(...).limit(n)`: the latter lowers to a
        # top-N that keeps a size-n heap per partition and finished in 40 ms, far inside the
        # sampler's cadence, so it measured the idle cluster rather than a sort.
        "sort+groupby": lambda w: (
            src()
            .sort("v", descending=True)
            .group_by("k")
            .agg(n=bt.col("w").count())
            .collect(distributed=True, num_workers=w)
        ),
        "distinct": lambda w: src().select("k").distinct().collect(distributed=True, num_workers=w),
        # The control. Every other shape here is IO- or shuffle-bound and finishes in a
        # fraction of a second, so a low ratio could mean either "the engine over-reserves" or
        # "these queries are too small to fill what they asked for" -- and those call for
        # opposite responses. This one is arithmetic-bound per row over the same data, so it
        # is the shape a 384-core reservation would be *right* about. If its ratio does not
        # rise, the reservation is not tracking demand; if it does, the low ratios above are a
        # statement about the queries rather than about the scheduler.
        "arith-heavy": lambda w: (
            src()
            .with_columns(
                z=(bt.col("v") * 1.0000001 + bt.col("w")).sqrt().log().abs()
                + (bt.col("v") - bt.col("w")).sqrt().exp().abs()
            )
            .group_by("k")
            .agg(m=bt.col("z").mean(), s=bt.col("z").sum())
            .collect(distributed=True, num_workers=w)
        ),
    }


def _measure(name: str, run, workers: int, monitor, total_cores: float, node_cores: float) -> Row:
    """Run one shape repeatedly inside one sampled window, and report both halves.

    **Repeated, because a single run is not measurable.** The sampler's cadence is 250 ms and
    these shapes finish in 90-300 ms, so one run per window returns the idle cluster: measured
    that way every shape reported 3.8-5.6 busy cores against a 4.0-4.7 baseline, which is the
    noise floor and not an answer. Looping until `_MIN_WINDOW_S` has elapsed makes the window
    dominated by query execution rather than by the gap between queries, and amortizes the
    fleet setup that a first run pays.

    The wall time reported is the per-iteration mean, so it stays comparable with any other
    timing of the same shape.
    """
    # Let the previous shape's fleet finish tearing down before sampling the baseline. Without
    # this the idle reading trails the run before it: measured `idle=30.7` against `busy=5.4`
    # for a shape whose own window was quiet, which is the preceding aggregate's teardown and
    # not the cluster's resting state.
    time.sleep(_SETTLE_S)
    monitor.start()
    time.sleep(_BASELINE_S)
    baseline = monitor.stop()["mean_busy_pct"] / 100.0 * total_cores

    installs, restore = _capture_envelopes()
    try:
        monitor.start()
        t0 = time.perf_counter()
        iterations = 0
        while time.perf_counter() - t0 < _MIN_WINDOW_S:
            run(workers)
            iterations += 1
        elapsed = time.perf_counter() - t0
        sample = monitor.stop()
        busy_pct, peak_pct = sample["mean_busy_pct"], sample["peak_busy_pct"]
    finally:
        restore()
    wall = elapsed / max(1, iterations)

    final = installs[-1] if installs else None
    per_task = float(final.num_cpus) if final else float("nan")
    n_tasks = int(final.n_tasks) if final else 0
    return Row(
        shape=name,
        wall_s=wall,
        reserved_cores=per_task * n_tasks if final else float("nan"),
        busy_cores=busy_pct / 100.0 * total_cores,
        baseline_cores=baseline,
        workers=n_tasks,
        per_task_cpus=per_task,
        iterations=iterations,
        # The single busiest node-sample in the window, as cores of one node. It separates the
        # two readings a low mean allows: a fleet that never uses its cores, against one that
        # does but spends most of the window in the query's serial phases (planning, fan-out,
        # result collection) with the cluster idle. Those call for opposite fixes.
        peak_cores=peak_pct / 100.0 * node_cores,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000_000)
    parser.add_argument("--workers", type=int, default=0, help="0 lets the cluster shape decide")
    args = parser.parse_args()
    require_release_build()

    import ray

    from batcher.dist.executors.ray_runtime import _ensure_ray

    _ensure_ray(4)
    total_cores = float(ray.cluster_resources().get("CPU", 0.0))
    # The widest worker node, for the per-node peak. Taken from the node records rather than
    # by dividing the cluster total, because the head node is probed but advertises no CPU.
    node_cores = max(
        (float(n.get("Resources", {}).get("CPU", 0.0)) for n in ray.nodes() if n.get("Alive")),
        default=total_cores,
    )
    if total_cores <= 0:
        print("no cluster CPUs reported; nothing to measure")
        return 1

    if not os.path.isdir(_SHARED):
        print(f"{_SHARED} is not present: a distributed scan needs shared storage")
        return 1
    path = os.path.join(_SHARED, "reservation_fidelity_fixture")
    print(f"{machine_fingerprint()}\ncluster: {total_cores:.0f} cores")
    print(f"writing {args.rows:,}-row fixture to {path} ...")
    _write_fixture(path, args.rows)

    from cluster_util import ClusterMonitor

    monitor = ClusterMonitor(interval_s=_INTERVAL_S)
    rows: list[Row] = []
    try:
        for name, run in _shapes(path).items():
            run(args.workers or 4)  # warm the fleet and the page cache; not measured
            rows.append(_measure(name, run, args.workers or 4, monitor, total_cores, node_cores))
    finally:
        monitor.shutdown()
        _clear(path)

    print(
        f"\n{'shape':<22}{'wall':>7}{'runs':>6}{'wrk':>5}{'cpus/w':>8}"
        f"{'reserved':>10}{'busy':>7}{'idle':>7}{'net':>7}{'pk/node':>9}{'net/reserved':>14}"
    )
    for r in rows:
        verdict = "  below noise" if not r.measurable else f"{r.ratio:>13.2f}x"
        print(
            f"{r.shape:<22}{r.wall_s:>6.2f}s{r.iterations:>6}{r.workers:>5}"
            f"{r.per_task_cpus:>8.2f}{r.reserved_cores:>10.1f}{r.busy_cores:>7.1f}"
            f"{r.baseline_cores:>7.1f}{r.net_busy:>7.1f}{r.peak_cores:>9.1f}{verdict}"
        )
    ratios = [r.ratio for r in rows if r.measurable]
    if ratios:
        print(f"\nmedian net/reserved: {statistics.median(ratios):.2f}x")
        print("1.00x means the reservation matched the work; below it the cluster reads as")
        print("full while idle, above it the tasks oversubscribed their nodes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
