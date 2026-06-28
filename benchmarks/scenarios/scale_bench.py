"""At-scale distributed benchmark: batcher across the whole cluster vs Daft.

The sf1 suites measure single-node engine quality; this measures **scale-out** — the
reason a distributed engine exists. It reads TPC-H lineitem at a large scale factor
directly from S3 (no driver materialization) and runs a scan-heavy aggregation, then
compares:

- **batcher distributed** — fanned across the cluster's worker nodes (here 8 nodes x
  16 cores = 128 CPUs) via the Flight shuffle. A persistent worker fleet is installed
  for the run so per-query fleet-spawn overhead is amortized (what a long-lived Session
  would do). Per-node row counts are reported so work-distribution skew is visible.
- **Daft native** — its fast multithreaded local engine on the driver node (16 cores).
- **batcher single-node** — the same 16-core driver, for the speedup reference.

The head node has 0 schedulable task CPUs (Anyscale reserves it), so Daft-native and
batcher-single-node run on the head's 16 physical cores while batcher-distributed uses
the 8 worker nodes — the whole point: bring 128 CPUs to bear.

Run:
    python benchmarks/scenarios/scale_bench.py --scale 10 --workers 8
    python benchmarks/scenarios/scale_bench.py --scale 100 --workers 8 --engines batcher-dist,daft
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time

import pyarrow as pa

sys.path.insert(0, "benchmarks")
from harness import results_match

_PKG = "/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/batcher/python/batcher"
_REGION = "us-west-2"
# positional -> canonical for the lineitem columns the aggregation needs
_REN = {
    "column04": "l_quantity",
    "column05": "l_extendedprice",
    "column06": "l_discount",
    "column07": "l_tax",
    "column08": "l_returnflag",
    "column09": "l_linestatus",
}


def _glob(scale: int) -> str:
    return f"s3://ray-benchmark-data/tpch/parquet/sf{scale}/lineitem/*.parquet"


def _time(fn, runs: int) -> tuple[float, pa.Table]:
    out = fn()
    best = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best, out


def _batcher(scale: int, workers: int, distributed: bool):
    import batcher as bt
    from batcher.config import active_config, set_config

    rt = {"py_modules": [_PKG], "env_vars": {"AWS_DEFAULT_REGION": _REGION, "AWS_REGION": _REGION}}
    cfg = active_config()
    set_config(
        cfg.replace(
            distributed=dataclasses.replace(cfg.distributed, ray_address="auto", runtime_env=rt)
        )
    )
    ds = bt.read.parquet(_glob(scale)).select(*[bt.col(k).alias(v) for k, v in _REN.items()])
    if distributed:
        # Persistent worker fleet for the whole run (a long-lived Session amortizes the
        # ~per-query fleet-spawn this way). Set the envelope FIRST — with the same even
        # CPU share `execute_distributed` now applies — so the pre-spawned actors get a
        # full node's cores instead of the cgroup-starved 1-core default.
        from batcher.dist.executor import _even_cpu_share
        from batcher.dist.executors.ray_runtime import _ensure_ray, engine_config_json
        from batcher.dist.executors.ray_runtime.scheduling import set_scheduling_envelope
        from batcher.dist.fleet import _fleet as F
        from batcher.dist.flight_aggregate import _shuffle_credits
        from batcher.plan.resource import SchedulingEnvelope

        _ensure_ray(workers)
        set_scheduling_envelope(
            SchedulingEnvelope(num_cpus=_even_cpu_share(workers), n_tasks=workers)
        )
        F.set_fleet(F.ShuffleFleet.spawn(workers, _shuffle_credits(), engine_config_json()))

    def q():
        return (
            ds.group_by("l_returnflag", "l_linestatus")
            .agg(
                rev=(bt.col("l_extendedprice") * (1 - bt.col("l_discount"))).sum(),
                charge=(
                    bt.col("l_extendedprice") * (1 - bt.col("l_discount")) * (1 + bt.col("l_tax"))
                ).sum(),
                qty=bt.col("l_quantity").sum(),
                n=bt.col("l_quantity").count(),
            )
            .collect(distributed=distributed, num_workers=workers)
        )

    return q


def _daft(scale: int):
    import os

    os.environ.setdefault("DAFT_RUNNER", "native")
    import daft
    from daft import col

    def q():
        # Fresh frame each call so the timed run re-reads S3 (cold), matching batcher,
        # which re-reads every query — Daft otherwise caches the (projected, RAM-fitting)
        # read in memory after the warmup and the timed run never touches S3, an unfair
        # cache hit batcher's distributed re-read can't get.
        ddf = daft.read_parquet(_glob(scale)).select(*[col(k).alias(v) for k, v in _REN.items()])
        net = col("l_extendedprice") * (1 - col("l_discount"))
        return (
            ddf.groupby("l_returnflag", "l_linestatus")
            .agg(
                net.sum().alias("rev"),
                (net * (1 + col("l_tax"))).sum().alias("charge"),
                col("l_quantity").sum().alias("qty"),
                col("l_quantity").count().alias("n"),
            )
            .to_arrow()
        )

    return q


def _daft_dist(scale: int):
    """Daft on its Ray runner across the whole cluster.

    Daft is not on the worker image, so — exactly like batcher — its package + native
    extension are shipped to workers via Ray's job-level `runtime_env` py_modules. Ray is
    initialized here (with that env) before Daft's Ray runner is selected, so the flotilla
    workers can import daft. Run this engine in its own process (the runner is global).
    """
    import os

    os.environ["DAFT_RUNNER"] = "ray"
    import daft
    import ray

    daft_pkg = os.path.dirname(daft.__file__)
    if not ray.is_initialized():
        ray.init(
            address="auto",
            ignore_reinit_error=True,
            log_to_driver=False,
            runtime_env={
                "py_modules": [daft_pkg],
                "env_vars": {"AWS_DEFAULT_REGION": _REGION, "AWS_REGION": _REGION},
            },
        )
    daft.set_runner_ray()
    from daft import col

    def q():
        # Fresh frame each call → cold S3 read each timed run (fair vs batcher's re-read).
        ddf = daft.read_parquet(_glob(scale)).select(*[col(k).alias(v) for k, v in _REN.items()])
        net = col("l_extendedprice") * (1 - col("l_discount"))
        return (
            ddf.groupby("l_returnflag", "l_linestatus")
            .agg(
                net.sum().alias("rev"),
                (net * (1 + col("l_tax"))).sum().alias("charge"),
                col("l_quantity").sum().alias("qty"),
                col("l_quantity").count().alias("n"),
            )
            .to_arrow()
        )

    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--engines", default="batcher-sn,batcher-dist,daft")
    args = ap.parse_args()
    names = args.engines.split(",")
    print(f"scale=sf{args.scale}  workers={args.workers}  engines={names}")

    results: dict[str, tuple[float, pa.Table]] = {}
    if "daft" in names:
        try:
            results["daft"] = _time(_daft(args.scale), args.runs)
            print(f"daft native: {results['daft'][0]:.0f}ms", flush=True)
        except Exception as exc:
            print(f"daft FAILED: {type(exc).__name__}: {exc}")
    if "batcher-sn" in names:
        try:
            results["batcher-sn"] = _time(_batcher(args.scale, args.workers, False), args.runs)
            print(f"batcher single-node(16c): {results['batcher-sn'][0]:.0f}ms", flush=True)
        except Exception as exc:
            print(f"batcher-sn FAILED: {type(exc).__name__}: {exc}")
    if "daft-dist" in names:
        try:
            results["daft-dist"] = _time(_daft_dist(args.scale), args.runs)
            print(f"daft distributed (ray runner): {results['daft-dist'][0]:.0f}ms", flush=True)
        except Exception as exc:
            print(f"daft-dist FAILED: {type(exc).__name__}: {exc}")
    if "batcher-dist" in names:
        try:
            results["batcher-dist"] = _time(_batcher(args.scale, args.workers, True), args.runs)
            print(
                f"batcher distributed({args.workers}w): {results['batcher-dist'][0]:.0f}ms",
                flush=True,
            )
        except Exception as exc:
            print(f"batcher-dist FAILED: {type(exc).__name__}: {exc}")

    # correctness gate across whatever produced a result
    ref_name = next(iter(results), None)
    if ref_name:
        ref = results[ref_name][1]
        for n, (_, out) in results.items():
            ok, msg = results_match(ref, out)
            print(f"  correctness {n} vs {ref_name}: {'OK' if ok else 'MISMATCH: ' + msg}")

    print("\nsummary (ms, lower better):")
    for n, (ms, _) in results.items():
        print(f"  {n:<26} {ms:>10.0f}")
    if "batcher-dist" in results and "daft" in results:
        print(f"\n  daft / batcher-dist = {results['daft'][0] / results['batcher-dist'][0]:.2f}x")


if __name__ == "__main__":
    main()
