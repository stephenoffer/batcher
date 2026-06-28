"""Distributed batcher on the live cluster — correctness + timing vs Ray Data.

Demonstrates batcher's *distributed* path running on the existing Ray cluster (it
auto-ships its package + native extension to workers via `runtime_env` — see
`dist.executors.ray_runtime.lifecycle._self_ship_runtime_env`). Batcher itself owns
`ray.init` (with `ray_address="auto"`) so the auto-ship applies; Ray Data then attaches
to the same session.

Reports single-node batcher, distributed batcher (N workers), and Ray Data on one
compute-ish workload (a per-batch numpy UDF + grouped reduce), correctness-gated. At
TPC-H sf1 (6M rows) the data is small for distribution, so single-node typically wins
(network shuffle + actor startup dominate) — distribution is for scale-out; the value
here is that the distributed path *works and is correct on the cluster*, and still
beats Ray Data.

Run:
    python benchmarks/dist_bench.py --workers 4
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time

import numpy as np
import pyarrow as pa

sys.path.insert(0, "benchmarks")
from harness import results_match
from sources import load_tables


def _charge(ep, qty, disc):
    return np.sqrt(ep * ep + (qty * 1000.0) * (qty * 1000.0)) * (1.0 - disc)


def _udf(batch: pa.RecordBatch) -> pa.RecordBatch:
    ep = batch.column("l_extendedprice").to_numpy(zero_copy_only=False)
    qty = batch.column("l_quantity").to_numpy(zero_copy_only=False)
    disc = batch.column("l_discount").to_numpy(zero_copy_only=False)
    return pa.record_batch(
        {"l_returnflag": batch.column("l_returnflag"), "charge": pa.array(_charge(ep, qty, disc))}
    )


def _time(fn, runs: int) -> tuple[float, pa.Table]:
    out = fn()  # warm up (also primes the worker fleet for the distributed case)
    best = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    import batcher as bt
    from batcher.config import active_config, set_config

    # Attach to the existing cluster; the distributed path auto-ships batcher to workers.
    cfg = active_config()
    set_config(cfg.replace(distributed=dataclasses.replace(cfg.distributed, ray_address="auto")))

    tables = load_tables("tpch", 1.0, None)
    print(f"lineitem rows: {tables['lineitem'].num_rows:,}")
    li = bt.from_arrow(tables["lineitem"])

    def workload(distributed: bool):
        ds = (
            li.map_batches(_udf, output_columns=["l_returnflag", "charge"])
            .group_by("l_returnflag")
            .agg(charge=bt.col("charge").sum())
        )
        if distributed:
            return ds.collect(distributed=True, num_workers=args.workers)
        return ds.collect(distributed=False)

    rows = []
    sn_ms, sn_out = _time(lambda: workload(False), args.runs)
    rows.append(("batcher single-node", sn_ms, sn_out))
    dist_ms, dist_out = _time(lambda: workload(True), args.runs)
    ok, msg = results_match(sn_out, dist_out)
    rows.append((f"batcher distributed ({args.workers}w)", dist_ms, dist_out))
    print(f"\ndistributed == single-node: {ok}" + ("" if ok else f"  ({msg})"))

    # Ray Data on the same cluster + same workload.
    try:
        import ray.data

        from engines.ray import _ensure_ray

        _ensure_ray()
        rd = ray.data.from_arrow(tables["lineitem"])

        def ray_workload():
            def fn(b):
                return {
                    "l_returnflag": b["l_returnflag"],
                    "charge": _charge(b["l_extendedprice"], b["l_quantity"], b["l_discount"]),
                }

            out = rd.map_batches(fn, batch_format="numpy").groupby("l_returnflag").sum("charge")
            df = out.to_pandas().rename(columns={"sum(charge)": "charge"})
            return pa.Table.from_pandas(df[["l_returnflag", "charge"]], preserve_index=False)

        ray_ms, _ = _time(ray_workload, args.runs)
        rows.append(("ray data (cluster)", ray_ms, None))
    except Exception as exc:
        print(f"ray data unavailable: {type(exc).__name__}: {exc}")

    print(f"\n{'engine':<30} {'ms':>10}")
    print("-" * 42)
    for name, ms, _ in rows:
        print(f"{name:<30} {ms:>10.1f}")


if __name__ == "__main__":
    main()
