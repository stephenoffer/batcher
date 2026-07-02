"""GPU-accelerated transform vs Batcher's CPU engine — group-by aggregation (TPC-H Q1 core).

The first step of the CPU-and-GPU-backends goal: show a core relational transform — a keyed
group-by SUM/COUNT (the heart of TPC-H Q1) — running on the GPU and compare it to Batcher's
native CPU engine on the same data.

The GPU path uses **torch** (already CUDA-13 in this env; RAPIDS cuDF is the richer backend but
needs the cluster env to sync `cudf-cu13`): a group-by is `scatter_add_` over the key column —
a memory-bandwidth-bound reduction the T4 does well. Runs on a GPU worker via Ray so the driver
(no GPU) orchestrates. Correctness-gated: the GPU per-group sums must match Batcher's before any
timing.

Run:
    python benchmarks/gpu_backend/tpch_gpu_agg.py            # 50M rows, 1000 groups
    BENCH_GPUAGG_N=100000000 python benchmarks/gpu_backend/tpch_gpu_agg.py
"""

from __future__ import annotations

import functools
import os
import time

import numpy as np
import pyarrow as pa

print = functools.partial(print, flush=True)

_SEED = 1234


def _cfg() -> dict:
    return {
        "n": int(os.environ.get("BENCH_GPUAGG_N", "50000000")),
        "groups": int(os.environ.get("BENCH_GPUAGG_GROUPS", "1000")),
        "runs": int(os.environ.get("BENCH_RUNS", "3")),
    }


def _init() -> None:
    import importlib.util

    for var in ("RAY_RUNTIME_ENV_HOOK", "RAY_RUNTIME_ENV_PLUGINS"):
        v = os.environ.get(var)
        if v:
            head = v.lstrip("[{\"' ").split(".")[0].split("[")[0]
            if head and importlib.util.find_spec(head) is None:
                os.environ.pop(var, None)
    import ray

    if not ray.is_initialized():
        ray.init(
            address="auto", runtime_env={"pip": None}, logging_level="ERROR", log_to_driver=False
        )


def _gpu_groupby_sum(keys_np, vals_np, n_groups: int, runs: int) -> tuple[list[float], float, float]:
    """A group-by SUM on the GPU via torch scatter_add.

    Returns `(per_group_sums, best_compute_s, best_e2e_s)`: compute-only (data already resident
    on the GPU — the ceiling when ops are fused so one transfer is amortized) and end-to-end
    (host→device transfer + compute + device→host result — the honest cost of a single op)."""
    import torch

    dev = torch.device("cuda")
    best_compute = float("inf")
    best_e2e = float("inf")
    out = None
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        keys = torch.from_numpy(keys_np).to(dev)
        vals = torch.from_numpy(vals_np).to(dev)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out = torch.zeros(n_groups, device=dev, dtype=torch.float64).scatter_add_(0, keys, vals)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        result = out.cpu().numpy()
        t3 = time.perf_counter()
        best_compute = min(best_compute, t2 - t1)
        best_e2e = min(best_e2e, t3 - t0)
    return result.tolist(), best_compute, best_e2e


def main() -> int:
    cfg = _cfg()
    _init()
    import ray

    import batcher as bt
    from batcher import col

    n, g = cfg["n"], cfg["groups"]
    rng = np.random.default_rng(_SEED)
    keys = (rng.integers(0, g, size=n)).astype(np.int64)
    vals = rng.random(n).astype(np.float64)
    print(f"rows={n}  groups={g}  runs={cfg['runs']}\n")

    tbl = pa.table({"k": keys, "v": vals})

    # Batcher CPU engine (native Rust group-by, all driver cores).
    def batcher_run():
        return bt.from_arrow(tbl).group_by("k").agg(s=col("v").sum()).collect()

    b_out = batcher_run()
    b_map = dict(zip(b_out.column("k").to_pylist(), b_out.column("s").to_pylist(), strict=True))
    best_cpu = float("inf")
    for _ in range(cfg["runs"]):
        t0 = time.perf_counter()
        batcher_run()
        best_cpu = min(best_cpu, time.perf_counter() - t0)

    # GPU transform on a worker (the driver has no GPU).
    gpu_task = ray.remote(num_gpus=1)(_gpu_groupby_sum)
    gpu_sums, best_compute, best_e2e = ray.get(gpu_task.remote(keys, vals, g, cfg["runs"]))

    max_err = max(abs(b_map.get(i, 0.0) - gpu_sums[i]) for i in range(g))
    print(f"per-group sum agreement over {g} groups: max abs err {max_err:.2e}")
    if max_err > 1e-3:
        print("FAIL: GPU and CPU group sums diverge — not reporting")
        return 1

    print(f"batcher-cpu       {n / best_cpu / 1e6:8.1f} M rows/s  ({best_cpu * 1000:.0f} ms)")
    print(f"torch-gpu e2e     {n / best_e2e / 1e6:8.1f} M rows/s  ({best_e2e * 1000:.1f} ms)  [+H2D transfer]")
    print(f"torch-gpu compute {n / best_compute / 1e6:8.1f} M rows/s  ({best_compute * 1000:.1f} ms)  [data resident]")
    print(
        f"\nGPU vs Batcher CPU: {best_cpu / best_e2e:.1f}x end-to-end (single op, transfer-bound), "
        f"{best_cpu / best_compute:.0f}x compute-only (fused/resident ceiling)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
