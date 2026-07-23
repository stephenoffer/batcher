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
from _ray_env import init_ray

print = functools.partial(print, flush=True)

_SEED = 1234


def _cfg() -> dict:
    return {
        "n": int(os.environ.get("BENCH_GPUAGG_N", "50000000")),
        "groups": int(os.environ.get("BENCH_GPUAGG_GROUPS", "1000")),
        "runs": int(os.environ.get("BENCH_RUNS", "3")),
    }


def _gpu_groupby_sum(keys_np, vals_np, runs: int) -> tuple[dict, float]:
    """Run Batcher's GPU kernel (`core.gpu_transform.gpu_groupby_agg`) on a GPU worker.

    Returns `(key->sum map, best_end_to_end_seconds)` — the honest single-op cost including the
    host→device transfer done inside the kernel. Exercises the productized module on real GPU
    hardware (the module's algorithm is separately unit-tested on CPU-torch vs the CPU engine)."""
    import pyarrow as pa

    from batcher.core.gpu_transform import gpu_groupby_agg

    tbl = pa.table({"k": keys_np, "v": vals_np})
    out = None
    best_e2e = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        out = gpu_groupby_agg(tbl, "k", {"s": ("v", "sum")})
        best_e2e = min(best_e2e, time.perf_counter() - t0)
    d = out.to_pydict()
    return dict(zip(d["k"], d["s"], strict=True)), best_e2e


def main() -> int:
    cfg = _cfg()
    init_ray()
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

    # GPU transform on a worker (the driver has no GPU), running the real Batcher kernel.
    # `worker_runtime_env()` ships the driver's batcher (package + native .so) so the GPU
    # worker can `import batcher.core.gpu_transform` regardless of Ray init order.
    from batcher.dist.executors.ray_runtime.scheduling import worker_runtime_env

    opts = {"num_gpus": 1}
    rt = worker_runtime_env()
    if rt is not None:
        opts["runtime_env"] = rt
    gpu_task = ray.remote(**opts)(_gpu_groupby_sum)
    gpu_sums, best_e2e = ray.get(gpu_task.remote(keys, vals, cfg["runs"]))

    max_err = max(abs(b_map.get(i, 0.0) - gpu_sums.get(i, 0.0)) for i in range(g))
    print(f"per-group sum agreement over {g} groups: max abs err {max_err:.2e}")
    if max_err > 1e-3:
        print("FAIL: GPU and CPU group sums diverge — not reporting")
        return 1

    cpu_r, e2e_r = n / best_cpu / 1e6, n / best_e2e / 1e6
    print(f"batcher-cpu           {cpu_r:8.1f} M rows/s  ({best_cpu * 1000:.0f} ms)")
    print(f"batcher-gpu (kernel)  {e2e_r:8.1f} M rows/s  ({best_e2e * 1000:.1f} ms)  [end-to-end]")
    print(f"\nGPU group-by (core.gpu_transform) vs Batcher CPU: {best_cpu / best_e2e:.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
