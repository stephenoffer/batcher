"""TPC-H Q6 on GPU (fused scan→filter→revenue-sum) vs Batcher's CPU engine.

Q6 is the canonical scan-heavy TPC-H query: filter `lineitem` on a shipdate range, a discount
band, and a quantity cap, then `sum(extendedprice * discount)`. It is a **fused pipeline** —
filter + multiply + reduce over the same columns — so it is the case a GPU backend wins on:
one host→device transfer amortized over several ops, unlike a single op where PCIe dominates
(see `tpch_gpu_agg.py`). The GPU path uses torch (the env's CUDA-13 vehicle; cuDF-cu13 is the
richer backend once the cluster env syncs it to workers) and runs on a GPU worker via Ray.

Correctness-gated: the GPU revenue must match Batcher's before any timing.

Run:
    python benchmarks/gpu_backend/tpch_q6_gpu.py                 # 100M lineitem rows
    BENCH_Q6_N=200000000 python benchmarks/gpu_backend/tpch_q6_gpu.py
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
# Q6 constants (TPC-H standard-ish): shipdate in [D1, D2), discount in [0.05, 0.07], qty < 24.
_D1, _D2 = 8766, 9131  # day indices (a one-year window)
_DISC_LO, _DISC_HI, _QTY_MAX = 0.05, 0.07, 24.0


def _cfg() -> dict:
    return {
        "n": int(os.environ.get("BENCH_Q6_N", "100000000")),
        "runs": int(os.environ.get("BENCH_RUNS", "3")),
    }


def _gpu_q6(shipdate, discount, quantity, extprice, runs: int) -> tuple[float, float, float]:
    """Fused TPC-H Q6 on the GPU. Returns (revenue, best_compute_s, best_e2e_s)."""
    import torch

    dev = torch.device("cuda")
    rev = 0.0
    best_compute = float("inf")
    best_e2e = float("inf")
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sd = torch.from_numpy(shipdate).to(dev)
        di = torch.from_numpy(discount).to(dev)
        qt = torch.from_numpy(quantity).to(dev)
        ep = torch.from_numpy(extprice).to(dev)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        mask = (sd >= _D1) & (sd < _D2) & (di >= _DISC_LO) & (di <= _DISC_HI) & (qt < _QTY_MAX)
        rev_t = torch.sum(torch.where(mask, ep * di, torch.zeros((), device=dev, dtype=ep.dtype)))
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        rev = float(rev_t.cpu())
        t3 = time.perf_counter()
        best_compute = min(best_compute, t2 - t1)
        best_e2e = min(best_e2e, t3 - t0)
    return rev, best_compute, best_e2e


def main() -> int:
    cfg = _cfg()
    init_ray()
    import ray

    import batcher as bt
    from batcher import col

    n = cfg["n"]
    rng = np.random.default_rng(_SEED)
    shipdate = rng.integers(8000, 9500, size=n).astype(np.int64)
    discount = np.round(rng.uniform(0.0, 0.10, size=n), 2).astype(np.float64)
    quantity = rng.integers(1, 51, size=n).astype(np.float64)
    extprice = rng.uniform(1000.0, 100000.0, size=n).astype(np.float64)
    print(f"lineitem rows={n}  runs={cfg['runs']}\n")

    tbl = pa.table(
        {"shipdate": shipdate, "discount": discount, "quantity": quantity, "extprice": extprice}
    )

    # Batcher CPU engine: filter + sum(extprice * discount), all native.
    def batcher_run():
        return (
            bt.from_arrow(tbl)
            .filter(
                (col("shipdate") >= _D1)
                & (col("shipdate") < _D2)
                & (col("discount") >= _DISC_LO)
                & (col("discount") <= _DISC_HI)
                & (col("quantity") < _QTY_MAX)
            )
            .agg(revenue=(col("extprice") * col("discount")).sum())
            .collect()
        )

    b_out = batcher_run()
    b_rev = b_out.column("revenue")[0].as_py()
    best_cpu = float("inf")
    for _ in range(cfg["runs"]):
        t0 = time.perf_counter()
        batcher_run()
        best_cpu = min(best_cpu, time.perf_counter() - t0)

    gpu_task = ray.remote(num_gpus=1)(_gpu_q6)
    g_rev, best_compute, best_e2e = ray.get(
        gpu_task.remote(shipdate, discount, quantity, extprice, cfg["runs"])
    )

    rel_err = abs(b_rev - g_rev) / max(abs(b_rev), 1.0)
    print(f"Q6 revenue agreement: batcher={b_rev:.2f} gpu={g_rev:.2f}  rel_err={rel_err:.2e}")
    if rel_err > 1e-6:
        print("FAIL: revenue diverges — not reporting")
        return 1

    cpu_r, e2e_r, comp_r = n / best_cpu / 1e6, n / best_e2e / 1e6, n / best_compute / 1e6
    print(f"batcher-cpu       {cpu_r:8.1f} M rows/s  ({best_cpu * 1000:.0f} ms)")
    print(f"torch-gpu e2e     {e2e_r:8.1f} M rows/s  ({best_e2e * 1000:.1f} ms)  [+H2D transfer]")
    print(f"torch-gpu compute {comp_r:8.1f} M rows/s  ({best_compute * 1000:.1f} ms)  [resident]")
    print(
        f"\nfused Q6 (filter+revenue) GPU vs Batcher CPU: {best_cpu / best_e2e:.1f}x end-to-end, "
        f"{best_cpu / best_compute:.0f}x compute-only"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
