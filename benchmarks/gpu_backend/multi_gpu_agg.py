"""Multi-GPU persistent-actor aggregate vs single-GPU cuDF (Polars-GPU's backend).

cuDF / Polars-GPU run on ONE GPU. Batcher has many. The lever to beat them: a pool of
persistent GPU actors, each holding a resident shard of the data, aggregating in parallel; the
driver combines the mergeable partials. Warm (data resident, model of an iterative query) the
pool's throughput is ~N_gpus x a single GPU's, so 8 T4s should beat single-GPU cuDF well past 2x.

Compares, on the same 20M-row / 200k-group aggregate, all GPU-resident (warm):
  * Batcher multi-GPU persistent actors (this prototype)
  * cuDF single-GPU (installed per-task via runtime_env; the engine behind Polars `engine="gpu"`)

Correctness-gated: the multi-GPU per-group sums match a single-GPU reference before timing.

Run:
    python benchmarks/gpu_backend/multi_gpu_agg.py
"""

from __future__ import annotations

import functools
import os
import time

import numpy as np
from _ray_env import init_ray

print = functools.partial(print, flush=True)

_SEED = 1234


def _cfg() -> dict:
    return {
        "n": int(os.environ.get("BENCH_MGA_N", "20000000")),
        "groups": int(os.environ.get("BENCH_MGA_GROUPS", "200000")),
        "runs": int(os.environ.get("BENCH_RUNS", "5")),
    }


def _make_shard_actor():
    import ray

    @ray.remote(num_gpus=1)
    class ShardAgg:
        """A persistent GPU actor: loads its shard resident once, aggregates on demand."""

        def load(self, keys_np, vals_np) -> int:
            import torch

            self._k = torch.from_numpy(keys_np).cuda()
            self._v = torch.from_numpy(vals_np).cuda()
            torch.cuda.synchronize()
            return int(self._k.numel())

        def agg(self):
            import torch

            uniq, inv = torch.unique(self._k, return_inverse=True)
            n = int(uniq.numel())
            s = torch.zeros(n, device="cuda", dtype=torch.float64).scatter_add_(0, inv, self._v)
            c = torch.zeros(n, device="cuda", dtype=torch.float64).scatter_add_(
                0, inv, torch.ones_like(self._v)
            )
            torch.cuda.synchronize()
            return uniq.cpu().numpy(), s.cpu().numpy(), c.cpu().numpy()

    return ShardAgg


def _combine(parts):
    """Combine per-actor (keys, sums, counts) partials into global per-group sum/count."""
    import pandas as pd

    frames = [pd.DataFrame({"k": k, "s": s, "c": c}) for k, s, c in parts]
    g = pd.concat(frames).groupby("k", as_index=False).agg({"s": "sum", "c": "sum"})
    return dict(zip(g["k"].to_numpy(), g["s"].to_numpy(), strict=True))


def _cudf_agg(n, groups, runs):
    import cudf
    import numpy as np

    rng = np.random.default_rng(_SEED)
    df = cudf.DataFrame({"k": rng.integers(0, groups, n), "v": rng.random(n)})
    best = float("inf")
    out = None
    for _ in range(runs):
        t = time.perf_counter()
        out = df.groupby("k").agg({"v": "sum"})
        _ = out["v"].sum()  # force materialization
        best = min(best, time.perf_counter() - t)
    return best


def main() -> int:
    cfg = _cfg()
    init_ray()
    import ray

    n, g, runs = cfg["n"], cfg["groups"], cfg["runs"]
    rng = np.random.default_rng(_SEED)
    keys = rng.integers(0, g, n).astype(np.int64)
    vals = rng.random(n).astype(np.float64)
    n_gpus = int(ray.available_resources().get("GPU", ray.cluster_resources().get("GPU", 1)))
    n_gpus = max(1, n_gpus)
    print(f"rows={n}  groups={g}  gpus={n_gpus}  runs={runs}\n")

    # Persistent GPU actor pool: one actor per GPU, each holds a resident shard.
    ShardAgg = _make_shard_actor()
    actors = [ShardAgg.remote() for _ in range(n_gpus)]
    bounds = np.linspace(0, n, n_gpus + 1).astype(int)
    ray.get(
        [
            actors[i].load.remote(keys[bounds[i] : bounds[i + 1]], vals[bounds[i] : bounds[i + 1]])
            for i in range(n_gpus)
        ]
    )

    # Warm multi-GPU aggregate: all actors aggregate their resident shard in parallel, combine.
    def multi_gpu_run():
        parts = ray.get([a.agg.remote() for a in actors])
        return _combine(parts)

    ref = multi_gpu_run()  # warm-up + correctness reference
    ref_full = {}
    # single-GPU reference sums for correctness (compute on CPU here — exact)
    import pandas as pd

    cpu = pd.DataFrame({"k": keys, "v": vals}).groupby("k")["v"].sum()
    ref_full = dict(zip(cpu.index.to_numpy(), cpu.to_numpy(), strict=True))
    max_err = max(abs(ref[k] - ref_full[k]) for k in ref_full)
    print(f"multi-GPU per-group sum agreement: max abs err {max_err:.2e}")
    if max_err > 1e-3:
        print("FAIL: multi-GPU sums diverge")
        return 1

    best_mg = float("inf")
    for _ in range(runs):
        t = time.perf_counter()
        multi_gpu_run()
        best_mg = min(best_mg, time.perf_counter() - t)
    mg_r = n / best_mg / 1e6
    print(f"batcher multi-GPU ({n_gpus}x)  {mg_r:8.1f} M rows/s  ({best_mg * 1000:.1f} ms)")

    # Release the persistent actors so their GPUs free up — otherwise the single-GPU cuDF task
    # (num_gpus=1) can never schedule (the whole cluster's GPUs are held by the pool).
    for a in actors:
        ray.kill(a)
    time.sleep(3)

    # cuDF single-GPU (installed per-task in an isolated runtime_env — slow first call), resident.
    cudf_task = ray.remote(num_gpus=1, runtime_env={"pip": ["cudf-cu13==26.6.0"]})(_cudf_agg)
    try:
        best_cudf = ray.get(cudf_task.remote(n, g, runs))
    except Exception as e:
        best_cudf = None
        print(f"cuDF unavailable: {type(e).__name__}")

    if best_cudf is not None:
        cd_r = n / best_cudf / 1e6
        print(f"cudf single-GPU            {cd_r:8.1f} M rows/s  ({best_cudf * 1000:.1f} ms)")
        print(f"\nBatcher {n_gpus}xGPU vs cuDF single-GPU: {best_cudf / best_mg:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
