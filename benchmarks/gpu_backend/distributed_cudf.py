"""Distributed multi-GPU cuDF (Batcher-style) vs single-GPU cuDF / Polars-GPU.

The honest lesson from `multi_gpu_agg.py`: torch can't out-compute cuDF on one GPU, so the
right architecture is to USE cuDF as the per-GPU data plane and add Batcher's distribution.
The distribution's payoff is SCALE: a single GPU's memory (a 16 GB T4) caps how much data
cuDF / Polars-GPU can hold, so at large N single-GPU cuDF OOMs while N GPUs (here 8 → ~128 GB)
still fit — each GPU runs cuDF on its shard, the driver combines the mergeable partials.

Each actor generates + holds its shard resident *on its own GPU* (no driver-memory or transfer
bottleneck), then cuDF-aggregates. cuDF is shipped per-task via `runtime_env` (Ray caches it
per node after the first install — slow first call only). Correctness-gated.

Run:
    python benchmarks/gpu_backend/distributed_cudf.py                 # sweeps N up to an OOM
    BENCH_DC_N=800000000 python benchmarks/gpu_backend/distributed_cudf.py
"""

from __future__ import annotations

import functools
import os
import time

print = functools.partial(print, flush=True)

# Pin numpy to the cluster's version so numpy arrays returned from the task unpickle on the
# driver (cudf's install otherwise pulls numpy 2.x → `No module named 'numpy._core'` mismatch).
_CUDF_RT = {"pip": ["cudf-cu13==26.6.0", "numpy==1.26.4"]}
_GROUPS = int(os.environ.get("BENCH_DC_GROUPS", "1000"))


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


def _shard_agg(n: int, groups: int, seed: int, runs: int):
    """On a GPU worker: generate `n` rows resident on the GPU, cuDF-aggregate. Returns
    ((keys, sums), best_seconds) or the string 'OOM' if the GPU can't hold the data."""
    import cudf
    import cupy as cp

    try:
        rng = cp.random.default_rng(seed)
        k = rng.integers(0, groups, n)
        v = rng.random(n)
        df = cudf.DataFrame({"k": k, "v": v})
        best = float("inf")
        out = None
        for _ in range(runs):
            t = time.perf_counter()
            out = df.groupby("k").agg({"v": "sum"})
            cp.cuda.runtime.deviceSynchronize()
            best = min(best, time.perf_counter() - t)
        keys = out.index.to_numpy()
        sums = out["v"].to_numpy()
        return (keys, sums), best
    except Exception as e:  # a GPU out-of-memory is the expected outcome at large N
        blob = (type(e).__name__ + " " + str(e)).lower()
        if any(m in blob for m in ("memoryerror", "outofmemory", "bad_alloc", "out of memory")):
            return "OOM"
        raise


def _combine(parts):
    import pandas as pd

    frames = [pd.DataFrame({"k": k, "s": s}) for k, s in parts]
    g = pd.concat(frames).groupby("k", as_index=False)["s"].sum()
    return dict(zip(g["k"].to_numpy(), g["s"].to_numpy(), strict=True))


def main() -> int:
    _init()
    import ray

    n_gpus = max(1, int(ray.cluster_resources().get("GPU", 1)))
    runs = int(os.environ.get("BENCH_RUNS", "3"))
    task = ray.remote(num_gpus=1, runtime_env=_CUDF_RT)(_shard_agg)

    ns = (
        [int(os.environ["BENCH_DC_N"])]
        if os.environ.get("BENCH_DC_N")
        else [200_000_000, 600_000_000, 1_200_000_000, 2_000_000_000]
    )
    print(f"gpus={n_gpus}  groups={_GROUPS}  runs={runs}  (first call installs cuDF, ~min)\n")
    for n in ns:
        # single-GPU: all N on one GPU (cuDF / Polars-GPU's limit).
        single = ray.get(task.remote(n, _GROUPS, 0, runs))
        # distributed: N split across every GPU, each runs cuDF on its shard, driver combines.
        per = -(-n // n_gpus)
        refs = [task.remote(min(per, n - i * per), _GROUPS, i, runs) for i in range(n_gpus)]
        shards = ray.get(refs)
        dist_ok = all(s != "OOM" for s in shards)
        t0 = time.perf_counter()
        dist_map = _combine([s[0] for s in shards]) if dist_ok else None
        combine_ms = (time.perf_counter() - t0) * 1000.0
        dist_wall = (max(s[1] for s in shards) if dist_ok else 0.0) + combine_ms / 1000.0

        if single == "OOM":
            tag = "distributed OK" if dist_ok else "both OOM"
            print(f"N={n / 1e6:6.0f}M  single-GPU cuDF: OOM  |  distributed 8xGPU: {tag}"
                  + (f" ({n / dist_wall / 1e6:.0f} M rows/s)" if dist_ok else ""))
        elif dist_ok:
            # correctness: distributed combined sums match single-GPU
            sk, ss = single[0]
            smap = dict(zip(sk, ss, strict=True))
            err = max(abs(smap[k] - dist_map[k]) for k in smap) if smap else 0.0
            speed = single[1] / dist_wall
            ok = "OK" if err < 1e-3 * max(1, n) else "MISMATCH"
            print(f"N={n / 1e6:6.0f}M  single-GPU {n / single[1] / 1e6:6.0f} M rows/s  |  "
                  f"distributed {n / dist_wall / 1e6:6.0f} M rows/s  -> {speed:.2f}x  [{ok}]")
        else:
            print(f"N={n / 1e6:6.0f}M  single-GPU OK, distributed OOM (unexpected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
