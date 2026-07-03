"""Cluster-wide GPU utilization during distributed inference — is EVERY GPU saturated?

The goal is close-to-full GPU capacity across the WHOLE cluster, not just one device. This runs
compute-bound inference on one actor per GPU, each sampling its OWN GPU via NVML during a steady
window, and reports per-GPU + aggregate utilization. A starved GPU (imbalance, feeding gap,
dispatch stall) shows up as a low per-device number even when the total looks busy.

Run:
    python benchmarks/gpu_backend/cluster_gpu_util.py
    BENCH_CU_BATCH=256 BENCH_CU_ITERS=200 python benchmarks/gpu_backend/cluster_gpu_util.py
"""

from __future__ import annotations

import functools
import os
import time

print = functools.partial(print, flush=True)


def _cfg() -> dict:
    return {
        "batch": int(os.environ.get("BENCH_CU_BATCH", "128")),
        "iters": int(os.environ.get("BENCH_CU_ITERS", "150")),
        "warmup": int(os.environ.get("BENCH_CU_WARMUP", "20")),
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


def _worker(gpu_id: int, batch: int, iters: int, warmup: int) -> dict:
    """Run compute-bound inference on one GPU while sampling its NVML utilization."""
    import threading

    import numpy as np
    import torch

    dev = torch.device("cuda")
    model = __import__("torchvision").models.resnet50().to(dev).eval().half()

    samples: list[int] = []
    stop = threading.Event()

    def sample():
        try:
            import pynvml

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            while not stop.is_set():
                samples.append(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
                time.sleep(0.05)
            pynvml.nvmlShutdown()
        except Exception:
            pass

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()

    host = torch.from_numpy(np.random.rand(batch, 3, 224, 224).astype(np.float16)).pin_memory()
    copy_stream = torch.cuda.Stream()
    done = 0
    with torch.no_grad():
        nxt = host.to(dev, non_blocking=True)
        for i in range(iters):
            cur = nxt
            with torch.cuda.stream(copy_stream):
                nxt = host.to(dev, non_blocking=True)
            _ = model(cur)
            torch.cuda.synchronize()
            if i >= warmup:
                done += batch
    stop.set()
    sampler.join(timeout=1.0)
    steady = samples[len(samples) * warmup // max(1, iters) :]
    mean = sum(steady) / len(steady) if steady else 0.0
    return {"gpu": gpu_id, "mean_util": mean, "imgs": done}


def main() -> int:
    cfg = _cfg()
    _init()
    import ray

    n_gpus = max(1, int(ray.cluster_resources().get("GPU", 1)))
    print(f"model=resnet50(fp16)  gpus={n_gpus}  batch={cfg['batch']}  iters={cfg['iters']}\n")

    task = ray.remote(num_gpus=1)(_worker)
    t0 = time.perf_counter()
    results = ray.get([task.remote(i, cfg["batch"], cfg["iters"], cfg["warmup"]) for i in range(n_gpus)])
    wall = time.perf_counter() - t0

    utils = [r["mean_util"] for r in results]
    for r in sorted(results, key=lambda r: r["gpu"]):
        print(f"  gpu[{r['gpu']}]  util={r['mean_util']:5.1f}%")
    agg = sum(utils) / len(utils)
    total_imgs = sum(r["imgs"] for r in results)
    flag = "  >= FULL CAPACITY" if agg >= 90 and min(utils) >= 85 else ""
    print(
        f"\ncluster GPU util: mean={agg:.1f}%  min={min(utils):.1f}%  max={max(utils):.1f}%{flag}"
    )
    print(f"cluster throughput: {total_imgs / wall:.0f} img/s across {n_gpus} GPUs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
