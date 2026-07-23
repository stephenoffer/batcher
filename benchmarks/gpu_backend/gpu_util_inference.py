"""Measure steady-state GPU utilization for compute-bound inference (target: >=90%).

The goal's utilization half: keep the GPU busy. A compute-bound model (ResNet-50 at a real
batch size) with CPU preprocessing overlapped against the GPU forward pass should saturate the
device. This runs the model on a GPU worker while an NVML thread samples utilization every
50 ms, and reports the steady-state mean/p50 (warm-up skipped). Two feeding modes:

* **overlapped** — a background thread prepares (H2D-copies) batch k+1 on a second CUDA stream
  while the model runs batch k, so the GPU never waits on the host copy.
* **naive** — copy-then-compute serially (the gap that drops utilization).

Run:
    python benchmarks/gpu_backend/gpu_util_inference.py
    BENCH_UTIL_BATCH=256 BENCH_UTIL_ITERS=200 python benchmarks/gpu_backend/gpu_util_inference.py
"""

from __future__ import annotations

import functools
import os
import time

from _ray_env import init_ray

print = functools.partial(print, flush=True)


def _cfg() -> dict:
    return {
        "batch": int(os.environ.get("BENCH_UTIL_BATCH", "128")),
        "iters": int(os.environ.get("BENCH_UTIL_ITERS", "150")),
        "warmup": int(os.environ.get("BENCH_UTIL_WARMUP", "20")),
    }


def _run_inference(batch: int, iters: int, warmup: int, overlap: bool) -> dict:
    """Run ResNet-50 inference on the GPU, sampling utilization via NVML. Returns util stats."""
    import threading

    import numpy as np
    import torch

    dev = torch.device("cuda")
    model = __import__("torchvision").models.resnet50().to(dev).eval().half()

    # NVML utilization sampler (a background thread; 50 ms cadence).
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

    host = torch.from_numpy(np.random.rand(batch, 3, 224, 224).astype(np.float16))
    host = host.pin_memory()
    copy_stream = torch.cuda.Stream()
    steady_ms: list[float] = []
    with torch.no_grad():
        next_gpu = host.to(dev, non_blocking=True)
        for i in range(iters):
            cur = next_gpu
            if overlap:
                # Stage the next batch's H2D copy on a side stream while the model runs.
                with torch.cuda.stream(copy_stream):
                    next_gpu = host.to(dev, non_blocking=True)
            t0 = time.perf_counter()
            _ = model(cur)
            if not overlap:
                next_gpu = host.to(dev, non_blocking=True)
            torch.cuda.synchronize()
            if i >= warmup:
                steady_ms.append((time.perf_counter() - t0) * 1000.0)

    stop.set()
    sampler.join(timeout=1.0)
    steady = [s for s in samples[len(samples) * warmup // iters :] if s is not None]
    steady.sort()
    mean = sum(steady) / len(steady) if steady else 0.0
    p50 = steady[len(steady) // 2] if steady else 0
    imgs_s = batch / (sum(steady_ms) / len(steady_ms) / 1000.0) if steady_ms else 0.0
    return {"mean": mean, "p50": p50, "imgs_s": imgs_s, "n": len(steady)}


def main() -> int:
    cfg = _cfg()
    init_ray()
    import ray

    print(f"model=resnet50(fp16)  batch={cfg['batch']}  iters={cfg['iters']}\n")
    task = ray.remote(num_gpus=1)(_run_inference)
    for mode, overlap in (("naive", False), ("overlapped", True)):
        r = ray.get(task.remote(cfg["batch"], cfg["iters"], cfg["warmup"], overlap))
        flag = "  >=90% TARGET MET" if r["mean"] >= 90 else ""
        print(
            f"{mode:<11} GPU util mean={r['mean']:.1f}%  p50={r['p50']}%  "
            f"({r['imgs_s']:.0f} img/s, {r['n']} samples){flag}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
