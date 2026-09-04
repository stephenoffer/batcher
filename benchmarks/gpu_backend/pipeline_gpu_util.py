"""Cluster GPU utilization for a preprocessing-HEAVY pipeline (CPU decode -> GPU inference).

The hard case for keeping GPUs fed: real per-image CPU work (JPEG decode + resize) upstream of
the GPU forward pass. If decode can't keep up, the GPU starves and utilization drops — the Ray
Data "GPU starvation from slow CPU preprocessing" failure. This runs, per GPU, a producer pool
of CPU decode threads feeding a bounded queue that the GPU consumer (ResNet-50) drains, and
samples each GPU's NVML utilization. `workers` = CPU decode threads per GPU (the CPU:GPU feed
ratio); the whole point is that enough parallel decode keeps every GPU near 100%.

Run:
    python benchmarks/gpu_backend/pipeline_gpu_util.py
    BENCH_PU_WORKERS=12 python benchmarks/gpu_backend/pipeline_gpu_util.py
"""

from __future__ import annotations

import functools
import os
import sys
import time
from pathlib import Path

from _ray_env import init_ray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envinfo import machine_fingerprint, require_release_build

print = functools.partial(print, flush=True)


def _cfg() -> dict:
    return {
        "batch": int(os.environ.get("BENCH_PU_BATCH", "64")),
        "iters": int(os.environ.get("BENCH_PU_ITERS", "120")),
        "warmup": int(os.environ.get("BENCH_PU_WARMUP", "15")),
        "workers": int(os.environ.get("BENCH_PU_WORKERS", "12")),
    }


def _worker(gpu_id: int, batch: int, iters: int, warmup: int, workers: int) -> dict:
    """Per-GPU pipeline: `workers` CPU decode threads feed a queue; the GPU drains it. Samples
    this GPU's NVML util through the steady window."""
    import io
    import queue
    import threading

    import numpy as np
    import torch
    from PIL import Image

    dev = torch.device("cuda")
    model = __import__("torchvision").models.resnet50().to(dev).eval().half()

    # One real JPEG to decode repeatedly (the CPU cost is decode+resize, as in a real pipeline).
    buf = io.BytesIO()
    Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8)).save(buf, format="JPEG")
    jpeg = buf.getvalue()

    def decode_one():
        img = Image.open(io.BytesIO(jpeg)).convert("RGB").resize((224, 224))
        return np.asarray(img, dtype=np.float16).transpose(2, 0, 1) / 255.0

    q: queue.Queue = queue.Queue(maxsize=workers * 2)
    stop = threading.Event()

    def producer():
        while not stop.is_set():
            arr = np.stack([decode_one() for _ in range(batch)])
            try:
                q.put(torch.from_numpy(arr).pin_memory(), timeout=1.0)
            except queue.Full:
                continue

    for _ in range(workers):
        threading.Thread(target=producer, daemon=True).start()

    util: list[int] = []

    def sample():
        try:
            import pynvml

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            while not stop.is_set():
                util.append(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
                time.sleep(0.05)
            pynvml.nvmlShutdown()
        except Exception:
            pass

    threading.Thread(target=sample, daemon=True).start()

    done = 0
    with torch.no_grad():
        for i in range(iters):
            host = q.get()
            cur = host.to(dev, non_blocking=True)
            _ = model(cur)
            torch.cuda.synchronize()
            if i >= warmup:
                done += batch
    stop.set()
    time.sleep(0.2)
    steady = util[len(util) * warmup // max(1, iters) :]
    mean = sum(steady) / len(steady) if steady else 0.0
    return {"gpu": gpu_id, "mean_util": mean, "imgs": done}


def main() -> int:
    # Refuse a dev-profile engine (8-60x slower) and a contended box (a neighbour's load
    # is not a fact about any engine), and print the machine, because a timing is only
    # reproducible beside the box that produced it. `BENCH_ALLOW_DEBUG_BUILD=1` /
    # `BENCH_ALLOW_BUSY_BOX=1` override.
    # No `require_quiet_box()` here, deliberately: the work in a cluster benchmark
    # happens on Ray workers, so the *driver's* run queue is not the contention
    # signal that would invalidate the measurement, and refusing on it is a false
    # negative on the multi-node deployment these scripts are written for.
    require_release_build()
    print(machine_fingerprint())
    cfg = _cfg()
    init_ray()
    import ray

    n_gpus = max(1, int(ray.cluster_resources().get("GPU", 1)))
    print(
        f"pipeline=jpeg-decode->resnet50(fp16)  gpus={n_gpus}  batch={cfg['batch']}  "
        f"decode_threads/gpu={cfg['workers']}\n"
    )
    task = ray.remote(num_gpus=1, num_cpus=cfg["workers"])(_worker)
    t0 = time.perf_counter()
    refs = [
        task.remote(i, cfg["batch"], cfg["iters"], cfg["warmup"], cfg["workers"])
        for i in range(n_gpus)
    ]
    results = ray.get(refs)
    wall = time.perf_counter() - t0
    utils = [r["mean_util"] for r in results]
    for r in sorted(results, key=lambda r: r["gpu"]):
        print(f"  gpu[{r['gpu']}]  util={r['mean_util']:5.1f}%")
    agg = sum(utils) / len(utils)
    flag = "  >= FULL CAPACITY" if agg >= 90 and min(utils) >= 85 else "  (GPU-starved)"
    print(f"\ncluster GPU util: mean={agg:.1f}%  min={min(utils):.1f}%{flag}")
    print(f"cluster throughput: {sum(r['imgs'] for r in results) / wall:.0f} img/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
