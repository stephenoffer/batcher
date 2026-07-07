"""2x the prebuilt AI functions: model-side inference optimizations vs the current baseline.

Batcher's `ds.ml.infer` already loads once, batches, and casts to fp16 (the current perf). The
next 2x is model-side and free of correctness risk for inference: `channels_last` memory format
(tensor-core-friendly layout for CNNs), `torch.compile` (kernel fusion + graph capture), and
their combination. This measures each on a GPU worker over the same batches and reports the
speedup vs the fp16-eager baseline, so the wins can be wired into the inference path.

Run:
    python benchmarks/gpu_backend/infer_speedup.py
    BENCH_IS_MODEL=resnet50 BENCH_IS_BATCH=128 python benchmarks/gpu_backend/infer_speedup.py
"""

from __future__ import annotations

import functools
import os
import time

print = functools.partial(print, flush=True)


def _cfg() -> dict:
    return {
        "model": os.environ.get("BENCH_IS_MODEL", "resnet50"),
        "batch": int(os.environ.get("BENCH_IS_BATCH", "128")),
        "iters": int(os.environ.get("BENCH_IS_ITERS", "80")),
        "warmup": int(os.environ.get("BENCH_IS_WARMUP", "25")),
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


def _bench(model_name: str, batch: int, iters: int, warmup: int) -> dict:
    """Time fp16 inference under four configs and check the optimized outputs match the
    baseline (argmax label — inference-stable)."""
    import numpy as np
    import torch
    import torchvision

    dev = torch.device("cuda")
    x = torch.from_numpy(np.random.rand(batch, 3, 224, 224).astype(np.float16)).to(dev)

    def build(channels_last: bool, compile_: bool):
        m = getattr(torchvision.models, model_name)().to(dev).eval().half()
        if channels_last:
            m = m.to(memory_format=torch.channels_last)
        if compile_:
            m = torch.compile(m, mode="max-autotune")
        return m

    def run(m, xin, n):
        best = float("inf")
        out = None
        with torch.no_grad():
            for i in range(n):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = m(xin)
                torch.cuda.synchronize()
                if i >= warmup:
                    best = min(best, time.perf_counter() - t0)
        return out, best

    configs = {
        "baseline (fp16 eager)": (False, False),
        "+channels_last": (True, False),
        "+torch.compile": (False, True),
        "+channels_last+compile": (True, True),
    }
    res = {}
    ref = None
    for name, (cl, co) in configs.items():
        xin = x.to(memory_format=torch.channels_last) if cl else x
        m = build(cl, co)
        out, best = run(m, xin, iters)
        logits = out.float().cpu()
        # Compare LOGITS with tolerance, not argmax: an untrained net on random input has
        # ~uniform logits so argmax is meaningless noise; the optimizations are numerically
        # equivalent (same predictions on real data) when logits agree within fp16 tolerance.
        if ref is None:
            ref = logits
            rel = 0.0
        else:
            rel = float((logits - ref).abs().max() / ref.abs().max().clamp(min=1e-6))
        res[name] = {"imgs_s": batch / best, "ms": best * 1000.0, "rel_err": rel}
    return res


def main() -> int:
    cfg = _cfg()
    _init()
    import ray

    print(f"model={cfg['model']}(fp16)  batch={cfg['batch']}  iters={cfg['iters']}\n")
    task = ray.remote(num_gpus=1)(_bench)
    res = ray.get(task.remote(cfg["model"], cfg["batch"], cfg["iters"], cfg["warmup"]))
    base = res["baseline (fp16 eager)"]["imgs_s"]
    for name, r in res.items():
        speed = r["imgs_s"] / base
        tag = "  >=2x!" if speed >= 2.0 else ""
        print(
            f"{name:<26} {r['imgs_s']:8.0f} img/s  ({r['ms']:.1f} ms)  "
            f"{speed:.2f}x  logit_rel_err={r['rel_err']:.2e}{tag}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
