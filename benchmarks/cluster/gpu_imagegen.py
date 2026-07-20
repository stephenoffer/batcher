"""Batcher vs Ray Data — distributed image generation / diffusion (the image-generation workload).

Batch image generation with a diffusion model (diffusers `google/ddpm-cifar10-32` UNet +
DDIM), K denoising steps per image, fanned across every GPU. Like LLM batch inference this
is **model-load-dominated** (the UNet loads in ~4 s and each execution's generation is a few
seconds), so Ray Data — which respawns the actor pool per execution — pays the load every
time, while Batcher keeps it warm across `collect()`s. Per-id-seeded initial noise makes the
generated image deterministic (independent of batch composition), so a per-id checksum agrees
across engines regardless of batching.

Both engines read the same Parquet id shards distributed and run the same seeded UNet.

Run:
    python benchmarks/cluster/gpu_imagegen.py            # ddpm-cifar10, 2048 images, 20 steps
"""

from __future__ import annotations

import contextlib
import functools
import os
import time

import numpy as np
import pyarrow as pa
from _ray_env import init_batcher_ray

print = functools.partial(print, flush=True)

_MODEL = "google/ddpm-cifar10-32"


def _cfg() -> dict:
    return {
        "n": int(os.environ.get("BENCH_GEN_N", "2048")),
        "shards": int(os.environ.get("BENCH_GEN_SHARDS", "32")),
        "batch": int(os.environ.get("BENCH_GEN_BATCH", "64")),
        "steps": int(os.environ.get("BENCH_GEN_STEPS", "20")),
        "num_gpus": float(os.environ.get("BENCH_GPU_NUM_GPUS", "1")),
        "dir": os.environ.get("BENCH_GEN_PARQUET", "/mnt/cluster_storage/gpu_imagegen"),
        "runs": int(os.environ.get("BENCH_RUNS", "3")),
        "timeout": float(os.environ.get("BENCH_ENGINE_TIMEOUT", "400")),
    }


def write_shards(directory: str, n: int, shards: int) -> str:
    import pyarrow.parquet as pq

    os.makedirs(directory, exist_ok=True)
    per = -(-n // shards)
    written = 0
    for s in range(shards):
        lo, hi = s * per, min((s + 1) * per, n)
        if lo >= hi:
            break
        pq.write_table(
            pa.table({"id": pa.array(np.arange(lo, hi, dtype=np.int64))}),
            os.path.join(directory, f"shard_{s:04d}.parquet"),
        )
        written = hi
    return f"{written} generation requests -> {directory}"


class Generator:
    """Model-load-once diffusion generator: build the UNet + DDIM scheduler once, then per
    batch of ids generate an image each (K denoising steps) from per-id-seeded noise."""

    def __init__(self) -> None:
        import torch
        from diffusers import DDIMScheduler, UNet2DModel

        self._steps = int(os.environ.get("BENCH_GEN_STEPS", "20"))
        self._dev = "cuda" if torch.cuda.is_available() else "cpu"
        if self._dev == "cuda":
            torch.set_float32_matmul_precision("high")
        self._unet = UNet2DModel.from_pretrained(_MODEL).to(self._dev).eval()
        self._sched = DDIMScheduler.from_pretrained(_MODEL)
        self._sched.set_timesteps(self._steps)

    def __call__(self, batch) -> dict:
        import torch

        ids = batch.column("id") if hasattr(batch, "column") else batch["id"]
        if hasattr(ids, "combine_chunks"):
            ids = ids.combine_chunks() if isinstance(ids, pa.ChunkedArray) else ids
        ids = np.asarray(ids.to_numpy(zero_copy_only=False) if hasattr(ids, "to_numpy") else ids)
        noise = torch.stack(
            [torch.randn((3, 32, 32), generator=torch.Generator().manual_seed(int(i))) for i in ids]
        ).to(self._dev)
        img = noise
        with torch.inference_mode():
            for t in self._sched.timesteps:
                pred = self._unet(img, t).sample
                img = self._sched.step(pred, t, img).prev_sample
        chk = img.reshape(len(ids), -1).sum(1).to("cpu").numpy()
        return {"id": ids, "chk": chk}


def batcher_thunk(cfg: dict):
    import batcher as bt

    ds = bt.read.parquet(f"{cfg['dir']}/*.parquet").map_batches(
        Generator,
        output_columns=["id", "chk"],
        batch_format="pyarrow",
        num_gpus=cfg["num_gpus"],
        batch_size=cfg["batch"],
    )

    def run():
        return _sig(ds.collect(distributed=True))

    return run


def ray_thunk(cfg: dict):
    import ray.data as rd

    conc = _auto_gpu_actors(cfg["num_gpus"])

    class _RayGen:
        def __init__(self) -> None:
            self._m = Generator()

        def __call__(self, b):
            return self._m(b) if len(b["id"]) else {"id": [], "chk": []}

    ds = rd.read_parquet(cfg["dir"]).map_batches(
        _RayGen,
        concurrency=conc,
        num_gpus=cfg["num_gpus"],
        num_cpus=0,
        batch_size=cfg["batch"],
        batch_format="numpy",
    )

    def run():
        rows = [b for b in ds.iter_batches(batch_format="pyarrow") if b.num_rows]
        return _sig(pa.Table.from_batches([b.combine_chunks().to_batches()[0] for b in rows]))

    return run


def _auto_gpu_actors(num_gpus: float) -> int:
    import ray

    return max(1, int(float(ray.cluster_resources().get("GPU", 1.0)) / max(num_gpus, 0.01)))


def _sig(tbl: pa.Table) -> dict:
    d = tbl.to_pydict()
    return {
        "rows": tbl.num_rows,
        "preds": {i: round(float(c), 1) for i, c in zip(d["id"], d["chk"], strict=False)},
    }


def _agreement(a: dict, b: dict) -> float:
    pa_, pb = a["preds"], b["preds"]
    common = set(pa_) & set(pb)
    # tolerance for GPU float nondeterminism in a 3072-elt image-checksum sum
    return sum(1 for k in common if abs(pa_[k] - pb[k]) <= 2.0) / len(common) if common else 0.0


def _with_timeout(fn, t):
    import threading

    def wrapped():
        box: dict = {}

        def run():
            try:
                box["v"] = fn()
            except BaseException as e:
                box["e"] = e

        th = threading.Thread(target=run, daemon=True)
        th.start()
        th.join(t)
        if th.is_alive():
            raise TimeoutError
        if "e" in box:
            raise box["e"]
        return box.get("v")

    return wrapped


def bench(cfg: dict, n: int) -> dict:
    out: dict = {}
    for eng, builder in (("batcher", batcher_thunk), ("ray", ray_thunk)):
        try:
            run = _with_timeout(builder(cfg), cfg["timeout"])
            print(f"  [{eng}] warmup (pays UNet load) ...")
            warm = run()
            best = float("inf")
            for _ in range(cfg["runs"]):
                t0 = time.perf_counter()
                run()
                best = min(best, time.perf_counter() - t0)
            out[eng] = {"s": best, "sig": warm}
            print(f"  [{eng}] {best:.2f}s  {n / best:.1f} img/s")
        except TimeoutError:
            out[eng] = {"error": "TIMEOUT"}
        except Exception as e:
            out[eng] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  [{eng}] ERROR {type(e).__name__}: {e}")
        finally:
            if eng == "batcher":
                with contextlib.suppress(Exception):
                    from batcher.dist.executors.map import release_inference_pools

                    release_inference_pools()
                with contextlib.suppress(Exception):
                    from batcher.dist.fleet import release_session_fleet

                    release_session_fleet()
    return out


def _init() -> None:
    init_batcher_ray(forward=("BENCH_GEN_STEPS",), hf_cache="/mnt/cluster_storage/hf_cache")


def main() -> int:
    cfg = _cfg()
    _init()
    import pyarrow.parquet as pq
    import ray

    if not os.path.isdir(cfg["dir"]) or not os.listdir(cfg["dir"]):
        print("generating:", write_shards(cfg["dir"], cfg["n"], cfg["shards"]))
    n = sum(
        pq.read_metadata(os.path.join(cfg["dir"], f)).num_rows
        for f in os.listdir(cfg["dir"])
        if f.endswith(".parquet")
    )
    print(f"cluster: {ray.cluster_resources().get('GPU')} GPU, {len(ray.nodes())} nodes")
    print(
        f"model=ddpm-cifar10 images={n} batch={cfg['batch']} "
        f"steps={cfg['steps']} best-of-{cfg['runs']}\n"
    )
    res = bench(cfg, n)
    bm, rm = res.get("batcher", {}).get("s"), res.get("ray", {}).get("s")
    print("\nengine     time_s    img/s")
    print("-" * 30)
    for eng in ("batcher", "ray"):
        r = res.get(eng, {})
        print(
            f"{eng:<10} {r['error']}"
            if "error" in r
            else f"{eng:<10} {r['s']:>6.2f}  {n / r['s']:>7.1f}"
        )
    if bm and rm:
        print(f"\nbatcher vs ray: {rm / bm:.2f}x  (>1 = batcher faster)")
    sb, sr = res.get("batcher", {}).get("sig"), res.get("ray", {}).get("sig")
    if sb and sr:
        agree = _agreement(sb, sr)
        print(f"correctness: agreement={agree:.3%}  [{'OK' if agree >= 0.99 else 'MISMATCH'}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
