"""Batcher vs Ray Data — distributed video-clip inference (large-intermediate multimodal).

The multimodal-preprocessing *video* workload: each row is a short clip (K frames), so a
row is large (~0.6 MB) and expands to K times the compute — the regime that stresses memory
bounding and batch sizing (a fixed batch OOMs on wide rows; the guides drop block size to
64 MiB for video). A clip → K frames → a per-frame model → mean-pool → clip label.
Batcher's byte-aware morselization isolates wide rows and its zero-config batch shrinks by
row width; warm pools reuse the model across `collect()`s.

Both engines read the same Parquet clip shards distributed, run the same seeded model.

Run:
    python benchmarks/cluster/gpu_video.py            # resnet18, 4096 clips x 16 frames
"""

from __future__ import annotations

import contextlib
import functools
import os
import time

import numpy as np
import pyarrow as pa
from _ray_env import init_batcher_ray, with_timeout

print = functools.partial(print, flush=True)

_SEED = 1234
_H = _W = 112
_FRAME = 3 * _H * _W


def _cfg() -> dict:
    return {
        "n": int(os.environ.get("BENCH_VIDEO_N", "4096")),
        "frames": int(os.environ.get("BENCH_VIDEO_FRAMES", "16")),
        "shards": int(os.environ.get("BENCH_VIDEO_SHARDS", "32")),
        "num_gpus": float(os.environ.get("BENCH_GPU_NUM_GPUS", "1")),
        "dir": os.environ.get("BENCH_VIDEO_PARQUET", "/mnt/cluster_storage/gpu_video"),
        "runs": int(os.environ.get("BENCH_RUNS", "2")),
        "timeout": float(os.environ.get("BENCH_ENGINE_TIMEOUT", "400")),
    }


def write_shards(directory: str, n: int, shards: int, frames: int) -> str:
    import pyarrow.parquet as pq

    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(_SEED)
    per = -(-n // shards)
    written = 0
    for s in range(shards):
        lo, hi = s * per, min((s + 1) * per, n)
        if lo >= hi:
            break
        clips = rng.integers(0, 256, size=(hi - lo, frames * _FRAME), dtype=np.uint8)
        tbl = pa.table(
            {
                "id": pa.array(np.arange(lo, hi, dtype=np.int64)),
                "clip": pa.FixedSizeListArray.from_arrays(
                    pa.array(clips.reshape(-1), pa.uint8()), frames * _FRAME
                ),
            }
        )
        pq.write_table(tbl, os.path.join(directory, f"shard_{s:04d}.parquet"))
        written = hi
    return f"{written} clips x {frames} frames -> {directory}"


class VideoModel:
    """Model-load-once: a clip (K frames) → per-frame ResNet-18 → mean-pool logits → label."""

    def __init__(self) -> None:
        import torch
        import torchvision

        self._frames = int(os.environ.get("BENCH_VIDEO_FRAMES", "16"))
        torch.manual_seed(_SEED)
        self._dev = "cuda" if torch.cuda.is_available() else "cpu"
        if self._dev == "cuda":
            torch.set_float32_matmul_precision("high")
        self._m = torchvision.models.resnet18(weights=None).to(self._dev).eval()

    def __call__(self, batch) -> dict:
        import torch

        col = batch.column("clip")
        col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
        flat = col.flatten().to_numpy(zero_copy_only=False).astype(np.uint8, copy=False)
        b = len(batch)
        frames = flat.reshape(b * self._frames, 3, _H, _W)
        x = torch.from_numpy(frames).to(self._dev).float().div_(255.0)
        with torch.inference_mode():
            logits = self._m(x).reshape(b, self._frames, -1).mean(dim=1)
            pred = logits.argmax(1).to("cpu").numpy()
        ids = batch.column("id")
        ids = ids.combine_chunks() if isinstance(ids, pa.ChunkedArray) else ids
        return {"id": ids.to_numpy(zero_copy_only=False), "pred": pred}


def batcher_thunk(cfg: dict):
    import batcher as bt

    ds = bt.read.parquet(f"{cfg['dir']}/*.parquet").map_batches(
        VideoModel, output_columns=["id", "pred"], batch_format="pyarrow", num_gpus=cfg["num_gpus"]
    )

    def run():
        return _sig(ds.collect(distributed=True))

    return run


def ray_thunk(cfg: dict):
    import ray.data as rd

    conc = _auto_gpu_actors(cfg["num_gpus"])

    class _RayVideo:
        def __init__(self) -> None:
            self._m = VideoModel()

        def __call__(self, b):

            if not b.num_rows:
                return {"id": [], "pred": []}
            return self._m(b.combine_chunks().to_batches()[0])

    # Ray needs an explicit batch_size for GPU; use a wide-row-safe value (clips are ~0.6 MB).
    ds = rd.read_parquet(cfg["dir"]).map_batches(
        _RayVideo,
        concurrency=conc,
        num_gpus=cfg["num_gpus"],
        num_cpus=0,
        batch_size=64,
        batch_format="pyarrow",
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
    return {"rows": tbl.num_rows, "preds": dict(zip(d["id"], d["pred"], strict=False))}


def _agreement(a: dict, b: dict) -> float:
    pa_, pb = a["preds"], b["preds"]
    common = set(pa_) & set(pb)
    return sum(1 for k in common if pa_[k] == pb[k]) / len(common) if common else 0.0


def bench(cfg: dict, n: int) -> dict:
    out: dict = {}
    for eng, builder in (("batcher", batcher_thunk), ("ray", ray_thunk)):
        try:
            run = with_timeout(builder(cfg), cfg["timeout"])
            print(f"  [{eng}] warmup ...")
            warm = run()
            best = float("inf")
            for _ in range(cfg["runs"]):
                t0 = time.perf_counter()
                run()
                best = min(best, time.perf_counter() - t0)
            out[eng] = {"s": best, "sig": warm}
            print(f"  [{eng}] {best:.2f}s  {n / best:.1f} clip/s")
        except TimeoutError:
            out[eng] = {"error": "TIMEOUT"}
        except Exception as e:
            out[eng] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  [{eng}] ERROR {type(e).__name__}: {e}")
        finally:
            if eng == "batcher":
                for mod, fn in (
                    ("map", "release_inference_pools"),
                    ("fleet", "release_session_fleet"),
                ):
                    with contextlib.suppress(Exception):
                        m = __import__(
                            f"batcher.dist.{'executors.map' if mod == 'map' else 'fleet'}",
                            fromlist=[fn],
                        )
                        getattr(m, fn)()
    return out


def _init() -> None:
    init_batcher_ray(forward=("BENCH_VIDEO_FRAMES",))


def main() -> int:
    cfg = _cfg()
    _init()
    import pyarrow.parquet as pq
    import ray

    if not os.path.isdir(cfg["dir"]) or not os.listdir(cfg["dir"]):
        print("generating:", write_shards(cfg["dir"], cfg["n"], cfg["shards"], cfg["frames"]))
    n = sum(
        pq.read_metadata(os.path.join(cfg["dir"], f)).num_rows
        for f in os.listdir(cfg["dir"])
        if f.endswith(".parquet")
    )
    print(f"cluster: {ray.cluster_resources().get('GPU')} GPU, {len(ray.nodes())} nodes")
    print(
        f"model=resnet18 clips={n} frames={cfg['frames']} "
        f"num_gpus={cfg['num_gpus']} best-of-{cfg['runs']}\n"
    )
    res = bench(cfg, n)
    bm, rm = res.get("batcher", {}).get("s"), res.get("ray", {}).get("s")
    print("\nengine     time_s    clip/s")
    print("-" * 32)
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
        print(f"correctness: agreement={agree:.3%}  [{'OK' if agree >= 0.999 else 'MISMATCH'}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
