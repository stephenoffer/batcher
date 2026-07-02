"""Batcher vs Ray Data — distributed **two-stage** CPU→GPU image pipeline.

The canonical Ray Data batch-inference shape: a CPU stage decodes/resizes images and a
GPU stage runs the model, as two `map_batches` stages so the CPU decode of block k+1
overlaps the GPU forward of block k. This is where a data engine wins or loses on GPU
*utilization*: if the CPU stage can't keep the GPU fed, the GPU idles.

Batcher's moat here is `dist/streaming/pipeline.py` — it splits the chain at the GPU
stage and streams morsels producer→consumer over Carbonite Arrow Flight with credit
backpressure, so CPU and GPU overlap with bounded memory. This benchmark exercises that
path (`distributed.stream_inference`) against Ray Data's streaming executor on a real
JPEG-decode workload, correctness-gated, reporting images/sec + GPU utilization.

Both engines read the same Parquet shards (JPEG bytes) distributed from shared storage,
run the same seeded model, and are checked for prediction agreement before timing.

Run:
    python benchmarks/cluster/gpu_pipeline.py                 # resnet50, streamed
    BENCH_GPU_STREAM=0 python benchmarks/cluster/gpu_pipeline.py   # non-overlapped baseline
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import io
import os
import time

import numpy as np
import pyarrow as pa

print = functools.partial(print, flush=True)

_SRC_HW = 256  # stored JPEG size (decode + resize to 224 is the real CPU cost)
_HW = 224
_CHW = 3 * _HW * _HW
_SEED = 1234
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _cfg() -> dict:
    return {
        "model": os.environ.get("BENCH_GPU_MODEL", "resnet50"),
        "batch": int(os.environ.get("BENCH_GPU_BATCH", "128")),
        "n": int(os.environ.get("BENCH_GPU_N", "12288")),
        "shards": int(os.environ.get("BENCH_GPU_SHARDS", "48")),
        "num_gpus": float(os.environ.get("BENCH_GPU_NUM_GPUS", "1")),
        "concurrency": os.environ.get("BENCH_GPU_CONCURRENCY", ""),
        "stream": os.environ.get("BENCH_GPU_STREAM", "1") == "1",
        "dir": os.environ.get("BENCH_GPU_PARQUET", "/mnt/cluster_storage/gpu_pipeline"),
        "runs": int(os.environ.get("BENCH_RUNS", "2")),
        "timeout": float(os.environ.get("BENCH_ENGINE_TIMEOUT", "600")),
    }


def write_shards(directory: str, n: int, shards: int) -> str:
    """Write `n` fixed-seed synthetic images as JPEG bytes in `shards` Parquet files."""
    import pyarrow.parquet as pq
    from PIL import Image

    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(_SEED)
    per = -(-n // shards)
    written = 0
    for s in range(shards):
        lo, hi = s * per, min((s + 1) * per, n)
        if lo >= hi:
            break
        jpegs = []
        for _ in range(hi - lo):
            arr = rng.integers(0, 256, size=(_SRC_HW, _SRC_HW, 3), dtype=np.uint8)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=90)
            jpegs.append(buf.getvalue())
        tbl = pa.table(
            {
                "id": pa.array(np.arange(lo, hi, dtype=np.int64)),
                "jpeg": pa.array(jpegs, pa.binary()),
            }
        )
        pq.write_table(tbl, os.path.join(directory, f"shard_{s:04d}.parquet"))
        written = hi
    return f"{written} jpeg imgs in {shards} shards -> {directory}"


# --------------------------------------------------------------------------- #
# Stage 1 (CPU): JPEG decode + resize to 224 + transpose to CHW uint8
# --------------------------------------------------------------------------- #
def decode_batch(batch) -> dict:
    """JPEG bytes -> (B, 3, 224, 224) uint8 CHW. The real CPU work; keeps the inter-stage
    payload uint8 as a **numpy tensor** (normalize happens on the GPU) — the format both
    engines carry natively (Ray Data's tensor blocks, Batcher's Arrow tensor column)."""
    from PIL import Image

    col = batch.column("jpeg")
    col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    out = np.empty((len(col), 3, _HW, _HW), dtype=np.uint8)
    for i in range(len(col)):
        img = Image.open(io.BytesIO(col[i].as_py())).convert("RGB").resize((_HW, _HW))
        out[i] = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)
    ids = batch.column("id")
    ids = ids.combine_chunks() if isinstance(ids, pa.ChunkedArray) else ids
    return {"id": ids.to_numpy(zero_copy_only=False), "chw": out}


# --------------------------------------------------------------------------- #
# Stage 2 (GPU): uint8 CHW -> normalize on device -> model -> argmax
# --------------------------------------------------------------------------- #
class GPUModel:
    """GPU stage: consumes a numpy batch ``{"id", "chw"}`` (chw = (B,3,224,224) uint8),
    normalizes on device, runs the model, returns ``{"id", "pred"}``."""

    def __init__(self) -> None:
        import torch
        import torchvision

        name = os.environ.get("BENCH_GPU_MODEL", "resnet50")
        torch.manual_seed(_SEED)
        self._dev = "cuda" if torch.cuda.is_available() else "cpu"
        if self._dev == "cuda":
            torch.set_float32_matmul_precision("high")
        self._m = getattr(torchvision.models, name)(weights=None).to(self._dev).eval()
        self._mean = torch.tensor(_MEAN, device=self._dev).view(1, 3, 1, 1)
        self._std = torch.tensor(_STD, device=self._dev).view(1, 3, 1, 1)

    def __call__(self, batch: dict) -> dict:
        import torch

        chw = np.ascontiguousarray(batch["chw"]).reshape(-1, 3, _HW, _HW)
        x = torch.from_numpy(chw).to(self._dev, non_blocking=True)
        with torch.inference_mode():
            xf = x.float().div_(255.0).sub_(self._mean).div_(self._std)  # normalize on GPU
            pred = self._m(xf).argmax(1).to("cpu").numpy()
        return {"id": np.asarray(batch["id"]), "pred": pred}


# --------------------------------------------------------------------------- #
# Engine pipelines
# --------------------------------------------------------------------------- #
def batcher_thunk(cfg: dict):
    import batcher as bt

    conc = int(cfg["concurrency"]) if cfg["concurrency"] else None
    ds = (
        bt.read.parquet(f"{cfg['dir']}/*.parquet")
        .map_batches(decode_batch, output_columns=["id", "chw"], batch_format="pyarrow")
        .map_batches(
            GPUModel,
            output_columns=["id", "pred"],
            batch_format="numpy",
            num_gpus=cfg["num_gpus"],
            concurrency=conc,
            batch_size=cfg["batch"] or None,
        )
    )

    def run():
        return _sig(ds.collect(distributed=True))

    return run


def ray_thunk(cfg: dict):
    import ray.data as rd

    conc = int(cfg["concurrency"]) if cfg["concurrency"] else _auto_gpu_actors(cfg["num_gpus"])

    def _decode(b):
        return decode_batch(b)

    class _RayModel:
        def __init__(self) -> None:
            self._m = GPUModel()

        def __call__(self, b):
            return self._m(b) if len(b["id"]) else {"id": [], "pred": []}

    ds = (
        rd.read_parquet(cfg["dir"])
        .map_batches(_decode, num_cpus=1, batch_format="pyarrow", batch_size=cfg["batch"] or None)
        .map_batches(
            _RayModel,
            concurrency=conc,
            num_gpus=cfg["num_gpus"],
            num_cpus=0,
            batch_size=cfg["batch"] or None,
            batch_format="numpy",
        )
    )

    def run():
        rows = [b for b in ds.iter_batches(batch_format="pyarrow") if b.num_rows]
        return _sig(pa.Table.from_batches([b.combine_chunks().to_batches()[0] for b in rows]))

    return run


def _auto_gpu_actors(num_gpus: float) -> int:
    import ray

    return max(1, int(float(ray.cluster_resources().get("GPU", 1.0)) / max(num_gpus, 0.01)))


# --------------------------------------------------------------------------- #
# Correctness + timing
# --------------------------------------------------------------------------- #
def _sig(tbl: pa.Table) -> dict:
    d = tbl.to_pydict()
    return {"rows": tbl.num_rows, "preds": dict(zip(d["id"], d["pred"], strict=False))}


def _agreement(a: dict, b: dict) -> float:
    pa_, pb = a["preds"], b["preds"]
    common = set(pa_) & set(pb)
    return sum(1 for k in common if pa_[k] == pb[k]) / len(common) if common else 0.0


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


def _time(run, monitor):
    if monitor:
        monitor.start()
    t0 = time.perf_counter()
    sig = run()
    dt = time.perf_counter() - t0
    return dt, sig, (monitor.stop() if monitor else {})


def _fmt(u: dict) -> str:
    if not u:
        return "-"
    return f"{u.get('mean_gpu_pct', 0):.0f}%/{u.get('peak_gpu_pct', 0):.0f}%peak"


def bench(cfg: dict, n: int) -> dict:
    from gpu_util import GpuMonitor

    out: dict = {}
    for eng, builder in (("batcher", batcher_thunk), ("ray", ray_thunk)):
        try:
            run = _with_timeout(builder(cfg), cfg["timeout"])
            print(f"  [{eng}] warmup ...")
            warm = run()
            best, util = float("inf"), {}
            for i in range(cfg["runs"]):
                mon = GpuMonitor() if i == 0 else None
                dt, _s, u = _time(run, mon)
                if mon:
                    mon.shutdown()
                    util = u
                best = min(best, dt)
            out[eng] = {"s": best, "sig": warm, "util": util}
            print(f"  [{eng}] {best:.2f}s  {n / best:.0f} img/s  gpu={_fmt(util)}")
        except TimeoutError:
            out[eng] = {"error": "TIMEOUT"}
            print(f"  [{eng}] TIMEOUT")
        except Exception as e:
            out[eng] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  [{eng}] ERROR {type(e).__name__}: {e}")
        finally:
            if eng == "batcher":
                with contextlib.suppress(Exception):
                    from batcher.dist.fleet import release_session_fleet

                    release_session_fleet()
    return out


def _init(cfg: dict) -> None:
    import importlib.util

    for var in ("RAY_RUNTIME_ENV_HOOK", "RAY_RUNTIME_ENV_PLUGINS"):
        v = os.environ.get(var)
        if v:
            head = v.lstrip("[{\"' ").split(".")[0].split("[")[0]
            if head and importlib.util.find_spec(head) is None:
                os.environ.pop(var, None)
    import batcher
    from batcher.config import active_config, set_config

    pkg = os.path.dirname(os.path.abspath(batcher.__file__))
    renv = {"py_modules": [pkg], "pip": None, "env_vars": {}}
    base = active_config()
    set_config(
        base.replace(
            distributed=dataclasses.replace(
                base.distributed,
                ray_address="auto",
                runtime_env=renv,
                stream_inference=cfg["stream"],
            )
        )
    )
    import ray

    if not ray.is_initialized():
        ray.init(address="auto", runtime_env=renv, logging_level="ERROR", log_to_driver=False)


def main() -> int:
    cfg = _cfg()
    _init(cfg)
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
        f"model={cfg['model']} n={n} batch={cfg['batch']} num_gpus={cfg['num_gpus']} "
        f"stream_inference={cfg['stream']} best-of-{cfg['runs']}\n"
    )
    res = bench(cfg, n)
    bm, rm = res.get("batcher", {}).get("s"), res.get("ray", {}).get("s")
    print("\nengine     time_s    img/s    gpu_util")
    print("-" * 44)
    for eng in ("batcher", "ray"):
        r = res.get(eng, {})
        if "error" in r:
            print(f"{eng:<10} {r['error']}")
        else:
            print(f"{eng:<10} {r['s']:>6.2f}  {n / r['s']:>7.0f}   {_fmt(r['util'])}")
    if bm and rm:
        print(f"\nbatcher vs ray: {rm / bm:.2f}x  (>1 = batcher faster)")
    sb, sr = res.get("batcher", {}).get("sig"), res.get("ray", {}).get("sig")
    if sb and sr:
        agree = _agreement(sb, sr)
        print(
            f"correctness: pred-agreement={agree:.3%}  [{'OK' if agree >= 0.999 else 'MISMATCH'}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
