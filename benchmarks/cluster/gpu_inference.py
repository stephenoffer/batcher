"""Batcher vs Ray Data — distributed GPU batch-inference on a live GPU cluster.

The workload Ray Data is built for: a two-stage image pipeline — a CPU stage that
normalizes decoded images, then a GPU stage that runs a vision model (torchvision
ResNet-50 / EfficientNet-B0) as a **model-load-once actor pool** — fanned across every
GPU in the cluster. Both engines consume the *same* in-memory Arrow table (fixed seed),
run the *same* seeded model weights, and are checked for prediction agreement before any
timing is trusted; then we report images/sec, the Batcher/Ray speedup, and — the number
that actually matters for a data engine — the **mean/peak GPU utilization** each engine
sustained (a fast model on a starved GPU is the failure Ray Data users hit).

Why in-memory synthetic input rather than ``read_images`` from S3: it isolates the
compute + scheduling comparison (the ask) from each engine's divergent image reader and
from S3 variance, so the GPU is the bottleneck and the number is stable and reproducible.
The CPU normalize stage keeps the CPU→GPU hand-off (where overlap wins or loses) in play.

Run (from the env that carries ray + torch + torchvision, batcher installed):
    python benchmarks/cluster/gpu_inference.py                 # resnet50, 8192 imgs
    BENCH_GPU_MODEL=efficientnet_b0 python benchmarks/cluster/gpu_inference.py
    BENCH_GPU_N=16384 BENCH_GPU_BATCH=128 python benchmarks/cluster/gpu_inference.py
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

_HW = 224
_PIX = _HW * _HW * 3
_SEED = 1234
# ImageNet normalization constants (the standard torchvision preprocessing).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _cfg() -> dict:
    return {
        "n": int(os.environ.get("BENCH_GPU_N", "8192")),
        "model": os.environ.get("BENCH_GPU_MODEL", "resnet50"),
        "batch": int(os.environ.get("BENCH_GPU_BATCH", "128")),
        "num_gpus": float(os.environ.get("BENCH_GPU_NUM_GPUS", "1")),
        "concurrency": os.environ.get("BENCH_GPU_CONCURRENCY", ""),  # "" -> auto (all GPUs)
        "runs": int(os.environ.get("BENCH_RUNS", "2")),
        "timeout": float(os.environ.get("BENCH_ENGINE_TIMEOUT", "600")),
        # A shared-filesystem/S3 dir of image Parquet shards. When set, both engines read it
        # DISTRIBUTED (each worker reads its shards directly) — the faithful Ray Data pattern,
        # with no single-driver data-serving bottleneck. Empty -> in-memory from_arrow.
        "parquet": os.environ.get("BENCH_GPU_PARQUET", ""),
        "shards": int(os.environ.get("BENCH_GPU_SHARDS", "32")),
    }


def write_shards(directory: str, n: int, shards: int) -> str:
    """Write `n` fixed-seed synthetic images as `shards` Parquet files under `directory`.

    Each shard carries a contiguous id range so the union across shards is exactly the
    single-source table — the correctness oracle is identical to the in-memory path."""
    import pyarrow.parquet as pq

    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(_SEED)
    per = -(-n // shards)
    written = 0
    for s in range(shards):
        lo, hi = s * per, min((s + 1) * per, n)
        if lo >= hi:
            break
        imgs = rng.integers(0, 256, size=(hi - lo, _PIX), dtype=np.uint8)
        tbl = pa.table(
            {
                "id": pa.array(np.arange(lo, hi, dtype=np.int64)),
                "img": pa.FixedSizeListArray.from_arrays(
                    pa.array(imgs.reshape(-1), pa.uint8()), _PIX
                ),
            }
        )
        pq.write_table(tbl, os.path.join(directory, f"shard_{s:04d}.parquet"))
        written = hi
    return f"{written} imgs in {shards} shards -> {directory}"


def make_table(n: int) -> pa.Table:
    """A fixed-seed table of ``n`` synthetic uint8 images (flattened HWC) + an id."""
    rng = np.random.default_rng(_SEED)
    imgs = rng.integers(0, 256, size=(n, _PIX), dtype=np.uint8)
    return pa.table(
        {
            "id": pa.array(np.arange(n, dtype=np.int64)),
            "img": pa.FixedSizeListArray.from_arrays(pa.array(imgs.reshape(-1), pa.uint8()), _PIX),
        }
    )


# --------------------------------------------------------------------------- #
# The shared model + preprocessing (identical math in both engines)
# --------------------------------------------------------------------------- #
def _decode_uint8(col) -> np.ndarray:
    """An image column → (B, 224, 224, 3) uint8, robust to Array/ChunkedArray + slices.

    ``flatten()`` on a FixedSizeList yields the child uint8 values honoring the logical
    slice (unlike ``.values``, which returns the whole parent buffer)."""
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    flat = col.flatten().to_numpy(zero_copy_only=False).astype(np.uint8, copy=False)
    return flat.reshape(-1, _HW, _HW, 3)


def preprocess_np(imgs_u8: np.ndarray) -> np.ndarray:
    """(B,H,W,3) uint8 → (B,3,H,W) float32, ImageNet-normalized (the CPU stage)."""
    x = imgs_u8.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def _build_model(model_name: str):
    import torch
    import torchvision

    torch.manual_seed(_SEED)  # identical weights across every actor / engine
    ctor = getattr(torchvision.models, model_name)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.set_float32_matmul_precision("high")
    model = ctor(weights=None).to(dev).eval()
    return model, dev


class GPUModel:
    """Model-load-once class UDF: build the model once per actor, classify each batch.

    Input/output are pyarrow ``RecordBatch``es carrying ``id`` + ``img`` (flattened
    uint8); output carries ``id`` + ``pred`` (argmax class). The CPU normalize is folded
    in so the batch shipped between stages stays compact uint8 (the guides' "never move
    float tensors between stages") — the realistic single-fused inference stage.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._name = model_name or os.environ.get("BENCH_GPU_MODEL", "resnet50")
        self._model, self._dev = _build_model(self._name)

    def __call__(self, batch) -> dict:
        """Classify a columnar batch (pyarrow RecordBatch or Table); return id+pred dict.

        Returning a dict keeps the UDF portable across Batcher and Ray Data (both coerce
        a column dict), and avoids RecordBatch-vs-Table conversion pitfalls."""
        import torch

        imgs = _decode_uint8(batch.column("img"))
        ids = batch.column("id")
        ids = ids.combine_chunks() if isinstance(ids, pa.ChunkedArray) else ids
        x = torch.from_numpy(preprocess_np(imgs)).to(self._dev, non_blocking=True)
        with torch.inference_mode():
            pred = self._model(x).argmax(1).to("cpu").numpy()
        return {"id": ids.to_numpy(zero_copy_only=False), "pred": pred}


# --------------------------------------------------------------------------- #
# Batcher pipeline
# --------------------------------------------------------------------------- #
def batcher_thunk(table: pa.Table, cfg: dict):
    import batcher as bt

    conc = int(cfg["concurrency"]) if cfg["concurrency"] else None
    src = bt.read.parquet(f"{cfg['parquet']}/*.parquet") if cfg["parquet"] else bt.from_arrow(table)
    ds = src.map_batches(
        GPUModel,
        output_columns=["id", "pred"],
        batch_format="pyarrow",
        num_gpus=cfg["num_gpus"],
        concurrency=conc,
        batch_size=cfg["batch"],
    )

    def run():
        return _sig(ds.collect(distributed=True))

    return run


# --------------------------------------------------------------------------- #
# Ray Data pipeline
# --------------------------------------------------------------------------- #
def ray_thunk(table: pa.Table, cfg: dict):
    import ray.data as rd

    conc = int(cfg["concurrency"]) if cfg["concurrency"] else _auto_gpu_actors(cfg["num_gpus"])
    # Idiomatic Ray Data: a single `from_arrow` table is ONE block, so an actor pool would
    # run on one GPU. Repartition into several blocks per GPU actor so Ray fans across every
    # GPU and streams — the fair counterpart to Batcher's automatic balanced partitioning.
    n_blocks = conc * int(os.environ.get("BATCHER_GPU_STREAM_FACTOR", "3"))

    class _RayModel:
        def __init__(self) -> None:
            self._m = GPUModel()

        def __call__(self, batch):
            # batch_format="pyarrow" hands a pyarrow.Table; the UDF returns an id+pred dict.
            return self._m(batch) if batch.num_rows else {"id": [], "pred": []}

    base = (
        rd.read_parquet(cfg["parquet"])
        if cfg["parquet"]
        else rd.from_arrow(table).repartition(n_blocks)
    )
    ds = base.map_batches(
        _RayModel,
        concurrency=conc,
        num_gpus=cfg["num_gpus"],
        num_cpus=0,
        batch_size=cfg["batch"],
        batch_format="pyarrow",
    )

    def run():
        rows = list(ds.iter_batches(batch_format="pyarrow"))
        return _sig(
            pa.Table.from_batches([b.combine_chunks().to_batches()[0] for b in rows if b.num_rows])
        )

    return run


def _auto_gpu_actors(num_gpus: float) -> int:
    import ray

    total = float(ray.cluster_resources().get("GPU", 1.0))
    return max(1, int(total / max(num_gpus, 0.01)))


# --------------------------------------------------------------------------- #
# Correctness signature + timing
# --------------------------------------------------------------------------- #
def _sig(tbl: pa.Table) -> dict:
    """A slice-order-independent signature: row count + per-id predictions map."""
    d = tbl.to_pydict()
    preds = dict(zip(d["id"], d["pred"], strict=False))
    return {"rows": tbl.num_rows, "preds": preds}


def _agreement(a: dict, b: dict) -> float:
    """Fraction of ids on which two prediction maps agree (GPU nondeterminism tolerant)."""
    pa_, pb = a["preds"], b["preds"]
    common = set(pa_) & set(pb)
    if not common:
        return 0.0
    return sum(1 for k in common if pa_[k] == pb[k]) / len(common)


def _time(run, monitor=None):
    if monitor:
        monitor.start()
    t0 = time.perf_counter()
    sig = run()
    dt = time.perf_counter() - t0
    util = monitor.stop() if monitor else {}
    return dt, sig, util


ENGINES = {"batcher": batcher_thunk, "ray": ray_thunk}


def bench(table: pa.Table, cfg: dict) -> dict:
    from gpu_util import GpuMonitor

    out: dict = {}
    for eng, builder in ENGINES.items():
        thunk = builder(table, cfg)
        try:
            run = with_timeout(thunk, cfg["timeout"])
            print(f"  [{eng}] warmup (pays actor spawn + model load) ...")
            warm = run()
            best, util = float("inf"), {}
            for i in range(cfg["runs"]):
                mon = GpuMonitor() if i == 0 else None
                dt, _, u = _time(run, mon)
                if mon:
                    mon.shutdown()
                    util = u
                best = min(best, dt)
            out[eng] = {"s": best, "sig": warm, "util": util}
            print(f"  [{eng}] {best:.2f}s  {cfg['n'] / best:.0f} img/s  gpu={_fmt_util(util)}")
        except TimeoutError:
            out[eng] = {"error": f"TIMEOUT (>{cfg['timeout']:.0f}s)"}
            print(f"  [{eng}] TIMEOUT")
        except Exception as e:
            out[eng] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  [{eng}] ERROR {type(e).__name__}: {e}")
        finally:
            if eng == "batcher":
                with contextlib.suppress(Exception):
                    from batcher.dist.fleet import release_session_fleet

                    release_session_fleet()
                with contextlib.suppress(Exception):
                    from batcher.dist.executors.map import release_inference_pools

                    release_inference_pools()
    return out


def _fmt_util(u: dict) -> str:
    if not u:
        return "-"
    return (
        f"{u.get('mean_gpu_pct', 0):.0f}%/{u.get('peak_gpu_pct', 0):.0f}%peak "
        f"{int(u.get('active_gpu_nodes', 0))}/{int(u.get('total_gpu_nodes', 0))}gnodes"
    )


def _init_ray() -> None:
    """Ship the working-tree Batcher to the cluster and attach to it.

    Delegates to the shared bootstrap so this script cannot drift from its siblings — in
    particular it inherits the driver-matching numpy pin, without which every actor here
    dies in its constructor (see `_ray_env.worker_pip`).
    """
    init_batcher_ray(forward=("BENCH_GPU_MODEL",))


def main() -> int:
    cfg = _cfg()
    _init_ray()
    import ray

    print(
        f"cluster: {ray.cluster_resources().get('GPU')} GPU, "
        f"{ray.cluster_resources().get('CPU')} CPU, {len(ray.nodes())} nodes"
    )
    src = f"parquet:{cfg['parquet']}" if cfg["parquet"] else "in-memory"
    print(
        f"model={cfg['model']} n={cfg['n']} batch={cfg['batch']} "
        f"num_gpus={cfg['num_gpus']} concurrency={cfg['concurrency'] or 'auto'} "
        f"source={src} best-of-{cfg['runs']}\n"
    )
    if cfg["parquet"]:
        import pyarrow.parquet as pq

        if not os.path.isdir(cfg["parquet"]) or not os.listdir(cfg["parquet"]):
            print("generating shards:", write_shards(cfg["parquet"], cfg["n"], cfg["shards"]))
        total = sum(
            pq.read_metadata(os.path.join(cfg["parquet"], f)).num_rows
            for f in os.listdir(cfg["parquet"])
            if f.endswith(".parquet")
        )
        cfg["n"] = total
        print(f"parquet dataset: {total} imgs")
    table = None if cfg["parquet"] else make_table(cfg["n"])
    res = bench(table, cfg)

    bm = res.get("batcher", {}).get("s")
    rm = res.get("ray", {}).get("s")
    print("\nengine     time_s    img/s     gpu_util")
    print("-" * 52)
    for eng in ("batcher", "ray"):
        r = res.get(eng, {})
        if "error" in r:
            print(f"{eng:<10} {r['error']}")
        else:
            print(f"{eng:<10} {r['s']:>6.2f}  {cfg['n'] / r['s']:>7.0f}   {_fmt_util(r['util'])}")
    if bm and rm:
        print(f"\nbatcher vs ray: {rm / bm:.2f}x  (>1 = batcher faster)")
    sb, sr = res.get("batcher", {}).get("sig"), res.get("ray", {}).get("sig")
    if sb and sr:
        agree = _agreement(sb, sr)
        status = "OK" if agree >= 0.999 else f"MISMATCH ({agree:.3%} agree)"
        print(
            f"correctness: rows b={sb['rows']} r={sr['rows']}  "
            f"pred-agreement={agree:.3%}  [{status}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
