"""Batcher vs Ray Data — distributed text embeddings (batch-embeddings text workload).

Text → sentence-transformers `all-MiniLM-L6-v2` (the guides' lightweight text embedder) →
384-d embedding vectors, across every GPU. A real HF embedding model (distinct from the
ResNet feature-extractor path): the model loads ~2 s and is reused warm across `collect()`s,
while Ray Data reloads it per execution. `encode()` is passed the full batch size (the
guides' SentenceTransformer internal-batch_size=32 foot-gun avoided) in both engines.

Reads the same prompt shards the LLM benchmark generates; deterministic model → checksum agrees.

Run:
    python benchmarks/cluster/gpu_text_embed.py            # all-MiniLM-L6-v2, prompt shards
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import os
import time

import numpy as np
import pyarrow as pa

print = functools.partial(print, flush=True)

_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_PROMPTS = [
    "The future of artificial intelligence is",
    "In a distant galaxy, a lone explorer discovered",
    "The most important lesson in life is",
    "Scientists recently announced that",
    "Once upon a time, in a village by the sea,",
    "The recipe for a perfect day begins with",
    "Climate change will reshape the world by",
    "The secret to writing good software is",
]


def _cfg() -> dict:
    return {
        "n": int(os.environ.get("BENCH_EMB_N", "8192")),
        "shards": int(os.environ.get("BENCH_EMB_SHARDS", "32")),
        "batch": int(os.environ.get("BENCH_EMB_BATCH", "256")),
        "num_gpus": float(os.environ.get("BENCH_GPU_NUM_GPUS", "1")),
        "dir": os.environ.get("BENCH_EMB_PARQUET", "/mnt/cluster_storage/gpu_text_embed"),
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
        prompts = [f"{_PROMPTS[i % len(_PROMPTS)]} (doc {i})" for i in range(lo, hi)]
        pq.write_table(
            pa.table(
                {"id": pa.array(np.arange(lo, hi, dtype=np.int64)), "prompt": pa.array(prompts)}
            ),
            os.path.join(directory, f"shard_{s:04d}.parquet"),
        )
        written = hi
    return f"{written} texts -> {directory}"


class Embedder:
    """Model-load-once text embedder: build SentenceTransformer once, encode each batch."""

    def __init__(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self._dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._m = SentenceTransformer(_MODEL, device=self._dev)
        self._m.eval()

    def __call__(self, batch) -> dict:
        prompts = [p.decode() if isinstance(p, bytes) else str(p) for p in batch["prompt"]]
        emb = self._m.encode(
            prompts, batch_size=len(prompts), normalize_embeddings=True, show_progress_bar=False
        )
        return {"id": np.asarray(batch["id"]), "emb": np.asarray(emb, dtype=np.float32)}


def batcher_thunk(cfg: dict):
    import batcher as bt

    ds = bt.read.parquet(f"{cfg['dir']}/*.parquet").map_batches(
        Embedder,
        output_columns=["id", "emb"],
        batch_format="numpy",
        num_gpus=cfg["num_gpus"],
        batch_size=cfg["batch"],
    )

    def run():
        return _sig(ds.collect(distributed=True))

    return run


def ray_thunk(cfg: dict):
    import ray.data as rd

    conc = _auto_gpu_actors(cfg["num_gpus"])

    class _RayEmb:
        def __init__(self) -> None:
            self._m = Embedder()

        def __call__(self, b):
            return self._m(b) if len(b["id"]) else {"id": [], "emb": []}

    ds = rd.read_parquet(cfg["dir"]).map_batches(
        _RayEmb,
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
    col = tbl.column("emb")
    col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    try:
        arr = np.asarray(col.to_numpy_ndarray())
    except Exception:
        arr = np.stack([np.asarray(v) for v in col.to_pylist()])
    ids = tbl.column("id").to_numpy(zero_copy_only=False)
    sums = arr.reshape(arr.shape[0], -1).sum(axis=1)
    return {
        "rows": tbl.num_rows,
        "preds": {int(i): round(float(s), 2) for i, s in zip(ids, sums, strict=False)},
    }


def _agreement(a: dict, b: dict) -> float:
    pa_, pb = a["preds"], b["preds"]
    common = set(pa_) & set(pb)
    return sum(1 for k in common if abs(pa_[k] - pb[k]) <= 0.05) / len(common) if common else 0.0


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
            print(f"  [{eng}] warmup (pays model load) ...")
            warm = run()
            best = float("inf")
            for _ in range(cfg["runs"]):
                t0 = time.perf_counter()
                run()
                best = min(best, time.perf_counter() - t0)
            out[eng] = {"s": best, "sig": warm}
            print(f"  [{eng}] {best:.2f}s  {n / best:.1f} text/s")
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
    renv = {
        "py_modules": [pkg],
        "pip": None,
        "env_vars": {"HF_HOME": os.environ.get("HF_HOME", "/mnt/cluster_storage/hf_cache")},
    }
    base = active_config()
    set_config(
        base.replace(
            distributed=dataclasses.replace(base.distributed, ray_address="auto", runtime_env=renv)
        )
    )
    import ray

    if not ray.is_initialized():
        ray.init(address="auto", runtime_env=renv, logging_level="ERROR", log_to_driver=False)


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
    print(f"model=all-MiniLM-L6-v2 texts={n} batch={cfg['batch']} best-of-{cfg['runs']}\n")
    res = bench(cfg, n)
    bm, rm = res.get("batcher", {}).get("s"), res.get("ray", {}).get("s")
    print("\nengine     time_s    text/s")
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
        print(f"correctness: agreement={agree:.3%}  [{'OK' if agree >= 0.99 else 'MISMATCH'}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
