"""Batcher vs Ray Data — distributed LLM batch inference (the vLLM/HF workload).

Text prompts → a causal LM (HF ``transformers`` gpt2, FP16) generating tokens, as a
model-load-once actor pool across every GPU. This is the workload where a data engine's
**cold start** matters most: an LLM load is 7-10 s (a multi-GB model, tens of seconds) —
and Ray Data reloads it on *every execution*, while Batcher keeps the pool warm across
`collect()`s (``distributed.warm_inference_pools``), loading once per session. Greedy
decoding (deterministic) lets the generated text be checked for agreement before timing.

Both engines read the same Parquet prompt shards distributed, run the same seeded model.

Run:
    python benchmarks/cluster/gpu_llm.py                 # gpt2, 2048 prompts
    BENCH_LLM_MODEL=EleutherAI/pythia-410m BENCH_LLM_N=4096 python benchmarks/cluster/gpu_llm.py
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
        "model": os.environ.get("BENCH_LLM_MODEL", "gpt2"),
        "n": int(os.environ.get("BENCH_LLM_N", "2048")),
        "shards": int(os.environ.get("BENCH_LLM_SHARDS", "32")),
        "batch": int(os.environ.get("BENCH_LLM_BATCH", "32")),
        "max_new": int(os.environ.get("BENCH_LLM_MAX_NEW", "32")),
        "num_gpus": float(os.environ.get("BENCH_GPU_NUM_GPUS", "1")),
        "concurrency": os.environ.get("BENCH_GPU_CONCURRENCY", ""),
        "dir": os.environ.get("BENCH_LLM_PARQUET", "/mnt/cluster_storage/gpu_llm"),
        "runs": int(os.environ.get("BENCH_RUNS", "3")),
        "timeout": float(os.environ.get("BENCH_ENGINE_TIMEOUT", "600")),
    }


def write_shards(directory: str, n: int, shards: int) -> str:
    """Write `n` prompts (cycled templates + an index, so each is distinct) as Parquet."""
    import pyarrow.parquet as pq

    os.makedirs(directory, exist_ok=True)
    per = -(-n // shards)
    written = 0
    for s in range(shards):
        lo, hi = s * per, min((s + 1) * per, n)
        if lo >= hi:
            break
        prompts = [f"{_PROMPTS[i % len(_PROMPTS)]} (story {i})" for i in range(lo, hi)]
        tbl = pa.table(
            {"id": pa.array(np.arange(lo, hi, dtype=np.int64)), "prompt": pa.array(prompts)}
        )
        pq.write_table(tbl, os.path.join(directory, f"shard_{s:04d}.parquet"))
        written = hi
    return f"{written} prompts in {shards} shards -> {directory}"


class LLMGen:
    """Model-load-once causal-LM generator: build gpt2 (FP16) once per actor, greedily
    generate ``BENCH_LLM_MAX_NEW`` tokens per prompt. Consumes a numpy batch ``{"id",
    "prompt"}``; returns ``{"id", "text"}`` (the generated continuation)."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = os.environ.get("BENCH_LLM_MODEL", "gpt2")
        self._max_new = int(os.environ.get("BENCH_LLM_MAX_NEW", "32"))
        self._dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok = AutoTokenizer.from_pretrained(model_id)
        self._tok.pad_token = self._tok.eos_token
        self._tok.padding_side = "left"
        dtype = torch.float16 if self._dev == "cuda" else torch.float32
        self._m = (
            AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(self._dev).eval()
        )

    def __call__(self, batch: dict) -> dict:
        import torch

        prompts = [p.decode() if isinstance(p, bytes) else str(p) for p in batch["prompt"]]
        enc = self._tok(prompts, return_tensors="pt", padding=True, truncation=True).to(self._dev)
        with torch.inference_mode():
            out = self._m.generate(
                **enc,
                max_new_tokens=self._max_new,
                do_sample=False,
                pad_token_id=self._tok.eos_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1] :]
        text = self._tok.batch_decode(gen, skip_special_tokens=True)
        return {"id": np.asarray(batch["id"]), "text": np.array(text, dtype=object)}


def batcher_thunk(cfg: dict):
    import batcher as bt

    conc = int(cfg["concurrency"]) if cfg["concurrency"] else None
    ds = bt.read.parquet(f"{cfg['dir']}/*.parquet").map_batches(
        LLMGen,
        output_columns=["id", "text"],
        batch_format="numpy",
        num_gpus=cfg["num_gpus"],
        concurrency=conc,
        batch_size=cfg["batch"],
    )

    def run():
        return _sig(ds.collect(distributed=True))

    return run


def ray_thunk(cfg: dict):
    import ray.data as rd

    conc = int(cfg["concurrency"]) if cfg["concurrency"] else _auto_gpu_actors(cfg["num_gpus"])

    class _RayLLM:
        def __init__(self) -> None:
            self._m = LLMGen()

        def __call__(self, b):
            return self._m(b) if len(b["id"]) else {"id": [], "text": []}

    ds = rd.read_parquet(cfg["dir"]).map_batches(
        _RayLLM,
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
    """Per-id generated text (greedy decode is deterministic → matches across engines)."""
    d = tbl.to_pydict()
    text = [t.decode() if isinstance(t, bytes) else str(t) for t in d["text"]]
    return {"rows": tbl.num_rows, "preds": dict(zip(d["id"], text, strict=False))}


def _agreement(a: dict, b: dict) -> float:
    pa_, pb = a["preds"], b["preds"]
    common = set(pa_) & set(pb)
    return sum(1 for k in common if pa_[k] == pb[k]) / len(common) if common else 0.0


def bench(cfg: dict, n: int) -> dict:
    out: dict = {}
    for eng, builder in (("batcher", batcher_thunk), ("ray", ray_thunk)):
        try:
            run = with_timeout(builder(cfg), cfg["timeout"])
            print(f"  [{eng}] warmup (pays model load) ...")
            warm = run()
            best = float("inf")
            for _ in range(cfg["runs"]):
                t0 = time.perf_counter()
                run()
                best = min(best, time.perf_counter() - t0)
            out[eng] = {"s": best, "sig": warm}
            print(f"  [{eng}] {best:.2f}s  {n / best:.1f} prompt/s")
        except TimeoutError:
            out[eng] = {"error": "TIMEOUT"}
            print(f"  [{eng}] TIMEOUT")
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
    init_batcher_ray(
        forward=("BENCH_LLM_MODEL", "BENCH_LLM_MAX_NEW"),
        hf_cache="/mnt/cluster_storage/hf_cache",
    )


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
        f"model={cfg['model']} prompts={n} batch={cfg['batch']} max_new={cfg['max_new']} "
        f"best-of-{cfg['runs']} (iterative: batcher warm, ray reloads per run)\n"
    )
    res = bench(cfg, n)
    bm, rm = res.get("batcher", {}).get("s"), res.get("ray", {}).get("s")
    print("\nengine     time_s    prompt/s")
    print("-" * 34)
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
        print(f"correctness: text-agreement={agree:.3%}  [{'OK' if agree >= 0.99 else 'MISMATCH'}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
