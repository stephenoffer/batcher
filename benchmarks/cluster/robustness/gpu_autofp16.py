"""Auto half-precision on the managed ``ds.ml.infer`` path — the compute-bound lever.

A compute-bound model saturates the GPU at the same FLOPs for every engine, so throughput
is set by the *precision* the model runs in. Ray Data's ``map_batches`` over a HF pipeline
defaults to FP32 unless the user hand-sets ``torch_dtype``; Batcher's ``ds.ml.infer(<model
id>)`` auto-detects a half-precision-capable GPU and casts to BF16/FP16 (`recommend_inference_
dtype`) — ~2x tensor-core throughput at negligible inference quality loss.

This isolates the dtype lever on **one** engine (Batcher) so nothing else varies: same
model, same data, same actor pool. FP16 is the managed auto path; FP32 is the identical
pipeline pinned to ``torch_dtype=float32`` via a class UDF. Correctness-gated: the two
precisions must agree on ≥ the required fraction of predicted labels before any timing.

Run:
    python benchmarks/cluster/robustness/gpu_autofp16.py
    BENCH_FP16_MODEL=<hf-classifier-id> python benchmarks/cluster/robustness/gpu_autofp16.py
"""

from __future__ import annotations

import functools
import os
import time

import numpy as np
import pyarrow as pa

print = functools.partial(print, flush=True)

_SEED = 1234
_TEXTS = [
    "This movie was an absolute delight from start to finish.",
    "A tedious, overlong slog that I could not wait to end.",
    "The performances elevate an otherwise thin script.",
    "I have never been so bored in a theater in my life.",
    "A quietly moving portrait of grief and recovery.",
    "Loud, incoherent, and utterly charmless.",
    "Easily the best film I have seen this year.",
    "Forgettable and derivative in every possible way.",
]


def _cfg() -> dict:
    return {
        "model": os.environ.get(
            "BENCH_FP16_MODEL", "distilbert-base-uncased-finetuned-sst-2-english"
        ),
        "n": int(os.environ.get("BENCH_FP16_N", "8192")),
        "shards": int(os.environ.get("BENCH_FP16_SHARDS", "32")),
        "batch": int(os.environ.get("BENCH_FP16_BATCH", "64")),
        "num_gpus": float(os.environ.get("BENCH_GPU_NUM_GPUS", "1")),
        "concurrency": int(os.environ.get("BENCH_GPU_CONCURRENCY", "0")) or None,
        "dir": os.environ.get("BENCH_FP16_PARQUET", "/mnt/cluster_storage/gpu_autofp16"),
        "min_agree": float(os.environ.get("BENCH_FP16_MIN_AGREE", "0.99")),
        "runs": int(os.environ.get("BENCH_RUNS", "3")),
    }


def write_shards(directory: str, n: int, shards: int) -> str:
    import pyarrow.parquet as pq

    os.makedirs(directory, exist_ok=True)
    per = -(-n // shards)
    idx = 0
    for s in range(shards):
        rows = min(per, n - idx)
        if rows <= 0:
            break
        texts = [f"{_TEXTS[(idx + i) % len(_TEXTS)]} (#{idx + i})" for i in range(rows)]
        pq.write_table(pa.table({"text": texts}), os.path.join(directory, f"part-{s:05d}.parquet"))
        idx += rows
    return directory


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
        # Propagate to worker actors: HF cache location and the model id the FP32 class UDF
        # reads (driver-process os.environ does NOT reach a remote actor otherwise).
        env = {
            "pip": None,
            "env_vars": {
                "HF_HOME": os.environ.get("HF_HOME", "/mnt/cluster_storage/hf"),
                "BENCH_FP16_MODEL_ID": os.environ["BENCH_FP16_MODEL_ID"],
            },
        }
        ray.init(address="auto", runtime_env=env, logging_level="ERROR", log_to_driver=False)


class _Fp32Pipeline:
    """The identical HF classification pipeline pinned to FP32 (the Ray Data default)."""

    def __init__(self) -> None:
        import torch
        from transformers import pipeline

        model = os.environ["BENCH_FP16_MODEL_ID"]
        device = 0 if torch.cuda.is_available() else -1
        self._pipe = pipeline(
            "text-classification", model=model, device=device, torch_dtype=torch.float32
        )

    def __call__(self, batch: pa.RecordBatch) -> dict:
        preds = self._pipe(batch.column("text").to_pylist(), truncation=True)
        return {"label": [p["label"] for p in preds]}


def _labels(out) -> list[str]:
    tbl = out if isinstance(out, pa.Table) else pa.Table.from_batches(list(out))
    return tbl.column("label").to_pylist()


def main() -> int:
    cfg = _cfg()
    os.environ["BENCH_FP16_MODEL_ID"] = cfg["model"]
    _init()
    import batcher as bt

    directory = write_shards(cfg["dir"], cfg["n"], cfg["shards"])
    print(f"model={cfg['model']}  rows={cfg['n']}  shards={cfg['shards']}  batch={cfg['batch']}\n")

    def read():
        return bt.read.parquet(directory).repartition(cfg["shards"])

    # FP16 (managed auto path) and FP32 (identical pipeline, pinned) — same engine/model/data.
    def fp16_run():
        return _labels(
            read()
            .ml.infer(
                cfg["model"],
                column="text",
                output_column="label",
                task="text-classification",
                batch_size=cfg["batch"],
                num_gpus=cfg["num_gpus"],
                concurrency=cfg["concurrency"],
            )
            .collect()
        )

    def fp32_run():
        return _labels(
            read()
            .map_batches(
                _Fp32Pipeline,
                output_columns=["label"],
                batch_size=cfg["batch"],
                num_gpus=cfg["num_gpus"],
                concurrency=cfg["concurrency"],
                batch_format="pyarrow",
            )
            .collect()
        )

    # A GPU inference pool stays resident (holding every GPU) across `collect()`s, so the two
    # precisions MUST run one-at-a-time with a release between — otherwise the fp16 pool holds
    # all the GPUs and the fp32 pool can never schedule (a deadlock). Run each phase (warm-up
    # for correctness + timed runs) to completion, release, then the next.
    from batcher.dist.executors.map import release_inference_pools

    def phase(name: str, fn) -> tuple[list[str], float]:
        labels = fn()  # warm-up + correctness sample
        best = float("inf")
        for _ in range(cfg["runs"]):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        release_inference_pools()
        print(f"{name:<6} {cfg['n'] / best:8.0f} rows/s  ({best * 1000:.0f} ms)")
        return labels, best

    fp16_labels, fp16_s = phase("fp16", fp16_run)
    fp32_labels, fp32_s = phase("fp32", fp32_run)
    agree = np.mean([a == b for a, b in zip(fp16_labels, fp32_labels, strict=True)])
    print(f"label agreement fp16 vs fp32: {agree:.4f} (required ≥ {cfg['min_agree']})")
    if agree < cfg["min_agree"]:
        print("FAIL: precisions disagree beyond tolerance — result not trusted")
        return 1
    ratio = fp32_s / fp16_s
    print(f"\nauto-FP16 vs FP32 on the managed path: {ratio:.2f}x")
    # Honest read: half precision only wins when *compute* dominates. If the per-collect
    # wall time is dominated by model load / actor warm-up (small N, a small model), dtype
    # is noise and FP16 can even lose (its load is marginally heavier). A ratio < 1 here
    # means the run was setup-bound, not that FP16 is slow — scale N and the model up (or
    # keep the pool warm) until compute dominates before reading it as the FP16 speedup.
    if ratio < 1.0:
        print(
            "note: setup-bound run (load/warm-up >> compute) — not a valid compute-bound "
            "test; increase BENCH_FP16_N and use a larger model to isolate the FP16 win"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
