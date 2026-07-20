"""Batcher vs Ray Data — training-data ingest throughput (`iter_torch_batches`).

The distributed-training data-loading workload: stream a dataset to a PyTorch training
loop as ``{column: tensor}`` batches. The engine's job is to keep the GPU fed — read,
Arrow→tensor convert, collate, prefetch — without becoming the bottleneck (the guides
note Ray Data's ``iter_torch_batches`` runs ~20% slower than a native DataLoader from the
Arrow→tensor conversion + IPC). Batcher's loader is zero-copy (DLPack) with background
prefetch. We measure rows/sec delivered as tensors (device="cpu" isolates the loader from
H2D, which is identical for both), correctness-gated on the row count + a checksum.

Run:
    python benchmarks/cluster/gpu_train_ingest.py            # 200k rows x 1024-d float
    BENCH_INGEST_N=500000 BENCH_INGEST_DIM=512 python benchmarks/cluster/gpu_train_ingest.py
"""

from __future__ import annotations

import functools
import os
import time

import numpy as np
import pyarrow as pa
from _ray_env import init_ray

print = functools.partial(print, flush=True)

_SEED = 1234


def _cfg() -> dict:
    return {
        "n": int(os.environ.get("BENCH_INGEST_N", "200000")),
        "dim": int(os.environ.get("BENCH_INGEST_DIM", "1024")),
        "batch": int(os.environ.get("BENCH_INGEST_BATCH", "256")),
        "prefetch": int(os.environ.get("BENCH_INGEST_PREFETCH", "2")),
        # Training shuffle: 0 = none; >0 = a per-epoch local-shuffle buffer of that many rows
        # (the guides' cheap block-order + local buffer, vs a global random_shuffle O(n) OOM).
        "shuffle": int(os.environ.get("BENCH_INGEST_SHUFFLE", "0")),
        "runs": int(os.environ.get("BENCH_RUNS", "3")),
    }


def make_table(n: int, dim: int) -> pa.Table:
    rng = np.random.default_rng(_SEED)
    feats = rng.standard_normal((n, dim), dtype=np.float32)
    labels = rng.integers(0, 1000, size=n, dtype=np.int64)
    return pa.table(
        {
            "feat": pa.FixedSizeListArray.from_arrays(pa.array(feats.reshape(-1)), dim),
            "label": pa.array(labels),
        }
    )


def batcher_iter(table: pa.Table, cfg: dict):
    import batcher as bt

    ds = bt.from_arrow(table)

    shuf = cfg["shuffle"] or None

    def run():
        seen, checksum = 0, 0.0
        for b in ds.ml.iter_torch_batches(
            batch_size=cfg["batch"],
            prefetch_batches=cfg["prefetch"],
            device="cpu",
            local_shuffle_buffer_size=shuf,
        ):
            seen += int(b["label"].shape[0])
            checksum += float(b["label"].sum().item())
        return {"rows": seen, "checksum": round(checksum, 1)}

    return run


def ray_iter(table: pa.Table, cfg: dict):
    import ray.data as rd

    ds = rd.from_arrow(table)
    shuf = cfg["shuffle"] or None

    def run():
        seen, checksum = 0, 0.0
        for b in ds.iter_torch_batches(
            batch_size=cfg["batch"],
            prefetch_batches=cfg["prefetch"],
            local_shuffle_buffer_size=shuf,
        ):
            seen += int(b["label"].shape[0])
            checksum += float(b["label"].sum().item())
        return {"rows": seen, "checksum": round(checksum, 1)}

    return run


def main() -> int:
    cfg = _cfg()
    init_ray()
    table = make_table(cfg["n"], cfg["dim"])
    print(
        f"rows={cfg['n']} dim={cfg['dim']} batch={cfg['batch']} "
        f"prefetch={cfg['prefetch']} best-of-{cfg['runs']}\n"
    )
    res: dict = {}
    for eng, builder in (("batcher", batcher_iter), ("ray", ray_iter)):
        run = builder(table, cfg)
        sig = run()  # warmup
        best = float("inf")
        for _ in range(cfg["runs"]):
            t0 = time.perf_counter()
            run()
            best = min(best, time.perf_counter() - t0)
        res[eng] = {"s": best, "sig": sig}
        print(f"{eng:<10} {best:.2f}s  {cfg['n'] / best:>10.0f} rows/s")
    bm, rm = res["batcher"]["s"], res["ray"]["s"]
    print(f"\nbatcher vs ray: {rm / bm:.2f}x  (>1 = batcher faster)")
    sb, sr = res["batcher"]["sig"], res["ray"]["sig"]
    ok = sb["rows"] == sr["rows"] and abs(sb["checksum"] - sr["checksum"]) < 1.0
    print(f"correctness: rows b={sb['rows']} r={sr['rows']}  [{'OK' if ok else 'MISMATCH'}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
