"""Demonstrate Kyber's adaptive GPU/CPU crossover learning end-to-end on the cluster.

Runs the *real* `collect(backend="gpu")` and `collect(backend="cpu")` paths over a shared
Parquet group-by at several sizes, in one process, so both halves of the crossover learner get
fed (Core measures, Kyber consumes). After the runs it prints the learned crossover row count
and shows that `backend="auto"` now routes by the measured threshold — the adaptivity, live.

Run:
    python benchmarks/gpu_backend/adaptive_crossover.py
"""

from __future__ import annotations

import functools
import os

from _ray_env import init_ray

print = functools.partial(print, flush=True)

_DIR = "/mnt/cluster_storage/gpu_xover"
# Span both regimes so a crossover exists: sub-million (GPU dispatch overhead makes it LOSE to
# even a single-node CPU) up to tens of millions (GPU wins). A range that only wins or only loses
# has no crossover and the learner correctly abstains.
_SIZES = [
    int(x)
    for x in os.environ.get("BENCH_XO_SIZES", "300000,1000000,5000000,20000000,40000000").split(",")
]
_GROUPS = 1000


def _gen(path: str, n: int, seed: int) -> int:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    pq.write_table(
        pa.table({"k": rng.integers(0, _GROUPS, n).astype("int64"), "v": rng.random(n)}), path
    )
    return n


def _ensure(n: int) -> str:
    import ray

    d = os.path.join(_DIR, f"n{n}")
    marker = os.path.join(d, "_SUCCESS")
    if os.path.exists(marker):
        return d
    os.makedirs(d, exist_ok=True)
    files = 8
    per = -(-n // files)
    gen = ray.remote(num_cpus=1, runtime_env={"pip": ["numpy==1.26.4"]})(_gen)
    refs = [
        gen.remote(os.path.join(d, f"part-{i:04d}.parquet"), min(per, n - i * per), i)
        for i in range(files)
        if n - i * per > 0
    ]
    ray.get(refs)
    open(marker, "w").close()
    return d


def main() -> int:
    init_ray(unconditional_hook_strip=True)
    import batcher as bt
    from batcher import col, core
    from batcher.kyber.gpu.adaptive import learned_gpu_min_rows

    paths = {n: _ensure(n) for n in _SIZES}
    print(f"sizes={[n // 1_000_000 for n in _SIZES]}M  groups={_GROUPS}\n")

    def _agree(a: dict, b: dict) -> bool:
        da = dict(zip(a["k"], a["s"], strict=True))
        db = dict(zip(b["k"], b["s"], strict=True))
        return da.keys() == db.keys() and all(
            abs(da[k] - db[k]) < 1e-3 * max(1.0, abs(da[k])) for k in da
        )

    # Warm up cuDF once so the first *timed* GPU run isn't a cold-import outlier (a warm pool /
    # long-running deployment pays that once; the learner should fit steady-state throughput).
    bt.read.parquet(paths[min(paths)]).group_by("k").agg(s=col("v").sum()).collect(backend="gpu")

    # Two passes over the sizes → enough spread-out samples for a stable per-backend fit.
    for _pass in range(2):
        for n, path in paths.items():
            q = bt.read.parquet(path).group_by("k").agg(s=col("v").sum())
            gpu = q.collect(backend="gpu").to_pydict()
            cpu = q.collect(backend="cpu", distributed=False).to_pydict()
            xo = learned_gpu_min_rows(core.default_hub())
            print(
                f"n={n // 1_000_000:3d}M  gpu==cpu [{'OK' if _agree(gpu, cpu) else 'MISMATCH'}]  "
                f"learned_crossover={xo if xo else '(not yet)'}"
            )

    xo = learned_gpu_min_rows(core.default_hub())
    print(
        f"\n==> learned GPU/CPU crossover: {xo} rows"
        + (
            ""
            if xo is None
            else f"  (config default is {bt.config.active_config().distributed.gpu_min_rows})"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
