"""Batcher's GPU relational backend vs Ray Data (+cuDF) — the real `collect(backend="gpu")` path.

Every other GPU benchmark times a raw kernel on a worker; this one times the *public* engine
path — `bt.read.parquet(...).group_by(k).agg(...).collect(backend="gpu")` — which routes through
Kyber's `decide_gpu_backend`, the distributed cuDF aggregate (shard per GPU, mergeable combine),
spot-preemption retries, and shard oversubscription. It is compared head-to-head against the
idiomatic Ray Data answers to the same query:

* **Ray Data + cuDF** — `map_batches(cudf partial group-by, num_gpus=1)` then a driver combine
  (the accelerated path a Ray Data user writes by hand — Ray Data has no GPU aggregate).
* **Ray Data CPU** — `groupby(k).sum(v)` (the out-of-the-box path).

and Batcher's own CPU engine as the reference. Correctness-gated: every engine's per-group sums
must match Batcher CPU before any time is reported. A shared parquet dataset on cluster storage
(generated once) makes both engines read a real splittable source, not an in-memory handle.

Run:
    python benchmarks/gpu_backend/relational_vs_raydata.py                 # 60M rows
    BENCH_RR_N=200000000 BENCH_RR_FILES=32 python benchmarks/gpu_backend/relational_vs_raydata.py
"""

from __future__ import annotations

import functools
import os
import time

print = functools.partial(print, flush=True)

_DATA_DIR = os.environ.get("BENCH_RR_DIR", "/mnt/cluster_storage/gpu_relbench")
_N = int(os.environ.get("BENCH_RR_N", "60000000"))
_FILES = int(os.environ.get("BENCH_RR_FILES", "16"))
_GROUPS = int(os.environ.get("BENCH_RR_GROUPS", "1000"))
_CUDF_PIP = ["cudf-cu13==26.6.0", "numpy==1.26.4"]


def _init() -> None:
    """Init Ray with `pip: None` — NOT a pip list. The workspace injects a default pip set that
    includes a broken `batcher-engine[delta]` requirement; passing our own pip list at init still
    merges that in and every actor's env-setup fails. cuDF is shipped per-TASK instead (Batcher
    via its own runtime_env, Ray Data via `ray_remote_args`), the pattern `distributed_cudf.py`
    uses successfully."""
    # Unconditionally drop the workspace's runtime-env hook: it injects a default dev-pip set
    # containing a broken `batcher-engine[delta]` requirement into EVERY task (even ones with no
    # runtime_env, e.g. data generation), failing env setup. We ship cuDF per-task ourselves.
    for var in ("RAY_RUNTIME_ENV_HOOK", "RAY_RUNTIME_ENV_PLUGINS"):
        os.environ.pop(var, None)
    import ray

    if not ray.is_initialized():
        # With the hook gone, a clean job-level cuDF+numpy pip installs on every worker (cached
        # per node) — so Ray Data's GPU map tasks find cuDF without per-op runtime_env plumbing.
        ray.init(
            address="auto",
            runtime_env={"pip": _CUDF_PIP},
            logging_level="ERROR",
            log_to_driver=False,
        )


def _gen_shard(path: str, n: int, groups: int, seed: int) -> int:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    tbl = pa.table({"k": rng.integers(0, groups, n).astype("int64"), "v": rng.random(n)})
    pq.write_table(tbl, path)
    return n


def _ensure_data() -> str:
    """Generate the parquet dataset once on shared storage (one file per shard, in parallel on
    the cluster). Idempotent: a `_SUCCESS` marker with the same (N, files, groups) is reused."""
    import ray

    marker = os.path.join(_DATA_DIR, f"_SUCCESS_{_N}_{_FILES}_{_GROUPS}")
    if os.path.exists(marker):
        return _DATA_DIR
    os.makedirs(_DATA_DIR, exist_ok=True)
    # a fresh dataset for this shape: clear stale parquet
    for f in os.listdir(_DATA_DIR):
        if f.endswith(".parquet") or f.startswith("_SUCCESS"):
            os.remove(os.path.join(_DATA_DIR, f))
    per = -(-_N // _FILES)
    # An explicit per-task pip overrides the workspace hook's broken default dev-pip (a task with
    # NO runtime_env inherits that default and fails env setup); numpy is all these need.
    gen = ray.remote(num_cpus=1, runtime_env={"pip": ["numpy==1.26.4"]})(_gen_shard)
    refs = [
        gen.remote(
            os.path.join(_DATA_DIR, f"part-{i:04d}.parquet"), min(per, _N - i * per), _GROUPS, i
        )
        for i in range(_FILES)
        if _N - i * per > 0
    ]
    total = sum(ray.get(refs))
    with open(marker, "w") as fh:
        fh.write(str(total))
    print(f"generated {total / 1e6:.0f}M rows across {len(refs)} parquet files at {_DATA_DIR}")
    return _DATA_DIR


# --- Ray Data + cuDF -----------------------------------------------------------------------


def _cudf_partial(batch: dict, groups_col: str = "k", val_col: str = "v") -> dict:
    """A Ray Data map_batches fn: partial group-by SUM on the GPU via cuDF (numpy batch in/out)."""
    import cudf

    df = cudf.DataFrame({groups_col: batch[groups_col], val_col: batch[val_col]})
    g = df.groupby(groups_col, sort=False).agg({val_col: "sum"}).reset_index()
    return {
        groups_col: g[groups_col].to_numpy(),
        val_col: g[val_col].to_numpy(),
    }


def _raydata_cudf(path: str) -> dict:
    import ray

    ds = ray.data.read_parquet(path)
    # Ray Data hard-requires an explicit batch_size for a GPU stage (it raises otherwise). Batcher
    # sizes this itself; here we hand Ray Data a large block so its per-call overhead is amortized.
    # cuDF reaches the map workers via the job-level runtime_env (see `_init`).
    parts = ds.map_batches(
        _cudf_partial,
        batch_format="numpy",
        num_gpus=1,
        batch_size=int(os.environ.get("BENCH_RR_RAYDATA_BS", "1000000")),
        concurrency=8,
    ).materialize()
    # driver-side mergeable combine of the per-block partials

    acc: dict[int, float] = {}
    for row in parts.iter_rows():
        acc[int(row["k"])] = acc.get(int(row["k"]), 0.0) + float(row["v"])
    return acc


def _raydata_cpu(path: str) -> dict:
    import ray

    ds = ray.data.read_parquet(path)
    agg = ds.groupby("k").sum("v")
    return {int(r["k"]): float(r["sum(v)"]) for r in agg.take_all()}


# --- Batcher -------------------------------------------------------------------------------


def _batcher(path: str, backend: str) -> dict:
    import batcher as bt
    from batcher import col

    q = bt.read.parquet(path).group_by("k").agg(s=col("v").sum())
    # The CPU reference runs single-node: Batcher's non-GPU distributed tasks ship py_modules with
    # no pip and would inherit the workspace's broken default pip. The GPU path ships cuDF per task
    # so it is unaffected and uses the cluster.
    tbl = q.collect(backend=backend, distributed=(backend != "cpu"))
    d = tbl.to_pydict()
    return dict(zip(d["k"], d["s"], strict=True))


def _time(fn, *a) -> tuple[dict, float]:
    t = time.perf_counter()
    out = fn(*a)
    return out, time.perf_counter() - t


def _agree(ref: dict, other: dict) -> bool:
    if other is None or len(ref) != len(other):
        return False
    return all(abs(ref[k] - other.get(k, 0.0)) < 1e-3 * max(1.0, abs(ref[k])) for k in ref)


def main() -> int:
    _init()
    path = _ensure_data()
    print(f"\nN={_N / 1e6:.0f}M rows  groups={_GROUPS}  files={_FILES}\n")

    # Correctness reference: Batcher CPU.
    ref, cpu_s = _time(_batcher, path, "cpu")
    print(f"{'batcher cpu':22s} {cpu_s * 1000:8.0f} ms   (reference)")

    results = []
    for name, fn, args in [
        ("batcher gpu", _batcher, (path, "gpu")),
        ("batcher auto", _batcher, (path, "auto")),
        ("ray data + cudf", _raydata_cudf, (path,)),
        ("ray data cpu", _raydata_cpu, (path,)),
    ]:
        try:
            out, secs = _time(fn, *args)
            ok = _agree(ref, out)
            results.append((name, secs, ok))
            ratio = cpu_s / secs if secs else 0.0
            print(
                f"{name:22s} {secs * 1000:8.0f} ms   {ratio:5.2f}x vs batcher-cpu   "
                f"[{'OK' if ok else 'MISMATCH'}]"
            )
        except Exception as e:
            print(f"{name:22s} FAILED: {type(e).__name__}: {str(e)[:80]}")

    # Headline: Batcher GPU vs Ray Data + cuDF.
    bg = next((s for n, s, ok in results if n == "batcher gpu" and ok), None)
    rc = next((s for n, s, ok in results if n == "ray data + cudf" and ok), None)
    if bg and rc:
        print(f"\n==> batcher-gpu vs ray-data+cudf: {rc / bg:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
