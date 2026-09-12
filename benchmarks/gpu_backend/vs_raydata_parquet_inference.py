"""Batcher vs Ray Data — GPU batch inference over a **large** Parquet feature table.

`vs_ray_daft_gpu_inference.py` races the same engines over a JPEG corpus, and that corpus is
1 GiB of 211,742 tiny objects: the fastest arm finishes in about eleven seconds, of which most
is per-object S3 latency and each engine's fixed startup. A ratio taken there is a ratio
between two schedulers' warm-up costs, and it is not a throughput measurement of anything.

This is the same comparison sized so that it is one. The corpus is a generated Parquet feature
table on shared cluster storage -- a splittable source both engines read natively -- and the
model is wide enough that the accelerators, not the object store, decide the wall time. At the
defaults each arm runs for minutes, which is the point.

The workload is the ordinary one: score every row through a model that is **loaded once per
actor**, and reduce on the workers. Both engines get one actor per device, the same batch
size, the same seeded weights and the same reduction, so what is being compared is read,
batch assembly, actor dispatch and overlap.

**The reduction happens on the workers.** Every arm ends in a count and a sum, a few bytes per
engine, so no engine is charged a transfer the other is not. That shape is also what makes
this benchmark worth having: `map_batches -> aggregate` is the shape a batch-inference job
actually has, and on Batcher it is a different executor route from `map_batches` alone.

**cupy, not torch**, for the reason `vs_ray_daft_gpu_inference.py` gives: these workers carry
cupy and no torch, and shipping a torch wheel would charge every arm a multi-minute install
and then measure it. The scorer is dense matrix multiplication either way.

Run (needs ray on the driver and a GPU fleet with a shared mount):
    python benchmarks/gpu_backend/vs_raydata_parquet_inference.py
    BENCH_PI_ROWS=40000000 python benchmarks/gpu_backend/vs_raydata_parquet_inference.py
    BENCH_PI_WIDTH=2 BENCH_PI_RUNS=3 python benchmarks/gpu_backend/vs_raydata_parquet_inference.py
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

print = functools.partial(print, flush=True)

#: Weight seed. Every actor in every engine builds the same network from it.
_SEED = 20260910

#: Where the generated corpus lives. A shared mount, so both engines read the same bytes from
#: the same place and neither is charged a copy.
_DIR = os.environ.get("BENCH_PI_DIR", "/mnt/cluster_storage/gpu_inferbench")

#: Relative agreement required on the returned checksum. Both engines run the identical
#: `float32` network over identical rows, so the only disagreement available is the order a
#: float sum is reassociated in, which is the reassociation the distributed contract already
#: allows. Tight enough that a dropped or duplicated partition still fails the run.
_AGREE_RTOL = 1e-6


def _cfg() -> dict:
    return {
        "rows": int(os.environ.get("BENCH_PI_ROWS", "10000000")),
        "dim": int(os.environ.get("BENCH_PI_DIM", "256")),
        "files": int(os.environ.get("BENCH_PI_FILES", "64")),
        "batch": int(os.environ.get("BENCH_PI_BATCH", "16384")),
        "width": max(1, int(os.environ.get("BENCH_PI_WIDTH", "1"))),
        "runs": int(os.environ.get("BENCH_PI_RUNS", "2")),
        "engines": [e for e in os.environ.get("BENCH_PI_ENGINES", "batcher,ray").split(",") if e],
    }


def _layers(dim: int, width: int) -> tuple[tuple[int, int], ...]:
    """Dense layer shapes. Wide enough that the device, not the read, sets the wall time."""
    return (
        (dim, 2048 * width),
        (2048 * width, 2048 * width),
        (2048 * width, 1024 * width),
        (1024 * width, 16),
    )


def _weights(dim: int, width: int) -> list[np.ndarray]:
    rng = np.random.default_rng(_SEED)
    return [
        (rng.standard_normal((a, b)) / np.sqrt(a)).astype(np.float32)
        for a, b in _layers(dim, width)
    ]


class Scorer:
    """Model-load-once scorer whose weights live on a device.

    `__init__` is the expensive per-actor step every batch-inference framework exists to
    amortize: build the network on the host, move it to the GPU, once. `__call__` scores a
    batch of feature rows and returns one `float64` score each.
    """

    def __init__(self, dim: int, width: int) -> None:
        import cupy as cp

        self._cp = cp
        self._weights = [cp.asarray(w) for w in _weights(dim, width)]
        cp.cuda.Stream.null.synchronize()  # materialize the context and kernels now

    def score(self, feats: np.ndarray) -> np.ndarray:
        cp = self._cp
        if feats.shape[0] == 0:
            return np.empty(0, dtype="float64")
        x = cp.asarray(feats, dtype=cp.float32)
        for weight in self._weights:
            x = cp.maximum(x @ weight, 0)
        return cp.asnumpy(x.max(axis=1).astype(cp.float64))


def _feats(column) -> np.ndarray:
    """A batch's feature column as one `(B, D)` float32 array, with no Python object per row.

    **The batch format is the single largest thing a benchmark of this shape can get wrong**,
    and the two engines do not agree on it. Batcher's `batch_format="numpy"` hands a
    fixed-size-list column over as a real `(B, D)` array. Ray Data's hands the same column
    over as an **object** array holding one small ndarray per row, so the obvious
    `np.asarray(...)` there builds 16,384 Python objects per batch -- which is a measurement
    of a conversion, not of an engine, and it is the defect that made the sibling image
    benchmark report 20x the wrong way round before it was found.

    So each engine gets its own zero-copy path: the array it already has for Batcher, and for
    Ray Data the Arrow buffer underneath (`batch_format="pyarrow"` -> `flatten()`), which is
    the same bytes with no object per row. The object-array branch stays as a correctness
    fallback for a format neither of those covers, and is never on either engine's path.
    """
    import pyarrow as pa

    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    if isinstance(column, pa.Array):
        dim = column.type.list_size
        return column.flatten().to_numpy(zero_copy_only=False).reshape(len(column), dim)
    arr = np.asarray(column)
    if arr.dtype == object:
        return np.stack([np.asarray(row) for row in arr])
    return arr.reshape(arr.shape[0], -1)


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #
#: The shard generator's body, as source, so the remote function can be built **inside**
#: `_ensure_corpus` and therefore pickled by value.
#:
#: A module-level function is pickled by *module path*, which resolves only while this file is
#: `__main__` — the way it is normally run. Import it instead (a sweep driver, a second
#: benchmark reusing the corpus) and every generator task dies with
#: `ModuleNotFoundError: No module named 'vs_raydata_parquet_inference'` before a row is
#: written. Defining the function in a local scope makes cloudpickle serialize the code
#: itself, which works either way.
def _make_shard_generator():
    """A `_gen_shard(path, rows, dim, seed)` closure, defined locally so it ships by value."""

    def _gen_shard(path: str, rows: int, dim: int, seed: int) -> int:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        rng = np.random.default_rng(seed)
        feats = rng.standard_normal((rows, dim), dtype=np.float32)
        table = pa.table(
            {
                "feat": pa.FixedSizeListArray.from_arrays(pa.array(feats.reshape(-1)), dim),
                "id": pa.array(np.arange(rows, dtype="int64")),
            }
        )
        pq.write_table(table, path, compression="none")
        return rows

    return _gen_shard


def _ensure_corpus(cfg: dict) -> str:
    """Generate the corpus once on the shared mount, in parallel on the cluster.

    Idempotent through a `_SUCCESS` marker naming the shape, so a re-run of the benchmark
    reads the same bytes rather than regenerating them -- which also keeps a repeated run
    from measuring the generator.
    """
    import ray

    marker = os.path.join(_DIR, f"_SUCCESS_{cfg['rows']}_{cfg['dim']}_{cfg['files']}")
    if os.path.exists(marker):
        return _DIR
    os.makedirs(_DIR, exist_ok=True)
    for name in os.listdir(_DIR):
        if name.endswith(".parquet") or name.startswith("_SUCCESS"):
            os.remove(os.path.join(_DIR, name))
    per = -(-cfg["rows"] // cfg["files"])
    gen = ray.remote(num_cpus=1)(_make_shard_generator())
    refs = [
        gen.remote(
            os.path.join(_DIR, f"part-{i:04d}.parquet"),
            min(per, cfg["rows"] - i * per),
            cfg["dim"],
            i,
        )
        for i in range(cfg["files"])
        if cfg["rows"] - i * per > 0
    ]
    total = sum(ray.get(refs))
    Path(marker).write_text(str(total))
    gib = total * cfg["dim"] * 4 / 2**30
    print(f"# generated {total:,} rows ({gib:.1f} GiB) across {len(refs)} files at {_DIR}")
    return _DIR


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #
def batcher_thunk(path: str, devices: int, cfg: dict):
    """`read.parquet` -> a GPU class UDF over an actor pool -> a distributed aggregate."""
    import batcher as bt
    from batcher import col
    from batcher.api.functions import count

    dim, width = cfg["dim"], cfg["width"]

    class _Scorer:
        def __init__(self) -> None:
            self._scorer = Scorer(dim, width)

        def __call__(self, batch) -> dict:
            return {"pred": self._scorer.score(_feats(batch["feat"]))}

    ds = (
        bt.read.parquet(path)
        .map_batches(
            _Scorer,
            output_columns=["pred"],
            batch_format="numpy",
            concurrency=devices,
            batch_size=cfg["batch"],
            num_gpus=1,
        )
        .agg(n=count(), s=col("pred").sum())
    )

    def run():
        out = ds.collect(distributed=True).to_pydict()
        return int(out["n"][0]), float(out["s"][0])

    return run


def ray_thunk(path: str, devices: int, cfg: dict):
    """`read_parquet` -> `map_batches` over a GPU actor pool -> `aggregate`.

    The pattern Ray Data's own batch-inference guides are written around, so it is the arm
    this benchmark most has to get right rather than merely run.
    """
    import ray.data as rd
    from ray.data.aggregate import Count, Sum

    dim, width = cfg["dim"], cfg["width"]

    class _Scorer:
        def __init__(self) -> None:
            self._scorer = Scorer(dim, width)

        def __call__(self, batch) -> dict:
            return {"pred": self._scorer.score(_feats(batch.column("feat")))}

    def run():
        # `batch_format="pyarrow"` is Ray Data's zero-copy path for this column type; its
        # numpy format hands over one Python object per row. See `_feats`.
        ds = rd.read_parquet(path).map_batches(
            _Scorer,
            batch_format="pyarrow",
            concurrency=devices,
            num_gpus=1,
            batch_size=cfg["batch"],
        )
        agg = ds.aggregate(Sum("pred"), Count())
        return int(agg["count()"]), float(agg["sum(pred)"])

    return run


THUNKS = {"batcher": batcher_thunk, "ray": ray_thunk}


def _release_batcher_pools() -> None:
    """Give every device Batcher's session-warm pool holds back before the next arm runs.

    Batcher keeps an inference pool warm across `collect()`s so the model loads once per
    session, and its idle window is minutes -- sized for an interactive session, not for a
    sweep whose next arm starts a second later. Left held, the engine measured next cannot
    place its own pool at all, and nothing reports it: it simply never runs. Releasing here
    is the fair boundary rather than a handicap, since each arm is timed after its own
    untimed warm-up and so still reloads once and reuses across its timed runs.
    """
    with contextlib.suppress(Exception):
        from batcher.dist.executors.map import release_inference_pools

        release_inference_pools()
    with contextlib.suppress(Exception):
        from batcher.dist.fleet import release_session_fleet

        release_session_fleet()


def _device_share(devices: int, cfg: dict) -> float:
    """Seconds one device spends on the model for its share of the corpus, measured here.

    Printed beside every ratio because it is the number that says what the ratio is *about*.
    Two engines running the identical kernel on the identical devices can only differ in how
    well they keep those devices fed, so a ratio near 1.0 with a model share near 1.0 is the
    engines agreeing, and the same ratio with a model share of 0.6 is 40% of the wall clock
    sitting in each engine's feeding path and available to whichever one improves it.
    """
    import ray

    from batcher.dist.gpu.tasks import gpu_task_options

    @ray.remote(**dict(gpu_task_options(num_gpus=1.0)))
    def _time_the_model(rows: int, batch: int, dim: int, width: int) -> float:
        import time as t

        import numpy as np

        scorer = Scorer(dim, width)
        feats = np.zeros((batch, dim), dtype="float32")
        scorer.score(feats)  # warm the kernels
        t0 = t.perf_counter()
        done = 0
        while done < rows:
            scorer.score(feats)
            done += batch
        return t.perf_counter() - t0

    per_device = max(1, cfg["rows"] // max(1, devices))
    return float(
        ray.get(_time_the_model.remote(per_device, cfg["batch"], cfg["dim"], cfg["width"]))
    )


def _time(thunk, runs: int) -> tuple[float, tuple[int, float] | None, str]:
    """Best-of-`runs` after one untimed warm-up, plus the signature and any error."""
    try:
        run = thunk()
        signature = run()
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            signature = run()
            times.append(time.perf_counter() - t0)
    except Exception as exc:
        _release_batcher_pools()
        return float("nan"), None, f"{type(exc).__name__}: {str(exc)[:200]}"
    _release_batcher_pools()
    return min(times), signature, ""


def _verdict(results: dict, cfg: dict, share: float) -> None:
    """Agreement first, then the ratio. Two engines that disagree did different work."""
    good = {k: v for k, v in results.items() if not v[2] and v[1] is not None}
    if len(good) > 1:
        rows = {v[1][0] for v in good.values()}
        sums = [v[1][1] for v in good.values()]
        if len(rows) != 1:
            print(f"\n# DISAGREEMENT on row count: { {k: v[1][0] for k, v in good.items()} }")
            return
        spread = (max(sums) - min(sums)) / max(1e-9, abs(sum(sums) / len(sums)))
        if spread > _AGREE_RTOL:
            seen = {k: v[1][1] for k, v in good.items()}
            print(f"\n# DISAGREEMENT on checksum ({spread:.2%}): {seen}")
            return
        print(f"\n# all engines agree: {rows.pop():,} rows, checksum spread {spread:.1e}")
    base = good.get("batcher")
    if base is None:
        return
    print("# speedup, batcher vs:")
    for name, (secs, _sig, _err) in good.items():
        if name != "batcher":
            print(f"#   {name:>6}: {secs / base[0]:.2f}x   ({secs:.2f}s / {base[0]:.2f}s)")
    rate = cfg["rows"] / base[0]
    print(f"# batcher throughput: {rate:,.0f} rows/s over {cfg['rows']:,} rows")
    if share > 0:
        print(
            f"# of batcher's {base[0]:.2f}s, the model itself accounts for {share:.2f}s "
            f"({share / base[0]:.0%}); the rest is read, batch assembly and dispatch"
        )


def main() -> int:
    from envinfo import machine_fingerprint, require_release_build

    require_release_build()
    print(machine_fingerprint())

    from _ray_env import strip_broken_runtime_env_hook
    from cluster_env import RAPIDS_DIR, init_gpu_cluster

    strip_broken_runtime_env_hook(unconditional=True)
    root = Path(RAPIDS_DIR) / "nvidia"
    libs = (
        os.pathsep.join(str(d / "lib") for d in sorted(root.iterdir()) if (d / "lib").is_dir())
        if root.is_dir()
        else ""
    )
    env_vars = (
        {"LD_LIBRARY_PATH": f"{libs}{os.pathsep}/usr/local/nvidia/lib64:/usr/local/cuda/lib64"}
        if libs
        else {}
    )
    init_gpu_cluster(env_vars=env_vars)
    import ray

    devices = int(ray.cluster_resources().get("GPU", 0))
    if devices < 1:
        print("# no GPUs visible; nothing to measure")
        return 1

    cfg = _cfg()
    path = _ensure_corpus(cfg)
    print(
        f"# parquet gpu inference: {cfg['rows']:,} rows x {cfg['dim']} float32, "
        f"{devices} devices, batch {cfg['batch']}, width x{cfg['width']}, "
        f"best of {cfg['runs']}"
    )

    share = _device_share(devices, cfg)
    print(f"# model time on one device for its share of the corpus: {share:.2f}s")

    results: dict = {}
    for engine in cfg["engines"]:
        secs, signature, err = _time(lambda e=engine: THUNKS[e](path, devices, cfg), cfg["runs"])
        results[engine] = (secs, signature, err)
        if err:
            print(f"  {engine:>8}: FAILED  {err}")
        else:
            rows, checksum = signature
            print(f"  {engine:>8}: {secs:>8.2f}s   rows={rows:,} checksum={checksum:.3f}")

    _verdict(results, cfg, share)
    out = os.environ.get("BENCH_OUT", "")
    if out:
        Path(out).write_text(
            json.dumps(
                {
                    "config": cfg,
                    "devices": devices,
                    "engines": {
                        k: {"seconds": v[0], "signature": v[1], "error": v[2]}
                        for k, v in results.items()
                    },
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
