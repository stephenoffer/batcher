"""Batcher vs Ray Data on a batch-inference pipeline sized to saturate BOTH resource classes.

Every earlier image arm on this fleet measured `/mnt/cluster_storage` (0.9 GiB/s aggregate)
rather than an engine: 74 KB of JPEG an image against 14 MFLOP of model leaves the devices at
1-11% and the cores at 8-17% no matter who schedules it. This corpus inverts the ratio -- 256
bytes a row -- and sizes the two stages against each other so that the CPU stage and the device
stage finish a row at the same rate:

    cores / cpu_seconds_per_row  ==  devices / gpu_seconds_per_row

On this fleet that is 192 cores against 8 T4s, so the CPU stage must be **24x** the device
stage per row. `FEAT_PASSES` and `FEAT_LAYERS` are the two knobs; `--calibrate` measures one
batch of each on this machine and prints what they should be.

Both engines run the identical two UDF classes over the identical shards, and the answer is
checked before any timing is reported.
"""

from __future__ import annotations

import functools
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
print = functools.partial(print, flush=True)

DIR = os.environ.get("FEAT_DIR", "/mnt/cluster_storage/feat_infer_bench")
DIM = int(os.environ.get("FEAT_DIM", "256"))
HID = int(os.environ.get("FEAT_HID", "4096"))
PASSES = int(os.environ.get("FEAT_PASSES", "400"))
LAYERS = int(os.environ.get("FEAT_LAYERS", "16"))
BATCH = int(os.environ.get("FEAT_BATCH", "512"))
DEV = int(os.environ.get("FEAT_DEV", "8"))
ACTORS = int(os.environ.get("FEAT_ACTORS", "96"))
ENGINES = [e for e in os.environ.get("FEAT_ENGINES", "batcher,ray").split(",") if e]
NSHARDS = int(os.environ.get("FEAT_NSHARDS", "0"))  # 0 = the whole corpus
SEED = 20260910
SHARDS = sorted(str(p) for p in Path(DIR).glob("*.parquet"))
if NSHARDS:
    SHARDS = SHARDS[:NSHARDS]


def _weights():
    """The model: one projection, `LAYERS` square hidden layers, one head. Fixed seed, so both
    engines score with bit-identical weights and the checksum is comparable."""
    rng = np.random.default_rng(SEED)
    dims = [(DIM, HID), *[(HID, HID)] * LAYERS, (HID, 16)]
    return [(rng.standard_normal((a, b)) / np.sqrt(a)).astype(np.float32) for a, b in dims]


def _matrix(v) -> np.ndarray:
    """`(N, DIM)` float32 from whatever the engine handed over, without per-row Python.

    The two engines present a fixed-size-list column differently and the difference is not
    cosmetic: Ray Data's numpy `batch_format` materializes one Python object per row, which is
    a hot-path tuple touch and would show up as an engine difference that is really a format
    choice. Its own guide says to use `batch_format="pyarrow"` for exactly this, so the Ray arm
    does, and this reads either shape zero-copy.
    """
    if not isinstance(v, np.ndarray):  # a pyarrow (Chunked)Array of fixed-size lists
        if hasattr(v, "combine_chunks"):
            v = v.combine_chunks()
        return v.flatten().to_numpy(zero_copy_only=False).reshape(-1, DIM).astype(np.float32)
    if v.dtype == object:
        return np.concatenate([np.asarray(r) for r in v]).reshape(-1, DIM).astype(np.float32)
    return v.reshape(-1, DIM).astype(np.float32)


class CpuStage:
    """The preprocess: elementwise + sort passes over the row's own features.

    Deliberately BLAS-free. A matmul here would be multi-threaded by OpenBLAS inside whatever
    process the engine happened to put it in, so the stage's width would stop being the
    engine's decision and the comparison would measure two BLAS thread pools. `sort`, `tanh`
    and `sqrt` are single-threaded, release the GIL, and are what a tokenize/normalize stage
    actually costs.
    """

    def __call__(self, batch):
        x = _matrix(batch["feat"]) / 128.0
        for _ in range(PASSES):
            x = np.sort(np.sqrt(np.abs(x)) + np.tanh(x), axis=1)
        return {"x": x}


class GpuStage:
    def __init__(self):
        import cupy as cp

        self._w = [cp.asarray(w) for w in _weights()]

    def __call__(self, batch):
        import cupy as cp

        x = cp.asarray(np.ascontiguousarray(_matrix(batch["x"])), dtype=cp.float32)
        for w in self._w:
            x = cp.maximum(x @ w, 0)
        return {"pred": cp.asnumpy(x.max(axis=1).astype(cp.float64))}


def calibrate() -> None:
    """Per-row cost of each stage on this box, and the ratio the fleet needs."""
    rng = np.random.default_rng(1)
    raw = rng.integers(-128, 127, size=(BATCH, DIM), dtype=np.int8)
    cpu = CpuStage()
    t = time.perf_counter()
    out = cpu({"feat": raw})
    c = (time.perf_counter() - t) / BATCH
    print(f"  cpu stage: {c * 1e3:.3f} ms/row  (PASSES={PASSES})")
    try:
        gpu = GpuStage()
        gpu({"x": out["x"]})  # warm the kernels
        t = time.perf_counter()
        gpu({"x": out["x"]})
        g = (time.perf_counter() - t) / BATCH
        print(f"  gpu stage: {g * 1e3:.3f} ms/row  (LAYERS={LAYERS}, HID={HID})")
        print(
            f"  ratio cpu/gpu = {c / g:.1f}x; this fleet wants {192 / DEV:.0f}x "
            f"-> scale PASSES by {(192 / DEV) / (c / g):.2f}"
        )
    except Exception as exc:
        print(
            f"  gpu stage: unavailable here ({type(exc).__name__}) -- run --calibrate on a GPU node"
        )


def batcher_thunk():
    import batcher as bt
    from batcher import col
    from batcher.api.functions import count

    ds = (
        bt.read.parquet(SHARDS)
        .map_batches(
            CpuStage,
            output_columns=["x"],
            batch_format="numpy",
            concurrency=ACTORS,
            batch_size=BATCH,
        )
        .map_batches(
            GpuStage,
            output_columns=["pred"],
            batch_format="numpy",
            concurrency=DEV,
            batch_size=BATCH,
            num_gpus=1,
        )
        .agg(n=count(), s=col("pred").sum())
    )
    return lambda: (lambda o: (int(o["n"][0]), round(float(o["s"][0]), 2)))(
        ds.collect(distributed=True).to_pydict()
    )


def batcher_stream_thunk():
    """Map-terminal, so the chain stays linear and reaches the stage-overlapped route."""
    import batcher as bt

    ds = (
        bt.read.parquet(SHARDS)
        .map_batches(
            CpuStage,
            output_columns=["x"],
            batch_format="numpy",
            concurrency=ACTORS,
            batch_size=BATCH,
        )
        .map_batches(
            GpuStage,
            output_columns=["pred"],
            batch_format="numpy",
            concurrency=DEV,
            batch_size=BATCH,
            num_gpus=1,
        )
    )

    def run():
        import pyarrow.compute as pc

        t0 = time.perf_counter()
        t = ds.collect(distributed=True)
        t1 = time.perf_counter()
        out = int(t.num_rows), round(float(pc.sum(t.column("pred")).as_py()), 2)
        # A map-terminal arm has to reduce on the driver, so the split is printed: an arm that
        # is really paying for a 4M-row hand-back should not read as a slow pipeline.
        print(f"        [collect {t1 - t0:.1f}s + reduce {time.perf_counter() - t1:.1f}s]")
        return out

    return run


def ray_thunk():
    import ray.data as rd
    from ray.data.aggregate import Count, Sum

    def run():
        ds = (
            rd.read_parquet(SHARDS)
            .map_batches(CpuStage, concurrency=ACTORS, batch_size=BATCH, batch_format="pyarrow")
            # `num_cpus=0` on the device actors, which is what Batcher's `_gpu_options` does
            # and for the reason this arm found the hard way: the read tasks took 163 of the
            # fleet's 192 cores, the eight device actors then pended forever on the one CPU
            # each that Ray Data asks for by default, the CPU stage's output had nowhere to
            # drain, and a 4M-row run spent twenty minutes backpressured with every GPU idle.
            # A device actor needs a device, not a core, on either engine.
            .map_batches(GpuStage, concurrency=DEV, num_gpus=1, num_cpus=0, batch_size=BATCH)
        )
        agg = ds.aggregate(Sum("pred"), Count())
        return int(agg["count()"]), round(float(agg["sum(pred)"]), 2)

    return run


THUNKS = {"batcher": batcher_thunk, "batcher_stream": batcher_stream_thunk, "ray": ray_thunk}


def _release():
    try:
        from batcher.dist.executors.map import release_inference_pools

        release_inference_pools()
    except Exception:
        pass


def main() -> None:
    if "--calibrate" in sys.argv:
        calibrate()
        return
    import ray
    from cluster_utilization import ClusterUtil, fmt

    from envinfo import machine_fingerprint, require_release_build

    require_release_build()
    print(machine_fingerprint())
    ray.init(address="auto", log_to_driver=False, ignore_reinit_error=True)
    gib = sum(os.path.getsize(p) for p in SHARDS) / 2**30
    print(
        f"# feature inference: {DIR} ({len(SHARDS)} shards, {gib:.2f} GiB), "
        f"{DEV} devices, batch {BATCH}, {ACTORS} cpu actors, "
        f"PASSES={PASSES} LAYERS={LAYERS} HID={HID}"
    )
    results = {}
    for eng in ENGINES:
        try:
            run = THUNKS[eng]()
            sig = run()  # untimed warm-up: pools built, kernels compiled, shards in page cache
            mon = ClusterUtil()
            mon.start()
            t = time.perf_counter()
            sig = run()
            dt = time.perf_counter() - t
            u = mon.stop()
            mon.shutdown()
            results[eng] = (dt, sig)
            print(f"  {eng:>15}: {dt:8.1f}s  {sig[0]:,} rows  {sig[0] / dt:,.0f} row/s")
            print(f"                   {fmt(u)}")
        except Exception as exc:
            print(f"  {eng:>15}: FAILED {type(exc).__name__}: {str(exc)[:300]}")
        finally:
            _release()
    if len(results) > 1:
        rows = {v[1][0] for v in results.values()}
        sums = [v[1][1] for v in results.values()]
        # Row counts are exact; the checksum is a float reduction, and partition count changes
        # the summation order. `assert_same`'s own tolerance applies: relative, not bitwise.
        agree = len(rows) == 1 and (max(sums) - min(sums)) <= 1e-9 * max(abs(s) for s in sums)
        print(f"\n# agreement: {'IDENTICAL' if agree else f'DIVERGENT {sorted(sums)}'}")
        ray_t = results.get("ray")
        if ray_t:
            for k, v in results.items():
                if k != "ray":
                    print(f"# {k} vs ray: {ray_t[0] / v[0]:.2f}x")


if __name__ == "__main__":
    main()
