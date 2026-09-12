"""Batcher vs Ray Data vs Daft on GPU batch inference, across this fleet's six T4s.

`benchmarks/cluster/vs_ray_daft_ml.py` races the three engines on a CPU fleet. This is the same
race with the model on a **device**: every engine builds an identical seeded network, moves it
onto a GPU once per actor, and scores the corpus in batches through a GPU actor pool.

**cupy, not torch, and that is the fleet rather than a preference.** These workers carry
cupy 13.6 and cuDF and no torch (`ModuleNotFoundError` from inside the UDF). Shipping a torch
wheel in the runtime env would charge every engine's sweep a multi-minute install and measure
that. The scorer is dense matrix multiplication either way, so the library changes which BLAS
runs and not what is computed -- and it keeps the arithmetic bit-comparable with the CPU
benchmark's NumPy scorer, which is what lets the two be read together.

Fairness, in the places this shape makes it easy to lose:

**Every engine gets the same GPU pool.** One actor per device, `num_gpus=1` each, the same
batch size, so the comparison is engine machinery -- read, decode, batch assembly, actor
dispatch, reduction -- and not how many devices each engine happened to claim.

**The reduction happens on the workers.** Every arm ends in a count and a sum, a few bytes per
engine. An arm ending in `collect()` would charge whichever engine returns pixels a transfer the
others do not pay.

**The model is the same object everywhere.** Fixed seed, identical layer shapes, identical
preprocessing, so the returned checksum is comparable exactly rather than approximately. A
mismatch beyond `_AGREE_RTOL` fails the run rather than being reported as a speedup.

**Read where the time actually goes before quoting a ratio.** This corpus is 110x110 JPEGs of
about 5 KiB on S3, so a small model leaves the workload dominated by per-object reads and CPU
JPEG decode, which no accelerator touches. `BENCH_INFER_FLOPS` scales the network so the same
harness can be run at both ends of that: the default is the CPU benchmark's network, and a
larger one moves the balance onto the device. The summary prints the device's share so a ratio
is never read without it.

Run (needs ray + daft on the driver, and a GPU fleet):
    python benchmarks/gpu_backend/vs_ray_daft_gpu_inference.py
    BENCH_IMAGE_SCALE=100 python benchmarks/gpu_backend/vs_ray_daft_gpu_inference.py
    BENCH_INFER_FLOPS=8 python benchmarks/gpu_backend/vs_ray_daft_gpu_inference.py
    BENCH_ENGINES=batcher,ray python benchmarks/gpu_backend/vs_ray_daft_gpu_inference.py
"""

from __future__ import annotations

import functools
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

print = functools.partial(print, flush=True)

#: Weight seed. Every actor in every engine builds the same network from it.
_SEED = 20260906

#: Decode target (H, W) -- the corpus's native size, so no engine is resizing and the
#: comparison is not about three resize kernels.
_SIZE = (110, 110)

#: Pixel subsample stride into the first dense layer, matching the CPU benchmark's scorer.
_STRIDE = 2

#: Relative agreement required between engines on the returned checksum. Independent JPEG
#: decoders disagree in the last bits and a dense network amplifies rather than averages that;
#: the CPU benchmark measured 0.46% on this shape from 0.09% of pixel disagreement, so this
#: sits an order above it and still catches a different corpus, dropped rows or a skipped
#: decode. Row counts are compared exactly.
_AGREE_RTOL = 0.05


def _scale() -> int:
    return int(os.environ.get("BENCH_IMAGE_SCALE", "1000"))


def _runs() -> int:
    return int(os.environ.get("BENCH_RUNS", "3"))


def _batch() -> int:
    return int(os.environ.get("BENCH_INFER_BATCH", "512"))


def _flops() -> int:
    """Network width multiplier. 1 is the CPU benchmark's network; larger moves work onto the
    device, which is the axis this benchmark exists to be read along."""
    return max(1, int(os.environ.get("BENCH_INFER_FLOPS", "1")))


def _engines() -> list[str]:
    raw = os.environ.get("BENCH_ENGINES", "batcher,ray,daft")
    return [e.strip() for e in raw.split(",") if e.strip()]


def _layers(width: int) -> tuple[tuple[int, int], ...]:
    """Dense layer shapes. The input width is fixed by the corpus and the stride."""
    h, w = _SIZE
    inp = (h // _STRIDE) * (w // _STRIDE)

    return (
        (inp, 2048 * width),
        (2048 * width, 2048 * width),
        (2048 * width, 1024 * width),
        (1024 * width, 16),
    )


def _cuda_library_path() -> str:
    """`LD_LIBRARY_PATH` additions that let the workers' cupy find its CUDA 12 libraries.

    These workers carry `cupy-cuda12x` but the image's CUDA is 13, so cupy's first JIT raises
    `libnvrtc.so.12: cannot open shared object file` and every arm dies inside the model rather
    than in anything this benchmark is measuring. The staged RAPIDS tree already ships the
    CUDA 12 runtime libraries beside cuDF, so pointing at them costs nothing and installs
    nothing.

    Set on the **cluster**, so all three engines' workers inherit it identically — Ray Data's
    actors and Daft's Ray runner both take the session's runtime env. An engine given a working
    accelerator while another is not is not a benchmark.
    """
    from cluster_env import RAPIDS_DIR

    root = Path(RAPIDS_DIR) / "nvidia"
    if not root.is_dir():
        return ""
    libs = [str(d / "lib") for d in sorted(root.iterdir()) if (d / "lib").is_dir()]
    return os.pathsep.join(libs)


def _worker_pip(engines: list[str]) -> list[str] | None:
    """Packages this sweep's workers need, or `None`.

    `daft` is not in this cluster's worker image, so its Ray-runner (flotilla) actors cannot
    import it and the runner never starts — an environment failure that reads as a Daft
    failure. Shipping it in the job's runtime env is what makes the column real.

    Charged to every worker in the job, so it is only requested when Daft is the **sole**
    engine in the sweep: adding it to a Batcher or Ray Data sweep would tax an engine that does
    not need it and measure a pip install. Run Daft as `BENCH_ENGINES=daft`. This mirrors
    `benchmarks/cluster/vs_ray_daft.py::_worker_pip`, for the same reason and on the same
    fleet.
    """
    if engines != ["daft"]:
        return None
    import daft

    return [f"daft=={daft.__version__}"]


def _build_weights(width: int) -> list[np.ndarray]:
    """Seeded host-side weights -- identical in every engine and every actor."""
    rng = np.random.default_rng(_SEED)
    return [
        (rng.standard_normal((a, b)) / np.sqrt(a)).astype(np.float32) for a, b in _layers(width)
    ]


class GpuScorer:
    """Model-load-once scorer whose weights live on a **device**.

    The constructor is the expensive per-actor step every batch-inference framework is built to
    amortize: it builds the network on the host and moves it onto the GPU once. `__call__` takes
    a decoded uint8 image batch, moves it across, and runs the same dense stack the CPU
    benchmark runs, returning the winning activation per image.

    The **score**, not the class. This corpus is face crops of one shape and a random network
    puts every one of them in the same class, so an argmax checksum would equal `k x rows` from
    every engine and prove only the row count.

    `width` is passed in rather than read from the environment, and that is not a style choice.
    It is read in the **actor**, which is a different process on a different machine and does
    not inherit the driver's environment — so an env-var read there silently returned the
    default and made the width knob inert. The first sweep taken that way reported the model at
    1% of the workload at both x1 and x8, which looked like a finding about the corpus and was a
    finding about the harness.
    """

    def __init__(self, width: int = 1) -> None:
        import cupy as cp

        self._cp = cp
        self._weights = [cp.asarray(w) for w in _build_weights(width)]
        # Materialize the context and the kernels now, not inside the first timed batch.
        cp.cuda.Stream.null.synchronize()

    def score(self, images: np.ndarray) -> np.ndarray:
        cp = self._cp
        if images.shape[0] == 0:
            return np.empty(0, dtype="float64")
        x = cp.asarray(images, dtype=cp.float32).mean(axis=3)
        x = x[:, ::_STRIDE, ::_STRIDE].reshape(images.shape[0], -1)
        x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)
        for weight in self._weights:
            x = cp.maximum(x @ weight, 0)
        return cp.asnumpy(x.max(axis=1).astype(cp.float64))


def _stack(column) -> np.ndarray:
    """A batch of images as one `(B, H, W, 3)` uint8 array, without a Python object per image.

    **The batch format is the single largest thing this benchmark can get wrong, and the first
    version got it wrong.** Batcher's arm took `batch_format="pyarrow"` and called `.to_pylist()`
    -- one Python list of 36,300 integers per image -- while Ray Data's arm received a zero-copy
    tensor block. Measured on the identical query with the identical checksum, that one
    difference was **25.95 s against 1.28 s, 20x**, and it read as Batcher being three times
    slower than Ray Data at image inference.

    Every arm now hands the batch over as an array. The fast path is an object that is already
    one; the list path remains for Daft, whose UDF receives a Series.
    """
    if isinstance(column, np.ndarray):
        return column if column.size else np.empty((0, *_SIZE, 3), dtype="uint8")
    values = column.to_pylist() if hasattr(column, "to_pylist") else list(column)
    if not values:
        return np.empty((0, *_SIZE, 3), dtype="uint8")
    return np.stack([np.asarray(v, dtype="uint8").reshape(*_SIZE, 3) for v in values])


def _sig(rows: int, checksum: float) -> tuple[int, float]:
    return int(rows), round(float(checksum), 2)


# --------------------------------------------------------------------------- #
# Batcher
# --------------------------------------------------------------------------- #
def batcher_thunk(devices: int, width: int):
    """`read.images` -> a GPU class UDF over an actor pool -> a distributed aggregate."""
    import batcher as bt
    from batcher import col
    from batcher.api.functions import count
    from sources.corpora import image_corpus

    class _Scorer:
        def __init__(self) -> None:
            self._scorer = GpuScorer(width)

        def __call__(self, batch) -> dict:
            return {"pred": self._scorer.score(_stack(batch["image"]))}

    corpus = image_corpus(_scale())
    src = bt.read.images(corpus.glob, size=_SIZE)
    ds = src.map_batches(
        _Scorer,
        output_columns=["pred"],
        batch_format="numpy",
        concurrency=devices,
        batch_size=_batch(),
        num_gpus=1,
    ).agg(n=count(), s=col("pred").sum())

    def run():
        out = ds.collect(distributed=True).to_pydict()
        return _sig(out["n"][0], out["s"][0])

    return run


# --------------------------------------------------------------------------- #
# Ray Data
# --------------------------------------------------------------------------- #
def ray_thunk(devices: int, width: int):
    """`read_images` -> `map_batches` over a GPU actor pool -> `aggregate`.

    This is the pattern Ray Data's own batch-inference guides are written around, so it is the
    arm this benchmark most has to get right rather than merely run.
    """
    import ray.data as rd
    from ray.data.aggregate import Count, Sum

    from sources.corpora import image_corpus

    class _RayScorer:
        def __init__(self) -> None:
            self._scorer = GpuScorer(width)

        def __call__(self, batch) -> dict:
            return {"pred": self._scorer.score(_stack(batch["image"]))}

    corpus = image_corpus(_scale())
    filesystem, paths = corpus.open()
    ds = rd.read_images(paths, filesystem=filesystem, size=_SIZE).map_batches(
        _RayScorer, concurrency=devices, num_gpus=1, batch_size=_batch()
    )

    def run():
        agg = ds.aggregate(Sum("pred"), Count())
        return _sig(agg["count()"], agg["sum(pred)"])

    return run


# --------------------------------------------------------------------------- #
# Daft
# --------------------------------------------------------------------------- #
def daft_thunk(devices: int, width: int):
    """`download` + `decode_image` -> a `num_gpus=1` class UDF -> an aggregate."""
    import daft

    from sources.corpora import image_corpus

    corpus = image_corpus(_scale())
    image = daft.col("uri").download().decode_image().resize(_SIZE[1], _SIZE[0])

    @daft.udf(return_dtype=daft.DataType.float64(), num_gpus=1, batch_size=_batch())
    class _DaftScorer:
        def __init__(self) -> None:
            self._scorer = GpuScorer(width)

        def __call__(self, images):
            return list(self._scorer.score(np.stack([np.asarray(x) for x in images.to_pylist()])))

    udf = _DaftScorer.with_concurrency(devices)

    def run():
        df = daft.from_pydict({"uri": corpus.uris()}).with_column("img", image)
        df = df.with_column("v", udf(daft.col("img")))
        out = df.agg(daft.col("v").sum().alias("s"), daft.col("v").count().alias("n")).to_pydict()
        return _sig(out["n"][0], out["s"][0])

    return run


THUNKS = {"batcher": batcher_thunk, "ray": ray_thunk, "daft": daft_thunk}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _device_share(devices: int) -> float:
    """Seconds one device spends on the model for the whole corpus, measured here.

    Printed beside every ratio because it is the number that says whether a ratio is about the
    accelerator at all. On this corpus at the default width the scoring is a small fraction of
    a workload that is mostly per-object S3 reads and CPU JPEG decode, and a reader who does not
    know that will attribute an engine's IO scheduling to its GPU support.
    """
    import ray

    from batcher.dist.gpu.tasks import gpu_task_options

    @ray.remote(**dict(gpu_task_options(num_gpus=1.0)))
    def _time_the_model(rows: int, batch: int, width: int) -> float:
        import time as t

        import numpy as np

        scorer = GpuScorer(width)
        images = np.zeros((batch, *_SIZE, 3), dtype="uint8")
        scorer.score(images)  # warm the kernels
        t0 = t.perf_counter()
        done = 0
        while done < rows:
            scorer.score(images)
            done += batch
        return t.perf_counter() - t0

    from sources.corpora import image_corpus

    rows = image_corpus(_scale()).count
    per_device = max(1, rows // max(1, devices))
    return float(ray.get(_time_the_model.remote(per_device, _batch(), _flops())))


def _time(thunk, runs: int) -> tuple[float, tuple[int, float] | None, str]:
    """Best-of-`runs` after one untimed warm-up, plus the signature and any error."""
    try:
        run = thunk()
        signature = run()
    except Exception as exc:
        return float("nan"), None, f"{type(exc).__name__}: {str(exc)[:180]}"
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        try:
            signature = run()
        except Exception as exc:
            return float("nan"), None, f"{type(exc).__name__}: {str(exc)[:180]}"
        times.append(time.perf_counter() - t0)
    return min(times), signature, ""


def main() -> int:
    from envinfo import require_release_build

    require_release_build()

    from _ray_env import strip_broken_runtime_env_hook
    from cluster_env import init_gpu_cluster

    strip_broken_runtime_env_hook(unconditional=True)
    cuda_libs = _cuda_library_path()
    env_vars = {}
    if cuda_libs:
        existing = "/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
        env_vars["LD_LIBRARY_PATH"] = f"{cuda_libs}{os.pathsep}{existing}"
    init_gpu_cluster(env_vars=env_vars, pip=_worker_pip(_engines()))
    import ray

    devices = int(ray.cluster_resources().get("GPU", 0))
    if devices < 1:
        print("# no GPUs visible; nothing to measure")
        return 1

    from sources.corpora import image_corpus

    rows = image_corpus(_scale()).count
    print(
        f"# gpu batch inference: {rows:,} images, {devices} devices, batch {_batch()}, "
        f"width x{_flops()}, best of {_runs()}"
    )
    share = _device_share(devices)
    print(f"# model time on one device for its share of the corpus: {share:.2f}s")

    results: dict[str, tuple[float, tuple[int, float] | None, str]] = {}
    for engine in _engines():
        secs, signature, err = _time(lambda e=engine: THUNKS[e](devices, _flops()), _runs())
        results[engine] = (secs, signature, err)
        if err:
            print(f"  {engine:>8}: FAILED  {err}")
        else:
            print(f"  {engine:>8}: {secs:>8.2f}s   rows={signature[0]:,} checksum={signature[1]}")

    _verdict(results, share)
    out = os.environ.get("BENCH_OUT", "")
    if out:
        Path(out).write_text(
            json.dumps(
                {
                    "images": rows,
                    "devices": devices,
                    "batch": _batch(),
                    "width": _flops(),
                    "device_model_seconds": share,
                    "engines": {
                        k: {"seconds": v[0], "signature": v[1], "error": v[2]}
                        for k, v in results.items()
                    },
                },
                indent=2,
            )
        )
    return 0


def _verdict(results: dict, share: float) -> None:
    """Agreement first, then the ratios, then what fraction of the win the device can explain.

    Correctness gates the timing rather than accompanying it: two engines that disagree on the
    checksum did different work, and a ratio between them is meaningless whichever is faster.
    """
    good = {k: v for k, v in results.items() if not v[2] and v[1] is not None}
    if len(good) > 1:
        rows = {v[1][0] for v in good.values()}
        sums = [v[1][1] for v in good.values()]
        if len(rows) != 1:
            print(f"\n# DISAGREEMENT on row count: { {k: v[1][0] for k, v in good.items()} }")
            return
        spread = (max(sums) - min(sums)) / max(1e-9, abs(statistics.fmean(sums)))
        if spread > _AGREE_RTOL:
            seen = {k: v[1][1] for k, v in good.items()}
            print(f"\n# DISAGREEMENT on checksum ({spread:.1%}): {seen}")
            return
        print(f"\n# all engines agree: {rows.pop():,} rows, checksum spread {spread:.2%}")

    base = good.get("batcher")
    if base is None:
        return
    print("# speedup, batcher vs:")
    for engine, entry in good.items():
        secs = entry[0]
        if engine == "batcher":
            continue
        print(f"#   {engine:>6}: {secs / base[0]:.2f}x   ({secs:.2f}s / {base[0]:.2f}s)")
    print(
        f"# of batcher's {base[0]:.2f}s, the model itself accounts for {share:.2f}s "
        f"({share / base[0]:.0%}); the rest is read, decode and scheduling"
    )


if __name__ == "__main__":
    raise SystemExit(main())
