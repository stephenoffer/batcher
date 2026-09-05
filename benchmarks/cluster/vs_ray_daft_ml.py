"""Batcher vs Ray Data vs Daft on the two workloads they are *built* for.

`vs_ray_daft.py` races the three engines on relational shapes. This one races them where
the competitors are strongest and where a data engine is usually chosen:

- ``image_resize`` -- read a JPEG corpus from S3, decode every image, resize it to
  224x224, and reduce. Daft's flagship: a native, Rust-side image column with
  ``download``/``decode``/``resize`` fused into the plan.
- ``inference`` -- the same read and preprocessing, then a **model loaded once per
  worker** scoring every batch. Ray Data's flagship: `map_batches` over an actor pool,
  which is the pattern its batch-inference guides are written around.

Both run on a CPU cluster, because that is the cluster these are recorded on. The model
is torchvision ResNet-18 with fixed seeded weights, constructed identically in every
engine, so the predictions are comparable rather than merely similar.

Fairness, which on these two shapes is easier to get wrong than the timing:

**The reduction happens on the workers.** An image benchmark that ends in ``collect()``
or ``take_all()`` ships every decoded pixel to the driver -- at 10,000 images that is
1.5 GB, and at the top scale 15 GB -- so the engine that returns pixels is charged a
transfer the engine that returns two integers is not. Every arm here ends in an
aggregate: a row count and a sum, a few bytes per engine. `benchmarks/suites/multimodal`
has the shape this avoids, and its `img-decode` cases are not comparable across engines
for that reason.

**The decode cannot be pruned.** The reduction is the mean of every pixel of every image,
so an engine that skipped the decode would have nothing to average. A count alone would
let projection pushdown delete the whole workload and report a scan.

**Pixels are compared with a tolerance, and that is not a loosened gate.** Independent
JPEG decoders and resize kernels disagree in the last bits by construction; measured on
10 images of this corpus, the per-image mean summed to 1390.79 (Batcher), 1390.74 (Daft)
and 1389.53 (Ray Data) -- 0.004% and 0.09%. `_AGREE_RTOL` is set an order of magnitude
above the worst of those, so it still catches an engine that read a different corpus,
dropped rows, or skipped the resize, while not failing on a decoder difference nobody
can remove. Row counts are compared exactly.

A network amplifies that difference rather than averaging it away, and the tolerance has to
cover the amplified version: the summed confidences on `inference` came back 3,142.51 against
3,127.91, 0.46%, from the same 0.09% of pixel disagreement.

Run (from the env carrying ray + torch + torchvision, batcher installed):
    python benchmarks/cluster/vs_ray_daft_ml.py            # 10,000 images
    BENCH_IMAGE_SCALE=100 python benchmarks/cluster/vs_ray_daft_ml.py     # 1,000
    BENCH_PIPES=inference BENCH_ENGINE_ORDER=daft python benchmarks/cluster/vs_ray_daft_ml.py

Daft only competes when it is the sole engine in the sweep, for the reason
`vs_ray_daft._worker_pip` records: the `daft` wheel has to be in the job's runtime env,
and charging that install to a Batcher or Ray Data sweep would measure a pip install.
"""

from __future__ import annotations

import contextlib
import functools
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vs_ray_daft import _require_release, _worker_pip, bench_engine

from envinfo import machine_fingerprint, require_release_build
from sources.corpora import image_corpus

print = functools.partial(print, flush=True)

#: Model input size, and the resize every engine performs.
_SIZE = (224, 224)
#: Seed for the model weights, so every actor in every engine holds the same network.
_SEED = 1234
#: Decimation from the 224x224 decode to the network's input: every 4th pixel each way,
#: so 56x56 = 3,136 features.
_STRIDE = 4
#: How far two engines' pixel sums may differ before the ratio is withheld. See the module
#: docstring: the measured spread across three independent JPEG stacks is 0.09%.
_AGREE_RTOL = 1e-2


def _cfg() -> dict:
    return {
        # 1 -> 10 images, 10 -> 100, 100 -> 1,000, 1000 -> 10,000, 10000 -> 100,000.
        "scale": int(os.environ.get("BENCH_IMAGE_SCALE", "1000")),
        "batch": int(os.environ.get("BENCH_IMAGE_BATCH", "64")),
        "runs": int(os.environ.get("BENCH_RUNS", "2")),
        # Actors in the inference pool. Empty -> one per node, which is what an engine
        # that sizes a CPU pool from the fleet does on its own.
        "concurrency": os.environ.get("BENCH_INFER_CONCURRENCY", ""),
        "cpus": float(os.environ.get("BENCH_INFER_ACTOR_CPUS", "4")),
    }


def _pool_size(cfg: dict) -> int:
    """Actors in the inference pool: the setting, else as many as the fleet can host.

    Bounded by CPU capacity rather than by node count, because the two disagree: this
    cluster reported 85 alive nodes against 1,024 CPU, so one actor per node at
    `BENCH_INFER_ACTOR_CPUS` would have asked for more cores than exist and left the pool
    permanently short of its last members.
    """
    if cfg["concurrency"]:
        return int(cfg["concurrency"])
    import ray

    cpus = float(ray.cluster_resources().get("CPU", 0.0))
    nodes = len([n for n in ray.nodes() if n.get("Alive")])
    # Half the fleet, not all of it. A pool sized to every core leaves nothing for the
    # read stage feeding it, and the engines do not fail the same way when that happens:
    # Batcher's reader is inside the same query, while Ray Data streams `read_images`
    # tasks *beside* the actor pool, so a pool holding 1,024 of 1,024 CPU simply never
    # ran -- recorded as a 300 s timeout on 100 images, which would have read as a Ray
    # Data result rather than as a benchmark that asked for an impossible shape.
    placeable = int(cpus // (2.0 * max(1.0, cfg["cpus"])))
    return max(1, min(nodes - 1 or 1, placeable))


# --------------------------------------------------------------------------- #
# The two computations, shared by every engine
# --------------------------------------------------------------------------- #
def mean_pixels(images) -> np.ndarray:
    """``(B, H, W, 3)`` uint8 -> ``(B,)`` float64 per-image mean.

    The reduction every ``image_resize`` arm runs. It touches every pixel, so an engine
    that elided the decode cannot produce it.
    """
    a = np.asarray(images, dtype=np.float64)
    if a.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    return a.reshape(a.shape[0], -1).mean(axis=1)


def _stack(column) -> np.ndarray:
    """A batch's image column as one ``(B, H, W, 3)`` uint8 array.

    Arrow hands a fixed-shape tensor column back as an ndarray of ndarrays on some paths
    and as a single stacked array on others; both engines' Arrow batches land here. An
    **empty** batch is a real case rather than a guard against one -- a distributed read
    hands a worker a partition with no rows whenever the fan-out exceeds the file count,
    and `np.stack([])` raises `need at least one array to stack` from inside the UDF,
    which fails the query.
    """
    arr = column.to_numpy(zero_copy_only=False) if hasattr(column, "to_numpy") else column
    if not (isinstance(arr, np.ndarray) and arr.dtype != object and arr.ndim >= 2):
        parts = [np.asarray(x) for x in arr]
        if not parts:
            return np.empty((0, _SIZE[0], _SIZE[1], 3), dtype=np.uint8)
        arr = np.stack(parts)
    # A fixed-shape tensor column can arrive already flattened per row -- `(B, H*W*3)`
    # rather than `(B, H, W, 3)` -- which reads as a valid array and then fails four
    # frames later inside the UDF (`axis 3 is out of bounds for array of dimension 2`).
    # Normalizing here is what makes one UDF body work against both engines' Arrow.
    return arr.reshape(arr.shape[0], _SIZE[0], _SIZE[1], 3)


#: Layer widths of the scoring network, over the 56x56 grayscale thumbnail below
#: (3,136 features). ~14.7M parameters, so building it is a real per-actor cost and
#: scoring one image is ~29 MFLOPs -- the same order as a small CNN.
_LAYERS = ((3136, 2048), (2048, 2048), (2048, 1024), (1024, 16))


def _build_model() -> list[np.ndarray]:
    """Seeded weights for the scoring network -- the expensive per-actor initialization.

    **NumPy rather than torch, and that is the cluster's constraint rather than a
    preference.** This fleet's workers carry NumPy and not torch (`ModuleNotFoundError:
    No module named 'torch'` from inside the UDF), and shipping a torch wheel in the
    runtime env would charge every engine's sweep a multi-minute pip install and measure
    that instead. What the `inference` shape is here to compare is the engine machinery a
    batch-inference job runs on -- an actor pool, a model built once per worker, batches
    handed to it, results reduced -- and that is identical whichever library multiplies
    the matrices. Fixed seed, so every actor in every engine holds the same network and
    the predictions are comparable exactly rather than approximately.
    """
    rng = np.random.default_rng(_SEED)
    return [(rng.standard_normal((a, b)) / np.sqrt(a)).astype(np.float32) for a, b in _LAYERS]


class ImageScorer:
    """Model-load-once class UDF: build the network once per actor, score each batch.

    Takes an Arrow batch carrying a decoded ``image`` column, returns ``{"pred": ...}``
    with the argmax class per image. A dict keeps it portable across Batcher and Ray
    Data, both of which accept a column dict back from a batch UDF.
    """

    def __init__(self) -> None:
        self._weights = _build_model()

    def __call__(self, batch) -> dict:
        return {"pred": self.score(_stack(batch.column("image")))}

    def score(self, images: np.ndarray) -> np.ndarray:
        """``(B, H, W, 3)`` uint8 -> ``(B,)`` float64 top-class score per image.

        The **score**, not the class, and that is what makes the result a check. This corpus
        is 10,000 face crops of one shape, and a random network puts every one of them in the
        same class however the input is preprocessed -- so a checksum built on the argmax came
        back as `3 x rows` from all three engines and proved only the row count. The winning
        activation varies per image, sums to a figure an engine that read a different corpus
        or skipped the resize cannot reproduce, and is what an inference stage emits anyway.
        """
        if images.shape[0] == 0:
            return np.empty(0, dtype="float64")
        x = images.astype(np.float32).mean(axis=3)  # (B, H, W) grayscale
        x = x[:, ::_STRIDE, ::_STRIDE].reshape(images.shape[0], -1)
        # Standardize per image, the ordinary preprocessing step -- and here also what makes
        # the *result* a check. Feeding raw 0-255 pixels through a random ReLU stack put every
        # image in the same class, so the checksum was `4 x rows` and proved only the row
        # count: any engine returning 10,000 of anything would have agreed with any other.
        x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)
        for weight in self._weights:
            x = np.maximum(x @ weight, 0.0)
        return x.max(axis=1).astype("float64")


def _sig(rows: int, checksum: float) -> dict:
    return {"rows": int(rows), "checksum": round(float(checksum), 2)}


# --------------------------------------------------------------------------- #
# Batcher
# --------------------------------------------------------------------------- #
def batcher_thunk(name: str, cfg: dict):
    import batcher as bt
    from batcher import col, count

    corpus = image_corpus(cfg["scale"])
    src = bt.read.images(corpus.glob, decode=True, size=_SIZE)

    if name == "image_resize":

        def mean_batch(batch):
            return pa.table({"m": pa.array(mean_pixels(_stack(batch.column("image"))))})

        ds = src.map_batches(mean_batch, output_columns=["m"], batch_format="pyarrow").agg(
            n=count(), s=col("m").sum()
        )
    elif name == "inference":
        ds = src.map_batches(
            ImageScorer,
            output_columns=["pred"],
            batch_format="pyarrow",
            concurrency=_pool_size(cfg),
            batch_size=cfg["batch"],
        ).agg(n=count(), s=col("pred").sum())
    else:
        raise ValueError(name)

    def run():
        out = ds.collect(distributed=True).to_pydict()
        return _sig(out["n"][0], out["s"][0])

    return run


# --------------------------------------------------------------------------- #
# Ray Data
# --------------------------------------------------------------------------- #
def ray_thunk(name: str, cfg: dict):
    import ray.data as rd
    from ray.data.aggregate import Count, Sum

    corpus = image_corpus(cfg["scale"])
    filesystem, paths = corpus.open()
    src = rd.read_images(paths, filesystem=filesystem, size=_SIZE)

    if name == "image_resize":

        def mean_batch(batch):
            return {"m": mean_pixels(_stack(batch["image"]))}

        ds = src.map_batches(mean_batch)
        key = "m"
    elif name == "inference":

        class _RayScorer:
            def __init__(self) -> None:
                self._scorer = ImageScorer()

            def __call__(self, batch) -> dict:
                return {"pred": self._scorer.score(_stack(batch["image"]))}

        ds = src.map_batches(
            _RayScorer,
            concurrency=_pool_size(cfg),
            num_cpus=cfg["cpus"],
            batch_size=cfg["batch"],
        )
        key = "pred"
    else:
        raise ValueError(name)

    def run():
        agg = ds.aggregate(Sum(key), Count())
        return _sig(agg["count()"], agg[f"sum({key})"])

    return run


# --------------------------------------------------------------------------- #
# Daft
# --------------------------------------------------------------------------- #
def daft_thunk(name: str, cfg: dict):
    import daft

    corpus = image_corpus(cfg["scale"])
    image = daft.col("uri").download().decode_image().resize(_SIZE[1], _SIZE[0])

    if name == "image_resize":

        @daft.udf(return_dtype=daft.DataType.float64())
        def reduce_udf(images):
            return list(mean_pixels(np.stack([np.asarray(x) for x in images.to_pylist()])))

        udf = reduce_udf
    elif name == "inference":

        @daft.udf(
            return_dtype=daft.DataType.float64(), num_cpus=cfg["cpus"], batch_size=cfg["batch"]
        )
        class _DaftScorer:
            def __init__(self) -> None:
                self._scorer = ImageScorer()

            def __call__(self, images):
                return list(
                    self._scorer.score(np.stack([np.asarray(x) for x in images.to_pylist()]))
                )

        udf = _DaftScorer.with_concurrency(_pool_size(cfg))
    else:
        raise ValueError(name)

    def run():
        df = daft.from_pydict({"uri": corpus.uris()}).with_column("img", image)
        df = df.with_column("v", udf(daft.col("img")))
        out = df.agg(daft.col("v").sum().alias("s"), daft.col("v").count().alias("n")).to_pydict()
        return _sig(out["n"][0], out["s"][0])

    return run


ENGINES = {"batcher": batcher_thunk, "ray": ray_thunk, "daft": daft_thunk}
PIPELINES = ["image_resize", "inference"]


def _agreed(sigs: dict) -> bool:
    """Whether every engine returned the same rows and a pixel sum within `_AGREE_RTOL`.

    Exact on the row count, tolerant on the sum -- see the module docstring for why the
    second is a decoder fact rather than a weakened gate.
    """
    values = [s for s in sigs.values() if s]
    if len(values) < 2:
        return True
    if len({s["rows"] for s in values}) > 1:
        return False
    sums = [s["checksum"] for s in values]
    spread = max(sums) - min(sums)
    return spread <= _AGREE_RTOL * max(abs(v) for v in sums or [1.0])


def _report(pipelines: list[str], by_engine: dict) -> None:
    head = ("pipeline", "batcher_ms", "ray_ms", "daft_ms", "vs_ray", "vs_daft")
    widths = (16, 12, 11, 11, 9, 9)
    cols = "".join(c.ljust(widths[0]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(head))
    line = f"{cols}  util(batcher | ray)"
    print()
    print(line)
    print("-" * len(line))
    for name in pipelines:
        res = {eng: r[name] for eng, r in by_engine.items() if name in r}
        got = {e: res.get(e, {}).get("ms") for e in ("batcher", "ray", "daft")}
        agreed = _agreed({e: r.get("sig") for e, r in res.items() if "sig" in r})

        def ratio(other, got=got, agreed=agreed):
            b, o = got["batcher"], got[other]
            return f"{o / b:.2f}x" if (agreed and b and o) else ("n/c" if b and o else "-")

        def cell(v):
            return f"{v:.0f}" if isinstance(v, (int, float)) else "ERR"

        cells = f"{name:<16}{cell(got['batcher']):>12}{cell(got['ray']):>11}"
        cells += f"{cell(got['daft']):>11}{ratio('ray'):>9}{ratio('daft'):>9}"
        print(f"{cells}  {_util(res.get('batcher'))} | {_util(res.get('ray'))}")
        # Always, not only on a mismatch. Daft runs in its own process (the harness ships its
        # wheel only when it is the sole engine), so a `vs_daft` ratio is assembled by a
        # reader across two runs -- and `_agreed` cannot check a single engine against itself.
        # Printing what each arm returned is what makes that comparison checkable rather than
        # asserted.
        for eng, r in res.items():
            if "sig" in r:
                print(f"    {eng:>8} returned {r['sig']}")
        if not agreed:
            print("    !! signature mismatch: the ratios above are withheld")
        for eng, r in res.items():
            if "error" in r:
                print(f"    !! {eng}: {r['error']}")


def _util(entry) -> str:
    u = (entry or {}).get("util") or {}
    if not u:
        return "-"
    return (
        f"{u.get('mean_busy_pct', 0):.0f}%/{u.get('peak_busy_pct', 0):.0f}%peak "
        f"{int(u.get('active_nodes', 0))}/{int(u.get('total_nodes', 0))}n"
    )


def _warm_the_fleet(eng: str) -> None:
    """Pay an engine's *fleet* startup before the sweep, on a query with no data in it.

    `bench_engine` already runs each pipeline once untimed, which pays planning and the read.
    It does not help an engine whose worker startup has its own deadline: Daft's Ray runner
    spawns flotilla actors on first use and gives up on them after 120 s, and on this cluster
    the first pipeline of a Daft sweep died with `No flotilla workers became available within
    120s (64 attempted)` while the second -- with the actors up -- returned a number. That is
    the harness charging one pipeline for the fleet the whole sweep uses, and it reads as a
    Daft failure.

    Best-effort and untimed: an engine that cannot answer a one-row query here will fail in
    its own arm with its own error, which is where a reader should see it.
    """
    if eng != "daft":
        return
    with contextlib.suppress(Exception):
        import daft

        daft.from_pydict({"x": [1]}).agg(daft.col("x").sum().alias("s")).to_pydict()


def main() -> int:
    require_release_build()
    print(machine_fingerprint())
    _require_release()
    cfg = _cfg()
    pipes = os.environ.get("BENCH_PIPES")
    pipelines = [p for p in (pipes.split(",") if pipes else PIPELINES) if p in PIPELINES]
    order = [e for e in os.environ.get("BENCH_ENGINE_ORDER", "").split(",") if e in ENGINES]
    order = order or ["batcher", "ray"]

    import ray

    if not ray.is_initialized():
        os.environ.setdefault("RAY_ADDRESS", "auto")
        ray.init(
            address="auto",
            logging_level="ERROR",
            log_to_driver=False,
            runtime_env={"pip": _worker_pip(order)},
        )
    corpus = image_corpus(cfg["scale"])
    print(f"cluster: {ray.cluster_resources().get('CPU')} CPU, {len(ray.nodes())} nodes")
    print(f"{corpus.count} images -> {_SIZE[0]}x{_SIZE[1]}, best-of-{cfg['runs']}")
    print(f"engine sweep order: {' -> '.join(order)}\n")

    # `bench_engine` calls `builder(pipeline, scale)`; these workloads are sized by `cfg`
    # rather than by a TPC-H scale factor, so the scale argument is absorbed here.
    def builder_for(eng: str):
        return lambda name, _scale, fn=ENGINES[eng]: fn(name, cfg)

    for eng in order:
        _warm_the_fleet(eng)
    by_engine = {
        eng: bench_engine(eng, builder_for(eng), pipelines, 0, cfg["runs"]) for eng in order
    }
    _report(pipelines, by_engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
