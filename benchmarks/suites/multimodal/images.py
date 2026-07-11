"""Multimodal image ingest: read, decode, and resize a JPEG corpus, across engines.

The unstructured-data counterpart to the structured ``scan`` suite. Where ``scan`` reads
parquet, this reads a corpus of JPEG files (``profile-pictures`` in the Ray bucket, 110x110
RGB, ~5 KiB each) and runs the three stages of a real image-preprocessing pipeline:

- ``img-list`` — list the files and read their bytes (no decode). The pure I/O + per-file
  open cost, which dominates on a corpus of many tiny objects.
- ``img-decode`` — decode each JPEG to an ``(H, W, 3)`` pixel tensor at its native size.
- ``img-resize`` — decode and resize to ``224x224`` (the classic model-input preprocessing).

Both decode shapes pass an explicit target size to every engine (``img-decode`` the corpus's
native size, ``img-resize`` ``224x224``): Ray/Daft could infer the native size but Batcher's
``read.images`` reader requires the decode size be given, so passing it uniformly keeps the
engines doing the identical operation instead of special-casing one.

Only the multimodal engines compete: **Batcher** (``read.images``), **Ray Data**
(``read_images`` / ``read_binary_files``), and **Daft** (``url.download`` +
``decode_image`` + ``resize``). **PyArrow** joins ``img-list`` only (it reads bytes but does
not decode images). DuckDB/Polars have no image path and sit out entirely.

Cross-engine pixel equality is not a sound gate — JPEG decoders and resize kernels differ
by implementation — so each case returns a small aggregate the engines *do* agree on: the
image **count**, the exact **total bytes** (a strong content check for the list stage), and
the produced **height/width**. That is enough to prove every engine read and processed the
same corpus before its throughput is trusted, the same discipline the GPU cluster
benchmarks use (checksum/count agreement, not raw float pixels).

Each engine builds its reader *inside* the timed call, so per-file listing and open cost is
measured, not amortized (as in ``scan``). Reads run from S3 by default; the corpus size is
set by ``--scale`` (1 -> 10 images, 10 -> 100, ...). Because image reads are per-file, the
default is small and the suite is opt-in (excluded from ``--benchmark all``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from registry import EngineQueries, suite

if TYPE_CHECKING:
    from context import Context
    from sources import ImageCorpus

images = suite("multimodal", dataset="images")

_RESIZE = (224, 224)


def _row(**cols: int) -> pa.Table:
    """A single-row table of int64 scalars — the shape every image case returns."""
    return pa.table({k: pa.array([v], type=pa.int64()) for k, v in cols.items()})


# --------------------------------------------------------------------------- #
# Batcher — read.images over the corpus glob
# --------------------------------------------------------------------------- #
def _bt_list(c: ImageCorpus) -> pa.Table:
    import batcher as bt

    d = bt.read.images(c.glob).collect()
    return _row(n=d.num_rows, total_bytes=pc.sum(d.column("size")).as_py())


def _bt_decode(c: ImageCorpus, size: tuple[int, int]) -> pa.Table:
    import batcher as bt

    d = bt.read.images(c.glob, decode=True, size=size).collect()
    height, width, _ = d.column("image").type.shape  # fixed-shape tensor extension type
    return _row(n=d.num_rows, h=height, w=width)


# --------------------------------------------------------------------------- #
# Ray Data — read_binary_files (list) / read_images (decode, resize)
# --------------------------------------------------------------------------- #
def _ray_list(c: ImageCorpus) -> pa.Table:
    import ray.data

    filesystem, paths = c.open()
    rows = ray.data.read_binary_files(paths, filesystem=filesystem).take_all()
    return _row(n=len(rows), total_bytes=sum(len(r["bytes"]) for r in rows))


def _ray_decode(c: ImageCorpus, size: tuple[int, int]) -> pa.Table:
    import ray.data

    filesystem, paths = c.open()
    rows = ray.data.read_images(paths, filesystem=filesystem, size=size).take_all()
    image = rows[0]["image"]  # (H, W, C) ndarray
    return _row(n=len(rows), h=image.shape[0], w=image.shape[1])


# --------------------------------------------------------------------------- #
# Daft — url.download (list) + decode_image / resize
# --------------------------------------------------------------------------- #
def _daft_list(c: ImageCorpus) -> pa.Table:
    import daft

    blobs = daft.from_pydict({"uri": c.uris()}).with_column("b", daft.col("uri").download())
    data = blobs.to_pydict()
    return _row(n=len(data["b"]), total_bytes=sum(len(x) for x in data["b"]))


def _daft_decode(c: ImageCorpus, size: tuple[int, int]) -> pa.Table:
    import daft

    df = daft.from_pydict({"uri": c.uris()})
    img = daft.col("uri").download().decode_image().resize(size[1], size[0])  # resize(w, h)
    df = df.with_column("img", img)
    df = df.with_column("h", daft.col("img").image_height())
    df = df.with_column("w", daft.col("img").image_width())
    dims = df.select("h", "w").to_pydict()
    return _row(n=len(dims["h"]), h=dims["h"][0], w=dims["w"][0])


# --------------------------------------------------------------------------- #
# PyArrow — filesystem bytes read (list only; no image decode)
# --------------------------------------------------------------------------- #
def _pa_list(c: ImageCorpus) -> pa.Table:
    filesystem, paths = c.open()
    total = 0
    for path in paths:
        with filesystem.open_input_file(path) as handle:
            total += len(handle.readall())
    return _row(n=len(paths), total_bytes=total)


# Per shape: engine name -> callable taking the corpus, returning the agreed aggregate row.
# The decode functions per engine, keyed for the two decode shapes (native size / 224).
_DECODERS = {"batcher": _bt_decode, "ray": _ray_decode, "daft": _daft_decode}

_LIST = {"batcher": _bt_list, "ray": _ray_list, "daft": _daft_list, "pyarrow": _pa_list}
_DECODE = {name: (lambda c, fn=fn: fn(c, c.native_size)) for name, fn in _DECODERS.items()}
_RESIZE_FNS = {name: (lambda c, fn=fn: fn(c, _RESIZE)) for name, fn in _DECODERS.items()}


def _queries(ctx: Context, natives: dict[str, Callable[[ImageCorpus], pa.Table]]) -> EngineQueries:
    """Bind each engine that both supports the shape and is in the active lineup."""
    corpus = ctx.images
    if corpus is None:  # not a corpus context — nothing to run
        return {}
    active = set(ctx.names())
    out: EngineQueries = {}
    for name, fn in natives.items():
        if name in active:
            out[name] = lambda fn=fn, c=corpus: fn(c)
    return out


@images.case("img-list")
def img_list(ctx: Context) -> EngineQueries:
    """List the corpus and read every file's bytes (no decode) — the per-file I/O cost."""
    return _queries(ctx, _LIST)


@images.case("img-decode")
def img_decode(ctx: Context) -> EngineQueries:
    """Decode every JPEG to an ``(H, W, 3)`` pixel tensor."""
    return _queries(ctx, _DECODE)


@images.case("img-resize")
def img_resize(ctx: Context) -> EngineQueries:
    """Decode and resize every image to ``224x224`` — the model-input preprocessing step."""
    return _queries(ctx, _RESIZE_FNS)
