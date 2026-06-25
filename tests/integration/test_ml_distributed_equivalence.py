"""Every ML feature produces identical results single-node and distributed across Ray.

Preprocessors lower to mergeable aggregates/projects, image decode + download + UDFs
are map_batches chains — all route through the distributed dispatcher and must equal
the single-node result (the mergeable-algebra / single-node-fallback invariant).
"""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

import batcher as bt
from batcher import col, count

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = pytest.mark.integration

import sys  # noqa: E402

import ray  # noqa: E402

ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])


def _both(ds, key: str):
    single = ds.collect().sort_by(key).to_pydict()
    dist = ds.collect(distributed=True, num_workers=2).sort_by(key).to_pydict()
    return single, dist


def _png(w: int, h: int, rgb: tuple[int, int, int]) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes(rgb) * w
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(row * h))
        + chunk(b"IEND", b"")
    )


def test_preprocessor_fit_aggregate_distributed_equals_single_node():
    # A scaler's fit reduces to this global aggregate (mean + mean-of-squares); proving
    # the aggregate is partition-independent proves the fitted statistics are.
    ds = bt.from_pydict({"x": [float(i) for i in range(500)]})
    agg = ds.agg(m=col("x").mean(), sq=(col("x") * col("x")).mean(), n=count())
    single, dist = _both(agg, "m")
    assert single == dist


def test_preprocessor_transform_distributed_equals_single_node():
    from batcher.ml.preprocessors import SimpleImputer, StandardScaler

    ds = bt.from_pydict({"id": list(range(200)), "x": [float(i % 7) for i in range(200)]})
    # Sequence the steps: fit each on the prior step's output (deterministic stats).
    imputer = SimpleImputer(["x"], strategy="mean").fit(ds)
    scaler = StandardScaler(["x"]).fit(imputer.transform(ds))
    single, dist = _both(scaler.transform(imputer.transform(ds)), "id")
    assert single == dist


def test_image_decode_distributed_equals_single_node():
    from batcher.ml.decode import image_tensor_dataset

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    ds = bt.from_pydict({"id": list(range(4)), "bytes": [_png(8, 8, c) for c in colors]})
    decoded = image_tensor_dataset(ds, size=(4, 4))
    single = decoded.collect().sort_by("id")
    dist = decoded.collect(distributed=True, num_workers=2).sort_by("id")
    s = single.column("image").combine_chunks().to_numpy_ndarray()
    d = dist.column("image").combine_chunks().to_numpy_ndarray()
    assert np.array_equal(s, d)


def test_download_distributed_equals_single_node(tmp_path):
    paths = []
    for i in range(6):
        p = tmp_path / f"f{i}.bin"
        p.write_bytes(f"content-{i}".encode())
        paths.append(str(p))
    ds = bt.from_pydict({"id": list(range(6)), "url": paths})
    fetched = ds.ml.download("url", output_column="data")
    single, dist = _both(fetched, "id")
    assert single == dist


def test_map_batches_numpy_format_distributed_equals_single_node():
    def add(d):
        return {"id": d["id"], "z": d["x"] + d["y"]}

    ds = bt.from_pydict({"id": list(range(300)), "x": list(range(300)), "y": list(range(300))})
    out = ds.ml.map_batches(add, batch_format="numpy", output_columns=["id", "z"])
    single, dist = _both(out, "id")
    assert single == dist
