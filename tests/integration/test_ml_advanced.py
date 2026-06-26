"""Advanced AI surfaces: vision-LLM input, DLPack zero-copy, Lance vector search.

Live paths are gated on their optional deps (Pillow / torch / pylance); the request
shaping and error guards are always asserted.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt


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


# --- Vision-LLM input -------------------------------------------------------
def _record_engine(seen: list):
    def factory():
        def engine(requests):
            seen.extend(requests)
            return [
                "IMG" if isinstance(r, dict) and r.get("image") is not None else "TXT"
                for r in requests
            ]

        return engine

    return factory


def test_llm_text_only_passes_strings():
    from batcher.ml.llm import llm_generate

    seen: list = []
    batch = pa.record_batch({"q": ["hi", "yo"]})
    out = list(llm_generate([batch], _record_engine(seen), prompt_column="q"))
    assert out[0].column("response").to_pylist() == ["TXT", "TXT"]
    assert all(isinstance(r, str) for r in seen)


def test_llm_multimodal_builds_image_requests():
    pytest.importorskip("PIL", reason="Pillow not installed")
    from PIL import Image

    from batcher.ml.llm import llm_generate

    seen: list = []
    batch = pa.record_batch(
        {"q": ["describe", "caption"], "img": [_png(4, 4, (255, 0, 0)), _png(4, 4, (0, 255, 0))]}
    )
    out = list(llm_generate([batch], _record_engine(seen), prompt_column="q", image_column="img"))
    assert out[0].column("response").to_pylist() == ["IMG", "IMG"]
    assert all(isinstance(r, dict) and isinstance(r["image"], Image.Image) for r in seen)
    assert seen[0]["prompt"] == "describe"


def _pillow_installed() -> bool:
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(_pillow_installed(), reason="Pillow installed; guard not exercised")
def test_llm_multimodal_needs_pillow_when_absent():
    from batcher._internal.errors import BackendError
    from batcher.ml.llm import llm_generate

    batch = pa.record_batch({"q": ["x"], "img": [_png(2, 2, (1, 1, 1))]})
    with pytest.raises(BackendError, match=r"\[image\]"):
        list(llm_generate([batch], _record_engine([]), prompt_column="q", image_column="img"))


# --- DLPack zero-copy -------------------------------------------------------
def test_zero_copy_inference_path():
    pytest.importorskip("torch")
    ds = bt.from_pydict({"x": list(range(20)), "y": [float(i) for i in range(20)]})
    out = list(ds.ml.iter_torch_batches(batch_size=10, device=None, zero_copy=True))
    assert sorted(int(v) for b in out for v in b["x"].tolist()) == list(range(20))


def test_zero_copy_matches_copy_path():
    pytest.importorskip("torch")
    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0]})
    zc = [
        v
        for b in ds.ml.iter_torch_batches(batch_size=2, device=None, zero_copy=True)
        for v in b["x"].tolist()
    ]
    cp = [
        v
        for b in ds.ml.iter_torch_batches(batch_size=2, device=None, zero_copy=False)
        for v in b["x"].tolist()
    ]
    assert zc == cp == [1.0, 2.0, 3.0, 4.0]


# --- Lance vector search ----------------------------------------------------
def test_vector_search_roundtrip(tmp_path):
    lance = pytest.importorskip("lance", reason="pylance not installed")
    from batcher.ml import build_vector_index, vector_search

    # 300 random 8-d vectors (IVF_PQ needs >=256 training rows) + a planted match.
    n = 300
    rng = np.random.default_rng(0)
    vecs = rng.random((n, 8)).astype(np.float32)
    query = np.ones(8, dtype=np.float32)
    vecs[142] = query  # exact match
    table = pa.table(
        {
            "id": list(range(n)),
            "embedding": pa.FixedSizeListArray.from_arrays(pa.array(vecs.reshape(-1)), 8),
        }
    )
    uri = str(tmp_path / "vecs.lance")
    lance.write_dataset(table, uri)

    # Brute-force search (no index) returns the exact match first.
    out = vector_search(uri, query, column="embedding", k=5, columns=["id"]).collect()
    assert out.num_rows == 5
    assert out.column("id")[0].as_py() == 142  # nearest is the planted match
    assert "_distance" in out.column_names

    # Building an ANN index must not break search correctness.
    build_vector_index(uri, "embedding", num_partitions=4, num_sub_vectors=2)
    out2 = vector_search(uri, query, column="embedding", k=1, columns=["id"]).collect()
    assert out2.column("id")[0].as_py() == 142


# --- url.download + model-id infer/embed ---------------------------------------
def test_download_fetches_local_files(tmp_path):
    paths = []
    for i in range(4):
        p = tmp_path / f"f{i}.bin"
        p.write_bytes(f"data-{i}".encode())
        paths.append(str(p))
    ds = bt.from_pydict({"id": list(range(4)), "url": paths})
    out = ds.ml.download("url", output_column="blob").collect()
    assert [b.decode() for b in out.column("blob").to_pylist()] == [f"data-{i}" for i in range(4)]


def test_download_on_error_null(tmp_path):
    good = tmp_path / "ok.bin"
    good.write_bytes(b"ok")
    ds = bt.from_pydict({"url": [str(good), str(tmp_path / "missing.bin")]})
    out = ds.ml.download("url", output_column="blob", on_error="null").collect()
    vals = out.column("blob").to_pylist()
    assert vals[0] == b"ok" and vals[1] is None


def test_upload_writes_files_and_roundtrips(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    paths = []
    for i in range(3):
        p = src / f"s{i}.txt"
        p.write_bytes(f"payload-{i}".encode())
        paths.append(str(p))
    ds = bt.from_pydict({"id": list(range(3)), "url": paths})
    fetched = ds.ml.download("url", output_column="data")
    out = fetched.ml.upload("data", str(tmp_path / "out"), output_column="dest", extension=".bin")
    written = out.collect()
    for dest, expect in zip(written.column("dest").to_pylist(), range(3), strict=True):
        assert Path(dest).read_bytes().decode() == f"payload-{expect}"


def test_upload_with_name_column(tmp_path):
    ds = bt.from_pydict({"name": ["a", "b"], "data": [b"one", b"two"]})
    written = ds.ml.upload(
        "data", str(tmp_path / "o"), name_column="name", output_column="dest"
    ).collect()
    names = sorted(str(p).split("/")[-1] for p in written.column("dest").to_pylist())
    assert names == ["a", "b"]


def test_download_rejects_bad_on_error():
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="on_error"):
        bt.from_pydict({"url": ["x"]}).ml.download("url", on_error="oops")


def test_embed_model_id_builds_lazy_map_stage():
    from batcher.plan.logical import MapBatches

    # Building the plan must not require sentence-transformers (lazy load-once UDF).
    ds = bt.from_pydict({"text": ["hello", "world"]})
    plan = ds.ml.embed("all-MiniLM-L6-v2", column="text", num_gpus=0)._plan
    assert isinstance(plan, MapBatches)


def test_infer_model_id_requires_column():
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="column="):
        bt.from_pydict({"text": ["hi"]}).ml.infer("some-model-id")


def test_infer_model_id_builds_lazy_map_stage():
    from batcher.plan.logical import MapBatches

    # Building the plan must not require transformers (lazy load-once UDF).
    ds = bt.from_pydict({"text": ["great", "awful"]})
    plan = ds.ml.infer("distilbert-sst2", column="text")._plan
    assert isinstance(plan, MapBatches)


def test_vector_search_needs_lance_when_absent():
    try:
        import lance  # noqa: F401
    except ImportError:
        from batcher._internal.errors import BackendError
        from batcher.ml import vector_search

        with pytest.raises(BackendError, match=r"\[lance\]"):
            vector_search("/none.lance", [1.0, 2.0], column="e", k=1)
    else:
        pytest.skip("pylance installed; guard not exercised")
