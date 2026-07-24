"""Served-endpoint embedding encoders — `openai_embedding_encoder` / `tei_encoder`.

These call an HTTP embedding service instead of loading a local model. Every test fakes
`batcher.ml.serving.http.post_json`, so nothing here touches the network — the encoders
import `post_json` at call time, so the patch always precedes it.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher.ml.serving.http as http_mod
from batcher.ml import openai_embedding_encoder, tei_encoder


@pytest.fixture
def batch() -> pa.RecordBatch:
    return pa.RecordBatch.from_pydict({"text": ["cat", "dog", "bird"]})


def _fake_post(fn):
    """Swap `post_json` for `fn` for the duration of a `with` block."""
    import contextlib

    @contextlib.contextmanager
    def cm():
        real = http_mod.post_json
        http_mod.post_json = fn
        try:
            yield
        finally:
            http_mod.post_json = real

    return cm()


def test_openai_encoder_appends_a_fixed_size_list_column(batch):
    """The default output is a fixed_size_list<float32> — what Lance ANN indexing expects."""

    def fake(url, body, **kw):
        assert url.endswith("/embeddings")
        assert body["model"] == "m"
        vecs = {"cat": [1.0, 0.0], "dog": [0.0, 1.0], "bird": [1.0, 1.0]}
        return {"data": [{"index": i, "embedding": vecs[t]} for i, t in enumerate(body["input"])]}

    enc = openai_embedding_encoder("m", "text", base_url="http://x/v1")
    with _fake_post(fake):
        out = enc()(batch)
    col = out.column("embedding")
    assert pa.types.is_fixed_size_list(col.type)
    assert col.type.list_size == 2
    assert out.column("embedding").to_pylist() == [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]


def test_openai_encoder_reorders_by_response_index(batch):
    """A server that returns embeddings out of order still pairs each vector to its text."""

    def fake(url, body, **kw):
        data = [{"index": i, "embedding": [float(i)]} for i in range(len(body["input"]))]
        return {"data": list(reversed(data))}  # deliberately scrambled

    enc = openai_embedding_encoder("m", "text", base_url="http://x/v1")
    with _fake_post(fake):
        out = enc()(batch)
    assert out.column("embedding").to_pylist() == [[0.0], [1.0], [2.0]]


def test_openai_encoder_passes_dimensions_when_set(batch):
    """`dimensions` reaches the request body (Matryoshka truncation), and is omitted otherwise."""
    seen = {}

    def fake(url, body, **kw):
        seen.update(body)
        return {"data": [{"index": i, "embedding": [0.0]} for i in range(len(body["input"]))]}

    with _fake_post(fake):
        openai_embedding_encoder("m", "text", base_url="http://x/v1", dimensions=64)()(batch)
    assert seen["dimensions"] == 64
    seen.clear()
    with _fake_post(fake):
        openai_embedding_encoder("m", "text", base_url="http://x/v1")()(batch)
    assert "dimensions" not in seen


def test_openai_encoder_chunks_by_max_batch(batch):
    """More texts than `max_batch` are split across requests, in order, and reassembled."""
    calls = []

    def fake(url, body, **kw):
        calls.append(list(body["input"]))
        return {
            "data": [{"index": i, "embedding": [len(t) * 1.0]} for i, t in enumerate(body["input"])]
        }

    enc = openai_embedding_encoder("m", "text", base_url="http://x/v1", max_batch=2, concurrency=1)
    with _fake_post(fake):
        out = enc()(batch)
    assert calls == [["cat", "dog"], ["bird"]]  # two requests, order preserved
    assert out.column("embedding").to_pylist() == [[3.0], [3.0], [4.0]]


def test_openai_encoder_maps_null_text_to_empty_string():
    """A null cell still yields a vector aligned to its row (embeds the empty string)."""
    seen = {}

    def fake(url, body, **kw):
        seen["input"] = list(body["input"])
        return {"data": [{"index": i, "embedding": [0.0]} for i in range(len(body["input"]))]}

    b = pa.RecordBatch.from_pydict({"text": ["a", None, "c"]})
    with _fake_post(fake):
        out = openai_embedding_encoder("m", "text", base_url="http://x/v1")()(b)
    assert seen["input"] == ["a", "", "c"]
    assert out.num_rows == 3


def test_tei_encoder_reads_a_bare_list_response(batch):
    """TEI's /embed returns the vectors as a plain list, already in input order."""

    def fake(url, body, **kw):
        assert url.endswith("/embed")
        assert body["inputs"] == ["cat", "dog", "bird"]
        assert body["normalize"] is True and body["truncate"] is True
        return [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    enc = tei_encoder("text", base_url="http://tei:8080", concurrency=1)
    with _fake_post(fake):
        out = enc()(batch)
    assert out.column("embedding").to_pylist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


def test_output_type_is_validated():
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError):
        openai_embedding_encoder("m", "text", output_type="nonsense")
    with pytest.raises(PlanError):
        tei_encoder("text", base_url="http://x", output_type="nonsense")
