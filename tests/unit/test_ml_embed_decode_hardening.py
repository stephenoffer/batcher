"""Regression tests for the embedding and multimodal-decode hardening.

Every test here pins a defect that shipped: an embedding path that could not normalize,
pool, chunk, or accept a torch tensor; a Lance index build that failed only after the
whole GPU cost was paid; and a media path that reopened each video container three
times, decoded clips serially, resolved a filesystem per row, and retried nothing.

Nothing here needs a GPU, a network, or a real model. The encoders are plain callables,
`av` and `sentence_transformers` are fakes injected into `sys.modules`, and the
`Dataset` seam is a stub that captures the `map_batches` UDF so the per-batch work can
be exercised without the native engine.
"""

from __future__ import annotations

import importlib
import sys
import time
import types
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from batcher._internal.errors import BackendError, PlanError
from batcher.ml.embed import (
    _to_embedding_column,
    _validate_vector_field,
    embed,
    sentence_transformer_encoder,
)

# `batcher.ml` re-exports an `embed` *function*, which shadows the `embed` *module* on the
# package object, so `batcher.ml.decode` is resolved through importlib rather than by
# attribute access, which would break the same way if `decode` were ever re-exported too.
decode_mod = importlib.import_module("batcher.ml.decode")

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- helpers


class _StubDataset:
    """The `Dataset` seam the decode helpers use: `.columns` plus `.map_batches`."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.udf: Any = None
        self.output_columns: list[str] | None = None

    def map_batches(self, udf: Any, *, output_columns: list[str] | None = None) -> _StubDataset:
        self.udf = udf
        self.output_columns = output_columns
        return self


def _batches(rows: dict[str, list[Any]]) -> list[pa.RecordBatch]:
    return [pa.RecordBatch.from_pydict(rows)]


def _vectors(n: int, dims: int = 4, value: float = 3.0) -> np.ndarray:
    return np.full((n, dims), value, dtype=np.float32)


# ----------------------------------------------------------------- 1. normalization


def test_embed_normalizes_in_the_same_pass() -> None:
    """`normalize=True` yields unit vectors without a second normalize_embeddings scan."""
    out = list(
        embed(
            _batches({"text": ["a", "b", "c"]}),
            lambda: lambda texts: _vectors(len(texts)),
            text_column="text",
            normalize=True,
        )
    )
    vectors = np.stack(out[0].column("embedding").to_numpy(zero_copy_only=False))
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)


def test_normalize_leaves_a_zero_vector_at_zero_not_nan() -> None:
    """A zero vector has no direction; dividing by its norm would poison the column."""
    column = _to_embedding_column(np.zeros((2, 4), dtype=np.float32), normalize=True)
    assert not np.isnan(np.stack(column.to_numpy(zero_copy_only=False))).any()


# ------------------------------------------------- 2. sentence-transformers wiring


def _install_fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    class _FakeModel:
        def __init__(self, name: str, device: str | None = None) -> None:
            seen["model"] = name
            seen["device"] = device
            seen["halved"] = False

        def half(self) -> None:
            seen["halved"] = True

        def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
            seen["encode_kwargs"] = kwargs
            return _vectors(len(texts))

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return seen


def test_st_encoder_sizes_the_forward_pass_from_the_arrow_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an explicit batch_size the library caps the GPU at its internal default of 32."""
    seen = _install_fake_sentence_transformers(monkeypatch)
    encoder = sentence_transformer_encoder("m", "text")()
    encoder(pa.RecordBatch.from_pydict({"text": [f"t{i}" for i in range(50)]}))
    assert seen["encode_kwargs"]["batch_size"] == 50


def test_st_encoder_wires_device_normalize_and_fp16(monkeypatch: pytest.MonkeyPatch) -> None:
    """device/normalize_embeddings/fp16 all reach the model instead of being dropped."""
    seen = _install_fake_sentence_transformers(monkeypatch)
    monkeypatch.setattr("batcher.ml.gpu.torch_device", lambda backend=None: "cuda")
    encoder = sentence_transformer_encoder("m", "text", normalize=True, fp16=True)()
    encoder(pa.RecordBatch.from_pydict({"text": ["a", "b"]}))
    assert seen["device"] == "cuda"
    assert seen["halved"] is True
    assert seen["encode_kwargs"]["normalize_embeddings"] is True


def test_st_encoder_skips_fp16_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Half precision on CPU is slower, not faster, so it is not applied there."""
    seen = _install_fake_sentence_transformers(monkeypatch)
    monkeypatch.setattr("batcher.ml.gpu.torch_device", lambda backend=None: "cpu")
    sentence_transformer_encoder("m", "text", fp16=True)()
    assert seen["halved"] is False


# ------------------------------------------------------- 3. pooling and chunking


@pytest.mark.parametrize(
    ("pooling", "expected"),
    [("mean", 1.0), ("cls", 0.0), ("last", 2.0)],
)
def test_pooling_reduces_a_per_token_encoder_output(pooling: str, expected: float) -> None:
    """A 3-D (rows, tokens, dims) result is poolable rather than an unconditional error."""
    tokens = np.zeros((2, 3, 4), dtype=np.float32)
    tokens[:, 1, :] = 1.0
    tokens[:, 2, :] = 2.0
    column = _to_embedding_column(tokens, pooling=pooling)
    assert np.allclose(np.stack(column.to_numpy(zero_copy_only=False)), expected)


def test_three_dimensional_output_without_pooling_names_the_fix() -> None:
    with pytest.raises(BackendError, match="pooling='mean'"):
        _to_embedding_column(np.zeros((2, 3, 4), dtype=np.float32))


def test_chunking_encodes_a_long_document_in_windows_and_averages_them() -> None:
    """A document longer than the context window is chunked, not silently truncated."""
    seen: list[list[str]] = []

    def encoder(texts: list[str]) -> np.ndarray:
        seen.append(list(texts))
        return np.array([[float(len(t))] * 4 for t in texts], dtype=np.float32)

    out = list(
        embed(
            _batches({"text": ["x" * 250, "short"]}),
            lambda: encoder,
            text_column="text",
            chunk_size=100,
            chunk_overlap=20,
        )
    )
    # Row 0 becomes three overlapping windows, row 1 stays one.
    assert len(seen[0]) == 4
    assert out[0].num_rows == 2


def test_chunking_rejects_an_overlap_that_cannot_advance() -> None:
    with pytest.raises(PlanError, match="chunk_overlap"):
        list(
            embed(
                _batches({"text": ["a"]}),
                lambda: lambda texts: _vectors(len(texts)),
                text_column="text",
                chunk_size=10,
                chunk_overlap=10,
            )
        )


# --------------------------------------------------------------- 4. torch tensors


class _FakeCudaTensor:
    """A torch-like tensor that refuses conversion until detached and moved to host."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def detach(self) -> _FakeCudaTensor:
        return self

    def cpu(self) -> _FakeCudaTensor:
        return _HostTensor(self._array)

    def __array__(self, dtype: Any = None) -> np.ndarray:
        raise TypeError("can't convert cuda:0 device type tensor to numpy")


class _HostTensor(_FakeCudaTensor):
    def __init__(self, array: np.ndarray) -> None:
        super().__init__(array)
        self.dtype = "torch.float32"

    def numpy(self) -> np.ndarray:
        return self._array


def test_encoder_returning_a_device_tensor_is_detached_and_moved_to_host() -> None:
    column = _to_embedding_column(_FakeCudaTensor(_vectors(3)))
    assert np.stack(column.to_numpy(zero_copy_only=False)).shape == (3, 4)


def test_real_torch_cpu_tensor_round_trips() -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.ones(3, 4, requires_grad=True)  # grad makes a bare .numpy() raise
    column = _to_embedding_column(tensor)
    assert np.stack(column.to_numpy(zero_copy_only=False)).shape == (3, 4)


# ------------------------------------------------- 5. Lance-indexable vector types


def test_embed_can_emit_the_fixed_size_list_lance_indexes() -> None:
    out = list(
        embed(
            _batches({"text": ["a", "b"]}),
            lambda: lambda texts: _vectors(len(texts)),
            text_column="text",
            output_type="fixed_size_list",
        )
    )
    dtype = out[0].schema.field("embedding").type
    assert pa.types.is_fixed_size_list(dtype)
    assert dtype.value_type == pa.float32()


def test_build_vector_index_rejects_a_tensor_column_with_the_fix_in_the_message() -> None:
    """The mismatch must surface before the index build, not after the GPU cost is paid."""
    schema = pa.schema([pa.field("embedding", pa.fixed_shape_tensor(pa.float32(), [4]))])
    with pytest.raises(BackendError, match="output_type='fixed_size_list'"):
        _validate_vector_field(schema, "embedding")


def test_build_vector_index_rejects_a_non_vector_column() -> None:
    schema = pa.schema([pa.field("embedding", pa.list_(pa.float32()))])
    with pytest.raises(BackendError, match="fixed_size_list"):
        _validate_vector_field(schema, "embedding")


def test_build_vector_index_reports_a_missing_column() -> None:
    schema = pa.schema([pa.field("other", pa.int64())])
    with pytest.raises(BackendError, match="no column 'embedding'"):
        _validate_vector_field(schema, "embedding")


def test_build_vector_index_accepts_a_fixed_size_list_of_floats() -> None:
    schema = pa.schema([pa.field("embedding", pa.list_(pa.float32(), 4))])
    assert _validate_vector_field(schema, "embedding") is None


# ----------------------------------------------------- 6. bounded parallel decode


def test_bounded_map_preserves_order_and_caps_work_in_flight() -> None:
    """Executor.map would submit every clip up front, materializing the whole batch."""
    in_flight = {"now": 0, "peak": 0}

    def slow(item: int) -> int:
        in_flight["now"] += 1
        in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
        time.sleep(0.01)
        in_flight["now"] -= 1
        return item * 2

    result = list(decode_mod._bounded_map(slow, range(12), 3))
    assert result == [i * 2 for i in range(12)]
    assert in_flight["peak"] <= 3


def test_video_decode_consumes_clips_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    """At most `decode_concurrency` clips are pulled ahead of the decoder."""
    pytest.importorskip("PIL")
    _install_fake_av(monkeypatch, frames=6)
    ds = _StubDataset(["bytes"])
    decode_mod.video_dataset(
        ds, size=(4, 4), num_frames=2, source_column="bytes", decode_concurrency=2
    )
    batch = pa.RecordBatch.from_pydict({"bytes": [b"clip"] * 5})
    out = ds.udf(batch)
    assert out.num_rows == 5


# ------------------------------------------- 7 & 8. one container open, and seeking


def _install_fake_av(
    monkeypatch: pytest.MonkeyPatch, *, frames: int, advertised: int | None = None
) -> dict[str, Any]:
    stats: dict[str, Any] = {"opens": 0, "decoded": 0, "seeks": []}

    class _Frame:
        def __init__(self, index: int) -> None:
            self.index = index

        def to_ndarray(self, format: str = "rgb24") -> np.ndarray:
            return np.full((8, 8, 3), self.index % 256, dtype=np.uint8)

    header_frames = advertised if advertised is not None else frames

    class _Stream:
        frames = header_frames
        duration = frames
        time_base = 1.0

    class _Container:
        def __init__(self) -> None:
            self.streams = types.SimpleNamespace(video=[_Stream()])
            self.duration = frames * 1_000_000
            self._pos = 0

        def decode(self, video: int = 0) -> Any:
            while self._pos < frames:
                stats["decoded"] += 1
                index = self._pos
                self._pos += 1
                yield _Frame(index)

        def seek(self, offset: int, stream: Any = None) -> None:
            stats["seeks"].append(int(offset))
            self._pos = min(int(offset), frames - 1)

        def __enter__(self) -> _Container:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

    def _open(_buffer: Any) -> _Container:
        stats["opens"] += 1
        return _Container()

    module = types.ModuleType("av")
    module.open = _open  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "av", module)
    return stats


def test_video_opens_the_container_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Header probe, sampling, and recount used to cost three separate opens."""
    pytest.importorskip("PIL")
    stats = _install_fake_av(monkeypatch, frames=30)
    decode_mod._decode_video_bytes(b"clip", 4, 4, 4)
    assert stats["opens"] == 1


def test_video_recounts_a_lying_header_without_reopening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An advertised frame count that overshoots forces a recount — on the same handle."""
    pytest.importorskip("PIL")
    stats = _install_fake_av(monkeypatch, frames=30, advertised=45)
    out = decode_mod._decode_video_bytes(b"clip", 4, 4, 4)
    assert stats["opens"] == 1
    assert out.shape == (4, 4, 4, 3)


def test_seek_sampling_reads_a_handful_of_frames_not_the_whole_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The point of seeking: 8 frames out of 100k costs 8 decodes, not 100k."""
    pytest.importorskip("PIL")
    stats = _install_fake_av(monkeypatch, frames=100_000)
    out = decode_mod._decode_video_bytes(b"clip", 8, 4, 4, True)
    assert out.shape == (8, 4, 4, 3)
    assert stats["decoded"] <= 16
    assert len(stats["seeks"]) == 8


def test_in_order_sampling_still_decodes_sequentially(monkeypatch: pytest.MonkeyPatch) -> None:
    """seek=False keeps the exact, linear behavior — the seek path is opt-in."""
    pytest.importorskip("PIL")
    stats = _install_fake_av(monkeypatch, frames=40)
    decode_mod._decode_video_bytes(b"clip", 4, 4, 4)
    assert stats["seeks"] == []
    assert stats["decoded"] >= 40 - 1


# ----------------------------------------- 9. download retries, timeout, errors


class _FailingFilesystem:
    def __init__(self, fails: int, delay: float = 0.0) -> None:
        self.fails = fails
        self.delay = delay
        self.attempts = 0

    def open(self, path: str) -> Any:
        self.attempts += 1
        if self.delay:
            time.sleep(self.delay)
        if self.attempts <= self.fails:
            raise OSError("throttled")
        return _Handle(b"payload")


class _Handle:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> _Handle:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _download_udf(monkeypatch: pytest.MonkeyPatch, fs: Any, **kwargs: Any) -> Any:
    monkeypatch.setattr("batcher.io.filesystem.resolve_filesystem", lambda path, **kw: fs)
    ds = _StubDataset(["url"])
    decode_mod.download_dataset(ds, url_column="url", **kwargs)
    return ds


def test_download_retries_a_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A throttled object store is not a permanently bad URL."""
    fs = _FailingFilesystem(fails=2)
    ds = _download_udf(monkeypatch, fs, retries=3, retry_backoff=0.001)
    out = ds.udf(pa.RecordBatch.from_pydict({"url": ["s3://b/k"]}))
    assert out.column("bytes").to_pylist() == [b"payload"]
    assert fs.attempts == 3


def test_download_gives_up_after_the_retry_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    fs = _FailingFilesystem(fails=99)
    ds = _download_udf(monkeypatch, fs, retries=1, retry_backoff=0.001)
    with pytest.raises(PlanError, match="after 2 attempts"):
        ds.udf(pa.RecordBatch.from_pydict({"url": ["s3://b/k"]}))
    assert fs.attempts == 2


def test_download_records_why_each_row_is_null(monkeypatch: pytest.MonkeyPatch) -> None:
    """A null with no reason is an undiagnosable partial result."""
    fs = _FailingFilesystem(fails=99)
    ds = _download_udf(
        monkeypatch,
        fs,
        on_error="null",
        retries=0,
        error_column="download_error",
    )
    out = ds.udf(pa.RecordBatch.from_pydict({"url": ["s3://b/k"]}))
    assert out.column("bytes").to_pylist() == [None]
    assert "throttled" in out.column("download_error").to_pylist()[0]
    assert ds.output_columns == ["url", "bytes", "download_error"]


def test_download_error_column_requires_null_handling() -> None:
    with pytest.raises(PlanError, match="requires on_error='null'"):
        decode_mod.download_dataset(_StubDataset(["url"]), url_column="url", error_column="err")


def test_download_times_out_a_stalled_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A slow origin must not pin a worker for the life of the job."""
    fs = _FailingFilesystem(fails=0, delay=2.0)
    ds = _download_udf(
        monkeypatch,
        fs,
        on_error="null",
        retries=0,
        timeout=0.05,
        error_column="download_error",
    )
    start = time.monotonic()
    out = ds.udf(pa.RecordBatch.from_pydict({"url": ["s3://b/k"]}))
    assert time.monotonic() - start < 1.5
    assert out.column("bytes").to_pylist() == [None]
    assert "timeout" in out.column("download_error").to_pylist()[0]


def test_download_reuses_one_thread_pool_across_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh ThreadPoolExecutor per batch costs more than the transfers it overlaps."""
    fs = _FailingFilesystem(fails=0)
    ds = _download_udf(monkeypatch, fs, max_concurrency=4)
    batch = pa.RecordBatch.from_pydict({"url": ["s3://b/k"] * 3})
    ds.udf(batch)
    first = decode_mod._shared_pool(4)
    ds.udf(batch)
    assert decode_mod._shared_pool(4) is first


def test_download_rejects_a_negative_retry_budget() -> None:
    with pytest.raises(PlanError, match="retries must be >= 0"):
        decode_mod.download_dataset(_StubDataset(["url"]), url_column="url", retries=-1)


# ------------------------------------------------ 10. one filesystem per batch


def test_upload_resolves_the_filesystem_once_per_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """It used to be resolved inside the thread map, once for every single file."""
    calls: list[str] = []

    class _Writer:
        def __enter__(self) -> _Writer:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

        def write(self, data: bytes) -> None:
            return None

    class _FS:
        def atomic_writer(self, path: str) -> _Writer:
            return _Writer()

    def _resolve(path: str, **kwargs: Any) -> _FS:
        calls.append(path)
        return _FS()

    monkeypatch.setattr("batcher.io.filesystem.resolve_filesystem", _resolve)
    ds = _StubDataset(["data"])
    decode_mod.upload_dataset(ds, data_column="data", directory="s3://bucket/out")
    out = ds.udf(pa.RecordBatch.from_pydict({"data": [b"a", b"b", b"c", b"d", b"e"]}))
    assert len(calls) == 1
    assert out.num_rows == 5


# --------------------------------------------------------- 11. uint8 end to end


def test_video_frames_stay_uint8(monkeypatch: pytest.MonkeyPatch) -> None:
    """float32 is 4x the bytes and belongs at the GPU, not in the decode stage."""
    pytest.importorskip("PIL")
    _install_fake_av(monkeypatch, frames=10)
    ds = _StubDataset(["bytes"])
    decode_mod.video_dataset(ds, size=(4, 4), num_frames=2, source_column="bytes")
    out = ds.udf(pa.RecordBatch.from_pydict({"bytes": [b"clip", None]}))
    assert out.schema.field("frames").type.storage_type.value_type == pa.uint8()
