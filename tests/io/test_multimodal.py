"""Multimodal media source coverage.

The `MediaSource` base behavior — listing, file-batch Arrow assembly, the common
``uri/bytes/size/mime`` columns, MIME sniffing, splits, and header-only metadata
— is tested unconditionally via a tiny test-local subclass over real temp files
(no optional dependency). The concrete `ImageSource` / `AudioSource` /
`VideoSource` metadata paths are gated with ``pytest.importorskip`` so the suite
runs without Pillow / soundfile / PyAV installed.
"""

from __future__ import annotations

import pickle
import struct
import zlib

import pyarrow as pa
import pytest

from batcher.io.formats.base import SOURCES
from batcher.io.formats.multimodal.media import MediaSource, MediaSplit


def _make_png(width: int, height: int) -> bytes:
    """A minimal but valid PNG (header + IHDR + one IDAT + IEND), no real pixels.

    Valid enough that PIL reads its size/mode from the IHDR header without
    needing to decode the (tiny) image data.
    """

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    raw = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    idat = zlib.compress(raw)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


class _BlobSource(MediaSource):
    """A media source with no metadata columns — exercises the base behavior."""

    suffixes = (".bin",)
    format_name = "_blob_test"

    __slots__ = ()


# Register the test source so MediaSplit can reconstruct it on the worker side.
if "_blob_test" not in SOURCES:
    SOURCES.add("_blob_test", _BlobSource)


def _write_blobs(tmp_path, payloads: list[bytes]) -> str:
    for i, data in enumerate(payloads):
        (tmp_path / f"f{i:03d}.bin").write_bytes(data)
    return str(tmp_path)


def test_media_base_common_columns_and_schema(tmp_path):
    path = _write_blobs(tmp_path, [b"hello", b"world!!", b""])
    src = _BlobSource(path, batch_files=64)

    assert src.schema().names == ["uri", "bytes", "size", "mime"]
    assert src.row_count() == 3

    table = pa.Table.from_batches(src.read(), schema=src.schema())
    assert table.num_rows == 3
    assert sorted(table.column("size").to_pylist()) == [0, 5, 7]
    assert set(table.column("bytes").to_pylist()) == {b"hello", b"world!!", b""}
    assert all(uri.endswith(".bin") for uri in table.column("uri").to_pylist())
    # An unknown payload sniffs to the octet-stream default.
    assert set(table.column("mime").to_pylist()) == {"application/octet-stream"}


def test_reference_mode_emits_handles_without_payload(tmp_path):
    # Reference mode: bytes is null, but size/mime/uri come from header + stat —
    # the GB-per-row OOM fix (no full payload ever loaded).
    payloads = [_make_png(2, 3), b"x" * 5000, b""]
    path = _write_blobs(tmp_path, payloads)
    src = _BlobSource(path, batch_files=64, materialize_bytes=False)

    table = pa.Table.from_batches(src.read(), schema=src.schema())
    assert table.num_rows == 3
    # Payload column is entirely null (a handle, not the bytes).
    assert table.column("bytes").null_count == 3
    # Size is the true file size (from stat, not from reading the content).
    assert sorted(table.column("size").to_pylist()) == [0, len(payloads[0]), 5000]
    # MIME still sniffed from the header (the PNG magic is recognized).
    assert "image/png" in table.column("mime").to_pylist()


def test_reference_mode_round_trips_via_read_blob_bytes(tmp_path):
    from batcher.io.formats.multimodal.media import read_blob_bytes

    payloads = [b"alpha", b"beta-payload", b"gamma!!"]
    path = _write_blobs(tmp_path, payloads)
    src = _BlobSource(path, batch_files=64, materialize_bytes=False)

    handle_batch = src.read()[0]
    assert handle_batch.column("bytes").null_count == handle_batch.num_rows
    filled = read_blob_bytes(handle_batch)
    assert filled.column("bytes").null_count == 0
    assert set(filled.column("bytes").to_pylist()) == set(payloads)


def test_reference_mode_splits_carry_the_flag(tmp_path):
    path = _write_blobs(tmp_path, [b"a", b"bb", b"ccc"])
    src = _BlobSource(path, batch_files=2, materialize_bytes=False)
    splits = src.splits()
    assert all(s.materialize_bytes is False for s in splits)
    restored = pickle.loads(pickle.dumps(splits[0]))
    t = pa.Table.from_batches(restored.read(), schema=src.schema())
    assert t.column("bytes").null_count == t.num_rows


def test_media_base_batches_per_chunk(tmp_path):
    path = _write_blobs(tmp_path, [bytes([i]) for i in range(10)])
    src = _BlobSource(path, batch_files=4)
    batches = src.read()
    # 10 files / 4 per batch = 3 batches (4, 4, 2).
    assert [b.num_rows for b in batches] == [4, 4, 2]


def test_media_mime_sniffed_from_png_magic(tmp_path):
    (tmp_path / "img.bin").write_bytes(_make_png(2, 3))
    src = _BlobSource(str(tmp_path), batch_files=8)
    table = pa.Table.from_batches(src.read(), schema=src.schema())
    assert table.column("mime").to_pylist() == ["image/png"]


def test_media_splits_cover_and_are_picklable(tmp_path):
    path = _write_blobs(tmp_path, [bytes([i]) for i in range(7)])
    src = _BlobSource(path, batch_files=3)
    splits = src.splits()
    assert len(splits) == 3
    assert all(isinstance(s, MediaSplit) for s in splits)

    # Picklable (locator-only) and reconstructs to the same rows.
    for s in splits:
        restored = pickle.loads(pickle.dumps(s))
        assert restored == s

    covered = pa.concat_tables(
        [pa.Table.from_batches(s.read(), schema=src.schema()) for s in splits]
    )
    whole = pa.Table.from_batches(src.read(), schema=src.schema())
    assert covered.column("uri").to_pylist() == whole.column("uri").to_pylist()
    assert covered.num_rows == 7


def test_media_projection(tmp_path):
    path = _write_blobs(tmp_path, [b"a", b"bb"])
    src = _BlobSource(path, batch_files=8)
    got = pa.Table.from_batches(src.read(projection=["uri", "size"]))
    assert got.column_names == ["uri", "size"]


def test_media_empty_dir_raises(tmp_path):
    with pytest.raises(Exception, match="_blob_test"):
        _BlobSource(str(tmp_path)).read()


# ---- concrete sources (optional deps) ------------------------------------
def test_image_source_header_only_meta(tmp_path):
    pytest.importorskip("PIL")
    from batcher.io.formats.multimodal.images import ImageSource

    (tmp_path / "a.png").write_bytes(_make_png(4, 6))
    (tmp_path / "b.png").write_bytes(_make_png(8, 2))
    src = ImageSource(str(tmp_path), batch_files=8)

    assert "width" in src.schema().names and "height" in src.schema().names
    table = pa.Table.from_batches(src.read(), schema=src.schema())
    rows = {r["uri"].split("/")[-1]: r for r in table.to_pylist()}
    assert (rows["a.png"]["width"], rows["a.png"]["height"]) == (4, 6)
    assert (rows["b.png"]["width"], rows["b.png"]["height"]) == (8, 2)
    assert rows["a.png"]["mode"] == "RGB"
    assert rows["a.png"]["mime"] == "image/png"


def test_image_source_corrupt_header_yields_null_meta(tmp_path):
    pytest.importorskip("PIL")
    from batcher.io.formats.multimodal.images import ImageSource

    (tmp_path / "ok.png").write_bytes(_make_png(2, 2))
    (tmp_path / "bad.png").write_bytes(b"\x89PNG\r\n\x1a\nnot-a-real-png")
    src = ImageSource(str(tmp_path), batch_files=8)
    table = pa.Table.from_batches(src.read(), schema=src.schema())
    rows = {r["uri"].split("/")[-1]: r for r in table.to_pylist()}
    assert rows["ok.png"]["width"] == 2
    assert rows["bad.png"]["width"] is None  # bad header → null meta, batch survives


def test_audio_source_schema():
    pytest.importorskip("soundfile")
    from batcher.io.formats.multimodal.audio import AudioSource

    names = AudioSource.__new__(AudioSource)._meta_fields()
    assert [n for n, _ in names] == ["sample_rate", "channels", "duration"]


def test_video_source_schema():
    pytest.importorskip("av")
    from batcher.io.formats.multimodal.video import VideoSource

    names = VideoSource.__new__(VideoSource)._meta_fields()
    assert [n for n, _ in names] == ["fps", "frames", "width", "height", "duration"]


def test_embedding_source_npy_fixed_size_list(tmp_path):
    np = pytest.importorskip("numpy")
    from batcher.io.formats.multimodal.embeddings import EmbeddingSource

    np.save(str(tmp_path / "a.npy"), np.arange(6, dtype=np.float32).reshape(3, 2))
    np.save(str(tmp_path / "b.npy"), np.ones((2, 2), dtype=np.float32))
    src = EmbeddingSource(str(tmp_path), batch_files=8)

    assert pa.types.is_fixed_size_list(src.schema().field("embedding").type)
    table = pa.Table.from_batches(src.read(), schema=src.schema())
    assert table.num_rows == 5
    assert table.column("embedding").to_pylist()[0] == [0.0, 1.0]
