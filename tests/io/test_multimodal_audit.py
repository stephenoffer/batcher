"""Multimodal IO audit — the two silent-but-wrong bugs found in the media path.

Both bugs pass every existing test and a code review: the result is *correct*, only
the I/O it performs is wildly wrong. So each test here asserts on the bytes actually
read, not on the answer.

1. **Projection does not avoid materializing the payload (bug class 6).** A
   ``select("uri", "size")`` over a media source dropped ``bytes`` *after* the whole
   payload had already been read into the Arrow batch — so a metadata query over a
   directory of GB videos silently downloaded the entire corpus. The fix reads
   header-only whenever the projection excludes ``bytes``.
2. **`schema()` reads the whole first file (bug class 2).** `EmbeddingSource` learned
   its embedding dimension by ``np.load``-ing / ``read_table``-ing the first file and
   taking ``.shape[1]`` — a full read of a large matrix to discover a number stored in
   the file *header*. The fix reads only the ``.npy`` header block / the ``.parquet``
   schema.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from batcher.io.formats.base import SOURCES
from batcher.io.formats.multimodal.embeddings import EmbeddingSource
from batcher.io.formats.multimodal.media import MediaSource

pytestmark = pytest.mark.unit

# _read_payload reads a 64 KiB header in reference mode; test payloads must be larger
# than that so a header-only read is distinguishable from a full read by bytes counted.
_HEADER_BYTES = 1 << 16
_PAYLOAD = 300_000  # > _HEADER_BYTES


class _CountingHandle:
    """Wraps a real file handle, tallying every byte returned by ``read``.

    Delegates seek/tell/readable/... to the inner handle so it is a faithful stand-in
    for the pyarrow handle the real filesystem returns (numpy's ``.npy`` header readers
    and pyarrow's ``read_schema`` both drive it through ``read``/``seek``).
    """

    def __init__(self, inner, counter: list[int]) -> None:
        self._inner = inner
        self._counter = counter

    def read(self, *args, **kwargs):
        data = self._inner.read(*args, **kwargs)
        self._counter[0] += len(data)
        return data

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _CountingFS:
    """A filesystem façade that counts payload bytes read, delegating everything else.

    ``expand``/``size`` hit the real filesystem unchanged; only ``open`` is wrapped, so
    the count reflects exactly the bytes a read pulls off the file — the signal that
    tells a header-only read apart from a full download.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.bytes_read = [0]

    def open(self, path: str, mode: str = "rb"):
        return _CountingHandle(self._inner.open(path, mode), self.bytes_read)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _BlobAuditSource(MediaSource):
    """No-metadata media source over ``.bin`` files — exercises the base read path."""

    suffixes = (".bin",)
    format_name = "_blob_audit"

    __slots__ = ()


if "_blob_audit" not in SOURCES:
    SOURCES.add("_blob_audit", _BlobAuditSource)


def _write_blobs(tmp_path, n: int) -> str:
    for i in range(n):
        (tmp_path / f"f{i:03d}.bin").write_bytes(b"\xab" * _PAYLOAD)
    return str(tmp_path)


def _spy_source(path: str, **kw) -> tuple[_BlobAuditSource, _CountingFS]:
    src = _BlobAuditSource(path, **kw)
    spy = _CountingFS(src._fs)
    src._fs = spy
    return src, spy


# --------------------------------------------------------------------------- bug 6
def test_projection_dropping_bytes_reads_header_only():
    """A projection without ``bytes`` must not download any payload.

    Fails before the fix: ``read`` materialized every payload and the ``.select`` merely
    discarded it, so bytes-read == full corpus regardless of the projection.
    """
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    n = 4
    path = _write_blobs(tmp, n)

    # Control: a full read (no projection) pulls every payload byte.
    full_src, full_spy = _spy_source(path, batch_files=1)
    full = pa.Table.from_batches(full_src.read(), schema=full_src.schema())
    assert full_spy.bytes_read[0] == n * _PAYLOAD
    assert full.column("bytes").null_count == 0

    # Projection drops `bytes`: reads only headers (<= 64 KiB/file), not payloads.
    proj_src, proj_spy = _spy_source(path, batch_files=1)
    projected = proj_src.read(projection=["uri", "size", "mime"])
    tbl = pa.Table.from_batches(projected)
    assert proj_spy.bytes_read[0] <= n * _HEADER_BYTES
    assert proj_spy.bytes_read[0] < n * _PAYLOAD  # strictly cheaper than a full read
    # Correctness preserved: uri/size still exact; the dropped payload is simply absent.
    assert tbl.num_rows == n
    assert set(tbl.column("size").to_pylist()) == {_PAYLOAD}
    assert "bytes" not in tbl.schema.names


def test_iter_batches_projection_dropping_bytes_reads_header_only():
    """The streaming path honors the same rule as the bulk ``read`` path."""
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    n = 3
    path = _write_blobs(tmp, n)

    src, spy = _spy_source(path, batch_files=2)
    list(src.iter_batches(projection=["uri", "size"]))
    assert spy.bytes_read[0] <= n * _HEADER_BYTES
    assert spy.bytes_read[0] < n * _PAYLOAD


def test_projection_including_bytes_still_materializes():
    """Guard the other direction: asking for ``bytes`` must still read the payload."""
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    n = 2
    path = _write_blobs(tmp, n)

    src, spy = _spy_source(path, batch_files=1)
    out = src.read(projection=["uri", "bytes"])
    tbl = pa.Table.from_batches(out)
    assert spy.bytes_read[0] == n * _PAYLOAD
    assert tbl.column("bytes").null_count == 0


def test_effective_materialize_matrix(tmp_path):
    """The projection→materialize decision, unit-level."""
    path = _write_blobs(tmp_path, 1)
    src = _BlobAuditSource(path, materialize_bytes=True)
    assert src._effective_materialize(None) is True
    assert src._effective_materialize(["uri", "bytes", "size"]) is True
    assert src._effective_materialize(["uri", "size"]) is False
    # Reference mode never materializes regardless of projection.
    ref = _BlobAuditSource(path, materialize_bytes=False)
    assert ref._effective_materialize(None) is False
    assert ref._effective_materialize(["uri", "bytes"]) is False


# --------------------------------------------------------------------------- bug 2
class _NoFullReadEmbeddings(EmbeddingSource):
    """Embedding source whose full-read path raises — proves schema() never takes it."""

    __slots__ = ()

    def _file_vectors(self, path: str):
        raise AssertionError(f"schema() must not read the payload of {path!r}")


def test_embedding_schema_npy_reads_header_only(tmp_path):
    """`schema()` on ``.npy`` learns the dimension from the header, not the payload.

    Fails before the fix: ``_dimension`` called ``_file_vectors`` (a full ``np.load``),
    which the spy turns into an ``AssertionError``; and the byte count equalled the whole
    file rather than a few header bytes.
    """
    dim = 96
    mat = np.random.rand(4000, dim).astype(np.float32)  # ~1.5 MB, >> a header
    np.save(tmp_path / "a.npy", mat, allow_pickle=False)
    file_bytes = (tmp_path / "a.npy").stat().st_size

    # The full-read path must never be taken.
    src = _NoFullReadEmbeddings(str(tmp_path))
    assert src.schema().field("embedding").type.list_size == dim

    # And it reads only the header block, not the ~1.5 MB payload.
    counted = _NoFullReadEmbeddings(str(tmp_path))
    spy = _CountingFS(counted._fs)
    counted._fs = spy
    assert counted.schema().field("embedding").type.list_size == dim
    assert spy.bytes_read[0] < 8192
    assert spy.bytes_read[0] < file_bytes


def test_embedding_schema_npy_1d_vector(tmp_path):
    """A 1-D ``.npy`` vector is one row whose width is the vector length."""
    vec = np.arange(64, dtype=np.float32)
    np.save(tmp_path / "v.npy", vec, allow_pickle=False)
    src = _NoFullReadEmbeddings(str(tmp_path))
    assert src.schema().field("embedding").type.list_size == 64


def test_embedding_schema_parquet_fixed_size_list_uses_schema(tmp_path):
    """A ``fixed_size_list`` parquet's dimension comes from the schema, not a full read."""
    dim = 48
    values = pa.array(np.random.rand(10 * dim).astype(np.float32), pa.float32())
    fsl = pa.FixedSizeListArray.from_arrays(values, dim)
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"embedding": fsl}), tmp_path / "e.parquet")

    src = _NoFullReadEmbeddings(str(tmp_path))
    assert src.schema().field("embedding").type.list_size == dim


def test_with_meta_is_part_of_the_identity(tmp_path):
    """A `with_meta` toggle changes the *schema*, so it changes which relation this is.

    With metadata on, the relation carries the format's width/height/duration columns;
    off, it does not. Those columns' learned per-column statistics are meaningless to the
    without-meta relation, so the two must not share a learned-stats key — Kyber would
    otherwise hand one relation's column stats to the other. The default (`with_meta=True`)
    keeps the historical key, so existing statistics are not orphaned by the fix.
    """
    from batcher.io.formats.base import SOURCES

    images = SOURCES.get("images")
    with_meta = images(str(tmp_path), with_meta=True)
    without = images(str(tmp_path), with_meta=False)

    assert with_meta.identity() != without.identity()
    assert with_meta.identity() == f"images:{tmp_path}"  # default key unchanged
