"""Audit regression tests for the ML/array file formats (numpy, HDF5, Zarr, WebDataset).

Each test pins one bug class from the audit and fails against the pre-fix code:

  * numpy      — `iter_batches` was a fake stream (materialized every file) and
                 recursed infinitely under `n_rows` (bug class 1).
  * HDF5/Zarr  — `schema()` pulled a hyperslab/chunk to learn a schema the format's
                 metadata already states (bug class 2); neither exposed the exact
                 row/chunk counts its metadata carries (bug class 6).
  * WebDataset — `schema()` extracted every sample's payload to learn a column set
                 the member *names* fully determine (bug class 2).
"""

from __future__ import annotations

import io
import tarfile

import numpy as np
import pyarrow as pa
import pytest

import batcher.io.formats.ml.numpy as npmod
from batcher.io.formats.ml._ndarray import schema_from_array_meta, slice_to_batch
from batcher.io.formats.ml.hdf5 import HDF5Source
from batcher.io.formats.ml.numpy import NumpySource
from batcher.io.formats.ml.webdataset import WebDatasetSource
from batcher.io.formats.ml.zarr import ZarrSource

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- numpy


def test_numpy_read_with_n_rows_does_not_recurse(tmp_path):
    """`n_rows` on a NumpySource caps the read instead of recursing forever.

    The old `iter_batches` delegated to `read`, and `read` under `n_rows` delegated
    back to `iter_batches` — an unbounded mutual recursion (RecursionError) on any
    capped NumPy read.
    """
    p = tmp_path / "a.npy"
    np.save(p, np.arange(10, dtype=np.int64))
    batches = NumpySource(str(p), n_rows=3).read()
    assert sum(b.num_rows for b in batches) == 3


def test_numpy_iter_batches_streams_lazily(tmp_path, monkeypatch):
    """Consuming one batch must not decode every file first (real streaming)."""
    n_files = 64
    for i in range(n_files):
        np.save(tmp_path / f"a{i:03d}.npy", np.arange(4, dtype=np.int64))

    calls: list[int] = []
    orig = npmod._table_from_npy_handle

    def spy(fh):
        calls.append(1)
        return orig(fh)

    monkeypatch.setattr(npmod, "_table_from_npy_handle", spy)
    it = NumpySource(str(tmp_path)).iter_batches()
    next(it)
    # Old override called `read()`, decoding all 64 files before yielding the first
    # batch. The streaming base only decodes its bounded read-ahead window.
    assert len(calls) < n_files


# --------------------------------------------------------------- shared meta-schema


class _SpyArray:
    """A dense-array stand-in exposing dtype/ndim/shape metadata and recording reads."""

    def __init__(self, arr: np.ndarray) -> None:
        self._a = arr
        self.reads: list[object] = []

    @property
    def dtype(self):
        return self._a.dtype

    @property
    def ndim(self) -> int:
        return self._a.ndim

    @property
    def shape(self):
        return self._a.shape

    def __getitem__(self, item):
        self.reads.append(item)
        return self._a[item]


def test_schema_from_array_meta_reads_no_data_and_matches_a_real_read():
    """The metadata schema equals a real slice's schema, without touching array data."""
    a2 = np.zeros((7, 3), dtype=np.float32)
    spy2 = _SpyArray(a2)
    assert schema_from_array_meta(spy2) == slice_to_batch(a2, None).schema
    assert spy2.reads == []  # no hyperslab/chunk pulled to learn the schema

    a1 = np.arange(5, dtype=np.int64)
    spy1 = _SpyArray(a1)
    assert schema_from_array_meta(spy1) == slice_to_batch(a1, None).schema
    assert spy1.reads == []


# ---------------------------------------------------------------------------- HDF5


def _write_h5(path, data) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data)


def test_hdf5_statistics_exact_row_count(tmp_path):
    """HDF5 knows its leading-axis length exactly from the dataset shape (bug 6)."""
    p = tmp_path / "d.h5"
    _write_h5(p, np.arange(100, dtype=np.int64))
    st = HDF5Source(str(p), dataset="data").statistics()
    assert st is not None
    assert st.row_count == 100
    assert st.exact_rows is True


def test_hdf5_schema_from_metadata_is_correct(tmp_path):
    """The metadata-derived schema is byte-identical to what a read would produce."""
    p = tmp_path / "d.h5"
    _write_h5(p, np.zeros((50, 3), dtype=np.float32))
    src = HDF5Source(str(p), dataset="data")
    sch = src.schema()
    assert sch.names == ["c0", "c1", "c2"]
    assert all(pa.types.is_float32(sch.field(n).type) for n in sch.names)
    assert sch == src.read()[0].schema


# ---------------------------------------------------------------------------- Zarr


def _write_zarr(path, shape, chunks, dtype) -> None:
    zarr = pytest.importorskip("zarr")
    z = zarr.open(str(path), mode="w", shape=shape, chunks=chunks, dtype=dtype)
    z[...] = np.zeros(shape, dtype=dtype)


def test_zarr_statistics_exact_row_and_chunk_count(tmp_path):
    """Zarr exposes exact rows and the leading-axis chunk count from metadata (bug 6)."""
    p = tmp_path / "z.zarr"
    _write_zarr(p, (100,), (10,), "int64")
    st = ZarrSource(str(p)).statistics()
    assert st is not None
    assert st.row_count == 100
    assert st.exact_rows is True
    assert st.row_group_count == 10


def test_zarr_schema_reads_no_chunk_data(tmp_path, monkeypatch):
    """`schema()` must be answered from `.zarray` metadata, never by reading a chunk."""
    p = tmp_path / "z.zarr"
    _write_zarr(p, (100, 3), (10, 3), "float32")
    src = ZarrSource(str(p))
    array_cls = type(src._array())

    def guard(self, item):
        raise AssertionError(f"schema() read chunk data: {item!r}")

    monkeypatch.setattr(array_cls, "__getitem__", guard)
    sch = src.schema()  # pre-fix: array[0:1] triggers the guard
    assert sch.names == ["c0", "c1", "c2"]
    assert all(pa.types.is_float32(sch.field(n).type) for n in sch.names)


# ------------------------------------------------------------------------ WebDataset


def _make_tar(path, samples) -> None:
    """Write ``samples`` (``[(key, {ext: bytes})]``) as a plain tar shard."""
    with tarfile.open(path, "w") as tar:
        for key, fields in samples:
            for ext, payload in fields.items():
                info = tarfile.TarInfo(f"{key}.{ext}")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))


_SAMPLES = [("a", {"jpg": b"x" * 1000, "cls": b"1"}), ("b", {"jpg": b"y" * 1000, "cls": b"2"})]


def test_webdataset_schema_reads_no_payloads(tmp_path, monkeypatch):
    """The column set is determined by member names — `schema()` extracts no payload."""
    p = tmp_path / "shard.tar"
    _make_tar(p, _SAMPLES)

    extracted: list[str] = []
    orig = tarfile.TarFile.extractfile

    def spy(self, member):
        extracted.append(member.name)
        return orig(self, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", spy)
    sch = WebDatasetSource(str(p)).schema()
    assert extracted == []  # pre-fix: every sample's bytes were pulled to learn the schema
    assert sch.names == ["__key__", "jpg", "cls"]
    assert pa.types.is_string(sch.field("__key__").type)
    assert pa.types.is_binary(sch.field("jpg").type)


def test_webdataset_schema_matches_read_and_roundtrips(tmp_path):
    """The declared schema equals the read batch's, and the bytes round-trip."""
    p = tmp_path / "shard.tar"
    _make_tar(p, _SAMPLES)
    src = WebDatasetSource(str(p))
    batches = src.read()
    assert src.schema() == batches[0].schema
    data = batches[0].to_pydict()
    assert data["__key__"] == ["a", "b"]
    assert data["jpg"] == [b"x" * 1000, b"y" * 1000]
