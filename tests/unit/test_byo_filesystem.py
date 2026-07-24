"""Bring-your-own filesystem and credentials for the plain file formats.

Before this, a user could only reach an object store through env vars or a URI query
string; a pre-built `pyarrow.fs.FileSystem`, an fsspec handle, or an fsspec-style
`storage_options` dict — the mechanism every other engine speaks — had no way in for
parquet/csv/json. These tests cover the two entry shapes and, crucially, that a
distributed read reconstructs the *same* backend on the worker rather than falling back to
that worker's environment.
"""

from __future__ import annotations

import pickle

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

import batcher as bt

pytestmark = pytest.mark.unit


@pytest.fixture
def parquet_dir(tmp_path):
    for i in range(3):
        pq.write_table(pa.table({"x": [i, i + 10]}), tmp_path / f"p{i}.parquet")
    return tmp_path


def test_bring_your_own_pyarrow_filesystem(parquet_dir):
    ds = bt.read.parquet(str(parquet_dir / "*.parquet"), filesystem=pafs.LocalFileSystem())
    assert ds.collect().num_rows == 6


def test_bring_your_own_fsspec_filesystem(parquet_dir):
    fsspec = pytest.importorskip("fsspec")
    ds = bt.read.parquet(str(parquet_dir / "*.parquet"), filesystem=fsspec.filesystem("file"))
    assert ds.collect().num_rows == 6


def test_a_non_filesystem_object_is_rejected_clearly():
    from batcher._internal.errors import IOError as BatcherIOError
    from batcher.io.filesystem import resolve_filesystem

    with pytest.raises(BatcherIOError, match=r"must be a pyarrow\.fs\.FileSystem or an fsspec"):
        resolve_filesystem("s3://b/k", filesystem=object())


def test_storage_options_reach_the_native_backend():
    """The portable credential dict maps to the native `S3FileSystem`, not just the URI."""
    from batcher.io.filesystem import resolve_filesystem

    fs = resolve_filesystem(
        "s3://bucket/k.parquet",
        storage_options={
            "endpoint_override": "http://minio:9000",
            "force_virtual_addressing": "false",
        },
    )
    assert type(fs._fs).__name__ == "S3FileSystem"
    assert fs._p("s3://bucket/k.parquet") == "bucket/k.parquet"


def test_storage_options_survive_to_a_worker_split(parquet_dir):
    """A distributed read reconstructs the source from the split via `SOURCES.get(...)(path,
    **kwargs)`. If the options are not in those kwargs, the worker re-resolves against its
    own environment and reads the wrong store — so the property is that they are carried and
    picklable, asserted here rather than trusting the single-node path."""
    from batcher.io.formats.structured.parquet.source import ParquetSource

    opts = {"endpoint_override": "http://minio:9000"}
    kwargs = ParquetSource(str(parquet_dir / "p0.parquet"), storage_options=opts)._reader_kwargs()
    assert kwargs["storage_options"] == opts
    assert pickle.loads(pickle.dumps(kwargs))["storage_options"] == opts


def test_options_force_the_whole_file_split_that_carries_them(parquet_dir):
    """The row-group fast path re-resolves from a path-keyed footer cache and cannot carry
    options, so a source given options falls back to the whole-file `FileSplit` — which
    reconstructs through `_reader_kwargs` and does carry them. Finer granularity is traded
    for correct credentials, and only on the bring-your-own case."""
    from batcher.io.formats.structured.parquet.source import ParquetSource
    from batcher.io.splits import FileSplit, RowGroupSplit

    with_opts = ParquetSource(str(parquet_dir / "*.parquet"), storage_options={"region": "x"})
    assert all(isinstance(s, FileSplit) for s in with_opts.splits())
    plain = ParquetSource(str(parquet_dir / "*.parquet"))
    assert all(isinstance(s, RowGroupSplit) for s in plain.splits())


def test_a_subclass_reader_kwargs_carries_the_base_credentials(parquet_dir):
    """Every format that overrides `_reader_kwargs` must fold in the base's, or a
    distributed CSV/Excel/… read silently drops the caller's credentials."""
    from batcher.io.formats.structured.csv import CSVSource

    csv = tmp = parquet_dir / "t.csv"
    csv.write_text("a,b\n1,2\n")
    kwargs = CSVSource(str(tmp), storage_options={"region": "eu-west-1"})._reader_kwargs()
    assert kwargs.get("storage_options") == {"region": "eu-west-1"}


def test_storage_options_values_accept_secret_references(monkeypatch):
    """A `storage_options` value may be an `env:`/`file:` reference, resolved where the
    filesystem is built (the worker, on a distributed read) — so the split carries the
    reference, never the key, matching the crypto-key and connector-credential discipline."""
    import batcher.io.filesystem as fsmod

    monkeypatch.setenv("BATCHER_TEST_S3_SECRET", "AKIAsecretvalue")
    fsmod._resolve_uri_fs.cache_clear()
    fsmod._resolve_uri_fs_opts.cache_clear()
    fs = fsmod.resolve_filesystem(
        "s3://bucket/k.parquet",
        storage_options={
            "access_key": "env:BATCHER_TEST_S3_SECRET",
            "secret_key": "env:BATCHER_TEST_S3_SECRET",
            "endpoint_override": "http://minio:9000",
        },
    )
    assert type(fs._fs).__name__ == "S3FileSystem"
