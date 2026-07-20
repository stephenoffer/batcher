"""A bring-your-own backend must never reach the native reader.

The native Parquet FFI (`bc_py::read_parquet*`) takes a **bare URI** and resolves the object
store itself, from the process environment and the URI's own query string. It therefore
cannot see a caller-supplied ``filesystem=`` or ``storage_options=``.

That matters most for the case those options exist to serve: a dict-carried
``endpoint_override`` pointing at on-prem MinIO/Ceph. Hand the bare ``s3://bucket/key`` to the
native reader and it addresses **real S3** instead — a different object, or an auth failure,
not a slower read. `_file_splits` already declines row-group splits for exactly this reason;
these tests pin the same rule across every read path, because two of them did not have it.

The tests assert on *which reader was called*, not on returned rows: with no MinIO to point
at, a wrong-store read here would raise or return nothing rather than silently return the
wrong bytes, so asserting rows would pass for the wrong reason.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher.io.formats.structured import _parquet_native
from batcher.io.formats.structured.parquet.source import ParquetSource


@pytest.fixture
def parquet_dir(tmp_path):
    """Two small Parquet files, so the multi-file native path is eligible."""
    for i in range(2):
        table = pa.table({"a": pa.array([i * 10 + 1, i * 10 + 2], pa.int64())})
        pq.write_table(table, tmp_path / f"part-{i}.parquet", row_group_size=1)
    return str(tmp_path)


@pytest.fixture
def native_spy(monkeypatch):
    """Records every native reader entry point that gets called."""
    calls: list[str] = []

    def _record(name, result=None):
        def _fn(*_args, **_kwargs):
            calls.append(name)
            return result

        return _fn

    monkeypatch.setattr(_parquet_native, "read_one", _record("read_one"))
    monkeypatch.setattr(_parquet_native, "read_many", _record("read_many"))
    monkeypatch.setattr(
        _parquet_native, "read_row_groups_filtered", _record("read_row_groups_filtered")
    )
    return calls


_MINIO = {"endpoint_override": "http://minio.internal:9000"}
BYO = [
    pytest.param({"storage_options": _MINIO}, id="storage_options"),
    pytest.param({"filesystem": pa.fs.LocalFileSystem()}, id="filesystem"),
]


@pytest.mark.parametrize("kwargs", BYO)
def test_multi_file_read_does_not_reach_the_native_reader(parquet_dir, native_spy, kwargs):
    """`_native_read_many` used to gate only on the byte cache, missing this entirely."""
    source = ParquetSource(f"{parquet_dir}/*.parquet", **kwargs)
    source.read(["a"])
    assert native_spy == [], f"native reader was handed a BYO backend: {native_spy}"


@pytest.mark.parametrize("kwargs", BYO)
def test_single_file_read_does_not_reach_the_native_reader(tmp_path, native_spy, kwargs):
    """`_read_by_path` had the same gap."""
    pq.write_table(pa.table({"a": pa.array([1, 2], pa.int64())}), tmp_path / "one.parquet")
    source = ParquetSource(str(tmp_path / "one.parquet"), **kwargs)
    source.read(["a"])
    assert native_spy == [], f"native reader was handed a BYO backend: {native_spy}"


@pytest.mark.parametrize("kwargs", BYO)
def test_filtered_read_does_not_reach_the_native_reader(parquet_dir, native_spy, kwargs):
    """The predicate path gates correctly already — pinned so it stays that way."""
    from batcher.plan.expr_ir import col

    source = ParquetSource(f"{parquet_dir}/*.parquet", **kwargs)
    source.read(["a"], predicate=(col("a") > 1).to_ir())
    assert native_spy == [], f"native reader was handed a BYO backend: {native_spy}"


@pytest.mark.parametrize("kwargs", BYO)
def test_streaming_read_does_not_reach_the_native_reader(parquet_dir, native_spy, kwargs):
    source = ParquetSource(f"{parquet_dir}/*.parquet", **kwargs)
    list(source.iter_batches(["a"]))
    assert native_spy == [], f"native reader was handed a BYO backend: {native_spy}"


def test_a_plain_source_still_uses_the_native_reader(parquet_dir, native_spy):
    """The guard must not cost the common case its fast path."""
    ParquetSource(f"{parquet_dir}/*.parquet").read(["a"])
    assert native_spy, "a plain local source should still reach the native reader"


@pytest.mark.parametrize("kwargs", BYO)
def test_byo_read_still_returns_the_right_rows(parquet_dir, kwargs):
    """Declining the fast path must stay a pure performance trade, not a behavior change."""
    plain = ParquetSource(f"{parquet_dir}/*.parquet").read(["a"])
    byo = ParquetSource(f"{parquet_dir}/*.parquet", **kwargs).read(["a"])
    to_rows = lambda bs: sorted(r["a"] for b in bs for r in b.to_pylist())  # noqa: E731
    assert to_rows(byo) == to_rows(plain) == [1, 2, 11, 12]
