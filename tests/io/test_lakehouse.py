"""Lakehouse connector coverage — registration, import safety, Delta round-trip.

The connectors lazily import their optional backends, so registration and basic
construction work without the extras installed; tests that need a real backend
(`deltalake`) are gated with `pytest.importorskip` and skip cleanly otherwise.

Runs without the native engine — these exercise the Python IO layer only.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher._internal.errors import BackendError
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.lakehouse import (
    DeltaSharingSource,
    DeltaSink,
    DeltaSource,
    HudiSink,
    HudiSource,
    IcebergSink,
    IcebergSource,
)


def _sorted_rows(table: pa.Table) -> list[dict]:
    return sorted(table.to_pylist(), key=lambda r: tuple(str(r[c]) for c in table.column_names))


def test_connectors_registered() -> None:
    for name in ("delta", "iceberg", "hudi", "delta_sharing"):
        assert name in SOURCES
    for name in ("delta", "iceberg", "hudi"):
        assert name in SINKS
    assert SOURCES.get("delta") is DeltaSource
    assert SINKS.get("delta") is DeltaSink
    assert SOURCES.get("iceberg") is IcebergSource
    assert SINKS.get("iceberg") is IcebergSink
    assert SOURCES.get("hudi") is HudiSource
    assert SOURCES.get("delta_sharing") is DeltaSharingSource


def test_identity_does_not_require_backend() -> None:
    assert DeltaSource("/tmp/t", version=3).identity() == "delta:/tmp/t@3"
    assert DeltaSource("/tmp/t").identity() == "delta:/tmp/t@latest"
    assert IcebergSource("ns.t", snapshot_id=7).identity() == "iceberg:ns.t@7"
    assert HudiSource("/tmp/h", as_of_instant="20240101").identity() == "hudi:/tmp/h@20240101"
    assert DeltaSharingSource("p#s.sch.t").identity() == "delta_sharing:p#s.sch.t"


def test_delta_source_rejects_version_and_timestamp() -> None:
    with pytest.raises(BackendError):
        DeltaSource("/tmp/t", version=1, timestamp="2024-01-01")


def test_delta_sink_rejects_bad_mode() -> None:
    with pytest.raises(BackendError):
        DeltaSink(mode="upsert")


def test_iceberg_sink_rejects_bad_mode() -> None:
    with pytest.raises(BackendError):
        IcebergSink("ns.t", mode="merge")


def test_hudi_sink_always_raises() -> None:
    with pytest.raises(BackendError, match="Spark/Flink"):
        HudiSink("/tmp/h")


def test_delta_sharing_url_validation() -> None:
    src = DeltaSharingSource("no-hash-here")
    with pytest.raises(BackendError):
        # Resolving files parses the url; an invalid ref must raise a typed error.
        src.schema()


def test_missing_backend_raises_actionable_error(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _block(name, *args, **kwargs):
        if name == "deltalake" or name.startswith("deltalake."):
            raise ImportError("no deltalake")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block)
    with pytest.raises(BackendError, match=r"\[delta\]"):
        DeltaSource("/tmp/t").schema()


def test_delta_roundtrip_and_time_travel(tmp_path) -> None:
    pytest.importorskip("deltalake")
    path = str(tmp_path / "delta_tbl")

    v0 = pa.table({"id": [1, 2, 3], "v": ["a", "b", "c"]})
    sink_v0 = DeltaSink(mode="append")
    sink_v0.write_partitioned(v0, path)
    sink_v0.commit(_manifest(), path)

    src = DeltaSource(path)
    assert src.row_count() == 3
    assert _sorted_rows(pa.Table.from_batches(src.read())) == _sorted_rows(v0)

    v1 = pa.table({"id": [4, 5], "v": ["d", "e"]})
    sink_v1 = DeltaSink(mode="append")
    sink_v1.write_partitioned(v1, path)
    sink_v1.commit(_manifest(), path)

    assert DeltaSource(path).row_count() == 5
    # Time travel to the first version sees only the original rows.
    assert DeltaSource(path, version=0).row_count() == 3
    assert _sorted_rows(pa.Table.from_batches(DeltaSource(path, version=0).read())) == _sorted_rows(
        v0
    )


def test_delta_deletion_vectors_read(tmp_path) -> None:
    # A merge-on-read delete (deletion vectors) must be applied on read: the deleted
    # rows are absent, projection still works, and we don't claim a (now-overcounting)
    # exact row count. delta-rs's pyarrow path raises on DV tables; the connector
    # routes them through the DataFusion QueryBuilder, which masks the deletes.
    pytest.importorskip("deltalake")
    from deltalake import DeltaTable, write_deltalake

    path = str(tmp_path / "delta_dv")
    write_deltalake(
        path,
        pa.table({"id": list(range(10)), "v": [i * 10 for i in range(10)]}),
        configuration={"delta.enableDeletionVectors": "true"},
    )
    DeltaTable(path).delete("id < 4")  # logically delete ids 0..3, no file rewrite

    src = DeltaSource(path)
    assert src._has_deletion_vectors()
    out = pa.Table.from_batches(src.read())
    assert sorted(out.column("id").to_pylist()) == [4, 5, 6, 7, 8, 9]
    proj = pa.Table.from_batches(src.read(projection=["id"]))
    assert proj.schema.names == ["id"]
    assert sorted(proj.column("id").to_pylist()) == [4, 5, 6, 7, 8, 9]
    streamed = pa.Table.from_batches(list(src.iter_batches()))
    assert streamed.num_rows == 6
    # The add-action num_records counts pre-delete rows, so don't answer count/stats.
    assert src.row_count() is None
    assert src.statistics() is None


def test_delta_projection_pushdown(tmp_path) -> None:
    pytest.importorskip("deltalake")
    path = str(tmp_path / "delta_proj")
    table = pa.table({"id": [1, 2], "v": ["a", "b"], "w": [10, 20]})
    sink = DeltaSink(mode="append")
    sink.write_partitioned(table, path)
    sink.commit(_manifest(), path)

    out = pa.Table.from_batches(DeltaSource(path).read(projection=["id"]))
    assert out.column_names == ["id"]
    assert out.num_rows == 2


def _manifest():
    from batcher.io.manifest import WriteManifest

    return WriteManifest()
