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
    assert IcebergSource("ns.t", snapshot_id=7).identity() == "iceberg:default:ns.t@7"
    assert HudiSource("/tmp/h", as_of_instant="20240101").identity() == "hudi:/tmp/h@20240101"
    assert DeltaSharingSource("p#s.sch.t").identity() == "delta_sharing:p#s.sch.t"


def test_an_iceberg_identity_distinguishes_catalog_and_row_filter(tmp_path) -> None:
    """The identity keys the statistics cache, so it must name everything that changes the rows.

    Two things were missing, and both are real collisions: ``db.t`` in one catalog is a
    different table from ``db.t`` in another, and a *filtered* read returns fewer rows than
    an unfiltered one. Sharing a cache entry between the last two is how a filtered `count()`
    came back with the whole table's total.
    """
    plain = IcebergSource("db.t").identity()
    other_catalog = IcebergSource(
        "db.t", catalog={"type": "sql", "uri": f"sqlite:///{tmp_path}/other.db"}
    ).identity()
    filtered = IcebergSource("db.t", row_filter="id > 5").identity()

    assert plain != other_catalog
    assert plain != filtered
    assert other_catalog != filtered


def test_an_iceberg_split_reads_through_the_table_not_around_it(tmp_path) -> None:
    """A split must read its file the way *Iceberg* reads it — which means via the catalog.

    This test used to assert the opposite: that the split reads its Parquet directly, "no
    catalog/pyiceberg", as a virtue. That independence was the bug. Iceberg resolves columns
    by field id and records its deletes in separate files, so a reader that goes around the
    table sees neither — a renamed column broke the read outright, and a merge-on-read table
    silently returned rows that had been deleted.

    So the split now carries the planned `FileScanTask` and reads it with Iceberg's own
    scanner. Needing the catalog is the point, not a cost. (The behaviors this buys are
    covered in `test_lakehouse_iceberg.py`.)
    """
    pytest.importorskip("pyiceberg")
    from pyiceberg.catalog.sql import SqlCatalog

    from batcher.io.formats.lakehouse import IcebergSource
    from batcher.io.formats.lakehouse.iceberg import IcebergTableSplit

    warehouse = tmp_path / "wh"
    warehouse.mkdir()
    spec = {
        "type": "sql",
        "uri": f"sqlite:///{warehouse}/catalog.db",
        "warehouse": f"file://{warehouse}",
    }
    catalog = SqlCatalog("default", uri=spec["uri"], warehouse=spec["warehouse"])
    catalog.create_namespace("db")
    schema = pa.schema([pa.field("id", pa.int64()), pa.field("v", pa.float64())])
    catalog.create_table("db.t", schema=schema).append(
        pa.table({"id": [1, 2, 3], "v": [1.5, 2.5, 3.5]}, schema=schema)
    )

    splits = IcebergSource("db.t", catalog=spec).splits()
    assert len(splits) == 1
    split = splits[0]
    assert isinstance(split, IcebergTableSplit)
    assert split.row_count() == 3  # from the manifest, without opening a footer

    got = pa.Table.from_batches(split.read(projection=["v"]))
    assert got.column_names == ["v"]
    assert got.column("v").to_pylist() == [1.5, 2.5, 3.5]


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
    _commit_staged(sink_v0, v0, path)

    src = DeltaSource(path)
    assert src.row_count() == 3
    assert _sorted_rows(pa.Table.from_batches(src.read())) == _sorted_rows(v0)

    v1 = pa.table({"id": [4, 5], "v": ["d", "e"]})
    sink_v1 = DeltaSink(mode="append")
    _commit_staged(sink_v1, v1, path)

    assert DeltaSource(path).row_count() == 5
    # Time travel to the first version sees only the original rows.
    assert DeltaSource(path, version=0).row_count() == 3
    assert _sorted_rows(pa.Table.from_batches(DeltaSource(path, version=0).read())) == _sorted_rows(
        v0
    )


def test_delta_deletion_vectors_read(tmp_path) -> None:
    """A table with deletion vectors *enabled* must not be penalized for having none.

    delta-rs implements `delete()` as copy-on-write, so this table ends up with the
    `deletionVectors` reader feature in its protocol and **zero actual vectors**. The
    connector used to key off that flag, and so condemned every such table — which, since
    the feature is default-on for new Delta tables, is most of them — to a path with no
    splits, no file skipping, and no row count, in exchange for nothing. Worse, a split read
    of one raised `DeltaProtocolError` outright, so it could not be read distributed at all.

    The gate is now the *presence* of vectors, which the log states exactly.
    """
    pytest.importorskip("deltalake")
    from deltalake import DeltaTable, write_deltalake

    path = str(tmp_path / "delta_dv")
    write_deltalake(
        path,
        pa.table({"id": list(range(10)), "v": [i * 10 for i in range(10)]}),
        configuration={"delta.enableDeletionVectors": "true"},
    )
    DeltaTable(path).delete("id < 4")  # logically delete ids 0..3

    src = DeltaSource(path)
    assert not src._has_deletion_vectors(), "the flag is set but no file carries a vector"

    # the deleted rows are gone, and projection still works
    out = pa.Table.from_batches(src.read())
    assert sorted(out.column("id").to_pylist()) == [4, 5, 6, 7, 8, 9]
    proj = pa.Table.from_batches(src.read(projection=["id"]))
    assert proj.schema.names == ["id"]
    assert sorted(proj.column("id").to_pylist()) == [4, 5, 6, 7, 8, 9]
    assert pa.Table.from_batches(list(src.iter_batches())).num_rows == 6

    # ...and the metadata is now exact, where it used to refuse to answer
    assert src.row_count() == 6
    stats = src.statistics()
    assert stats is not None
    assert stats.row_count == 6

    # ...and the table can be read distributed, which used to raise DeltaProtocolError
    splits = src.splits()
    assert splits
    rows = sorted(i for s in splits for b in s.read() for i in b.column("id").to_pylist())
    assert rows == [4, 5, 6, 7, 8, 9]


def test_a_deletion_vector_is_applied_before_the_predicate(tmp_path) -> None:
    """The load-bearing ordering invariant, tested directly.

    A deletion vector is indexed by *physical row position*, so it only means anything
    against the file's rows exactly as written. If a predicate were pushed into the Parquet
    read, it would drop rows first and slide every position — the mask would then delete the
    wrong rows and report success. So a vectored file is read unfiltered, masked, and only
    then filtered.

    delta-rs cannot write a real deletion vector (its `delete()` rewrites the file), so the
    mask is supplied directly here. That is the honest way to test the invariant: it is a
    property of `read_fragment`, not of how the vector got there.
    """
    import pyarrow.dataset as pds
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq

    from batcher.io.formats.lakehouse.delta.source import read_fragment

    data = pa.table({"id": list(range(10)), "day": [i % 2 for i in range(10)]})
    file = tmp_path / "part.parquet"
    pq.write_table(data, file)
    fragment = pds.ParquetFileFormat().make_fragment(str(file), filesystem=pafs.LocalFileSystem())

    # the vector deletes ids 0..3; True = keep
    mask = pa.array([i >= 4 for i in range(10)], type=pa.bool_())
    predicate = {
        "e": "binary",
        "op": "eq",
        "left": {"e": "col", "name": "day"},
        "right": {"e": "lit", "value": {"int": 0}},
    }

    got = read_fragment(fragment, data.schema, None, predicate, mask)
    # day == 0 is ids 0,2,4,6,8; the vector removes 0 and 2 -> 4, 6, 8
    assert got.column("id").to_pylist() == [4, 6, 8]

    # and with no vector, the same file+predicate keeps every matching row
    unmasked = read_fragment(fragment, data.schema, None, predicate, None)
    assert unmasked.column("id").to_pylist() == [0, 2, 4, 6, 8]


def test_a_misaligned_deletion_vector_is_refused(tmp_path) -> None:
    """A vector that does not describe the file's rows must fail, not silently mis-delete."""
    import pyarrow.dataset as pds
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq

    from batcher.io.formats.lakehouse.delta.source import read_fragment

    data = pa.table({"id": list(range(10))})
    file = tmp_path / "part.parquet"
    pq.write_table(data, file)
    fragment = pds.ParquetFileFormat().make_fragment(str(file), filesystem=pafs.LocalFileSystem())

    with pytest.raises(BackendError, match="disagree"):
        read_fragment(fragment, data.schema, None, None, pa.array([True] * 3))


def test_delta_projection_pushdown(tmp_path) -> None:
    pytest.importorskip("deltalake")
    path = str(tmp_path / "delta_proj")
    table = pa.table({"id": [1, 2], "v": ["a", "b"], "w": [10, 20]})
    sink = DeltaSink(mode="append")
    _commit_staged(sink, table, path)

    out = pa.Table.from_batches(DeltaSource(path).read(projection=["id"]))
    assert out.column_names == ["id"]
    assert out.num_rows == 2


def test_delta_multi_shard_commit_loses_no_data(tmp_path) -> None:
    """Two shards (as distributed workers produce) must both reach the committed table.

    The regression guard for the data-loss bug: each shard's `write_partitioned` returns
    file locators, the driver merges them into one manifest, and `commit` writes every
    staged shard. The old design buffered shards in per-sink memory, so a worker's data
    never reached the driver's committing sink and the distributed write wrote nothing.
    """
    from batcher.io.manifest import WriteManifest

    path = str(tmp_path / "delta_shards")
    shard_a = pa.table({"id": [1, 2, 3], "v": ["a", "b", "c"]})
    shard_b = pa.table({"id": [4, 5], "v": ["d", "e"]})

    # A distributed write constructs an independent sink per shard (separate processes).
    files = []
    files += DeltaSink(mode="append").write_partitioned(shard_a, path, file_index=0)
    files += DeltaSink(mode="append").write_partitioned(shard_b, path, file_index=1)
    # The driver's sink (yet another instance) commits the merged manifest.
    DeltaSink(mode="append").commit(WriteManifest(tuple(files)), path)

    assert DeltaSource(path).row_count() == 5
    got = _sorted_rows(pa.Table.from_batches(DeltaSource(path).read()))
    assert got == _sorted_rows(pa.concat_tables([shard_a, shard_b]))


def test_delta_partitioned_multi_shard_commit(tmp_path) -> None:
    """A partitioned multi-shard commit lays out Hive dirs and keeps every row."""
    from batcher.io.manifest import WriteManifest

    path = str(tmp_path / "delta_part")
    shard_a = pa.table({"id": [1, 2], "p": ["x", "y"]})
    shard_b = pa.table({"id": [3, 4], "p": ["x", "y"]})
    files = []
    files += DeltaSink(mode="append", partition_by=["p"]).write_partitioned(
        shard_a, path, partition_by=["p"], file_index=0
    )
    files += DeltaSink(mode="append", partition_by=["p"]).write_partitioned(
        shard_b, path, partition_by=["p"], file_index=1
    )
    DeltaSink(mode="append", partition_by=["p"]).commit(WriteManifest(tuple(files)), path)

    import os

    assert DeltaSource(path).row_count() == 4
    assert sorted(d for d in os.listdir(path) if d.startswith("p=")) == ["p=x", "p=y"]


def test_iceberg_multi_shard_add_files_commit(tmp_path) -> None:
    """Two shards register into one Iceberg snapshot via `add_files` (no data lost)."""
    pytest.importorskip("pyiceberg")
    from batcher.io.catalog import resolve_catalog
    from batcher.io.formats.lakehouse.iceberg import IcebergSink, IcebergSource
    from batcher.io.manifest import WriteManifest

    wh = str(tmp_path / "wh")
    import os

    os.makedirs(wh)
    cat = {"type": "sql", "uri": f"sqlite:///{tmp_path}/c.db", "warehouse": f"file://{wh}"}
    resolve_catalog(cat).create_namespace("ns")

    shard_a = pa.table({"id": [1, 2, 3]})
    shard_b = pa.table({"id": [4, 5]})
    token = "wtok1"
    files = []
    files += IcebergSink("ns.t", catalog=cat, write_token=token).write_partitioned(
        shard_a, "ns.t", file_index=0
    )
    files += IcebergSink("ns.t", catalog=cat, write_token=token).write_partitioned(
        shard_b, "ns.t", file_index=1
    )
    IcebergSink("ns.t", catalog=cat, write_token=token).commit(WriteManifest(tuple(files)), "ns.t")

    got = pa.Table.from_batches(IcebergSource("ns.t", catalog=cat).read())
    assert sorted(got.column("id").to_pylist()) == [1, 2, 3, 4, 5]


def _commit_staged(sink, table, path) -> None:
    """Drive a lakehouse sink the way the engine does: stage the shard, then commit the
    manifest the stage returned (the corrected contract — commit reads the staged files,
    not hidden in-sink state, so the same call works across the distributed boundary).

    The schema is attached here for the same reason the driver attaches it in
    `api/terminal/core.py::_commit`: a transactional sink creating a table cannot recover
    the output type from the data files alone, because a partitioned write stores its
    partition columns in the path rather than in the file.
    """
    from batcher.io.manifest import WriteManifest

    files = sink.write_partitioned(table, path)
    sink.commit(WriteManifest(tuple(files), schema=table.schema), path)
