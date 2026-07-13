"""Iceberg gets the same metadata the engine already knows how to use.

Iceberg records, per data file, exactly the zone map Delta does — record count, per-column
lower/upper bounds, null counts. The connector declined to surface it, on the grounds that a
data file's bounds are *"field-id keyed, byte-encoded"* and fragile to decode. That is true
of the raw `lower_bounds`/`upper_bounds` — and it is not what has to be read.
`inspect.data_files()` also exposes `readable_metrics`, keyed by **column name** with values
already decoded to their Arrow type.

The consequence of declining was not marginal. Without column bounds an Iceberg scan has no
zone map at all: Kyber cannot prune a predicate, cannot prove one empty, and cannot answer a
`min()`/`max()` without reading the table — while a Delta table of the same shape does all
three from metadata.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytest.importorskip("pyiceberg", reason="pyiceberg not installed")

from batcher.io.formats.lakehouse import IcebergSource
from batcher.io.formats.lakehouse.iceberg._manifest import file_manifest
from batcher.io.stats.file_skipping import surviving_files

pytestmark = pytest.mark.integration


@pytest.fixture
def clustered(tmp_path) -> tuple[str, dict]:
    """Four data files, each holding a distinct 25-wide ``id`` range."""
    from pyiceberg.catalog.sql import SqlCatalog

    warehouse = tmp_path / "wh"
    warehouse.mkdir()
    spec = {
        "type": "sql",
        "uri": f"sqlite:///{warehouse}/catalog.db",
        "warehouse": f"file://{warehouse}",
    }
    catalog = SqlCatalog("default", uri=spec["uri"], warehouse=spec["warehouse"])
    catalog.create_namespace("db")
    schema = pa.schema([pa.field("id", pa.int64()), pa.field("cat", pa.string())])
    table = catalog.create_table("db.t", schema=schema)
    for k in range(4):
        table.append(
            pa.table(
                {
                    "id": list(range(k * 25, (k + 1) * 25)),
                    "cat": ["a" if k % 2 == 0 else "b"] * 25,
                },
                schema=schema,
            )
        )
    return "db.t", spec


def _predicate(op: str, column: str, value: int) -> dict:
    return {
        "e": "binary",
        "op": op,
        "left": {"e": "col", "name": column},
        "right": {"e": "lit", "value": {"int": value}},
    }


def test_the_manifest_normalizes_into_the_add_action_layout(clustered) -> None:
    """One layout, one aggregator, one pruner — for both formats."""
    from pyiceberg.catalog.sql import SqlCatalog

    identifier, spec = clustered
    catalog = SqlCatalog("default", uri=spec["uri"], warehouse=spec["warehouse"])

    manifest = file_manifest(catalog.load_table(identifier))

    assert manifest is not None
    assert manifest.num_rows == 4
    assert {"path", "num_records", "min.id", "max.id", "null_count.id"} <= set(
        manifest.column_names
    )


def test_statistics_carry_column_bounds(clustered) -> None:
    """They used to carry only a row count, so Iceberg had no zone map at all."""
    identifier, spec = clustered

    stats = IcebergSource(identifier, catalog=spec).statistics()

    assert stats is not None
    assert stats.row_count == 100
    assert stats.exact_rows is True
    assert stats.columns["id"].min == 0
    assert stats.columns["id"].max == 99


def test_file_skipping_prunes_an_iceberg_table(clustered) -> None:
    """The same `file_skipping` module that prunes Delta, now fed by Iceberg's metrics."""
    from pyiceberg.catalog.sql import SqlCatalog

    identifier, spec = clustered
    catalog = SqlCatalog("default", uri=spec["uri"], warehouse=spec["warehouse"])
    manifest = file_manifest(catalog.load_table(identifier))

    assert len(surviving_files(_predicate("gt", "id", 80), manifest)) == 1
    assert surviving_files(_predicate("gt", "id", 500), manifest) == []
    assert len(surviving_files(_predicate("ge", "id", 0), manifest)) == 4


def test_a_predicate_outside_the_bounds_is_answered_without_a_scan(clustered) -> None:
    """The payoff: Kyber can now prove an Iceberg predicate empty from metadata."""
    identifier, spec = clustered
    ds = bt.read.iceberg(identifier, catalog=spec)

    assert ds.filter(bt.col("id") > 500).count() == 0
    assert ds.filter(bt.col("id") > 80).count() == 19  # and a real one still reads
    assert ds.count() == 100


def test_statistics_are_withheld_when_they_would_not_describe_the_source(clustered) -> None:
    """A `row_filter` means the manifest overstates what the source returns."""
    identifier, spec = clustered

    filtered = IcebergSource(identifier, catalog=spec, row_filter="id >= 90")
    assert filtered.statistics() is None
    assert filtered.row_count() is None


# --- partitioned writes ----------------------------------------------------


@pytest.fixture
def partitioned(tmp_path) -> tuple[str, dict]:
    """A partitioned Iceberg table (identity on ``cat``), typed the way pyiceberg types it."""
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.schema import Schema
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.types import LongType, NestedField, StringType

    warehouse = tmp_path / "wh"
    warehouse.mkdir()
    spec = {
        "type": "sql",
        "uri": f"sqlite:///{warehouse}/catalog.db",
        "warehouse": f"file://{warehouse}",
    }
    catalog = SqlCatalog("default", uri=spec["uri"], warehouse=spec["warehouse"])
    catalog.create_namespace("db")
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "cat", StringType(), required=False),
    )
    partition = PartitionSpec(
        PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="cat")
    )
    catalog.create_table("db.p", schema=schema, partition_spec=partition)
    catalog.create_table("db.u", schema=schema)
    return "db.p", spec


def test_a_partitioned_table_can_be_written_at_all(partitioned) -> None:
    """It could not be. The write failed outright, so a whole class of table was unusable.

    An Iceberg table's partitioning lives in the catalog's spec, so it never appears as a
    `partition_by` argument — and the writer keyed off that argument to decide whether to lay
    the output out. A shard was therefore written as one flat file spanning every partition
    value, and the commit rejected it: *"Cannot infer partition value ... more than one
    partition values for Partition Field: cat"*.
    """
    from pyiceberg.catalog.sql import SqlCatalog

    identifier, spec = partitioned

    bt.from_pydict({"id": [1, 2, 3, 4], "cat": ["a", "b", "a", "b"]}).write.iceberg(
        identifier, mode="append", catalog=spec
    )

    data = bt.read.iceberg(identifier, catalog=spec).collect().to_pydict()
    assert sorted(zip(data["id"], data["cat"], strict=True)) == [
        (1, "a"),
        (2, "b"),
        (3, "a"),
        (4, "b"),
    ]

    catalog = SqlCatalog("default", uri=spec["uri"], warehouse=spec["warehouse"])
    table = catalog.load_table(identifier)
    files = table.inspect.data_files()
    assert files.num_rows == 2, "one data file per partition"
    assert len(table.snapshots()) == 1, "still one atomic commit"


def test_an_unpartitioned_table_is_unaffected(partitioned) -> None:
    """The flat-file path is still the right one when the table owns no partitioning."""
    _, spec = partitioned

    bt.from_pydict({"id": [9], "cat": ["z"]}).write.iceberg("db.u", mode="append", catalog=spec)

    assert bt.read.iceberg("db.u", catalog=spec).collect().to_pydict() == {
        "id": [9],
        "cat": ["z"],
    }


def test_a_string_filter_does_not_crash_the_engine(partitioned) -> None:
    """It used to. `Invalid comparison operation: LargeUtf8 == Utf8`, from the Rust engine.

    pyiceberg maps Iceberg's `StringType` to Arrow `large_string`, and the engine's kernels
    compare against a plain `string` literal. Every Iceberg table written by Spark or Flink
    carries that mapping, so a string filter on an ordinary Iceberg table simply crashed.
    """
    identifier, spec = partitioned
    bt.from_pydict({"id": [1, 2, 3, 4], "cat": ["a", "b", "a", "b"]}).write.iceberg(
        identifier, mode="append", catalog=spec
    )

    ds = bt.read.iceberg(identifier, catalog=spec)
    assert ds.schema.field("cat").type == pa.string(), "normalized for the engine"

    got = ds.filter(bt.col("cat") == "a").collect().to_pydict()
    assert sorted(got["id"]) == [1, 3]


# --- reading a task the way Iceberg means it to be read --------------------


def _catalog(tmp_path):
    from pyiceberg.catalog.sql import SqlCatalog

    warehouse = tmp_path / "wh"
    warehouse.mkdir(exist_ok=True)
    spec = {
        "type": "sql",
        "uri": f"sqlite:///{warehouse}/catalog.db",
        "warehouse": f"file://{warehouse}",
    }
    catalog = SqlCatalog("default", uri=spec["uri"], warehouse=spec["warehouse"])
    catalog.create_namespace("db")
    return catalog, spec, str(warehouse)


def test_a_split_read_survives_a_renamed_column(tmp_path) -> None:
    """Iceberg resolves columns by field id; reading the raw Parquet resolves by name.

    So on a table whose column was renamed, the pre-rename data file still carries the old
    name. The split read either failed to find the column or returned a differently-named
    schema its siblings could not be concatenated with::

        Schema at index 1 was different: id, value  vs  id, v

    Schema evolution is routine in Iceberg, and a *distributed* read of an evolved table was
    therefore broken outright — while the single-node scan, which goes through pyiceberg, was
    fine. Reading each task with Iceberg's own scanner closes that gap.
    """
    catalog, spec, _ = _catalog(tmp_path)
    schema = pa.schema([pa.field("id", pa.int64()), pa.field("v", pa.int64())])
    table = catalog.create_table("db.t", schema=schema)
    table.append(pa.table({"id": [1, 2], "v": [10, 20]}, schema=schema))

    with table.update_schema() as update:
        update.rename_column("v", "value")
    table = catalog.load_table("db.t")
    table.append(pa.table({"id": [3], "value": [30]}, schema=table.schema().as_arrow()))

    source = IcebergSource("db.t", catalog=spec)
    splits = source.splits()
    assert len(splits) == 2  # one file written before the rename, one after

    rows = pa.Table.from_batches([b for s in splits for b in s.read()])
    assert rows.schema.names == ["id", "value"]
    assert sorted(
        zip(rows.column("id").to_pylist(), rows.column("value").to_pylist(), strict=True)
    ) == [
        (1, 10),
        (2, 20),
        (3, 30),
    ]


def test_a_split_read_backfills_an_added_column(tmp_path) -> None:
    """A file written before the column existed must read it back as NULL, not fail."""
    from pyiceberg.types import DoubleType

    catalog, spec, _ = _catalog(tmp_path)
    schema = pa.schema([pa.field("id", pa.int64())])
    table = catalog.create_table("db.t", schema=schema)
    table.append(pa.table({"id": [1]}, schema=schema))

    with table.update_schema() as update:
        update.add_column("extra", DoubleType())
    table = catalog.load_table("db.t")
    table.append(pa.table({"id": [2], "extra": [1.5]}, schema=table.schema().as_arrow()))

    splits = IcebergSource("db.t", catalog=spec).splits()
    rows = pa.Table.from_batches([b for s in splits for b in s.read()])

    assert rows.schema.names == ["id", "extra"]
    got = dict(zip(rows.column("id").to_pylist(), rows.column("extra").to_pylist(), strict=True))
    assert got == {1: None, 2: 1.5}


def test_a_merge_on_read_table_splits_and_applies_its_deletes(tmp_path) -> None:
    """Merge-on-read tables used to lose all parallelism, to avoid resurrecting deleted rows.

    A task carries positional delete files; reading its data file directly returns every row
    the deletes removed. `splits()` therefore refused to split such a table at all — correct,
    and it cost those tables every worker. Reading each task through Iceberg's own scanner
    applies the deletes, so they split like any other table.

    pyiceberg cannot *write* a positional delete (`table.delete()` is copy-on-write and warns
    "Merge on read is not yet supported"), so the delete manifest is constructed here. That
    reaches into pyiceberg internals, so a version bump skips this rather than failing it —
    the behavior under test is ours, not theirs.
    """
    mor = pytest.importorskip("pyiceberg.manifest", reason="pyiceberg internals moved")
    assert mor

    catalog, spec, warehouse = _catalog(tmp_path)
    try:
        table = _build_merge_on_read(catalog, warehouse)
    except Exception as exc:  # pragma: no cover - a pyiceberg internals change
        pytest.skip(f"cannot construct a merge-on-read table on this pyiceberg: {exc}")

    oracle = sorted(table.scan().to_arrow().column("id").to_pylist())
    assert oracle == [0, 2, 4, 5], "positions 1 and 3 are deleted"

    source = IcebergSource("db.mor", catalog=spec)
    splits = source.splits()
    assert splits and type(splits[0]).__name__ == "IcebergTableSplit", "no longer whole-source"

    rows = pa.Table.from_batches([b for s in splits for b in s.read()])
    assert sorted(rows.column("id").to_pylist()) == oracle, "the deletes must be applied"


def _build_merge_on_read(catalog, warehouse: str):
    """A real merge-on-read table: 6 rows, with positions 1 and 3 positionally deleted."""
    import os
    import uuid

    import pyarrow.parquet as pq
    from pyiceberg.manifest import (
        DataFile,
        DataFileContent,
        FileFormat,
        ManifestContent,
        ManifestEntry,
        ManifestEntryStatus,
        ManifestWriterV2,
        write_manifest_list,
    )
    from pyiceberg.table.snapshots import Operation, Snapshot, Summary
    from pyiceberg.table.update import AddSnapshotUpdate, AssertTableUUID, SetSnapshotRefUpdate
    from pyiceberg.typedef import Record

    rows = pa.table({"id": pa.array(range(6), pa.int64())})
    table = catalog.create_table("db.mor", schema=rows.schema)
    table.append(rows)
    table = catalog.load_table("db.mor")
    target = next(iter(table.scan().plan_files())).file.file_path

    delete_positions = (1, 3)
    delete_schema = pa.schema(
        [
            pa.field(
                "file_path",
                pa.string(),
                nullable=False,
                metadata={b"PARQUET:field_id": b"2147483546"},
            ),
            pa.field(
                "pos", pa.int64(), nullable=False, metadata={b"PARQUET:field_id": b"2147483545"}
            ),
        ]
    )
    delete_path = os.path.join(
        warehouse, "db.db", "mor", "data", f"deletes-{uuid.uuid4().hex}.parquet"
    )
    os.makedirs(os.path.dirname(delete_path), exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "file_path": [target] * len(delete_positions),
                "pos": list(delete_positions),
            },
            schema=delete_schema,
        ),
        delete_path,
    )

    delete_file = DataFile.from_args(
        content=DataFileContent.POSITION_DELETES,
        file_path=f"file://{delete_path}",
        file_format=FileFormat.PARQUET,
        partition=Record(),
        record_count=len(delete_positions),
        file_size_in_bytes=os.path.getsize(delete_path),
        sort_order_id=None,
        spec_id=0,
        equality_ids=None,
        key_metadata=None,
    )

    snapshot_id = int(uuid.uuid4().int >> 96)
    parent = table.current_snapshot()
    sequence = parent.sequence_number + 1
    io = table.io

    class _DeleteManifestWriter(ManifestWriterV2):
        def content(self):
            return ManifestContent.DELETES

        @property
        def _meta(self):
            return {**super()._meta, "content": "deletes"}

    manifest_path = os.path.join(
        warehouse, "db.db", "mor", "metadata", f"del-{uuid.uuid4().hex}.avro"
    )
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with _DeleteManifestWriter(
        spec=table.spec(),
        schema=table.schema(),
        output_file=io.new_output(f"file://{manifest_path}"),
        snapshot_id=snapshot_id,
        avro_compression="null",
    ) as writer:
        writer.add_entry(
            ManifestEntry.from_args(
                status=ManifestEntryStatus.ADDED,
                snapshot_id=snapshot_id,
                sequence_number=sequence,
                file_sequence_number=sequence,
                data_file=delete_file,
            )
        )
    manifest = writer.to_manifest_file()
    fields = (
        "manifest_path",
        "manifest_length",
        "partition_spec_id",
        "added_snapshot_id",
        "added_files_count",
        "existing_files_count",
        "deleted_files_count",
        "added_rows_count",
        "existing_rows_count",
        "deleted_rows_count",
        "partitions",
    )
    manifest = manifest.__class__.from_args(
        **{f: getattr(manifest, f) for f in fields},
        content=ManifestContent.DELETES,
        sequence_number=sequence,
        min_sequence_number=sequence,
        _table_format_version=2,
    )

    list_path = os.path.join(warehouse, "db.db", "mor", "metadata", f"snap-{snapshot_id}.avro")
    with write_manifest_list(
        format_version=2,
        output_file=io.new_output(f"file://{list_path}"),
        snapshot_id=snapshot_id,
        parent_snapshot_id=parent.snapshot_id,
        sequence_number=sequence,
        avro_compression="null",
    ) as lister:
        lister.add_manifests([*parent.manifests(io), manifest])

    snapshot = Snapshot(
        snapshot_id=snapshot_id,
        parent_snapshot_id=parent.snapshot_id,
        sequence_number=sequence,
        timestamp_ms=parent.timestamp_ms + 1000,
        manifest_list=f"file://{list_path}",
        summary=Summary(Operation.OVERWRITE),
        schema_id=table.schema().schema_id,
    )
    catalog.commit_table(
        table,
        (AssertTableUUID(uuid=table.metadata.table_uuid),),
        (
            AddSnapshotUpdate(snapshot=snapshot),
            SetSnapshotRefUpdate(
                ref_name="main",
                type="branch",
                snapshot_id=snapshot_id,
                max_ref_age_ms=None,
                max_snapshot_age_ms=None,
                min_snapshots_to_keep=None,
            ),
        ),
    )
    return catalog.load_table("db.mor")
