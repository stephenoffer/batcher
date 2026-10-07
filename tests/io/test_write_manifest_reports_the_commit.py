"""A write's manifest names where it landed and the version it committed (AP-438).

`WriteManifest` carried per-file stats but neither the destination nor the table version a
transactional commit created, so a caller wanting "which Delta version did I just write"
had to re-read the log, which a concurrent writer can race.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.io.manifest import WriteManifest, WrittenFile

pytestmark = pytest.mark.integration


def test_merge_keeps_the_first_set_destination_and_version():
    a = WriteManifest((WrittenFile("a", 1, 1),))
    b = WriteManifest((WrittenFile("b", 1, 1),), destination="t", version=3)
    assert a.merge(b).destination == "t" and a.merge(b).version == 3
    assert b.merge(a).destination == "t" and b.merge(a).version == 3
    assert a.merge(a).version is None


def test_delta_manifest_reports_the_version_each_commit_created(tmp_path):
    deltalake = pytest.importorskip("deltalake")
    path = str(tmp_path / "t")
    ds = bt.from_pydict({"id": [1, 2]})
    first = ds.write.delta(path)
    second = ds.write.delta(path, mode="append")
    assert (first.version, second.version) == (0, 1)
    assert second.version == deltalake.DeltaTable(path).version()
    assert second.destination == path


def test_delta_merge_reports_its_version(tmp_path):
    deltalake = pytest.importorskip("deltalake")
    path = str(tmp_path / "t")
    bt.from_pydict({"id": [1, 2], "v": [1, 1]}).write.delta(path)
    merged = bt.from_pydict({"id": [2, 3], "v": [9, 9]}).write.delta(path, merge_on="id")
    assert merged.version == deltalake.DeltaTable(path).version() == 1


def test_iceberg_manifest_reports_the_snapshot_it_committed(tmp_path):
    pytest.importorskip("pyiceberg")
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
    catalog.create_table("db.t", schema=pa.schema([pa.field("id", pa.int64())]))
    manifest = bt.from_pydict({"id": [1, 2]}).write.iceberg("db.t", catalog=spec)
    snapshot = catalog.load_table("db.t").current_snapshot()
    assert manifest.version == snapshot.snapshot_id
    assert manifest.destination == "db.t"


def test_a_file_write_has_a_destination_and_no_version(tmp_path):
    path = str(tmp_path / "out")
    manifest = bt.from_pydict({"id": [1]}).write.parquet(path)
    assert manifest.destination == path and manifest.version is None
