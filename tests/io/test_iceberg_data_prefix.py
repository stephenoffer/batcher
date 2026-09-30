"""An Iceberg write's data files live under a durable name, not a "staging" one.

`add_files` registers each written Parquet file in place, so the directory the writer puts
them in holds live table data after the commit. It used to be ``_batcher_staging``, and a
cleanup job sweeping a staging prefix deleted files that every snapshot still referenced.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytest.importorskip("pyiceberg", reason="pyiceberg not installed")

pytestmark = pytest.mark.integration


def test_committed_data_files_are_not_under_a_staging_prefix(tmp_path):
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

    bt.from_pydict({"id": [1, 2, 3]}).write.iceberg("db.t", catalog=spec, mode="append")

    table = catalog.load_table("db.t")
    paths = [task.file.file_path for task in table.scan().plan_files()]
    assert paths, "the write registered no data file, so the check below would be vacuous"
    assert all("/data/" in p for p in paths), paths
    assert not any("staging" in p for p in paths), paths
    assert sorted(bt.read.iceberg("db.t", catalog=spec).to_pydict()["id"]) == [1, 2, 3]
