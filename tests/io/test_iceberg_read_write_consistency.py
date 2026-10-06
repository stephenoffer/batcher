"""Iceberg reads and writes that used to answer for a table other than the one in front of them.

Four defects, each a wrong answer rather than an error:

* `count()` ignored **equality** deletes, so a Flink-CDC table counted every row it ever
  inserted (BT-243).
* A distributed split of a latest read carried no snapshot, so a worker resolved the schema
  again at execution time and could not find a column renamed after planning (BT-250).
* A streaming micro-batch's idempotency marker lived only in snapshot summaries, so expiring
  the snapshot that carried it made a replay append the batch twice (BT-248).
* An overwrite or `replace_where` whose input was empty returned before its delete, leaving
  the rows it was told to replace (the empty-overwrite finding on BT-242).
"""

from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pytest

import batcher as bt

pytest.importorskip("pyiceberg", reason="pyiceberg not installed")

from batcher.io.formats.lakehouse.iceberg.source import IcebergSource

pytestmark = pytest.mark.integration

_SCHEMA = pa.schema([pa.field("id", pa.int64()), pa.field("v", pa.int64())])


@pytest.fixture
def spec(tmp_path) -> dict:
    """A SQL catalog with an empty ``db.t`` of ``(id, v)``."""
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
    catalog.create_table("db.t", schema=_SCHEMA)
    return spec


def _table(spec: dict):
    from batcher.io.catalog import resolve_catalog

    return resolve_catalog(dict(spec)).load_table("db.t")


def _append(spec: dict, ids: list[int]) -> None:
    _table(spec).append(pa.table({"id": ids, "v": [i * 10 for i in ids]}, schema=_SCHEMA))


def _ids(spec: dict) -> list[int]:
    return sorted(_table(spec).scan().to_arrow()["id"].to_pylist())


# --- BT-243: equality deletes ---------------------------------------------------------------


def _with_summary(monkeypatch, summary: dict) -> IcebergSource:
    """A source whose snapshot summary is `summary`.

    Constructed rather than written: pyiceberg cannot write an equality-delete file, so the
    Flink-CDC shape is reproduced at the one place the source reads it.
    """
    source = IcebergSource("db.t", catalog={"type": "sql"})
    snapshot = SimpleNamespace(summary=summary)
    monkeypatch.setattr(IcebergSource, "_snapshot", lambda self: snapshot)
    return source


def test_a_summary_without_deletes_is_the_count(monkeypatch):
    # Positive control: the decline below is about the deletes, not the stub.
    source = _with_summary(monkeypatch, {"total-records": "100"})
    assert source.row_count() == 100


@pytest.mark.parametrize(
    "summary",
    [
        {"total-records": "100", "total-equality-deletes": "30", "total-delete-files": "1"},
        {"total-records": "100", "total-delete-files": "1"},
        {"total-records": "100", "total-position-deletes": "30"},
    ],
    ids=["equality", "delete-files-only", "position"],
)
def test_any_delete_file_declines_the_summary_count(monkeypatch, summary):
    source = _with_summary(monkeypatch, summary)
    assert source.row_count() is None
    assert source.statistics() is None  # nothing may claim an exact count of 100


# --- BT-250: splits carry the planned snapshot and schema ------------------------------------


def test_a_latest_split_carries_a_concrete_snapshot(spec):
    _append(spec, [1, 2, 3])
    current = _table(spec).current_snapshot().snapshot_id
    splits = IcebergSource("db.t", catalog=spec).splits()
    assert splits
    assert all(s._snapshot_id == current for s in splits)


def test_a_split_reads_under_the_schema_it_was_planned_with(spec):
    _append(spec, [1, 2, 3])
    (split,) = IcebergSource("db.t", catalog=spec).splits()

    # Between planning on the driver and reading on a worker, someone renames a column.
    with _table(spec).update_schema() as update:
        update.rename_column("v", "w")

    assert split.schema().names == ["id", "v"]
    got = pa.Table.from_batches(split.read(projection=["v"]))
    assert got.column_names == ["v"]
    assert sorted(got["v"].to_pylist()) == [10, 20, 30]
    # The planned predicate also binds against the planned schema.
    filtered = pa.Table.from_batches(
        split.read(
            projection=["id"],
            predicate=(bt.col("v") > 15).to_ir(),
        )
    )
    assert sorted(filtered["id"].to_pylist()) == [2, 3]


def test_a_time_travel_split_keeps_its_snapshot(spec):
    _append(spec, [1])
    pinned = _table(spec).current_snapshot().snapshot_id
    _append(spec, [2])
    splits = IcebergSource("db.t", catalog=spec, snapshot_id=pinned).splits()
    assert [s._snapshot_id for s in splits] == [pinned]
    rows = [r for s in splits for b in s.read() for r in b.to_pydict()["id"]]
    assert rows == [1]
