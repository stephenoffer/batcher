"""MERGE INTO on a lakehouse table runs natively, with every clause it was given.

Two defects are pinned here, both of which let a merge do something other than what it was
asked:

* **`merge_into` did not work on a lakehouse table at all.** It went straight to the
  copy-on-write file path, which rebuilds the target from an explicit file list — a thing a
  `DeltaSource` cannot be constructed with — so the general MERGE API raised a `TypeError`
  against the one format that has a native MERGE.

* **The shorthand silently dropped half its arguments.** `write.merge` routed a Delta
  target into a hard-coded ``update_all`` + ``insert_all``, so ``when_matched="delete"``
  and ``when_not_matched="ignore"`` were accepted and then ignored. The merge reported
  success and left the rows it was told to remove.

The theme of both, and of the Iceberg tests below: a merge must either run the statement it
was given or refuse. Quietly running a different one corrupts the table.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import lit, source_col, target_col
from batcher._internal.errors import PlanError

deltalake = pytest.importorskip("deltalake", reason="deltalake not installed")

pytestmark = pytest.mark.integration

BASE = {"id": [1, 2, 3], "v": [10, 20, 30]}


@pytest.fixture
def table(tmp_path) -> str:
    uri = str(tmp_path / "t")
    bt.from_pydict(BASE).write.delta(uri, mode="overwrite")
    return uri


def _rows(uri: str) -> list[tuple[int, int]]:
    data = bt.read.delta(uri).collect().to_pydict()
    return sorted(zip(data["id"], data["v"], strict=True))


# --- the two regressions ---------------------------------------------------


def test_merge_into_works_on_a_delta_table(table: str) -> None:
    """It used to raise TypeError — the flagship MERGE API on the format with a real MERGE."""
    manifest = (
        bt.from_pydict({"id": [2, 4], "v": [999, 40]})
        .write.merge_into(table, on="id")
        .when_matched()
        .update_all()
        .when_not_matched()
        .insert_all()
        .execute()
    )

    assert _rows(table) == [(1, 10), (2, 999), (3, 30), (4, 40)]
    assert manifest.num_files >= 1, "the manifest must describe the files the merge wrote"


def test_the_shorthand_honors_when_matched_delete(table: str) -> None:
    """It used to accept this and then do an update_all instead."""
    bt.from_pydict({"id": [2], "v": [0]}).write.merge(
        table, on="id", when_matched="delete", when_not_matched="ignore"
    )

    assert _rows(table) == [(1, 10), (3, 30)], "id=2 was told to be deleted"


def test_the_shorthand_honors_when_not_matched_ignore(table: str) -> None:
    bt.from_pydict({"id": [9], "v": [90]}).write.merge(
        table, on="id", when_matched="update", when_not_matched="ignore"
    )

    assert _rows(table) == [(1, 10), (2, 20), (3, 30)], "id=9 must not be inserted"


# --- the full clause set ---------------------------------------------------


def test_a_guarded_update_leaves_unchanged_rows_alone(table: str) -> None:
    (
        bt.from_pydict({"id": [1, 2], "v": [10, 777]})
        .write.merge_into(table, on="id")
        .when_matched(condition=source_col("v") != target_col("v"))
        .update({"v": source_col("v")})
        .when_not_matched()
        .insert_all()
        .execute()
    )

    assert _rows(table) == [(1, 10), (2, 777), (3, 30)]


def test_a_guarded_delete(table: str) -> None:
    (
        bt.from_pydict({"id": [1, 2], "v": [-1, 5]})
        .write.merge_into(table, on="id")
        .when_matched(condition=source_col("v") < lit(0))
        .delete()
        .when_matched()
        .update_all()
        .execute()
    )

    assert _rows(table) == [(2, 5), (3, 30)], "id=1 deleted (v < 0), id=2 updated"


def test_not_matched_by_source_expires_the_rows_the_change_set_never_mentioned(
    table: str,
) -> None:
    """The clause a plain upsert forgets — how a snapshot load retires departed rows."""
    (
        bt.from_pydict({"id": [1], "v": [111]})
        .write.merge_into(table, on="id")
        .when_matched()
        .update_all()
        .when_not_matched_by_source()
        .delete()
        .execute()
    )

    assert _rows(table) == [(1, 111)], "ids 2 and 3 were absent from the source"


def test_the_merge_is_one_transaction(table: str) -> None:
    import os

    log = os.path.join(table, "_delta_log")
    before = len([f for f in os.listdir(log) if f.endswith(".json")])

    bt.from_pydict({"id": [2, 4], "v": [999, 40]}).write.merge(table, on="id")

    after = len([f for f in os.listdir(log) if f.endswith(".json")])
    assert after == before + 1, "a merge must commit exactly one version"


def test_an_unrenderable_condition_is_refused_not_approximated(table: str) -> None:
    """A merge that quietly ran a *different* condition would corrupt the table."""
    with pytest.raises(PlanError, match="cannot be expressed"):
        (
            bt.from_pydict({"id": [1], "v": [1]})
            .write.merge_into(table, on="id")
            .when_matched(condition=source_col("v").cast("string") == lit("1"))
            .update_all()
            .execute()
        )


# --- Iceberg ---------------------------------------------------------------

pyiceberg = pytest.importorskip("pyiceberg", reason="pyiceberg not installed")


@pytest.fixture
def iceberg_table(tmp_path) -> tuple[str, dict]:
    """A real Iceberg table on a local sqlite catalog."""
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
    schema = pa.schema([pa.field("id", pa.int64()), pa.field("v", pa.int64())])
    catalog.create_table("db.t", schema=schema).append(pa.table(BASE, schema=schema))
    return "db.t", spec


def _iceberg_rows(identifier: str, spec: dict) -> list[tuple[int, int]]:
    data = bt.read.iceberg(identifier, catalog=spec).collect().to_pydict()
    return sorted(zip(data["id"], data["v"], strict=True))


def test_iceberg_merge_runs_natively(iceberg_table) -> None:
    """pyiceberg has a real upsert; it was simply never called."""
    identifier, spec = iceberg_table

    (
        bt.from_pydict({"id": [2, 4], "v": [999, 40]})
        .write.merge_into(identifier, on="id", format="iceberg", catalog=spec)
        .when_matched()
        .update_all()
        .when_not_matched()
        .insert_all()
        .execute()
    )

    assert _iceberg_rows(identifier, spec) == [(1, 10), (2, 999), (3, 30), (4, 40)]


def test_iceberg_shorthand_merge(iceberg_table) -> None:
    identifier, spec = iceberg_table
    bt.from_pydict({"id": [5], "v": [50]}).write.merge(
        identifier, on="id", format="iceberg", catalog=spec
    )
    assert _iceberg_rows(identifier, spec) == [(1, 10), (2, 20), (3, 30), (5, 50)]


def test_iceberg_refuses_a_clause_its_upsert_cannot_express(iceberg_table) -> None:
    """Silently dropping a DELETE clause would leave rows the user asked to remove."""
    identifier, spec = iceberg_table

    with pytest.raises(PlanError, match="cannot delete"):
        (
            bt.from_pydict({"id": [1], "v": [0]})
            .write.merge_into(identifier, on="id", format="iceberg", catalog=spec)
            .when_matched()
            .delete()
            .execute()
        )


# --- Iceberg correctness bugs ----------------------------------------------


def test_iceberg_replace_where_keeps_the_rows_it_did_not_match(iceberg_table) -> None:
    """It used to delete them. This is the data-loss regression.

    `replace_where` was dropped entirely on the Iceberg path: the writer's fallback tested
    ``exists(path)``, and an Iceberg "path" is a catalog identifier rather than a file, so
    the check was always False. The predicate went nowhere and the write ran as a plain
    ``mode="overwrite"`` — which wiped the whole table and reported success.
    """
    identifier, spec = iceberg_table

    bt.from_pydict({"id": [100], "v": [999]}).write.iceberg(
        identifier, mode="overwrite", replace_where=bt.col("id") == 1, catalog=spec
    )

    assert _iceberg_rows(identifier, spec) == [(2, 20), (3, 30), (100, 999)]


def test_iceberg_overwrite_is_one_atomic_commit(iceberg_table) -> None:
    """An overwrite used to be two commits, with the table *committed empty* between them.

    A concurrent reader saw zero rows, and a driver that died in the gap left the table
    that way. Both the delete and the add now land in one transaction.
    """
    from pyiceberg.catalog.sql import SqlCatalog

    identifier, spec = iceberg_table
    seen = {"commits": 0, "empty": False}
    original = SqlCatalog.commit_table

    def spy(self, table, requirements, updates):
        seen["commits"] += 1
        result = original(self, table, requirements, updates)
        snapshot = result.metadata.current_snapshot()
        if snapshot is not None and (snapshot.summary or {}).get("total-records") == "0":
            seen["empty"] = True
        return result

    SqlCatalog.commit_table = spy
    try:
        bt.from_pydict({"id": [7], "v": [70]}).write.iceberg(
            identifier, mode="overwrite", catalog=spec
        )
    finally:
        SqlCatalog.commit_table = original

    assert seen["commits"] == 1, "an overwrite must be a single catalog commit"
    assert not seen["empty"], "no committed state may show an empty table"
    assert _iceberg_rows(identifier, spec) == [(7, 70)]


def test_iceberg_count_is_not_answered_from_a_stale_summary(iceberg_table) -> None:
    """`count()` returned 10 for a table that yields 4.

    The snapshot summary's ``total-records`` counts every row the data files hold. It does
    not know about a `row_filter`, and it does not subtract merge-on-read deletes — so
    answering `count()` from it reports a number the table does not have.
    """
    identifier, spec = iceberg_table

    ds = bt.read.iceberg(identifier, row_filter="id >= 2", catalog=spec)

    assert ds.count() == ds.collect().num_rows == 2


def test_iceberg_count_still_uses_the_summary_when_it_is_right(iceberg_table) -> None:
    """The metadata fast path must survive the fix — an unfiltered count needs no scan."""
    from batcher.io.formats.lakehouse import IcebergSource

    identifier, spec = iceberg_table
    source = IcebergSource(identifier, catalog=spec)

    assert source.row_count() == 3
    stats = source.statistics()
    assert stats is not None and stats.exact_rows is True
