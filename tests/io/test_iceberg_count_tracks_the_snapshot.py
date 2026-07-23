"""Regression: an Iceberg `count()` must not outlive the snapshot it described.

`count()`, `is_empty()`, and `min()`/`max()` are answered from a source's cached statistics
without executing the plan. Those statistics are keyed by `Source.identity()`, so an
identity that does not name the snapshot lets an old row count survive a commit: after
appending a row, `count()` kept returning 3 while `collect()` returned 4. The same table,
asked twice, gave two different answers — and the cheap one was the wrong one.

Delta had this exact bug and fixed it by resolving `latest` to the concrete version in its
identity. This pins the same property for Iceberg. Resolving the snapshot also fixes it
against writers that are not us: invalidating a cache on our own commits never covered a
table appended to by Spark or another process.
"""

from __future__ import annotations

from typing import Any

import pytest

import batcher as bt

pytestmark = pytest.mark.io

pytest.importorskip("pyiceberg", reason="iceberg extra not installed")


@pytest.fixture
def catalog(tmp_path) -> dict[str, Any]:
    from batcher.io.catalog import resolve_catalog

    warehouse = tmp_path / "warehouse"
    warehouse.mkdir()
    cfg = {
        "type": "sql",
        "uri": f"sqlite:///{tmp_path}/catalog.db",
        "warehouse": f"file://{warehouse}",
    }
    resolve_catalog(dict(cfg)).create_namespace_if_not_exists("db")
    return cfg


def test_count_follows_an_append(catalog):
    bt.from_pydict({"id": [1, 2, 3]}).write.iceberg("db.t", mode="append", catalog=catalog)
    assert bt.read.iceberg("db.t", catalog=catalog).count() == 3

    bt.from_pydict({"id": [4]}).write.iceberg("db.t", mode="append", catalog=catalog)

    ds = bt.read.iceberg("db.t", catalog=catalog)
    # The cheap answer and the expensive one must agree.
    assert sorted(ds.to_pydict()["id"]) == [1, 2, 3, 4]
    assert ds.count() == 4


def test_identity_changes_with_the_snapshot(catalog):
    from batcher.io.formats.lakehouse.iceberg.source import IcebergSource

    bt.from_pydict({"id": [1]}).write.iceberg("db.t2", mode="append", catalog=catalog)
    before = IcebergSource("db.t2", catalog=catalog).identity()

    bt.from_pydict({"id": [2]}).write.iceberg("db.t2", mode="append", catalog=catalog)
    after = IcebergSource("db.t2", catalog=catalog).identity()

    # A new commit is a new source: that is what leaves no stale entry to serve.
    assert before != after
    assert "@latest" not in after


def test_a_pinned_snapshot_keeps_its_own_count(catalog):
    from batcher.io.catalog import resolve_catalog

    bt.from_pydict({"id": [1, 2, 3]}).write.iceberg("db.t3", mode="append", catalog=catalog)
    pinned = resolve_catalog(dict(catalog)).load_table("db.t3").current_snapshot().snapshot_id

    bt.from_pydict({"id": [4]}).write.iceberg("db.t3", mode="append", catalog=catalog)

    # Time travel still sees the old table, and counts it correctly.
    assert bt.read.iceberg("db.t3", catalog=catalog, snapshot_id=pinned).count() == 3
    assert bt.read.iceberg("db.t3", catalog=catalog).count() == 4
