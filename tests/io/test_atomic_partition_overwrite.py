"""`mode="overwrite_partitions"` replaces every covered partition in ONE commit.

It used to commit once per partition, because a Delta overwrite filter is a single AND. A
reader between two of those commits saw some partitions reloaded and others not. The Delta
sink now turns an OR of partition equalities into explicit removals committed with the new
files, so the table moves by exactly one version.
"""

from __future__ import annotations

import tempfile

import pytest

import batcher as bt

deltalake = pytest.importorskip("deltalake")

pytestmark = pytest.mark.integration


def _session_with_orders():
    session = bt.Session()
    lake = bt.Catalog.from_directory(tempfile.mkdtemp(), name="lake")
    session.catalog.attach(lake)
    orders = bt.from_pydict(
        {"id": [1, 2, 3], "region": ["eu", "us", "ap"], "amount": [1.0, 2.0, 3.0]}
    )
    orders.write.table("lake.main.orders", partition_by=["region"], session=session)
    return session, lake


def test_reloading_two_partitions_is_one_table_version():
    session, lake = _session_with_orders()
    uri = lake._backend._uri("main", "orders")
    before = deltalake.DeltaTable(uri).version()

    reload = bt.from_pydict({"id": [7, 8], "region": ["eu", "ap"], "amount": [7.0, 8.0]})
    reload.write.table("lake.main.orders", mode="overwrite_partitions", session=session)

    assert deltalake.DeltaTable(uri).version() == before + 1
    got = session.table("lake.main.orders").sort("id").to_pydict()
    assert got["id"] == [2, 7, 8]
    assert got["region"] == ["us", "eu", "ap"]


def test_a_delta_replace_where_takes_an_or_of_partitions(tmp_path):
    path = str(tmp_path / "t")
    bt.from_pydict({"p": ["a", "b", "c"], "v": [1, 2, 3]}).write.delta(path, partition_by=["p"])
    before = deltalake.DeltaTable(path).version()
    scope = (bt.col("p") == "a") | (bt.col("p") == "c")
    bt.from_pydict({"p": ["a", "c"], "v": [10, 30]}).write.delta(
        path, mode="overwrite", replace_where=scope
    )
    assert deltalake.DeltaTable(path).version() == before + 1
    got = bt.read.delta(path).sort("p").to_pydict()
    assert got == {"p": ["a", "b", "c"], "v": [10, 2, 30]}
