"""Phase 7 — named-table catalog and per-column data profiling."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError


@pytest.fixture(autouse=True)
def _clear_catalog():
    bt.catalog.clear()
    yield
    bt.catalog.clear()


def test_catalog_register_and_table():
    ds = bt.from_arrow(pa.table({"id": [1, 2, 3]}))
    bt.catalog.register("nums", ds)
    assert bt.catalog.list() == ["nums"]
    assert bt.catalog.table("nums").collect().num_rows == 3


def test_catalog_missing_raises():
    with pytest.raises(PlanError, match="no table"):
        bt.catalog.table("absent")


def test_catalog_resolves_in_sql():
    bt.catalog.register("customers", bt.from_arrow(pa.table({"id": [1, 2], "name": ["a", "b"]})))
    out = bt.sql("SELECT name FROM customers WHERE id = 1").collect()
    assert out.to_pydict() == {"name": ["a"]}


def test_catalog_explicit_table_overrides_registry():
    bt.catalog.register("t", bt.from_arrow(pa.table({"v": [1]})))
    out = bt.sql("SELECT * FROM t", t=bt.from_arrow(pa.table({"v": [99]}))).collect()
    assert out.to_pydict() == {"v": [99]}


def test_catalog_drop():
    bt.catalog.register("t", bt.from_arrow(pa.table({"v": [1]})))
    bt.catalog.drop("t")
    assert bt.catalog.list() == []


def test_profile_reports_per_column_stats():
    t = pa.table(
        {
            "id": pa.array([1, 2, 3, 4], pa.int64()),
            "cat": ["a", "a", "b", None],
        }
    )
    prof = {r["column"]: r for r in _rows(bt.from_arrow(t).profile().collect())}
    assert prof["id"]["null_count"] == 0
    assert prof["id"]["approx_distinct"] == 4
    assert prof["cat"]["null_count"] == 1
    assert prof["cat"]["null_fraction"] == 0.25
    assert prof["cat"]["approx_distinct"] == 2  # 'a','b' (null excluded)


def _rows(table: pa.Table) -> list[dict]:
    d = table.to_pydict()
    return [dict(zip(d.keys(), vals, strict=True)) for vals in zip(*d.values(), strict=True)]
