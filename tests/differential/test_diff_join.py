"""Differential tests for joins against DuckDB (USING-style key semantics)."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt


def _tables(duck):
    emp = pa.table({"id": [1, 2, 3, 4, 5], "name": list("abcde"), "dept_id": [10, 20, 10, 99, 20]})
    dept = pa.table({"dept_id": [10, 20, 30], "dept": ["eng", "sales", "ops"]})
    duck.register("emp", emp)
    duck.register("dept", dept)
    return bt.from_arrow(emp), bt.from_arrow(dept)


def test_inner_join_vs_duckdb(duck):
    from conftest import assert_same

    emp, dept = _tables(duck)
    out = emp.join(dept, on="dept_id").collect()
    expected = duck.sql("SELECT * FROM emp JOIN dept USING (dept_id)")
    assert_same(out, expected)


def test_left_join_vs_duckdb(duck):
    from conftest import assert_same

    emp, dept = _tables(duck)
    out = emp.join(dept, on="dept_id", how="left").collect()
    expected = duck.sql("SELECT * FROM emp LEFT JOIN dept USING (dept_id)")
    assert_same(out, expected)


def test_right_join_vs_duckdb(duck):
    from conftest import assert_same

    emp, dept = _tables(duck)
    out = emp.join(dept, on="dept_id", how="right").collect()
    expected = duck.sql("SELECT * FROM emp RIGHT JOIN dept USING (dept_id)")
    assert_same(out, expected)


def test_semi_join_vs_duckdb(duck):
    from conftest import assert_same

    emp, dept = _tables(duck)
    out = emp.join(dept, on="dept_id", how="semi").collect()
    expected = duck.sql("SELECT emp.* FROM emp SEMI JOIN dept USING (dept_id)")
    assert_same(out, expected)


def test_anti_join_vs_duckdb(duck):
    from conftest import assert_same

    emp, dept = _tables(duck)
    out = emp.join(dept, on="dept_id", how="anti").collect()
    expected = duck.sql("SELECT emp.* FROM emp ANTI JOIN dept USING (dept_id)")
    assert_same(out, expected)


def test_join_then_aggregate_vs_duckdb(duck):
    from batcher import col, count
    from conftest import assert_same

    emp, dept = _tables(duck)
    out = (
        emp.join(dept, on="dept_id")
        .group_by("dept")
        .agg(headcount=count(), max_id=col("id").max())
        .collect()
    )
    expected = duck.sql(
        "SELECT dept, COUNT(*) AS headcount, MAX(id) AS max_id "
        "FROM emp JOIN dept USING (dept_id) GROUP BY dept"
    )
    assert_same(out, expected)


def test_sort_merge_join_strategy_vs_duckdb(duck, monkeypatch):
    """Force the sort-merge strategy (thresholds) and check every join type still
    matches DuckDB end-to-end (Python plan → IR strategy=sort_merge → engine)."""
    from batcher.kyber.rules import selection
    from conftest import assert_same

    # No broadcast, and any join qualifies as sort-merge → exercise the SMJ path.
    monkeypatch.setattr(selection, "BROADCAST_MAX_BYTES", -1)
    monkeypatch.setattr(selection, "SORT_MERGE_MIN_ROWS", 0.0)

    emp, dept = _tables(duck)
    for how, sql in [
        ("inner", "JOIN"),
        ("left", "LEFT JOIN"),
        ("right", "RIGHT JOIN"),
    ]:
        out = emp.join(dept, on="dept_id", how=how).collect()
        expected = duck.sql(f"SELECT * FROM emp {sql} dept USING (dept_id)")
        assert_same(out, expected)

    semi = emp.join(dept, on="dept_id", how="semi").collect()
    assert_same(semi, duck.sql("SELECT emp.* FROM emp SEMI JOIN dept USING (dept_id)"))
    anti = emp.join(dept, on="dept_id", how="anti").collect()
    assert_same(anti, duck.sql("SELECT emp.* FROM emp ANTI JOIN dept USING (dept_id)"))
