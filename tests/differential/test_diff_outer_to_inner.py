"""Outer→inner join rewrite vs DuckDB.

A predicate that rejects an outer join's null-extended rows must produce the same
result whether or not Kyber strengthens the join type — DuckDB is the oracle for
the true answer in every case below.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt


def _tables(duck):
    emp = pa.table({"id": [1, 2, 3, 4, 5], "name": list("abcde"), "dept_id": [10, 20, 10, 99, 20]})
    dept = pa.table({"dept_id": [10, 20, 30], "dept": ["eng", "sales", "ops"]})
    duck.register("emp", emp)
    duck.register("dept", dept)
    return bt.from_arrow(emp), bt.from_arrow(dept)


def test_left_join_null_rejecting_right_eq(duck):
    """`WHERE dept = 'eng'` rejects null right rows → left join behaves as inner."""
    from conftest import assert_same

    emp, dept = _tables(duck)
    out = emp.join(dept, on="dept_id", how="left").filter(bt.col("dept") == "eng").collect()
    expected = duck.sql("SELECT * FROM emp LEFT JOIN dept USING (dept_id) WHERE dept = 'eng'")
    assert_same(out, expected)


def test_left_join_null_rejecting_right_isnotnull(duck):
    from conftest import assert_same

    emp, dept = _tables(duck)
    out = emp.join(dept, on="dept_id", how="left").filter(bt.col("dept").is_not_null()).collect()
    expected = duck.sql("SELECT * FROM emp LEFT JOIN dept USING (dept_id) WHERE dept IS NOT NULL")
    assert_same(out, expected)


def test_left_join_null_accepting_isnull_stays_left(duck):
    """`WHERE dept IS NULL` keeps exactly the null-extended rows — must NOT collapse."""
    from conftest import assert_same

    emp, dept = _tables(duck)
    out = emp.join(dept, on="dept_id", how="left").filter(bt.col("dept").is_null()).collect()
    expected = duck.sql("SELECT * FROM emp LEFT JOIN dept USING (dept_id) WHERE dept IS NULL")
    assert_same(out, expected)


def test_left_join_predicate_on_left_col_stays_left(duck):
    """A predicate on a left (preserved) column does not strengthen the join."""
    from conftest import assert_same

    emp, dept = _tables(duck)
    out = emp.join(dept, on="dept_id", how="left").filter(bt.col("name") > "a").collect()
    expected = duck.sql("SELECT * FROM emp LEFT JOIN dept USING (dept_id) WHERE name > 'a'")
    assert_same(out, expected)


def test_right_join_null_rejecting_left_col(duck):
    """`WHERE name = 'a'` rejects null left rows → right join behaves as inner."""
    from conftest import assert_same

    emp, dept = _tables(duck)
    out = emp.join(dept, on="dept_id", how="right").filter(bt.col("name") == "a").collect()
    expected = duck.sql("SELECT * FROM emp RIGHT JOIN dept USING (dept_id) WHERE name = 'a'")
    assert_same(out, expected)


def test_left_join_or_predicate_mixed(duck):
    """`WHERE dept='eng' OR name='d'`: the `name='d'` branch can be true on a
    null-extended row, so the join must NOT collapse. DuckDB is the oracle."""
    from conftest import assert_same

    emp, dept = _tables(duck)
    out = (
        emp.join(dept, on="dept_id", how="left")
        .filter((bt.col("dept") == "eng") | (bt.col("name") == "d"))
        .collect()
    )
    expected = duck.sql(
        "SELECT * FROM emp LEFT JOIN dept USING (dept_id) WHERE dept = 'eng' OR name = 'd'"
    )
    assert_same(out, expected)


def test_full_join_with_predicate(duck):
    """Full outer join carries a coalescing projection, so a top-level filter does
    not sit directly above the join — the rule leaves it alone, and the result
    still matches DuckDB."""
    from conftest import assert_same

    emp, dept = _tables(duck)
    out = emp.join(dept, on="dept_id", how="full").filter(bt.col("dept") == "eng").collect()
    expected = duck.sql("SELECT * FROM emp FULL OUTER JOIN dept USING (dept_id) WHERE dept = 'eng'")
    assert_same(out, expected)
