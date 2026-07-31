"""Outer→inner join rewrite vs DuckDB.

A predicate that rejects an outer join's null-extended rows must produce the same
result whether or not Kyber strengthens the join type — DuckDB is the oracle for
the true answer in every case below.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from _harness import assert_same


def _tables(duck):
    emp = pa.table({"id": [1, 2, 3, 4, 5], "name": list("abcde"), "dept_id": [10, 20, 10, 99, 20]})
    dept = pa.table({"dept_id": [10, 20, 30], "dept": ["eng", "sales", "ops"]})
    duck.register("emp", emp)
    duck.register("dept", dept)
    return bt.from_arrow(emp), bt.from_arrow(dept)


def test_left_join_null_rejecting_right_eq(duck):
    """`WHERE dept = 'eng'` rejects null right rows → left join behaves as inner."""
    emp, dept = _tables(duck)
    out = emp.join(dept, on="dept_id", how="left").filter(bt.col("dept") == "eng").collect()
    expected = duck.sql("SELECT * FROM emp LEFT JOIN dept USING (dept_id) WHERE dept = 'eng'")
    assert_same(out, expected)


def test_left_join_null_rejecting_right_isnotnull(duck):
    emp, dept = _tables(duck)
    out = emp.join(dept, on="dept_id", how="left").filter(bt.col("dept").is_not_null()).collect()
    expected = duck.sql("SELECT * FROM emp LEFT JOIN dept USING (dept_id) WHERE dept IS NOT NULL")
    assert_same(out, expected)


def test_left_join_null_accepting_isnull_stays_left(duck):
    """`WHERE dept IS NULL` keeps exactly the null-extended rows — must NOT collapse."""
    emp, dept = _tables(duck)
    out = emp.join(dept, on="dept_id", how="left").filter(bt.col("dept").is_null()).collect()
    expected = duck.sql("SELECT * FROM emp LEFT JOIN dept USING (dept_id) WHERE dept IS NULL")
    assert_same(out, expected)


def test_left_join_predicate_on_left_col_stays_left(duck):
    """A predicate on a left (preserved) column does not strengthen the join."""
    emp, dept = _tables(duck)
    out = emp.join(dept, on="dept_id", how="left").filter(bt.col("name") > "a").collect()
    expected = duck.sql("SELECT * FROM emp LEFT JOIN dept USING (dept_id) WHERE name > 'a'")
    assert_same(out, expected)


def test_right_join_null_rejecting_left_col(duck):
    """`WHERE name = 'a'` rejects null left rows → right join behaves as inner."""
    emp, dept = _tables(duck)
    out = emp.join(dept, on="dept_id", how="right").filter(bt.col("name") == "a").collect()
    expected = duck.sql("SELECT * FROM emp RIGHT JOIN dept USING (dept_id) WHERE name = 'a'")
    assert_same(out, expected)


def test_left_join_or_predicate_mixed(duck):
    """`WHERE dept='eng' OR name='d'`: the `name='d'` branch can be true on a
    null-extended row, so the join must NOT collapse. DuckDB is the oracle."""
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


# --- full outer join ---------------------------------------------------------
#
# A full join carries a coalescing projection between itself and any filter above it, so the
# rewrite has to look through one projection to see the join at all. Until it did, the whole
# `full` branch was unreachable and this section asserted only that the un-rewritten answer was
# right. The fixture matters here: `emp` has a row with no department (dept_id 99) and `dept` has
# a department with no employees (30), so all three parts of a full join -- matched, left-only,
# right-only -- are non-empty, and a rewrite that dropped the wrong part would show up as lost
# rows rather than as an equal result.


def test_full_join_null_rejecting_right_behaves_as_right(duck):
    """`WHERE dept = 'eng'` kills every left-only row → the full join behaves as a RIGHT join."""
    emp, dept = _tables(duck)
    out = emp.join(dept, on="dept_id", how="full").filter(bt.col("dept") == "eng").collect()
    expected = duck.sql("SELECT * FROM emp FULL OUTER JOIN dept USING (dept_id) WHERE dept = 'eng'")
    assert_same(out, expected)


def test_full_join_null_rejecting_left_behaves_as_left(duck):
    """`WHERE name = 'a'` kills every right-only row → the full join behaves as a LEFT join."""
    emp, dept = _tables(duck)
    out = emp.join(dept, on="dept_id", how="full").filter(bt.col("name") == "a").collect()
    expected = duck.sql("SELECT * FROM emp FULL OUTER JOIN dept USING (dept_id) WHERE name = 'a'")
    assert_same(out, expected)


def test_full_join_rejecting_both_sides_behaves_as_inner(duck):
    emp, dept = _tables(duck)
    out = (
        emp.join(dept, on="dept_id", how="full")
        .filter((bt.col("dept") == "eng") & (bt.col("name") == "a"))
        .collect()
    )
    expected = duck.sql(
        "SELECT * FROM emp FULL OUTER JOIN dept USING (dept_id) WHERE dept = 'eng' AND name = 'a'"
    )
    assert_same(out, expected)


def test_full_join_is_not_null_on_the_right(duck):
    emp, dept = _tables(duck)
    out = emp.join(dept, on="dept_id", how="full").filter(bt.col("dept").is_not_null()).collect()
    expected = duck.sql(
        "SELECT * FROM emp FULL OUTER JOIN dept USING (dept_id) WHERE dept IS NOT NULL"
    )
    assert_same(out, expected)


def test_full_join_on_the_coalesced_key_keeps_every_part(duck):
    """The case a wrong rewrite would break, and the reason the look-through maps only bare
    column references.

    The output key is `coalesce(emp.dept_id, dept.dept_id)`, which is non-null on a left-only
    row *and* on a right-only row — so a predicate on it rejects neither side and the join must
    stay `full`. Treating the coalesced key as a reference to either side would drop the
    unmatched rows on the other, and the answer would be silently short.
    """
    emp, dept = _tables(duck)
    out = emp.join(dept, on="dept_id", how="full").filter(bt.col("dept_id") > 5).collect()
    expected = duck.sql("SELECT * FROM emp FULL OUTER JOIN dept USING (dept_id) WHERE dept_id > 5")
    assert_same(out, expected)


def test_full_join_is_null_stays_full(duck):
    """`IS NULL` selects exactly the null-extended rows, so it must not strengthen anything."""
    emp, dept = _tables(duck)
    out = emp.join(dept, on="dept_id", how="full").filter(bt.col("dept").is_null()).collect()
    expected = duck.sql("SELECT * FROM emp FULL OUTER JOIN dept USING (dept_id) WHERE dept IS NULL")
    assert_same(out, expected)


def test_full_join_disjunction_needs_both_disjuncts_rejecting(duck):
    """`a OR b` rejects only what *both* sides reject — here neither, so nothing strengthens."""
    emp, dept = _tables(duck)
    out = (
        emp.join(dept, on="dept_id", how="full")
        .filter((bt.col("dept") == "eng") | bt.col("dept").is_null())
        .collect()
    )
    expected = duck.sql(
        "SELECT * FROM emp FULL OUTER JOIN dept USING (dept_id) WHERE dept = 'eng' OR dept IS NULL"
    )
    assert_same(out, expected)


def test_full_join_rewrite_agrees_on_every_execution_path(duck):
    from _harness import assert_tables_equal

    emp, dept = _tables(duck)

    def build():
        return emp.join(dept, on="dept_id", how="full").filter(bt.col("dept") == "eng")

    oracle = build().collect()
    assert_tables_equal(build().collect(spill=True), oracle)
    batches = list(build().iter_batches())
    streamed = (
        pa.Table.from_batches(batches, schema=batches[0].schema) if batches else oracle.slice(0, 0)
    )
    assert_tables_equal(streamed, oracle)
