"""SQL-frontend bug-hunt regression tests vs DuckDB.

Each case pins a distinct defect found in the `_sql` translator: run the same SQL
string through Batcher and DuckDB and assert the results agree. Every query here
returned a wrong answer (or errored on valid SQL) before its fix.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered


@pytest.fixture
def tables(duck):
    emp = pa.table(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "name": ["ann", "bob", "cat", "dan", "eve", None],
            "dept_id": [10, 20, 10, 30, 20, None],
            "salary": [100, 200, 150, 300, 250, 175],
            "bonus": [10.5, None, 20.25, 0.0, -5.5, 7.0],
        }
    )
    dept = pa.table({"dept_id": [10, 20, 40], "dept": ["eng", "sales", "hr"]})
    nums = pa.table({"i": [1, 2, 3, 4, 5, None, -7]})
    wn = pa.table({"k": [1, 2, None, 3]})
    strs = pa.table({"s": ["a", "B", "c_d", "  pad  ", None]})
    for n, t in (("emp", emp), ("dept", dept), ("nums", nums), ("wn", wn), ("strs", strs)):
        duck.register(n, t)
    return {"emp": emp, "dept": dept, "nums": nums, "wn": wn, "strs": strs}


def _run(duck, tables, q):
    return bt.sql(q, **tables).collect(), duck.sql(q)


# ---- NOT IN (subquery): SQL three-valued logic -------------------------------
@pytest.mark.differential
@pytest.mark.parametrize(
    "q",
    [
        # A NULL outer key must NOT survive NOT IN (was kept by the plain anti-join).
        "SELECT id FROM emp WHERE dept_id NOT IN (SELECT dept_id FROM dept)",
        # A NULL inside the subquery makes NOT IN return NO rows (was ignored).
        "SELECT i FROM nums WHERE i NOT IN (SELECT k FROM wn)",
        # Non-null baseline still matches.
        "SELECT id FROM emp WHERE dept_id NOT IN (SELECT dept_id FROM dept WHERE dept_id < 40)",
        # Empty subquery → NOT IN is TRUE for every row, incl. the NULL-keyed one.
        "SELECT id FROM emp WHERE dept_id NOT IN (SELECT dept_id FROM dept WHERE 1=0)",
        # IN (semi) must be unaffected.
        "SELECT i FROM nums WHERE i IN (SELECT k FROM wn)",
    ],
)
def test_not_in_subquery_null_semantics(duck, tables, q):
    got, exp = _run(duck, tables, q)
    assert_same(got, exp)


# ---- set-op trailing ORDER BY / LIMIT / OFFSET -------------------------------
@pytest.mark.differential
def test_setop_limit_applies_to_result(duck, tables):
    # LIMIT trailing a UNION ALL applies to the combined result (was ignored).
    q = "SELECT dept_id FROM emp UNION ALL SELECT dept_id FROM dept LIMIT 3"
    got = bt.sql(q, **tables).collect()
    assert got.num_rows == 3
    assert duck.sql(q).to_arrow_table().num_rows == 3


@pytest.mark.differential
def test_setop_order_by(duck, tables):
    q = "SELECT dept_id FROM emp UNION SELECT dept_id FROM dept ORDER BY 1 NULLS LAST"
    got, exp = _run(duck, tables, q)
    assert_same_ordered(got, exp)


@pytest.mark.differential
def test_setop_order_limit_offset(duck, tables):
    q = "SELECT id FROM emp UNION ALL SELECT id FROM emp ORDER BY id LIMIT 3 OFFSET 2"
    got, exp = _run(duck, tables, q)
    assert_same_ordered(got, exp)


# ---- set ops combine by position, not by name --------------------------------
@pytest.mark.differential
@pytest.mark.parametrize(
    "q",
    [
        "SELECT id FROM emp UNION ALL SELECT dept_id FROM dept",
        "SELECT k FROM wn INTERSECT SELECT i FROM nums",
        "SELECT k FROM wn EXCEPT SELECT i FROM nums",
    ],
)
def test_setop_by_position(duck, tables, q):
    got, exp = _run(duck, tables, q)
    assert_same(got, exp)


@pytest.mark.differential
def test_setop_by_position_swapped_names(duck):
    # Matching names but SWAPPED positions: SQL unions by position, so the second
    # SELECT's `b`/`a` pair into the output `a`/`b`. Union-by-name silently produced
    # the wrong assignment (a values under `a`, not the intended positional column).
    t1 = pa.table({"a": [1, 2], "b": [10, 20]})
    t2 = pa.table({"a": [3], "b": [30]})
    duck.register("t1", t1)
    duck.register("t2", t2)
    q = "SELECT a, b FROM t1 UNION ALL SELECT b, a FROM t2"
    assert_same(bt.sql(q, t1=t1, t2=t2).collect(), duck.sql(q))


# ---- concat_ws skips NULL args (no separator emitted) ------------------------
@pytest.mark.differential
def test_concat_ws_null_args(duck, tables):
    q = "SELECT concat_ws(',', name, 'x', CAST(dept_id AS VARCHAR)) AS c FROM emp"
    got, exp = _run(duck, tables, q)
    assert_same(got, exp)


# ---- CAST to narrow / decimal types is numeric, not string -------------------
@pytest.mark.differential
@pytest.mark.parametrize(
    "q",
    [
        "SELECT CAST(salary AS SMALLINT) c FROM emp",
        "SELECT CAST(salary AS TINYINT) c FROM emp WHERE salary < 128",
        "SELECT CAST(bonus AS DECIMAL(10,2)) c FROM emp",
        "SELECT CAST(id AS BIGINT) c FROM emp",
    ],
)
def test_cast_numeric_types(duck, tables, q):
    got, exp = _run(duck, tables, q)
    assert_same(got, exp)


# ---- TRY_CAST returns NULL on unconvertible values (does not error) ----------
@pytest.mark.differential
def test_try_cast_returns_null(duck, tables):
    q = "SELECT TRY_CAST(s AS INTEGER) c FROM strs"
    got, exp = _run(duck, tables, q)
    assert_same(got, exp)


# ---- ltrim / rtrim strip only one side --------------------------------------
@pytest.mark.differential
@pytest.mark.parametrize(
    "q",
    [
        "SELECT ltrim(s) t FROM strs",
        "SELECT rtrim(s) t FROM strs",
        "SELECT trim(s) t FROM strs",
    ],
)
def test_trim_sides(duck, tables, q):
    got, exp = _run(duck, tables, q)
    assert_same(got, exp)


# ---- GROUP BY with no aggregate is a DISTINCT over the keys ------------------
@pytest.mark.differential
@pytest.mark.parametrize(
    "q",
    [
        "SELECT dept_id FROM emp GROUP BY dept_id",
        "SELECT dept_id FROM emp GROUP BY dept_id HAVING dept_id > 10",
        "SELECT dept_id, dept_id % 20 AS m FROM emp GROUP BY dept_id, dept_id % 20",
    ],
)
def test_group_by_no_aggregate(duck, tables, q):
    got, exp = _run(duck, tables, q)
    assert_same(got, exp)


# ---- power (^, power(), **) and integer floor division (//) ------------------
@pytest.mark.differential
@pytest.mark.parametrize(
    "q",
    [
        "SELECT -2 ^ 2 AS a",
        "SELECT power(salary, 2) AS p FROM emp",
        "SELECT id, salary // 100 AS d FROM emp",
    ],
)
def test_pow_and_intdiv(duck, tables, q):
    got, exp = _run(duck, tables, q)
    assert_same(got, exp)


# ---- newly-wired scalar string functions ------------------------------------
@pytest.mark.differential
@pytest.mark.parametrize(
    "q",
    [
        "SELECT replace(s, 'a', 'z') t FROM strs",
        "SELECT split_part(s, '_', 1) t FROM strs",
        "SELECT s FROM strs WHERE starts_with(s, 'a')",
        "SELECT repeat('ab', 3) r",
        "SELECT s FROM strs WHERE regexp_matches(s, '^[a-c]')",
    ],
)
def test_scalar_string_functions(duck, tables, q):
    got, exp = _run(duck, tables, q)
    assert_same(got, exp)


# ---- E'...' escape string literal -------------------------------------------
@pytest.mark.differential
def test_escape_string_literal(duck, tables):
    q = r"SELECT E'a\tb' AS a"
    got, exp = _run(duck, tables, q)
    assert_same(got, exp)
