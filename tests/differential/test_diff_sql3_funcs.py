"""SQL-frontend function-coverage regression tests vs DuckDB (wave 8).

Each case pins a distinct defect in the `_sql` translator's scalar-function,
literal-relation, or statement handling: run the same SQL through Batcher and
DuckDB and assert they agree. Every query here returned a wrong answer, crashed,
or errored on valid DuckDB SQL before its fix.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from conftest import assert_same


@pytest.fixture
def tables(duck):
    emp = pa.table(
        {
            "id": pa.array([1, 2, 3, 4, 5, 6, 7], pa.int64()),
            "name": pa.array(["al", "bo", "cy", "di", "ed", "fi", None], pa.string()),
            "dept": pa.array([10, 10, 20, 20, 30, None, 30], pa.int64()),
            "sal": pa.array([100, 200, 150, None, 300, 50, 300], pa.int64()),
        }
    )
    for n, t in (("emp", emp),):
        duck.register(n, t)
    return {"emp": emp}


def _run(duck, tables, q):
    return bt.sql(q, **tables).collect(), duck.sql(q)


# ---- S1: concat over numeric / mixed args (cast + null-skip) -----------------
@pytest.mark.differential
@pytest.mark.parametrize(
    "q",
    [
        # `concat` casts every arg to text (numeric column) — was a type error.
        "SELECT concat(id, name) AS x FROM emp",
        # `concat` drops NULL args (treats as '') — DuckDB semantics.
        "SELECT concat(name, '!') AS x FROM emp",
        "SELECT concat(id, '-', dept) AS x FROM emp",
    ],
)
def test_concat_numeric_and_null(duck, tables, q):
    got, exp = _run(duck, tables, q)
    assert_same(got, exp)


# ---- S1: trunc(x, n) truncates to n decimals (was silently n=0) --------------
@pytest.mark.differential
@pytest.mark.parametrize(
    "q",
    [
        "SELECT trunc(2.567, 1) AS x FROM emp LIMIT 1",
        "SELECT trunc(2.567, 2) AS x FROM emp LIMIT 1",
        "SELECT trunc(-2.567, 1) AS x FROM emp LIMIT 1",
        "SELECT trunc(2.567) AS x FROM emp LIMIT 1",
    ],
)
def test_trunc_digits(duck, tables, q):
    got, exp = _run(duck, tables, q)
    assert_same(got, exp)


# ---- S2: substr with a negative / signed start no longer crashes -------------
@pytest.mark.differential
@pytest.mark.parametrize(
    "q",
    [
        "SELECT substr('hello', -2) AS x FROM emp LIMIT 1",
        "SELECT substr('hello', -3, 2) AS x FROM emp LIMIT 1",
        "SELECT substr(name, 2) AS x FROM emp",
    ],
)
def test_substr_negative_start(duck, tables, q):
    got, exp = _run(duck, tables, q)
    assert_same(got, exp)


# ---- S4: newly-wired scalar functions (were clean errors on valid SQL) -------
@pytest.mark.differential
@pytest.mark.parametrize(
    "q",
    [
        "SELECT left(name, 2) AS x FROM emp",
        "SELECT right(name, 2) AS x FROM emp",
        "SELECT ends_with(name, 'y') AS x FROM emp",
        "SELECT contains(name, 'y') AS x FROM emp",
        "SELECT ascii(name) AS x FROM emp",
        "SELECT regexp_extract(name, '([a-z])', 1) AS x FROM emp",
        "SELECT dayofweek(DATE '2021-03-15') AS x FROM emp LIMIT 1",
        "SELECT dayofmonth(DATE '2021-03-15') AS x FROM emp LIMIT 1",
        "SELECT dayofyear(DATE '2021-03-15') AS x FROM emp LIMIT 1",
        "SELECT date_part('month', DATE '2021-03-15') AS x FROM emp LIMIT 1",
        "SELECT date_part('doy', DATE '2021-03-15') AS x FROM emp LIMIT 1",
    ],
)
def test_new_scalar_functions(duck, tables, q):
    got, exp = _run(duck, tables, q)
    assert_same(got, exp)


# ---- VALUES as an inline literal relation ------------------------------------
@pytest.mark.differential
@pytest.mark.parametrize(
    "q",
    [
        "SELECT * FROM (VALUES (1, 'a'), (2, 'b')) AS t(id, nm)",
        "SELECT id, nm FROM (VALUES (1, 'a'), (2, NULL), (-3, 'c')) AS t(id, nm) WHERE id > 0",
        "SELECT sum(id) AS s FROM (VALUES (1, 'a'), (2, 'b'), (3, 'c')) AS t(id, nm)",
        "SELECT * FROM (VALUES (1, 2), (3, 4))",
        "VALUES (1, 'a'), (2, 'b')",
        "SELECT nm, count(*) AS c"
        " FROM (VALUES (1, 'x'), (2, 'x'), (3, 'y')) AS t(id, nm) GROUP BY nm",
    ],
)
def test_values_relation(duck, tables, q):
    got, exp = _run(duck, tables, q)
    assert_same(got, exp)


# ---- EXPLAIN returns a plan without executing (and without crashing) ---------
@pytest.mark.differential
def test_explain_returns_plan(tables):
    got = bt.sql("EXPLAIN SELECT id FROM emp WHERE sal > 150", **tables).collect()
    d = got.to_pydict()
    assert got.num_rows == 1
    assert "explain_value" in d
    # A real plan tree came back; the query itself was not executed as the result.
    assert len(d["explain_value"][0]) > 0
    assert "scan" in d["explain_value"][0]
