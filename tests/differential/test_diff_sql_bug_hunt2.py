"""SQL-frontend bug-hunt regression tests vs DuckDB (second batch).

Each case pins a distinct defect found in the `_sql` translator. Every query here
returned a wrong answer (or errored on valid SQL) before its fix:

* ``date_diff('week', ...)`` yielded the fractional ``days / 7`` (a float) instead of
  the whole week count truncated toward zero (an integer).
* ``//`` integer division floored (``-7 // 3 == -3``) and returned a float, where SQL
  truncates toward zero (``-7 // 3 == -2``) and returns an integer.
* ``x IS TRUE`` / ``x IS FALSE`` (and their ``IS NOT`` forms) raised
  ``unsupported SQL expression: Is`` instead of the three-valued boolean test.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered


@pytest.fixture
def tables(duck):
    base = dt.date(2021, 6, 15)
    weeks = pa.table(
        {
            "id": [1, 2, 3, 4, 5, 6, 7],
            "a": pa.array([base] * 7, pa.date32()),
            "b": pa.array(
                [base + dt.timedelta(days=n) for n in (-15, -8, -6, 0, 5, 7, 16)],
                pa.date32(),
            ),
        }
    )
    ints = pa.table({"i": [-7, -6, -1, 0, 1, 2, 3, 4, 5, 100, -100, None]})
    flags = pa.table({"id": [1, 2, 3], "flag": [True, False, None]})
    for n, t in (("weeks", weeks), ("ints", ints), ("flags", flags)):
        duck.register(n, t)
    return {"weeks": weeks, "ints": ints, "flags": flags}


def _run(duck, tables, q):
    return bt.sql(q, **tables).collect(), duck.sql(q)


# ---- date_diff('week') truncates toward zero to an integer -------------------
@pytest.mark.differential
@pytest.mark.parametrize(
    "q",
    [
        "SELECT id, date_diff('week', a, b) AS w FROM weeks",
        "SELECT id, date_diff('week', b, a) AS w FROM weeks",  # reversed → negative
        "SELECT id, date_diff('day', a, b) AS d FROM weeks",  # DAY path unchanged
    ],
)
def test_date_diff_week_is_truncated_integer(duck, tables, q):
    got, exp = _run(duck, tables, q)
    assert_same(got, exp)


# ---- `//` truncates toward zero (not floor) and stays integral ---------------
@pytest.mark.differential
@pytest.mark.parametrize(
    "q",
    [
        "SELECT i, i // 3 AS d FROM ints",
        "SELECT i, i // -3 AS d FROM ints",
        "SELECT i, (i - 1) // 7 AS d FROM ints",
    ],
)
def test_integer_division_truncates_toward_zero(duck, tables, q):
    got, exp = _run(duck, tables, q)
    assert_same(got, exp)


# ---- IS [NOT] TRUE / IS [NOT] FALSE — three-valued boolean tests -------------
@pytest.mark.differential
@pytest.mark.parametrize(
    "q",
    [
        "SELECT id, flag IS TRUE AS t FROM flags ORDER BY id",
        "SELECT id, flag IS NOT TRUE AS t FROM flags ORDER BY id",
        "SELECT id, flag IS FALSE AS t FROM flags ORDER BY id",
        "SELECT id, flag IS NOT FALSE AS t FROM flags ORDER BY id",
    ],
)
def test_is_true_false_projection(duck, tables, q):
    got, exp = _run(duck, tables, q)
    assert_same_ordered(got, exp)


@pytest.mark.differential
@pytest.mark.parametrize(
    "q",
    [
        "SELECT id FROM flags WHERE flag IS TRUE",
        "SELECT id FROM flags WHERE flag IS NOT TRUE",
        "SELECT id FROM flags WHERE flag IS FALSE",
        "SELECT id FROM flags WHERE flag IS NOT FALSE",
        "SELECT i FROM ints WHERE (i > 3) IS TRUE",
        "SELECT i FROM ints WHERE (i > 3) IS NOT TRUE",
    ],
)
def test_is_true_false_in_where(duck, tables, q):
    got, exp = _run(duck, tables, q)
    assert_same(got, exp)
