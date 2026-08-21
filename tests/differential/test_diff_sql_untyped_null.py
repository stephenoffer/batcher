"""A bare ``NULL`` takes the type its position requires.

SQL leaves ``NULL`` untyped and lets the surrounding operator decide. The IR has no
untyped null — a bare ``NULL`` lowers to the Int64 ``nullif(1, 1)`` — so every context
wanting a different type failed on a type error rather than answering:

* ``NULL OR x`` / ``NULL AND x`` reached the engine as ``or(Int64, Bool)``;
* ``s = NULL`` on a text column as ``Utf8 == Int64``;
* ``CASE s WHEN NULL THEN …`` likewise, and once the branch was skipped, as
  "CASE without WHEN";
* ``upper(NULL)`` (and every other string function) as "expected a Utf8 argument".

Each is now typed from its position. The results are compared against DuckDB rather than
asserted, because "returns NULL" is not the whole contract: three-valued ``OR`` returns
TRUE against a NULL when the other operand is true.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _table() -> pa.Table:
    return pa.table(
        {
            "i": pa.array([1, 2, None, 4], pa.int64()),
            "s": pa.array(["a", None, "c", "d"], pa.string()),
            "b": pa.array([True, False, None, True], pa.bool_()),
            "f": pa.array([1.5, None, -2.5, 0.0], pa.float64()),
        }
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT i, (NULL OR i = 1) AS r FROM t",
        "SELECT i, (NULL AND i = 1) AS r FROM t",
        "SELECT i, (i = 1 OR NULL) AS r FROM t",
        "SELECT i, (i = 1 AND NULL) AS r FROM t",
        "SELECT (NULL AND NULL) AS r",
        "SELECT (NULL OR NULL) AS r",
        "SELECT (NOT NULL) AS r",
        "SELECT s, s = NULL AS r FROM t",
        "SELECT s, s <> NULL AS r FROM t",
        "SELECT i, i > NULL AS r FROM t",
        "SELECT f, f <= NULL AS r FROM t",
        "SELECT i, i + NULL AS r FROM t",
        "SELECT f, f * NULL AS r FROM t",
        "SELECT s, s || NULL AS r FROM t",
        "SELECT s, CASE s WHEN NULL THEN 1 ELSE 0 END AS r FROM t",
        "SELECT s, CASE s WHEN NULL THEN 1 END AS r FROM t",
        "SELECT s, CASE s WHEN NULL THEN 'x' WHEN 'a' THEN 'y' END AS r FROM t",
        "SELECT upper(NULL) AS r",
        "SELECT lower(NULL) AS r",
        "SELECT length(NULL) AS r",
        "SELECT trim(NULL) AS r",
        "SELECT reverse(NULL) AS r",
        "SELECT md5(NULL) AS r",
        "SELECT substring(NULL, 1, 2) AS r",
        "SELECT replace(NULL, 'a', 'b') AS r",
        "SELECT s, concat(NULL, s) AS r FROM t",
        "SELECT s, coalesce(NULL, s) AS r FROM t",
        "SELECT s, coalesce(s, NULL) AS r FROM t",
        "SELECT coalesce(NULL, NULL) AS r",
        "SELECT s, nullif(s, NULL) AS r FROM t",
        "SELECT s, greatest(s, NULL) AS r FROM t",
        "SELECT s, least(NULL, s) AS r FROM t",
    ],
)
def test_a_bare_null_takes_the_type_its_position_needs(duck, sql):
    table = _table()
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_the_three_valued_or_is_not_simply_null():
    """`NULL OR TRUE` is TRUE, so "everything is NULL" would pass a weaker test."""
    table = _table()
    got = bt.sql("SELECT i, (NULL OR i = 1) AS r FROM t", t=table).collect().to_pydict()
    assert got["r"] == [True, None, None, None]
