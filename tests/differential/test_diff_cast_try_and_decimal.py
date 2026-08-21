"""``TRY_CAST`` never raises, and the two string/boolean decimal casts DuckDB answers.

``TRY_CAST`` is documented as "NULL rather than an error", and a reader does not
distinguish "this *value* does not convert" from "this *pair* has no kernel" — but arrow
raises for the second even under its safe mode, so the promise held only for the
conversions arrow implements. Two pairs made that visible:

* ``BOOLEAN -> DECIMAL`` had no kernel at all, so even the plain ``CAST`` failed where
  DuckDB answers ``1.00`` / ``0.00``;
* ``VARCHAR -> DECIMAL`` read the empty string as **0** (a blank CSV cell became a zero
  amount) and rejected scientific notation outright.
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
            "s": pa.array(["1.5", "abc", "", "12", None, "1e3", "  7.25  "], pa.string()),
            "b": pa.array([True, False, None, True, False, None, True], pa.bool_()),
            "i": pa.array([1, 2, 3, 4, 5, 6, 7], pa.int64()),
        }
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT s, TRY_CAST(s AS DECIMAL(10,2)) AS r FROM t",
        "SELECT b, TRY_CAST(b AS DECIMAL(10,2)) AS r FROM t",
        "SELECT b, CAST(b AS DECIMAL(10,2)) AS r FROM t",
        "SELECT s, TRY_CAST(s AS DOUBLE) AS r FROM t",
        "SELECT s, TRY_CAST(s AS INTEGER) AS r FROM t",
        "SELECT i, CAST(i AS DECIMAL(10,2)) AS r FROM t",
    ],
)
def test_a_decimal_cast_matches_duckdb(duck, sql):
    table = _table()
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_an_empty_string_is_null_not_zero():
    """The defect stated directly: a blank cell must not become a zero amount."""
    table = _table()
    got = bt.sql("SELECT TRY_CAST(s AS DECIMAL(10,2)) AS r FROM t", t=table).collect()
    assert got.to_pydict()["r"][2] is None


def test_try_cast_never_raises_even_for_a_pair_with_no_kernel():
    """Whatever the target, `TRY_CAST` answers — that is the whole of its contract."""
    table = _table()
    for target in ("DECIMAL(10,2)", "DOUBLE", "BIGINT", "VARCHAR", "BOOLEAN"):
        bt.sql(f"SELECT TRY_CAST(b AS {target}) AS r FROM t", t=table).collect()
