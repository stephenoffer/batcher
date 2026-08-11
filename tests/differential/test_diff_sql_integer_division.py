"""SQL `//` — integer division that truncates toward zero, checked against DuckDB.

The obvious lowering, `(a / b).trunc()`, is wrong in three ways at once and all three are
silent. True division casts to Float64, so the result type was DOUBLE where DuckDB gives
BIGINT; the cast happened *before* the truncation, so `9223372036854775807 // 2` came back
as `4.611686018427388e+18` instead of the exact `4611686018427387903`; and a zero divisor
produced `inf` rather than NULL.

None of that is visible in a small fixture — every value under 2^53 divides correctly, and
`assert_same` is type-tolerant, so only the extremes and the schema catch it. This file
asserts all three properties explicitly.

The semantics themselves differ from the DataFrame `//` on purpose: SQL truncates toward
zero (`-7 // 3` is -2) while Python and the `Expr` operator floor (-3). Both are pinned
here so the two cannot quietly converge.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential

# Sign combinations, an exact division, zero, a zero divisor, and both i64 extremes —
# where a detour through Float64 cannot represent the answer.
_PAIRS = [
    (7, 3), (-7, 3), (7, -3), (-7, -3), (9, 3), (0, 3), (7, 0), (-7, 0),
    (9223372036854775807, 2), (-9223372036854775808, 3), (9007199254740993, 2),
]  # fmt: skip


def _table():
    return pa.table(
        {
            "a": pa.array([p[0] for p in _PAIRS], type=pa.int64()),
            "b": pa.array([p[1] for p in _PAIRS], type=pa.int64()),
        }
    )


def test_integer_division_matches_duckdb_exactly(duck):
    table = _table()
    duck.register("t", table)
    sql = "SELECT a // b AS r FROM t"
    got = bt.sql(sql, t=table).to_pydict()["r"]
    want = [row[0] for row in duck.sql(sql).fetchall()]
    assert got == want


def test_the_result_is_an_integer_not_a_double():
    """The type is the tell: `assert_same` tolerates int-vs-float, so only this sees it."""
    schema = bt.sql("SELECT a // b AS r FROM t", t=_table()).schema
    assert schema.field("r").type == pa.int64(), schema


def test_a_value_above_2_to_the_53_stays_exact():
    """The precision claim, on the values where a Float64 detour provably cannot hold it."""
    table = _table()
    got = bt.sql("SELECT a // b AS r FROM t", t=table).to_pydict()["r"]
    by_pair = dict(zip(_PAIRS, got, strict=True))
    assert by_pair[(9223372036854775807, 2)] == 4611686018427387903
    assert by_pair[(9007199254740993, 2)] == 4503599627370496
    assert by_pair[(-9223372036854775808, 3)] == -3074457345618258602
    # ...and the float route really does lose it, so the test is not vacuous.
    assert int(9223372036854775807 / 2) != 4611686018427387903


def test_a_zero_divisor_is_null_not_infinity(duck):
    table = _table()
    duck.register("t", table)
    sql = "SELECT a // b AS r FROM t WHERE b = 0"
    got = bt.sql(sql, t=table).to_pydict()["r"]
    assert got == [None, None], got
    assert got == [row[0] for row in duck.sql(sql).fetchall()]


def test_sql_truncates_toward_zero_while_the_expression_operator_floors():
    """The deliberate divergence between the two surfaces, pinned in one place.

    SQL `//` is DuckDB/C integer division; `Expr.__floordiv__` is Python's and Polars'.
    They agree whenever the operands share a sign and differ by one when they do not.
    """
    table = _table()
    sql_result = bt.sql("SELECT a // b AS r FROM t", t=table).to_pydict()["r"]
    df_result = bt.from_arrow(table).select(r=bt.col("a") // bt.col("b")).to_pydict()["r"]
    by_sql = dict(zip(_PAIRS, sql_result, strict=True))
    by_df = dict(zip(_PAIRS, df_result, strict=True))
    assert by_sql[(-7, 3)] == -2 and by_df[(-7, 3)] == -3
    assert by_sql[(7, -3)] == -2 and by_df[(7, -3)] == -3
    assert by_sql[(7, 3)] == by_df[(7, 3)] == 2
    assert by_sql[(-7, -3)] == by_df[(-7, -3)] == 2


@pytest.mark.parametrize("expr", ["a // b", "(a + 1) // b", "a // (b + 1)", "-(a // b)"])
def test_integer_division_composes(duck, expr):
    """Away from the i64 extremes, where DuckDB itself raises on `a + 1`."""
    modest = pa.table(
        {
            "a": pa.array([7, -7, 9, 0, 100], type=pa.int64()),
            "b": pa.array([3, 3, 3, 3, 7], type=pa.int64()),
        }
    )
    duck.register("t", modest)
    sql = f"SELECT {expr} AS r FROM t"
    assert bt.sql(sql, t=modest).to_pydict()["r"] == [row[0] for row in duck.sql(sql).fetchall()]
