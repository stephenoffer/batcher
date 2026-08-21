"""Numeric and string-formatting edges the constant kernels answered differently.

Each of these returned a plausible value rather than raising, which is why they survived:
`hex(-3)` gave `'-3'`, `round(1e308, 1)` gave `inf`, `divide` floored where DuckDB
truncates, `fdiv` divided where DuckDB floors, and `ascii('')` gave 0 where DuckDB gives
-1 (0 is a real code point, so the two were indistinguishable).
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
            "i": pa.array([0, 1, -1, 5, -3, 255, -128, None], pa.int64()),
            "j": pa.array([3, 2, 3, 2, 2, 16, 16, None], pa.int64()),
            "f": pa.array([0.0, -0.0, 1.5, -1.5, 1e308, -1e308, 1e-308, None], pa.float64()),
            "s": pa.array(["", "a", "abc", "Z", " ", "ünï", None, "x"], pa.string()),
        }
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT i, hex(i) AS r FROM t",
        "SELECT i, to_hex(i) AS r FROM t",
        "SELECT i, format_bytes(i) AS r FROM t",
        "SELECT i, formatReadableDecimalSize(i) AS r FROM t",
        "SELECT s, ascii(s) AS r FROM t",
        "SELECT s, unicode(s) AS r FROM t",
        "SELECT s, ord(s) AS r FROM t",
        "SELECT f, round(f, 1) AS r FROM t",
        "SELECT f, round(f, 3) AS r FROM t",
        "SELECT f, trunc(f, 1) AS r FROM t",
        "SELECT i, j, divide(i, j) AS r FROM t",
        "SELECT i, j, fdiv(i, j) AS r FROM t",
        "SELECT i, j, i // j AS r FROM t",
        "SELECT f, divide(f, f) AS r FROM t",
        "SELECT f, fdiv(f, f) AS r FROM t",
    ],
)
def test_a_numeric_edge_matches_duckdb(duck, sql):
    table = _table()
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_ascii_and_ord_disagree_on_the_empty_string():
    """DuckDB draws a line here: `ascii('')` is 0 and `ord('')`/`unicode('')` are -1."""
    table = _table()
    got = bt.sql("SELECT ascii('') AS a, ord('') AS b, unicode('') AS c", t=table).collect()
    assert got.to_pydict() == {"a": [0], "b": [-1], "c": [-1]}


def test_hex_of_a_negative_is_the_twos_complement_pattern():
    """`hex(-3)` is `FFFFFFFFFFFFFFFD`; a sign-prefixed magnitude is a different string."""
    table = _table()
    got = bt.sql("SELECT hex(-3) AS r", t=table).collect().to_pydict()
    assert got["r"] == ["FFFFFFFFFFFFFFFD"]


def test_a_zero_divisor_is_null_on_floats_too():
    """IEEE would answer inf/NaN, and a NaN then compares false against everything."""
    table = _table()
    got = bt.sql("SELECT CAST(1 AS DOUBLE) // CAST(0 AS DOUBLE) AS r", t=table).collect()
    assert got.to_pydict()["r"] == [None]


def test_format_bytes_says_byte_for_exactly_one():
    table = _table()
    got = bt.sql("SELECT format_bytes(1) AS a, format_bytes(-1) AS b", t=table).collect()
    assert got.to_pydict() == {"a": ["1 byte"], "b": ["-1 byte"]}
