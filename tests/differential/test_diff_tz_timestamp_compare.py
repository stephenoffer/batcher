"""Comparing a tz-aware timestamp column to a tz-naive datetime literal — vs DuckDB.

A Delta / event-time column is UTC-normalized (`timestamp[us, "UTC"]`), while a bare
`lit(datetime)` is tz-naive (`timestamp[us, None]`). `col("ts") > lit(naive_dt)` used to
**crash** twice over: the comparison kernel rejected the mismatched types
(`Timestamp(us, Some(...)) > Timestamp(us, None)`), and even the fix's coercion tripped on a
*named* zone because the arrow build lacked the `chrono-tz` feature ("Invalid timezone 'UTC'").

Fixed by (1) coercing a tz-aware-vs-naive timestamp comparison by stripping the zone and
comparing the UTC instants — the naive literal read as that UTC instant, exactly DuckDB's
`TIMESTAMPTZ` vs naive-`TIMESTAMP` rule — and (2) enabling arrow's `chrono-tz` so a named zone
(`"UTC"`, `"America/New_York"`) is supported at all.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential

_TZS = ["UTC", "+00:00", "America/New_York"]


@pytest.mark.parametrize("tz", _TZS)
@pytest.mark.parametrize("op", [">", ">=", "<", "<=", "==", "!="])
def test_tz_aware_column_vs_naive_literal_matches_duckdb(duck, tz, op):
    """`ts <op> naive_datetime` over a tz-aware column agrees with DuckDB's instant compare."""
    rows = [dt.datetime(2021, 1, 1), dt.datetime(2021, 3, 1), dt.datetime(2021, 6, 1)]
    aware = [r.replace(tzinfo=dt.UTC) for r in rows]
    table = pa.table({"r": [0, 1, 2], "ts": pa.array(aware, pa.timestamp("us", tz))})
    lit = dt.datetime(2021, 3, 1)
    ops = {
        ">": bt.col("ts") > lit,
        ">=": bt.col("ts") >= lit,
        "<": bt.col("ts") < lit,
        "<=": bt.col("ts") <= lit,
        "==": bt.col("ts") == lit,
        "!=": bt.col("ts") != lit,
    }
    got = sorted(bt.from_arrow(table).filter(ops[op]).select("r").collect().to_pydict()["r"])

    # DuckDB oracle: a real TIMESTAMPTZ table (session UTC) compared to a naive literal.
    duck.execute("DROP TABLE IF EXISTS t")
    duck.execute("SET TimeZone='UTC'")
    duck.execute("CREATE TABLE t(r INT, ts TIMESTAMPTZ)")
    duck.executemany("INSERT INTO t VALUES (?, ?)", list(zip([0, 1, 2], aware, strict=True)))
    want = [
        row[0]
        for row in duck.sql(
            f"SELECT r FROM t WHERE ts {op} TIMESTAMP '2021-03-01 00:00:00' ORDER BY r"
        ).fetchall()
    ]
    assert got == sorted(want)


@pytest.mark.parametrize("op", [">", "<", "=="])
def test_tz_aware_column_vs_aware_literal_matches_duckdb(duck, op):
    """A tz-**aware** datetime literal must lower without a naive/aware subtraction crash and
    compare on the UTC instant. (Regression: `lit(aware_dt)` raised `TypeError: can't subtract
    offset-naive and offset-aware datetimes` in the IR literal lowering.)"""
    aware = [
        dt.datetime(2021, 1, 1, tzinfo=dt.UTC),
        dt.datetime(2021, 6, 1, tzinfo=dt.UTC),
    ]
    table = pa.table({"r": [0, 1], "ts": pa.array(aware, pa.timestamp("us", "UTC"))})
    lit = dt.datetime(2021, 3, 1, tzinfo=dt.UTC)
    ops = {">": bt.col("ts") > lit, "<": bt.col("ts") < lit, "==": bt.col("ts") == lit}
    got = sorted(bt.from_arrow(table).filter(ops[op]).select("r").collect().to_pydict()["r"])

    duck.execute("DROP TABLE IF EXISTS t")
    duck.execute("SET TimeZone='UTC'")
    duck.execute("CREATE TABLE t(r INT, ts TIMESTAMPTZ)")
    duck.executemany("INSERT INTO t VALUES (?, ?)", list(zip([0, 1], aware, strict=True)))
    want = [
        row[0]
        for row in duck.sql(
            f"SELECT r FROM t WHERE ts {op} TIMESTAMPTZ '2021-03-01 00:00:00+00:00' ORDER BY r"
        ).fetchall()
    ]
    assert got == sorted(want)


def test_named_tz_column_round_trips():
    """A named-tz timestamp column survives a collect (the `chrono-tz` prerequisite)."""
    aware = [dt.datetime(2021, 1, 1, tzinfo=dt.UTC)]
    table = pa.table({"ts": pa.array(aware, pa.timestamp("us", "UTC"))})
    out = bt.from_arrow(table).collect()
    assert out.num_rows == 1
