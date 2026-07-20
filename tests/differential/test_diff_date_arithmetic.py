"""DATE arithmetic vs DuckDB — `DATE - DATE` (day count) and `DATE ± int` (day shift).

`DATE - DATE` used to return a `duration[s]` interval (arrow's kernel default) where DuckDB
returns the integer count of days; `DATE ± <int>` **crashed** ("Invalid date arithmetic
operation: Date32 - Int64") where DuckDB shifts the date by that many days. Both are now
special-cased in the expression evaluator, with the public `Dataset.schema` reporting the
matching type (Int64 for date-date, the date type for date±int).
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _t():
    return pa.table(
        {
            "a": pa.array([dt.date(2023, 5, 15), dt.date(2023, 1, 1), None], pa.date32()),
            "b": pa.array(
                [dt.date(2023, 5, 10), dt.date(2022, 1, 1), dt.date(2020, 1, 1)], pa.date32()
            ),
            "n": [5, 10, 3],
        }
    )


def test_date_minus_date_is_day_count(duck):
    """`a - b` over two DATE columns is the integer day count (Int64), matching DuckDB."""
    t = _t()
    out = bt.from_arrow(t).select(d=bt.col("a") - bt.col("b")).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT (a - b) AS d FROM t"))


@pytest.mark.parametrize("op", ["-", "+"])
def test_date_plus_minus_int_shifts_days(duck, op):
    """`a - n` / `a + n` shifts the DATE by `n` days, staying a DATE, matching DuckDB."""
    t = _t()
    expr = bt.col("a") - bt.col("n") if op == "-" else bt.col("a") + bt.col("n")
    out = bt.from_arrow(t).select(d=expr).collect()
    duck.register("t", t)
    assert_same(out, duck.sql(f"SELECT (a {op} CAST(n AS INTEGER)) AS d FROM t"))


def test_date_minus_date_schema_is_int64():
    """The public schema reports Int64 for date-date (not date/duration)."""
    ds = bt.from_arrow(_t()).select(d=bt.col("a") - bt.col("b"))
    assert ds.schema.field("d").type == pa.int64()


def test_date_minus_int_schema_is_date():
    """The public schema keeps the date type for date±int."""
    ds = bt.from_arrow(_t()).select(d=bt.col("a") - bt.col("n"))
    assert pa.types.is_date(ds.schema.field("d").type)
