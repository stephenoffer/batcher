"""Two answers the engine gave that were wrong rather than merely different.

`[1, 2] || [3]` rendered both lists as text and concatenated the text, yielding the
string ``'[1, 2][3]'`` where every other engine returns the list ``[1, 2, 3]``. No error,
no warning — the shape of defect a differential census exists to find.

`last_day` returned midnight of that day as a *timestamp*, where DuckDB, Spark and Polars
all return a DATE. It typed the column wrongly in a `with_columns` and forced the
differential tests of it to cast DuckDB's answer before they could compare at all.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

CONCAT_QUERIES = [
    "SELECT [1, 2] || [3] AS r",
    "SELECT [1, 2] || [] AS r",
    "SELECT ['a'] || ['b', 'c'] AS r",
    "SELECT [1, 2] || [2, 3] AS r",
    "SELECT 'a' || 'b' AS r",
    "SELECT 1 || 'b' AS r",
]


@pytest.mark.parametrize("q", CONCAT_QUERIES)
def test_concat_operator_matches_duckdb(duck, q):
    assert_same(bt.sql(q).collect(), duck.sql(q))


def test_the_concat_operator_and_list_concat_agree(duck):
    # Compared value-by-value rather than with `=`: the engine has no equality kernel for
    # a nested type, which is a separate gap and not what this test is about.
    both = bt.sql("SELECT [1, 2] || [3] AS a, list_concat([1, 2], [3]) AS b").to_pydict()
    assert both["a"] == both["b"] == [[1, 2, 3]]
    duck_rows = duck.sql("SELECT [1, 2] || [3] AS a, list_concat([1, 2], [3]) AS b").fetchall()
    assert duck_rows[0][0] == duck_rows[0][1] == [1, 2, 3]


def test_concat_over_list_columns(duck):
    t = pa.table({"a": [[1, 2], [], [3]], "b": [[3], [4], []]})
    duck.register("t", t)
    q = "SELECT a || b AS r FROM t"
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT last_day(DATE '2024-02-15') AS r",
        "SELECT last_day(DATE '2023-02-15') AS r",
        "SELECT last_day(DATE '2024-12-01') AS r",
        "SELECT last_day(TIMESTAMP '2024-02-15 13:45:30') AS r",
    ],
)
def test_last_day_matches_duckdb_without_a_cast(duck, q):
    # The point of this test is the *absence* of a cast on the DuckDB side: both engines
    # now return a DATE, so the values can be compared as they are.
    assert_same(bt.sql(q).collect(), duck.sql(q))


def test_last_day_of_a_date_column_stays_a_date():
    ds = bt.from_pydict({"d": [dt.date(2024, 2, 15), dt.date(2024, 12, 3), None]})
    out = ds.select(r=bt.col("d").dt.last_day())
    assert out.schema.field("r").type == pa.date32()
    assert out.to_pydict()["r"] == [dt.date(2024, 2, 29), dt.date(2024, 12, 31), None]


def test_month_end_is_the_same_function_as_last_day():
    ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45)]})
    out = ds.select(a=bt.col("d").dt.month_end(), b=bt.col("d").dt.last_day()).to_pydict()
    assert out["a"] == out["b"] == [dt.date(2024, 2, 29)]


# --- the epoch readers on a DATE column ------------------------------------------------

EPOCH_QUERIES = [
    "SELECT epoch(DATE '2024-03-05') AS r",
    "SELECT epoch_ms(DATE '2024-03-05') AS r",
    "SELECT epoch_us(DATE '2024-03-05') AS r",
    "SELECT epoch_ns(DATE '2024-03-05') AS r",
    "SELECT epoch_ms(DATE '1969-12-31') AS r",
    "SELECT epoch_us(TIMESTAMP '2024-03-05 06:07:08') AS r",
    "SELECT epoch_ms(TIMESTAMP '2024-03-05 06:07:08.123') AS r",
]


@pytest.mark.parametrize("q", EPOCH_QUERIES)
def test_epoch_readers_match_duckdb_for_dates_and_timestamps(duck, q):
    # A Date32's integer value is a *day* count, so reading it as an integer reported
    # 19,787 microseconds for 2024-03-05 — a wrong answer with no error. The readers now
    # go through a timestamp cast, which is a no-op on a timestamp column.
    assert_same(bt.sql(q).collect(), duck.sql(q))


def test_the_epoch_readers_agree_with_each_other_on_a_date_column():
    ds = bt.from_pydict({"d": [dt.date(2024, 3, 5), None]})
    out = ds.select(
        s=bt.col("d").dt.epoch(),
        ms=bt.col("d").dt.epoch_ms(),
        us=bt.col("d").dt.epoch_us(),
        ns=bt.col("d").dt.epoch_ns(),
    ).to_pydict()
    assert out["ms"][0] == out["s"][0] * 1_000
    assert out["us"][0] == out["s"][0] * 1_000_000
    assert out["ns"][0] == out["s"][0] * 1_000_000_000
    assert [out[k][1] for k in ("s", "ms", "us", "ns")] == [None] * 4
