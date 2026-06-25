"""Differential coverage for Phase-2 namespace additions vs DuckDB.

`.str` (trim-with-chars, split_part, regexp_replace_all), `.dt` (is_leap_year,
days_in_month, iso_year), and `.list` (first/last/negative get) — each checked
against the equivalent DuckDB expression.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


# --- string ---------------------------------------------------------------
def test_str_trim_with_chars(duck):
    from conftest import assert_same

    t = pa.table({"s": ["xxhixx", "--a--", "  pad  ", None]})
    duck.register("t", t)
    out = bt.from_arrow(t).select(
        b=col("s").str.trim("x"),
        l=col("s").str.lstrip("-"),
        r=col("s").str.rstrip(" "),
        w=col("s").str.trim(),
    )
    assert_same(
        out.collect(),
        duck.sql("SELECT trim(s, 'x') b, ltrim(s, '-') l, rtrim(s, ' ') r, trim(s) w FROM t"),
    )


def test_str_split_part(duck):
    from conftest import assert_same

    t = pa.table({"s": ["a,b,c", "x", "", None]})
    duck.register("t", t)
    out = bt.from_arrow(t).select(
        p1=col("s").str.split_part(",", 1),
        p2=col("s").str.split_part(",", 2),
        p9=col("s").str.split_part(",", 9),
    )
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT split_part(s, ',', 1) p1, split_part(s, ',', 2) p2, "
            "split_part(s, ',', 9) p9 FROM t"
        ),
    )


def test_str_regexp_replace_all(duck):
    from conftest import assert_same

    t = pa.table({"s": ["a1b2c3", "no digits", None]})
    duck.register("t", t)
    out = bt.from_arrow(t).select(g=col("s").str.regexp_replace_all("[0-9]", "#"))
    assert_same(out.collect(), duck.sql("SELECT regexp_replace(s, '[0-9]', '#', 'g') g FROM t"))


# --- datetime -------------------------------------------------------------
def _dates():
    d = pa.array(["2024-02-15", "2023-02-15", "2024-12-31", None]).cast(pa.date32())
    return pa.table({"d": d})


def test_dt_leap_days_isoyear(duck):
    from conftest import assert_same

    t = _dates()
    duck.register("t", t)
    out = bt.from_arrow(t).select(
        leap=col("d").dt.is_leap_year(),
        dim=col("d").dt.days_in_month(),
        iy=col("d").dt.iso_year(),
    )
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT (extract('year' FROM d) % 4 = 0 AND "
            "(extract('year' FROM d) % 100 <> 0 OR extract('year' FROM d) % 400 = 0)) leap, "
            "extract('day' FROM (date_trunc('month', d) + INTERVAL 1 MONTH - INTERVAL 1 DAY)) dim, "
            "extract('isoyear' FROM d) iy FROM t"
        ),
    )


# --- list -----------------------------------------------------------------
def test_list_first_last_negative_get():
    # Structural (no DuckDB list literal parity needed): first/last/negative index.
    t = pa.table({"xs": [[10, 20, 30], [5], []]})
    out = (
        bt.from_arrow(t)
        .select(
            f=col("xs").list.get(0),
            l=col("xs").list.get(-1),
            penultimate=col("xs").list.get(-2),
        )
        .to_pydict()
    )
    assert out["f"] == [10, 5, None]
    assert out["l"] == [30, 5, None]
    assert out["penultimate"] == [20, None, None]
