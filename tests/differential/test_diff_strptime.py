"""`str.to_datetime` / `str.to_date` — parse string columns to Timestamp/Date.

Locks in parity with DuckDB ``try_strptime``: values that do not match the format
become NULL (the safe-ingest behavior for dirty date columns), and well-formed
values parse identically.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from _harness import assert_same
from batcher import col


def test_to_datetime_parses_and_nulls_bad(duck):
    """Parse ``%Y-%m-%d %H:%M:%S`` strings; junk → NULL (vs DuckDB try_strptime)."""
    t = pa.table(
        {
            "s": [
                "2024-02-15 13:45:30",
                "2023-12-31 00:00:00",
                "not a date",
                None,
                "2020-01-01 09:08:07",
            ]
        }
    )
    duck.register("ts", t)
    out = bt.from_arrow(t).select(d=col("s").str.to_datetime("%Y-%m-%d %H:%M:%S")).collect()
    assert_same(
        out,
        duck.sql("SELECT try_strptime(s, '%Y-%m-%d %H:%M:%S') d FROM ts"),
    )


def test_to_datetime_via_str_namespace(duck):
    """The accessor spelling ``col.str.to_datetime`` is equivalent."""
    t = pa.table({"s": ["2024-06-23 12:00:00", "bad", "2024-06-23 23:59:59"]})
    duck.register("ts2", t)
    out = bt.from_arrow(t).select(d=col("s").str.to_datetime("%Y-%m-%d %H:%M:%S")).collect()
    assert_same(out, duck.sql("SELECT try_strptime(s, '%Y-%m-%d %H:%M:%S') d FROM ts2"))


def test_to_date_parses_iso(duck):
    """`to_date` parses date-only strings to Date32 (junk → NULL)."""
    t = pa.table({"s": ["2024-02-15", "2023-12-31", "nope", None]})
    duck.register("td", t)
    out = bt.from_arrow(t).select(d=col("s").str.to_date()).collect()
    assert_same(
        out,
        duck.sql("SELECT try_cast(try_strptime(s, '%Y-%m-%d') AS DATE) d FROM td"),
    )


def test_a_partial_format_fills_the_fields_it_does_not_name(duck):
    """`%Y` alone, `%Y-%m`, an hour bucket — every one of these returned NULL.

    chrono's two whole-value parsers each demand a complete date or a complete instant, so
    a format naming only some fields matched neither and the column came back all-NULL.
    Silently: `strptime` is documented to null what it cannot parse, which is right for a
    malformed *value* and wrong for a format it simply could not represent. DuckDB fills
    the unnamed fields in, and every coarse rollup key is written this way.
    """
    t = pa.table({"s": ["1900", "2024", None, "not a year"]})
    duck.register("py", t)
    out = bt.from_arrow(t).select(d=col("s").str.to_datetime("%Y")).collect()
    assert_same(out, duck.sql("SELECT try_strptime(s, '%Y') d FROM py"))


def test_a_partial_format_keeps_the_fields_it_does_name(duck):
    """The narrower half: the old date-only fallback *threw the hour away*.

    `NaiveDate::parse_from_str` ignores time fields, so `'2024-03-05 13'` with
    `%Y-%m-%d %H` parsed as the date and answered midnight — a plausible instant thirteen
    hours from the right one, which is worse than the NULL the year-only case returned.
    """
    t = pa.table({"s": ["2024-03-05 13", "2024-03-05 00", None]})
    duck.register("ph", t)
    out = bt.from_arrow(t).select(d=col("s").str.to_datetime("%Y-%m-%d %H")).collect()
    assert_same(out, duck.sql("SELECT try_strptime(s, '%Y-%m-%d %H') d FROM ph"))


def test_a_format_with_no_year_stays_null(duck):
    """There is no instant to default to, so it must refuse rather than invent one."""
    t = pa.table({"s": ["12:30", None]})
    duck.register("pn", t)
    out = bt.from_arrow(t).select(d=col("s").str.to_datetime("%H:%M")).collect()
    assert out.to_pydict()["d"] == [None, None]
