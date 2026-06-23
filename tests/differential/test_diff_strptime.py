"""`str.to_datetime` / `str.to_date` — parse string columns to Timestamp/Date.

Locks in parity with DuckDB ``try_strptime``: values that do not match the format
become NULL (the safe-ingest behavior for dirty date columns), and well-formed
values parse identically.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def test_to_datetime_parses_and_nulls_bad(duck):
    """Parse ``%Y-%m-%d %H:%M:%S`` strings; junk → NULL (vs DuckDB try_strptime)."""
    from conftest import assert_same

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
    from conftest import assert_same

    t = pa.table({"s": ["2024-06-23 12:00:00", "bad", "2024-06-23 23:59:59"]})
    duck.register("ts2", t)
    out = bt.from_arrow(t).select(d=col("s").str.to_datetime("%Y-%m-%d %H:%M:%S")).collect()
    assert_same(out, duck.sql("SELECT try_strptime(s, '%Y-%m-%d %H:%M:%S') d FROM ts2"))


def test_to_date_parses_iso(duck):
    """`to_date` parses date-only strings to Date32 (junk → NULL)."""
    from conftest import assert_same

    t = pa.table({"s": ["2024-02-15", "2023-12-31", "nope", None]})
    duck.register("td", t)
    out = bt.from_arrow(t).select(d=col("s").str.to_date()).collect()
    assert_same(
        out,
        duck.sql("SELECT try_cast(try_strptime(s, '%Y-%m-%d') AS DATE) d FROM td"),
    )
