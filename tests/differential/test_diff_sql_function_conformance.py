"""SQL scalar functions whose DuckDB spelling and Batcher's disagreed on an edge.

Every case here is a *silent* disagreement found by running the same SQL text through both
engines: a plausible value of the wrong sign, the wrong unit, the wrong end of a list, or a
zero where the answer was unknown. None of them raised, and none of them is visible without
an oracle, which is why they are pinned as a family rather than left to the per-feature
suites they each belong to.

The families:

* ``fmod`` is not a spelling of ``mod`` — it takes the divisor's sign, and returns NaN on a
  zero divisor where ``mod`` returns NULL.
* ``era`` of a NULL date is NULL, not 0 (a ``when/otherwise`` on a null comparison takes the
  else branch, which reported a missing date as BCE).
* ``strlen`` counts bytes, ``length`` counts characters — the same only for ASCII.
* A negative subscript counts from the end, and sqlglot's 0-basing has to be undone there.
* ``list_reverse_sort`` keeps NULLs last, so it is not ``reverse(list_sort(...))``.
* ``make_timestamp`` keeps a fractional second instead of truncating it away.
* ``date_part('microsecond'/'millisecond')`` scales the *whole* seconds field.
* Casting a timestamp to text uses a space and a de-padded fraction.
"""

from __future__ import annotations

import datetime

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "f": [-2.25, 7.0, -7.0, 5.0, None],
            "g": [4.0, -4.0, 4.0, 0.0, 2.0],
            "d": [
                datetime.date(2021, 1, 2),
                None,
                datetime.date(1969, 12, 31),
                datetime.date(2024, 2, 29),
                datetime.date(1900, 1, 1),
            ],
            "s": ["Ünicode", "abc", "", None, "  pad  "],
            "li": [[1, 2, 3], [7], [4, None, 6], None, []],
            "ts": [
                datetime.datetime(2021, 1, 2, 3, 4, 5),
                datetime.datetime(2021, 1, 2, 3, 4, 5, 500000),
                datetime.datetime(1969, 12, 31, 23, 59, 59, 999999),
                None,
                datetime.datetime(1969, 7, 20, 20, 17, 40, 1),
            ],
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.mark.differential
@pytest.mark.parametrize(
    "expr",
    [
        # `fmod` carries the divisor's sign; `mod` carries the dividend's.
        "fmod(f, g)",
        "mod(f, g)",
        "fmod(f, 3.0)",
        # A null date has no era, and no year to compare against zero.
        "era(d)",
        # Bytes, not characters.
        "strlen(s)",
        "length(s)",
        # Negative subscripts count from the end.
        "li[-1]",
        "li[-2]",
        "li[-3]",
        "li[1]",
        "li[3]",
        # Descending list sort keeps nulls at the back.
        "list_reverse_sort(li)",
        "list_sort(li)",
        # A fractional second survives.
        "make_timestamp(2021, 1, 1, 0, 0, 0.5)",
        "make_timestamp(2021, 1, 1, 13, 45, 30)",
        # The seconds field scaled, not the sub-second remainder alone.
        "date_part('microsecond', ts)",
        "date_part('millisecond', ts)",
        "EXTRACT(microsecond FROM ts)",
        # A space separator and a de-padded fraction.
        "ts::VARCHAR",
        "d::VARCHAR",
        # `date_part('epoch')` is a DOUBLE that keeps the fraction; `epoch()` is the
        # integer-seconds spelling. Pointing the first at the second dropped sub-second
        # precision from every timestamp.
        "date_part('epoch', ts)",
        "EXTRACT(epoch FROM ts)",
        "date_part('epoch', d)",
        # Truncated toward zero, not floored — they differ only before 1970.
        "epoch_ms(ts)",
        "epoch_us(ts)",
        "epoch(ts)",
        # Fields the extract table never listed, each with a numbering of its own.
        "date_part('yearweek', d)",
        "date_part('weekday', d)",
        "date_part('era', d)",
        # A DATE has no sub-second part; reading its raw integer gave one out of the
        # day index.
        "date_part('microsecond', d)",
        "date_part('millisecond', d)",
    ],
)
def test_scalar_matches_duckdb(t, duck, expr):
    query = f"SELECT {expr} AS r FROM t"
    assert_same(bt.sql(query, t=bt.from_arrow(t)).collect(), duck.sql(query))


@pytest.mark.differential
def test_fmod_by_zero_is_nan(t):
    """A zero divisor gives NaN, as DuckDB does — not a null and not an error."""
    got = bt.sql("SELECT fmod(f, g) AS r FROM t", t=bt.from_arrow(t)).to_pydict()["r"]
    # Row 3 divides by zero; NaN is not equal to itself, which is how it is asserted.
    assert got[3] != got[3]


@pytest.mark.differential
@pytest.mark.parametrize("part", ["microsecond", "millisecond", "nanosecond"])
def test_subsecond_components_are_non_negative_before_1970(part):
    """A pre-epoch instant has a *negative* micro count, but its components do not.

    `hour`/`minute`/`second` floor the epoch and read correctly; the sub-second accessors
    took a truncated remainder, so `1969-07-20 20:17:40.000001` reported -999999
    microseconds past the second. Every docstring in the family promises `[0, 999999]`.
    """
    rows = [
        datetime.datetime(1969, 7, 20, 20, 17, 40, 1),
        datetime.datetime(1969, 12, 31, 23, 59, 59, 999999),
        datetime.datetime(1900, 1, 1, 10, 20, 30, 500000),
        datetime.datetime(2021, 1, 2, 3, 4, 5, 123456),
    ]
    ds = bt.from_arrow(pa.table({"ts": rows}))
    got = ds.select(r=getattr(bt.col("ts").dt, part)()).to_pydict()["r"]
    assert all(v >= 0 for v in got), got
    scale = {"microsecond": 1, "millisecond": 1000, "nanosecond": 1 / 1000}[part]
    assert got == [int(r.microsecond / scale) for r in rows]


@pytest.mark.differential
def test_timestamp_text_round_trips_through_duckdb(t, duck):
    """The rendered text parses back to the same instant, so the format is not merely close."""
    query = "SELECT ts::VARCHAR::TIMESTAMP AS r FROM t"
    assert_same(bt.sql(query, t=bt.from_arrow(t)).collect(), duck.sql(query))
