"""``ts +/- INTERVAL <literal>`` against DuckDB, across the whole literal vocabulary.

The unit half of this — that the *text* parses to the months/days/microseconds DuckDB
says it means — is `tests/unit/test_sql_interval_literals.py`. This half is the one that
matters for a user: that shifting a real timestamp by the parsed literal lands on the
instant DuckDB lands on, including where a calendar month clamps at end-of-month and where
a fractional count spills into a finer component.

Three spellings raised a bare ``ValueError: invalid literal for int() with base 10`` out
of `Session.sql()` before the lowering was given a real parser — a raw Python error, for
standard SQL, on the most ordinary temporal query there is:

* compound, ``INTERVAL '1 day 3 hours'``;
* fractional, ``INTERVAL '1.5 hours'``;
* clock, ``INTERVAL '04:05:06'``.

A fourth, the PostgreSQL abbreviations (``mon``, ``yr``, ``hrs``, ``mins``, ``secs``,
``ms``, ``us``), was declined by name.

The operand here is a **TIMESTAMP**. On a DATE operand Batcher keeps the DATE for a
whole-day or whole-month shift where DuckDB promotes to TIMESTAMP — same instant, narrower
type — so those cases are compared through an explicit cast below rather than being
dropped.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered

pytestmark = pytest.mark.differential

#: Every literal spelling the front end accepts. Kept as one list so the parametrization
#: below covers `+` and `-`, on a timestamp and (via a cast) on a date, for each.
LITERALS = [
    "1 day",
    "3 days",
    "1 month",
    "1 quarter",
    "1 year",
    "1 decade",
    "1 week",
    "2 hours",
    "90 minutes",
    "500 milliseconds",
    "1000 microseconds",
    "1 mon",
    "2 mons",
    "1 yr",
    "3 d",
    "2 w",
    "2 hrs",
    "2 mins",
    "2 secs",
    "2 ms",
    "2 us",
    "1.5 months",
    "0.5 years",
    "1.5 days",
    "0.25 days",
    "2.5 weeks",
    "1.5 hours",
    "1 day 3 hours",
    "2 years 3 months",
    "1 y 2 mons 3 d",
    "1 day -3 hours",
    "1 week 2 days 3 hours 4 minutes 5 seconds",
    "-1 day",
    "04:05:06",
]


@pytest.fixture
def t(duck):
    # A month-end date (2024-01-31) so a month shift has to clamp, a leap day, and a
    # timestamp with a sub-second component so a microsecond shift is visible.
    table = pa.table(
        {
            "id": pa.array([0, 1, 2, 3], pa.int64()),
            "d": pa.array(
                [
                    dt.date(2024, 1, 31),
                    dt.date(2024, 2, 29),
                    dt.date(2023, 12, 1),
                    None,
                ],
                pa.date32(),
            ),
            "ts": pa.array(
                [
                    dt.datetime(2024, 1, 31, 23, 59, 59, 999999),
                    dt.datetime(2024, 2, 29, 0, 0, 0),
                    dt.datetime(2023, 12, 1, 6, 30, 15, 250000),
                    None,
                ],
                pa.timestamp("us"),
            ),
        }
    )
    duck.register("t", table)
    return table


def _run(t, duck, sql):
    assert_same_ordered(bt.sql(sql, t=bt.from_arrow(t)).collect(), duck.sql(sql))


@pytest.mark.parametrize("literal", LITERALS)
@pytest.mark.parametrize("op", ["+", "-"])
def test_timestamp_shift_matches_duckdb(t, duck, op, literal):
    _run(t, duck, f"SELECT id, ts {op} INTERVAL '{literal}' AS e FROM t ORDER BY id")


@pytest.mark.parametrize("literal", LITERALS)
@pytest.mark.parametrize("op", ["+", "-"])
def test_date_shift_matches_duckdb(t, duck, op, literal):
    """The same shift on a DATE.

    Compared through an explicit cast because Batcher keeps the DATE for a whole-day or
    whole-month shift (`DateOffset` is type-preserving) where DuckDB always promotes to
    TIMESTAMP. The *instant* is what this asserts; the type difference is deliberate and
    pinned separately by `test_a_whole_day_shift_keeps_the_date_type`.
    """
    _run(
        t,
        duck,
        f"SELECT id, CAST(d {op} INTERVAL '{literal}' AS TIMESTAMP) AS e FROM t ORDER BY id",
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id, CAST(d + INTERVAL 3 DAY AS TIMESTAMP) AS e FROM t ORDER BY id",
        "SELECT id, CAST(d + INTERVAL '3' DAY AS TIMESTAMP) AS e FROM t ORDER BY id",
        "SELECT id, ts + INTERVAL 2 HOURS AS e FROM t ORDER BY id",
        "SELECT id, ts - INTERVAL 90 MINUTE AS e FROM t ORDER BY id",
        "SELECT id, CAST(date_add(d, 5) AS TIMESTAMP) AS e FROM t ORDER BY id",
    ],
)
def test_unit_beside_the_number_still_works(t, duck, sql):
    """The ``INTERVAL 3 DAY`` form, where the unit is a separate parse slot."""
    _run(t, duck, sql)


def _result_type(t, sql):
    return bt.sql(sql, t=bt.from_arrow(t)).schema.field("e").type


def test_a_whole_day_shift_keeps_the_date_type(t):
    """A DATE shifted by whole days is still a DATE — narrower than DuckDB's TIMESTAMP.

    Deliberate: `DateOffset` is type-preserving, and a date plus a whole number of days is
    a date. The promotion happens only when the literal carries a sub-day component, which
    a Date32 cannot represent — which is also the rule that decides whether a *compound*
    literal promotes.
    """
    assert pa.types.is_date(_result_type(t, "SELECT d + INTERVAL '1 day' AS e FROM t"))
    assert pa.types.is_date(_result_type(t, "SELECT d + INTERVAL '1 month' AS e FROM t"))
    assert pa.types.is_date(_result_type(t, "SELECT d + INTERVAL '1 y 2 mons 3 d' AS e FROM t"))
    assert pa.types.is_timestamp(_result_type(t, "SELECT d + INTERVAL '1 hour' AS e FROM t"))
    assert pa.types.is_timestamp(_result_type(t, "SELECT d + INTERVAL '1 day 3 hours' AS e FROM t"))
    # A timestamp operand stays a timestamp whatever the literal carries.
    assert pa.types.is_timestamp(_result_type(t, "SELECT ts + INTERVAL '1 day' AS e FROM t"))


def test_a_compound_literal_is_the_sum_of_its_terms(t, duck):
    """`'1 day 3 hours'` is one shift, not two roundings."""
    _run(
        t,
        duck,
        "SELECT id, ts + INTERVAL '1 day 3 hours' AS a, "
        "(ts + INTERVAL '1 day') + INTERVAL '3 hours' AS b FROM t ORDER BY id",
    )


def test_an_unknown_unit_is_declined_by_name(t):
    with pytest.raises(NotImplementedError, match="fortnight"):
        bt.sql("SELECT ts + INTERVAL '2 fortnights' AS e FROM t", t=bt.from_arrow(t)).collect()


def test_interval_in_a_filter_and_a_group_key(t, duck):
    """The literal is a plan-time constant, so it works anywhere an expression does."""
    _run(
        t,
        duck,
        "SELECT id, ts FROM t WHERE ts > TIMESTAMP '2023-01-01' - INTERVAL '1 y 6 mons' "
        "ORDER BY id",
    )
    assert_same(
        bt.sql(
            "SELECT date_trunc('month', ts + INTERVAL '1 day 12 hours') AS m, count(*) AS n "
            "FROM t GROUP BY 1",
            t=bt.from_arrow(t),
        ).collect(),
        duck.sql(
            "SELECT date_trunc('month', ts + INTERVAL '1 day 12 hours') AS m, count(*) AS n "
            "FROM t GROUP BY 1"
        ),
    )
