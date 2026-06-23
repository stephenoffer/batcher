"""Date arithmetic: INTERVAL (DAY/WEEK), date_add/date_diff, CAST AS DATE/TIMESTAMP."""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "id": [1, 2, 3],
            "d": pa.array(
                [dt.date(2021, 1, 1), dt.date(2021, 6, 15), dt.date(2020, 12, 25)], pa.date32()
            ),
            "d2": pa.array(
                [dt.date(2021, 1, 10), dt.date(2021, 7, 1), dt.date(2021, 1, 1)], pa.date32()
            ),
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.mark.parametrize(
    "q",
    [
        # date ± INTERVAL returns a DATE here (DuckDB promotes to timestamp); a CAST
        # normalizes both engines to DATE so the comparison is value-for-value.
        "SELECT id, CAST(d + INTERVAL 5 DAY AS DATE) r FROM t",
        "SELECT id, CAST(d - INTERVAL 10 DAY AS DATE) r FROM t",
        "SELECT id, CAST(d + INTERVAL 2 WEEK AS DATE) r FROM t",
        "SELECT id, CAST(date_add(d, INTERVAL 3 DAY) AS DATE) r FROM t",
    ],
)
def test_date_interval(duck, t, q):
    from conftest import assert_same

    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT id, date_diff('day', d, d2) n FROM t",
        "SELECT id FROM t WHERE d + INTERVAL 7 DAY > DATE '2021-01-05'",
        "SELECT id, CAST(d AS DATE) r FROM t",
        "SELECT CAST('2021-03-15' AS DATE) r",
        "SELECT CAST('2021-03-15 10:30:00' AS TIMESTAMP) r",
    ],
)
def test_date_functions(duck, t, q):
    from conftest import assert_same

    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        # MONTH/YEAR calendar arithmetic now works. Batcher keeps the DATE type
        # (consistent with day intervals); DuckDB promotes date+month/year to
        # TIMESTAMP, so the queries CAST back to DATE to compare the *values*
        # (incl. end-of-month clamping: Jan 31 + 1 month → Feb 28).
        "SELECT CAST(d + INTERVAL 1 MONTH AS DATE) m FROM t",
        "SELECT CAST(d + INTERVAL 1 YEAR AS DATE) y FROM t",
        "SELECT CAST(d - INTERVAL 2 MONTH AS DATE) m FROM t",
        "SELECT CAST(CAST('2021-01-31' AS DATE) + INTERVAL 1 MONTH AS DATE) m",
    ],
)
def test_month_year_interval(duck, t, q):
    from conftest import assert_same

    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))
