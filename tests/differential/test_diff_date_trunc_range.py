"""`date_trunc(unit, col) = lit` filters match DuckDB after the sargable rewrite.

The `date_trunc_to_range` NORMALIZE rule rewrites the equality to a half-open range
on the raw column; the result MUST stay identical to DuckDB evaluating the truncation
directly — for every truncation unit, and whether or not the literal is aligned.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col


@pytest.fixture
def t(duck):
    ts = [
        dt.datetime(2021, 3, 14, 9, 26, 53) + dt.timedelta(days=i * 23, minutes=i * 17)
        for i in range(60)
    ]
    tbl = pa.table({"ts": pa.array(ts, type=pa.timestamp("us")), "v": list(range(60))})
    duck.register("t", tbl)
    return tbl


# One aligned literal per unit that actually selects rows from the fixture.
_ALIGNED = {
    "year": dt.datetime(2022, 1, 1),
    "month": dt.datetime(2021, 6, 1),
    "day": dt.datetime(2021, 3, 14),
    "hour": dt.datetime(2021, 3, 14, 9),
    "minute": dt.datetime(2021, 3, 14, 9, 26),
    "second": dt.datetime(2021, 3, 14, 9, 26, 53),
}


@pytest.mark.parametrize("unit", list(_ALIGNED))
def test_date_trunc_eq_filter_vs_duckdb(duck, t, unit):
    from conftest import assert_same

    lit = _ALIGNED[unit]
    out = bt.from_arrow(t).filter(col("ts").dt.truncate(unit) == lit).collect()
    expected = duck.sql(
        f"SELECT * FROM t WHERE date_trunc('{unit}', ts) = TIMESTAMP '{lit.isoformat(sep=' ')}'"
    )
    assert_same(out, expected)


def test_unaligned_literal_still_matches_duckdb(duck, t):
    # A month literal on the 15th is unaligned: the rewrite must not fire, and the
    # (empty) result must still equal DuckDB's.
    from conftest import assert_same

    lit = dt.datetime(2021, 6, 15)
    out = bt.from_arrow(t).filter(col("ts").dt.truncate("month") == lit).collect()
    expected = duck.sql(
        f"SELECT * FROM t WHERE date_trunc('month', ts) = TIMESTAMP '{lit.isoformat(sep=' ')}'"
    )
    assert_same(out, expected)


def test_date_column_day_trunc_vs_duckdb(duck):
    # A DATE column (not timestamp): day-trunc equality rewrites to a date range.
    from conftest import assert_same

    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i * 11) for i in range(40)]
    tbl = pa.table({"d": pa.array(dates, type=pa.date32()), "v": list(range(40))})
    duck.register("dt_tbl", tbl)
    lit = dt.date(2020, 3, 6)
    out = bt.from_arrow(tbl).filter(col("d").dt.truncate("day") == lit).collect()
    expected = duck.sql(
        f"SELECT * FROM dt_tbl WHERE date_trunc('day', d) = DATE '{lit.isoformat()}'"
    )
    assert_same(out, expected)
