"""Coverage for the Phase 2 temporal functions.

`date_part`/`date_add`/`date_sub` dispatch onto the existing `.dt` accessor and are
checked against DuckDB; `now`/`current_date` bind a literal at plan-build time, so
they are unit-checked for shape (not differential — wall-clock isn't reproducible).
"""

from __future__ import annotations

import datetime

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, current_date, date_add, date_part, date_sub, now

pytestmark = pytest.mark.differential


def _dates():
    return pa.table({"d": pa.array([datetime.date(2024, 6, 23), datetime.date(2020, 2, 29), None])})


def test_date_part_matches_duckdb(duck):
    from conftest import assert_same

    duck.register("t", _dates())
    out = (
        bt.from_arrow(_dates())
        .select(
            y=date_part("year", col("d")),
            m=date_part("month", col("d")),
            dow=date_part("dow", col("d")),
            doy=date_part("doy", col("d")),
        )
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT date_part('year', d) AS y, date_part('month', d) AS m, "
            "date_part('dow', d) AS dow, date_part('doy', d) AS doy FROM t"
        ),
    )


def test_date_add_sub_matches_duckdb(duck):
    from conftest import assert_same

    duck.register("t", _dates())
    out = (
        bt.from_arrow(_dates()).select(a=date_add(col("d"), 10), s=date_sub(col("d"), 5)).collect()
    )
    # DuckDB `date + int` adds days and stays a DATE (INTERVAL would promote to TIMESTAMP).
    assert_same(out, duck.sql("SELECT d + 10 AS a, d - 5 AS s FROM t"))


def test_now_and_current_date_bind_literals():
    # Bound once at build time → a constant column, equal across all rows.
    ds = bt.from_pydict({"x": [1, 2, 3]})
    n = ds.select(t=now()).collect().to_pydict()["t"]
    d = ds.select(d=current_date()).collect().to_pydict()["d"]
    assert len(set(n)) == 1 and isinstance(n[0], datetime.datetime)
    assert len(set(d)) == 1 and d[0] == datetime.date.today()
