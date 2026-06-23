"""`.dt.offset_by(by)` — calendar+fixed date/timestamp shifting vs DuckDB intervals.

Checked against DuckDB ``date + INTERVAL`` (calendar months clamp end-of-month;
days and sub-day units are exact).
"""

from __future__ import annotations

import datetime

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def _dates():
    d = pa.array(["2024-01-31", "2023-02-28", "2024-12-15", None]).cast(pa.date32())
    return pa.table({"d": d})


def test_offset_months_and_days_vs_duckdb(duck):
    from conftest import assert_same

    t = _dates()
    duck.register("t", t)
    out = bt.from_arrow(t).select(
        plus_mo=col("d").dt.offset_by("1mo"),
        plus_yr=col("d").dt.offset_by("1y"),
        plus_days=col("d").dt.offset_by("10d"),
        minus_wk=col("d").dt.offset_by("-2w"),
    )
    # Batcher preserves Date32 (like Polars `dt.offset_by`); DuckDB's `date +
    # INTERVAL` promotes to TIMESTAMP, so cast its result back to DATE to compare.
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT (d + INTERVAL 1 MONTH)::DATE plus_mo, (d + INTERVAL 1 YEAR)::DATE plus_yr, "
            "(d + INTERVAL 10 DAY)::DATE plus_days, (d - INTERVAL 14 DAY)::DATE minus_wk FROM t"
        ),
    )


def test_offset_timestamp_subday_vs_duckdb(duck):
    from conftest import assert_same

    ts = pa.array(
        [datetime.datetime(2024, 1, 1, 10, 0, 0), datetime.datetime(2024, 2, 29, 23, 30, 0)]
    ).cast(pa.timestamp("us"))
    t = pa.table({"t": ts})
    duck.register("t2", t)
    out = bt.from_arrow(t).select(
        p=col("t").dt.offset_by("1h30m"),
        q=col("t").dt.offset_by("1mo2d"),
    )
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT (t + INTERVAL 90 MINUTE) p, (t + INTERVAL 1 MONTH + INTERVAL 2 DAY) q FROM t2"
        ),
    )


def test_offset_subday_on_date_raises():
    from batcher._internal.errors import BackendError

    t = _dates()
    with pytest.raises((BackendError, Exception)):
        bt.from_arrow(t).select(x=col("d").dt.offset_by("1h")).collect()


def test_parse_offset_unit():
    from batcher.plan.expr_ir.namespaces import parse_offset

    assert parse_offset("1mo15d") == (1, 15, 0)
    assert parse_offset("-3d") == (0, -3, 0)
    assert parse_offset("2h30m") == (0, 0, 2 * 3_600_000_000 + 30 * 60_000_000)
    assert parse_offset("1y") == (12, 0, 0)
    assert parse_offset("1w") == (0, 7, 0)
    with pytest.raises(ValueError, match="invalid offset"):
        parse_offset("1x")
    with pytest.raises(ValueError, match="invalid offset"):
        parse_offset("")
