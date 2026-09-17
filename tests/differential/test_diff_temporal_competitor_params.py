"""Each temporal parameter against the engine whose meaning it restores: Polars, Daft, Spark.

The DuckDB side of the same parameters is `test_diff_temporal_semantic_params.py`; this file
is the other half of the claim, that the non-default value really is the competitor's
answer. Polars and Daft run for real. There is no JVM here, so the Spark cases assert the
documented examples from the PySpark sources, each cited by function in
``python/pyspark/sql/functions/builtin.py`` (Spark 4.x), with ``extract``'s field table from
``sql/catalyst/.../expressions/datetimeExpressions.scala::DatePart.parseExtractField``.

A projection preserves row order, so the rows are compared positionally.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

_TS = [
    dt.datetime(2024, 2, 29, 13, 45, 30, 123456),
    dt.datetime(1969, 12, 31, 23, 59, 59, 500000),
    dt.datetime(2024, 3, 10, 2, 30),
    dt.datetime(2024, 2, 18),
    None,
]
_D = [dt.date(2024, 2, 29), dt.date(1969, 6, 1), dt.date(2024, 3, 10), dt.date(2024, 2, 18), None]
_D0 = [dt.date(2024, 1, 1), dt.date(1970, 1, 1), dt.date(2024, 3, 17), None, dt.date(2024, 1, 1)]


def _data() -> dict[str, list]:
    return {"ts": list(_TS), "d": list(_D), "d0": list(_D0)}


def _batcher(**exprs: bt.Expr) -> dict[str, list]:
    table = pa.table(
        {
            "ts": pa.array(_TS, pa.timestamp("us")),
            "d": pa.array(_D, pa.date32()),
            "d0": pa.array(_D0, pa.date32()),
        }
    )
    return bt.from_arrow(table).select(**exprs).to_pydict()


# --- Polars 1.40 ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pl():
    return pytest.importorskip("polars")


def _polars(pl, **exprs) -> dict[str, list]:
    frame = pl.DataFrame(_data(), schema={"ts": pl.Datetime("us"), "d": pl.Date, "d0": pl.Date})
    return frame.select(**exprs).to_dict(as_series=False)


def test_polars_truncate_keeps_a_date(pl):
    ours = _batcher(
        d=col("d").dt.truncate("month", preserve_type=True),
        ts=col("ts").dt.truncate("month", preserve_type=True),
    )
    theirs = _polars(pl, d=pl.col("d").dt.truncate("1mo"), ts=pl.col("ts").dt.truncate("1mo"))
    assert ours == theirs


def test_polars_month_start_and_month_end_keep_type_and_time(pl):
    ours = _batcher(
        ms_ts=col("ts").dt.month_start(keep_time=True),
        ms_d=col("d").dt.month_start(keep_time=True),
        me_ts=col("ts").dt.last_day(keep_time=True),
        me_d=col("d").dt.last_day(keep_time=True),
    )
    theirs = _polars(
        pl,
        ms_ts=pl.col("ts").dt.month_start(),
        ms_d=pl.col("d").dt.month_start(),
        me_ts=pl.col("ts").dt.month_end(),
        me_d=pl.col("d").dt.month_end(),
    )
    assert ours == theirs


def test_polars_epoch_units_are_timestamp_units(pl):
    ours = _batcher(
        us=col("ts").dt.timestamp("us"),
        ms=col("ts").dt.timestamp("ms"),
        s=col("ts").dt.timestamp("s"),
        ns=col("ts").dt.timestamp("ns"),
    )
    theirs = _polars(
        pl,
        us=pl.col("ts").dt.epoch(),
        ms=pl.col("ts").dt.epoch("ms"),
        s=pl.col("ts").dt.epoch("s"),
        ns=pl.col("ts").dt.epoch("ns"),
    )
    assert ours == theirs


def test_polars_weekday_is_iso(pl):
    ours = _batcher(r=col("d").dt.dayofweek(start="monday", base=1), w=col("d").dt.weekday())
    theirs = _polars(pl, r=pl.col("d").dt.weekday(), w=pl.col("d").dt.weekday())
    assert ours == theirs


def test_polars_to_string_default_is_this_strftime_per_type(pl):
    # The codemod template for `dt.to_string()`: the default rendering depends on the column's
    # type, so each type has its own pattern. Pinned here so the template cannot drift.
    ours = _batcher(
        ts=col("ts").dt.strftime("%Y-%m-%d %H:%M:%S%.6f"), d=col("d").dt.strftime("%Y-%m-%d")
    )
    theirs = _polars(pl, ts=pl.col("ts").dt.to_string(), d=pl.col("d").dt.to_string())
    assert ours == theirs


def test_polars_from_epoch_reads_a_column_name(pl):
    epochs = {"a": [0, 86_400, -1, None]}
    ours = bt.from_pydict(epochs).select(r=bt.from_epoch("a")).to_pydict()
    theirs = pl.DataFrame(epochs).select(r=pl.from_epoch("a")).to_dict(as_series=False)
    assert ours == theirs


# --- Daft 0.7.25 ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def daft():
    return pytest.importorskip("daft")


def _daft(daft, **exprs) -> dict[str, list]:
    return daft.from_pydict(_data()).select(*(e.alias(k) for k, e in exprs.items())).to_pydict()


def test_daft_day_of_week_and_datepart(daft):
    from daft import col as dcol
    from daft import functions as F

    ours = _batcher(
        d=col("d").dt.dayofweek(start="monday"),
        ts=col("ts").dt.dayofweek(start="monday"),
        part=col("d").dt.dayofweek(start="monday"),
    )
    theirs = _daft(
        daft,
        d=dcol("d").day_of_week(),
        ts=dcol("ts").day_of_week(),
        part=F.datepart("dayofweek", dcol("d")),
    )
    assert ours == theirs


def test_daft_date_diff_on_dates(daft):
    from daft import col as dcol
    from daft import functions as F

    ours = _batcher(r=col("d").dt.days_between(col("d0")), r2=col("d").dt.days_between("d0"))
    theirs = _daft(daft, r=F.date_diff(dcol("d"), dcol("d0")), r2=F.datediff(dcol("d"), dcol("d0")))
    assert ours == theirs


def test_daft_to_unix_epoch_defaults_to_seconds(daft):
    from daft import col as dcol

    # Row 1 is 1969-12-31 23:59:59.5, excluded from the seconds column on purpose: Daft
    # truncates a pre-1970 sub-second part toward zero (0) where `timestamp("s")` floors (-1),
    # the DuckDB `epoch` rule. That residual is asserted below rather than hidden.
    ours = _batcher(
        s=col("ts").dt.timestamp("s"),
        ms=col("ts").dt.timestamp("ms"),
        us=col("ts").dt.timestamp("us"),
        ns=col("ts").dt.timestamp("ns"),
        d=col("d").dt.timestamp("s"),
    )
    theirs = _daft(
        daft,
        s=dcol("ts").to_unix_epoch(),
        ms=dcol("ts").to_unix_epoch("ms"),
        us=dcol("ts").to_unix_epoch("us"),
        ns=dcol("ts").to_unix_epoch("ns"),
        d=dcol("d").to_unix_epoch(),
    )
    for name in ("ms", "us", "ns", "d"):
        assert ours[name] == theirs[name], name
    assert [v for i, v in enumerate(ours["s"]) if i != 1] == [
        v for i, v in enumerate(theirs["s"]) if i != 1
    ]
    assert (ours["s"][1], theirs["s"][1]) == (-1, 0)


def test_daft_partition_days_is_a_date(daft):
    from daft import col as dcol

    ours = _batcher(
        ts=bt.partition_days("ts", as_date=True), d=bt.partition_days("d", as_date=True)
    )
    theirs = _daft(daft, ts=dcol("ts").partition_days(), d=dcol("d").partition_days())
    assert ours == theirs


# --- Spark 4.x, documented examples --------------------------------------------------------


def _spark_dates() -> bt.Dataset:
    return bt.from_pydict({"dt": ["2015-04-08", "2024-10-31"]})


def test_spark_dayofweek_weekday_dayname_monthname_examples():
    # builtin.py::dayofweek -> 4, 5; ::weekday -> 2, 3; ::dayname -> Wed, Thu;
    # ::monthname -> Apr, Oct.
    out = _spark_dates().select(
        dayofweek=col("dt").dt.dayofweek(start="sunday", base=1),
        weekday=col("dt").dt.dayofweek(start="monday", base=0),
        dayname=col("dt").dt.dayname(abbreviated=True),
        monthname=col("dt").dt.monthname(abbreviated=True),
    )
    assert out.to_pydict() == {
        "dayofweek": [4, 5],
        "weekday": [2, 3],
        "dayname": ["Wed", "Thu"],
        "monthname": ["Apr", "Oct"],
    }


def test_spark_datediff_example():
    # builtin.py::datediff / ::date_diff: datediff('2015-04-08', '2015-05-10') is -32.
    ds = bt.from_pydict({"d1": ["2015-04-08"], "d2": ["2015-05-10"]})
    out = ds.select(a=col("d1").dt.days_between("d2"), b=col("d2").dt.days_between("d1"))
    assert out.to_pydict() == {"a": [-32], "b": [32]}


def test_spark_trunc_example_returns_a_date():
    # builtin.py::trunc: trunc('1997-02-28', 'year') -> 1997-01-01, 'mon' -> 1997-02-01, a DATE.
    ds = bt.from_pydict({"dt": [dt.date(1997, 2, 28)]})
    out = ds.select(
        y=col("dt").dt.truncate("year", preserve_type=True),
        m=col("dt").dt.truncate("month", preserve_type=True),
    )
    assert out.to_pydict() == {"y": [dt.date(1997, 1, 1)], "m": [dt.date(1997, 2, 1)]}


def test_spark_extract_dayofweek_and_fractional_second():
    # datetimeExpressions.scala::DatePart.parseExtractField: DAYOFWEEK/DOW is DayOfWeek
    # (Sunday = 1), SECOND is SecondWithFraction. builtin.py::extract shows 15.000000 for
    # 2015-04-08 13:08:15, a Wednesday.
    ds = bt.from_pydict({"ts": [dt.datetime(2015, 4, 8, 13, 8, 15, 250000)]})
    out = ds.select(
        dow=col("ts").dt.dayofweek(base=1),
        second=col("ts").dt.second() + col("ts").dt.microsecond() / 1_000_000,
    )
    assert out.to_pydict() == {"dow": [4], "second": [15.25]}


def test_spark_from_unixtime_example():
    # builtin.py::from_unixtime: 1428476400 renders as '2015-04-08 00:00:00' with the session
    # time zone America/Los_Angeles. Batcher's instants are UTC, so the session zone is an
    # explicit conversion, and the Java pattern yyyy-MM-dd HH:mm:ss is a strftime pattern.
    ds = bt.from_pydict({"unix_time": [1428476400]})
    rendered = (
        bt.from_epoch("unix_time")
        .dt.convert_timezone("UTC", "America/Los_Angeles")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    assert ds.select(r=rendered).to_pydict() == {"r": ["2015-04-08 00:00:00"]}
