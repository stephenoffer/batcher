"""The calendar accessors on ``.dt``, against DuckDB and against the calendar itself.

``test_diff_temporal.py`` covers extraction and arithmetic. This module covers the
*calendar* half of the accessor -- the period boundaries (``month_start``,
``quarter_end``, ``year_start``, ...), the boundary predicates (``is_month_end``,
``is_quarter_start``, ...), the names (``day_name``, ``month_name``) and the formatter --
none of which any test called.

Every one of these has a DuckDB equivalent, so that is the oracle. Where DuckDB spells
the same idea differently (``date_trunc('quarter')`` for ``quarter_start``, ``last_day``
for ``month_end``) the SQL is written from the *definition* rather than from Batcher's
output, which is what makes the comparison independent.

The fixture is chosen around the dates that separate a correct implementation from a
plausible one: a leap day, the last instant of a year, a quarter end that is also a month
end, a Sunday, and a null. A fixture of arbitrary Tuesdays passes for every wrong
boundary rule there is.
"""

from __future__ import annotations

import datetime as dt

import pytest

import batcher as bt

pytestmark = pytest.mark.differential

duckdb = pytest.importorskip("duckdb")

#: Timestamps chosen so each boundary predicate is true for at least one row and false
#: for at least one, and so the leap year, the year boundary and a weekend all appear.
STAMPS = [
    dt.datetime(2024, 1, 1, 0, 0, 0),  # Monday, year/quarter/month start
    dt.datetime(2024, 2, 29, 13, 45, 30),  # leap day, Thursday
    dt.datetime(2023, 12, 31, 23, 59, 59),  # Sunday, year/quarter/month end
    dt.datetime(2024, 3, 31, 12, 0, 0),  # Sunday, quarter and month end
    dt.datetime(2024, 7, 4, 9, 15, 0),  # Thursday, mid-month
    dt.datetime(2025, 6, 30, 0, 0, 0),  # Monday, quarter and month end
    None,
]


@pytest.fixture(scope="module")
def duck():
    """A DuckDB connection holding the fixture as a one-column table."""
    con = duckdb.connect()
    con.execute("CREATE TABLE t (i INTEGER, ts TIMESTAMP)")
    con.executemany("INSERT INTO t VALUES (?, ?)", [(i, s) for i, s in enumerate(STAMPS)])
    return con


@pytest.fixture(scope="module")
def ds():
    """The same fixture as a Batcher dataset."""
    return bt.from_pydict({"i": list(range(len(STAMPS))), "t": STAMPS})


def _duck_column(con, expression: str) -> list:
    """One DuckDB expression over the fixture, in fixture order."""
    return [row[0] for row in con.execute(f"SELECT {expression} FROM t ORDER BY i").fetchall()]


#: ``(accessor, DuckDB expression)`` for the period boundaries. Each DuckDB side is the
#: standard spelling of the definition, not a translation of Batcher's implementation.
BOUNDARIES = [
    ("month_start", "date_trunc('month', ts)"),
    ("quarter_start", "date_trunc('quarter', ts)"),
    ("year_start", "date_trunc('year', ts)"),
    ("quarter_end", "date_trunc('quarter', ts) + INTERVAL 3 MONTH - INTERVAL 1 DAY"),
    ("year_end", "make_timestamp(year(ts), 12, 31, 0, 0, 0)"),
]


@pytest.mark.parametrize(("accessor", "expression"), BOUNDARIES)
def test_period_boundary_matches_duckdb(ds, duck, accessor, expression):
    """Every period boundary, as a timestamp truncated to the day."""
    got = ds.select(v=getattr(bt.col("t").dt, accessor)()).to_pydict()["v"]
    want = _duck_column(duck, expression)
    assert got == want, f"{accessor}: {got} vs {want}"


#: ``(accessor, DuckDB expression)`` for the boundary predicates.
PREDICATES = [
    ("is_month_start", "ts::DATE = date_trunc('month', ts)::DATE"),
    ("is_month_end", "ts::DATE = last_day(ts)"),
    ("is_quarter_start", "ts::DATE = date_trunc('quarter', ts)::DATE"),
    (
        "is_quarter_end",
        "ts::DATE = (date_trunc('quarter', ts) + INTERVAL 3 MONTH - INTERVAL 1 DAY)::DATE",
    ),
    ("is_year_start", "month(ts) = 1 AND day(ts) = 1"),
    ("is_year_end", "month(ts) = 12 AND day(ts) = 31"),
    ("is_weekday", "isodow(ts) <= 5"),
    ("is_business_day", "isodow(ts) <= 5"),
]


@pytest.mark.parametrize(("accessor", "expression"), PREDICATES)
def test_boundary_predicate_matches_duckdb(ds, duck, accessor, expression):
    """Every boundary predicate, including on the leap day and the year boundary."""
    got = ds.select(v=getattr(bt.col("t").dt, accessor)()).to_pydict()["v"]
    want = _duck_column(duck, expression)
    assert got == want, f"{accessor}: {got} vs {want}"


def test_the_predicates_agree_with_the_boundaries_they_name(ds):
    """Cross-check: ``is_month_start`` must be true exactly where ``month_start`` is the day.

    The two families are computed independently inside the engine, so this catches a
    change to one that forgets the other -- which the DuckDB comparison above would not,
    since it checks each against the oracle rather than against its partner.
    """
    got = ds.select(
        t=bt.col("t"),
        starts=bt.col("t").dt.is_month_start(),
        start=bt.col("t").dt.month_start(),
        q_starts=bt.col("t").dt.is_quarter_start(),
        q_start=bt.col("t").dt.quarter_start(),
        y_starts=bt.col("t").dt.is_year_start(),
        y_start=bt.col("t").dt.year_start(),
    ).to_pydict()
    for i, stamp in enumerate(STAMPS):
        if stamp is None:
            assert got["starts"][i] is None
            continue
        assert got["starts"][i] == (stamp.date() == got["start"][i].date())
        assert got["q_starts"][i] == (stamp.date() == got["q_start"][i].date())
        assert got["y_starts"][i] == (stamp.date() == got["y_start"][i].date())


def test_day_and_month_names_match_duckdb(ds, duck):
    """The English names, which must be full words rather than abbreviations."""
    got = ds.select(day=bt.col("t").dt.day_name(), month=bt.col("t").dt.month_name()).to_pydict()
    assert got["day"] == _duck_column(duck, "dayname(ts)")
    assert got["month"] == _duck_column(duck, "monthname(ts)")


def test_ordinal_day_and_days_in_month_match_duckdb(ds, duck):
    """Day of year and month length, both of which the leap day is the test for."""
    got = ds.select(
        ordinal=bt.col("t").dt.ordinal_day(), length=bt.col("t").dt.daysinmonth()
    ).to_pydict()
    assert got["ordinal"] == _duck_column(duck, "dayofyear(ts)")
    assert got["length"] == _duck_column(duck, "day(last_day(ts))")
    assert got["ordinal"][1] == 60, "2024-02-29 is the 60th day of a leap year"
    assert got["length"][1] == 29, "February 2024 has 29 days"


def test_week_of_month_counts_from_the_first_of_the_month(ds):
    """``ceil(day / 7)``, recomputed here rather than taken from the engine.

    DuckDB has no equivalent -- its ``week()`` is the ISO week of the *year* -- so the
    definition is the oracle. Written this way the assertion fails if the implementation
    ever switches to counting whole weeks from the first Monday, which is the other
    common reading and a different answer for most days.
    """
    got = ds.select(v=bt.col("t").dt.week_of_month()).to_pydict()["v"]
    for stamp, value in zip(STAMPS, got, strict=True):
        if stamp is None:
            assert value is None
            continue
        assert value == -(-stamp.day // 7), f"week of month for {stamp}"


def test_to_string_matches_duckdbs_strftime(ds, duck):
    """The default ISO shape and an explicit format, both against ``strftime``."""
    got = ds.select(
        iso=bt.col("t").dt.to_string(),
        custom=bt.col("t").dt.to_string("%Y/%m/%d %H:%M"),
    ).to_pydict()
    assert got["iso"] == _duck_column(duck, "strftime(ts, '%Y-%m-%dT%H:%M:%S')")
    assert got["custom"] == _duck_column(duck, "strftime(ts, '%Y/%m/%d %H:%M')")


def test_every_calendar_accessor_nulls_on_a_null_input(ds):
    """One null row through the whole family, since a boundary rule can easily forget it."""
    accessors = [name for name, _ in BOUNDARIES] + [name for name, _ in PREDICATES]
    accessors += ["day_name", "month_name", "ordinal_day", "daysinmonth", "week_of_month"]
    projections = {f"c{i}": getattr(bt.col("t").dt, name)() for i, name in enumerate(accessors)}
    projections["fmt"] = bt.col("t").dt.to_string()
    got = ds.select(**projections).to_pydict()
    null_row = len(STAMPS) - 1
    for key, values in got.items():
        assert values[null_row] is None, f"{key} did not null on a null timestamp"


def test_the_calendar_accessors_work_on_a_date_column_too(duck):
    """A ``date32`` column, not just a timestamp -- the two take different Rust paths."""
    dates = [s.date() if s is not None else None for s in STAMPS]
    got = (
        bt.from_pydict({"d": dates})
        .select(
            start=bt.col("d").dt.month_start(),
            name=bt.col("d").dt.day_name(),
            ordinal=bt.col("d").dt.ordinal_day(),
            ends=bt.col("d").dt.is_month_end(),
        )
        .to_pydict()
    )
    con = duckdb.connect()
    con.execute("CREATE TABLE d (i INTEGER, v DATE)")
    con.executemany("INSERT INTO d VALUES (?, ?)", [(i, v) for i, v in enumerate(dates)])
    rows = con.execute(
        "SELECT date_trunc('month', v), dayname(v), dayofyear(v), v = last_day(v) FROM d ORDER BY i"
    ).fetchall()
    assert [r[1] for r in rows] == got["name"]
    assert [r[2] for r in rows] == got["ordinal"]
    assert [r[3] for r in rows] == got["ends"]
    for ours, theirs in zip(got["start"], [r[0] for r in rows], strict=True):
        if ours is None:
            assert theirs is None
            continue
        assert (ours.year, ours.month, ours.day) == (theirs.year, theirs.month, theirs.day)


def test_streaming_agrees_with_collect(ds):
    """The calendar family through ``iter_batches``, which is a different execution path."""
    projected = ds.select(
        start=bt.col("t").dt.quarter_start(),
        ends=bt.col("t").dt.is_year_end(),
        name=bt.col("t").dt.month_name(),
    )
    collected = projected.to_pydict()
    streamed: dict[str, list] = {"start": [], "ends": [], "name": []}
    for batch in projected.iter_batches():
        for key in streamed:
            streamed[key].extend(batch.column(key).to_pylist())
    assert streamed == collected
