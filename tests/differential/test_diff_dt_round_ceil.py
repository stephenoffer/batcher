"""`.dt.ceil` and `.dt.round`, checked against pandas and DuckDB.

`.dt.floor`/`.dt.truncate` bias every instant one way. Bucketing a series for a plot or
aligning two feeds sampled off each other's grid wants the nearest boundary instead, and
closing a half-open bucket wants the next one — so the accessor needs all three.

Two oracles, because neither covers the whole unit vocabulary:

* **pandas** has `dt.ceil`/`dt.round` for the fixed-length units (second through day), which
  is the arithmetic half.
* **DuckDB** has `date_trunc` plus interval arithmetic, which expresses the calendar units
  (month, quarter, year) that pandas' fixed-frequency rounding cannot — and expresses them
  through DuckDB's own calendar, not a restatement of Batcher's.

The half-way tie is asserted on its own, because the two references disagree there: pandas
breaks it to the even boundary and Batcher rounds up.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.differential

pd = pytest.importorskip("pandas")

# Instants chosen to land either side of every boundary, plus two exactly *on* one (which
# must be left alone by `ceil`) and one in a leap February. None sits exactly half way
# through a unit: that tie is where pandas and Batcher deliberately differ, and it gets its
# own assertion below rather than being smuggled into the reference comparison.
_STAMPS = [
    dt.datetime(2024, 2, 15, 13, 45, 30, 400000),
    dt.datetime(2024, 2, 15, 13, 0, 0),
    dt.datetime(2024, 1, 1, 0, 0, 0),
    dt.datetime(2024, 2, 29, 23, 59, 59),
    dt.datetime(2023, 11, 7, 6, 12, 3),
    dt.datetime(2024, 8, 20, 18, 40, 0),
    dt.datetime(1969, 7, 20, 20, 17, 40),  # before the epoch: flooring must go toward -inf
]

_FIXED_UNITS = {"second": "s", "minute": "min", "hour": "h", "day": "D"}
_CALENDAR_UNITS = {"month": "MONTH", "quarter": "QUARTER", "year": "YEAR"}


def _table():
    return pa.table({"d": pa.array(_STAMPS)})


def _batcher(method: str, unit: str):
    return bt.from_arrow(_table()).select(r=getattr(bt.col("d").dt, method)(unit)).to_pydict()["r"]


@pytest.mark.parametrize("unit", sorted(_FIXED_UNITS))
@pytest.mark.parametrize("method", ["ceil", "round"])
def test_a_fixed_unit_matches_pandas(unit, method):
    got = _batcher(method, unit)
    rounded = getattr(pd.Series(_STAMPS).dt, method)(_FIXED_UNITS[unit])
    want = [t.to_pydatetime() for t in rounded]
    assert got == want, f".dt.{method}({unit!r})"


@pytest.mark.parametrize("unit", sorted(_CALENDAR_UNITS))
def test_a_calendar_ceil_matches_duckdbs_own_calendar(duck, unit):
    duck.register("s", _table())
    duck_unit = _CALENDAR_UNITS[unit]
    # DuckDB's own formulation: the truncation, advanced by one whole unit unless the
    # instant already sits on a boundary.
    want = [
        r[0]
        for r in duck.sql(
            f"""
            SELECT CASE WHEN date_trunc('{duck_unit}', d) = d
                        THEN d
                        ELSE date_trunc('{duck_unit}', d) + INTERVAL 1 {duck_unit}
                   END
            FROM s
            """
        ).fetchall()
    ]
    assert _batcher("ceil", unit) == want


@pytest.mark.parametrize("unit", sorted(_CALENDAR_UNITS))
def test_a_calendar_round_picks_the_nearer_boundary(duck, unit):
    duck.register("s", _table())
    duck_unit = _CALENDAR_UNITS[unit]
    # Nearer in *elapsed time*, which is what makes a variable-length unit well defined:
    # mid-February is nearer to March than to February, and mid-March is not.
    want = [
        r[0]
        for r in duck.sql(
            f"""
            WITH b AS (
                SELECT d,
                       date_trunc('{duck_unit}', d) AS lo,
                       date_trunc('{duck_unit}', d) + INTERVAL 1 {duck_unit} AS hi
                FROM s
            )
            SELECT CASE WHEN epoch_us(d) - epoch_us(lo) < epoch_us(hi) - epoch_us(d)
                        THEN lo ELSE hi END
            FROM b
            """
        ).fetchall()
    ]
    assert _batcher("round", unit) == want


@pytest.mark.parametrize("unit", sorted(_FIXED_UNITS) + sorted(_CALENDAR_UNITS))
def test_ceil_leaves_an_instant_already_on_a_boundary_alone(unit):
    """The property that separates `ceil` from "floor then always advance"."""
    on_boundary = bt.from_pydict({"d": [dt.datetime(2024, 1, 1, 0, 0, 0)]})
    got = on_boundary.select(r=bt.col("d").dt.ceil(unit)).to_pydict()["r"]
    assert got == [dt.datetime(2024, 1, 1, 0, 0, 0)], unit


def test_an_exact_half_rounds_up():
    """The one place the references disagree, pinned so it cannot drift silently.

    pandas breaks a half to the even boundary; Batcher rounds up, which is the everyday
    reading and the only rule that does not depend on which boundary you started from.
    """
    ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 30)]})
    assert ds.select(r=bt.col("d").dt.round("hour")).to_pydict()["r"] == [
        dt.datetime(2024, 2, 15, 14, 0)
    ]


def test_rounding_and_flooring_bracket_the_instant():
    """`floor <= round <= ceil` for every unit and instant — a property, not a fixture."""
    for unit in sorted(_FIXED_UNITS) + sorted(_CALENDAR_UNITS):
        lo = _batcher("floor", unit)
        mid = _batcher("round", unit)
        hi = _batcher("ceil", unit)
        for i, stamp in enumerate(_STAMPS):
            assert lo[i] <= stamp <= hi[i], f"{unit} brackets {stamp}"
            assert mid[i] in (lo[i], hi[i]), f"{unit} rounds {stamp} to a boundary"


def test_a_unit_with_no_step_is_rejected_by_name():
    with pytest.raises(PlanError, match=r"\.dt\.ceil\('millisecond'\)"):
        bt.col("d").dt.ceil("millisecond")
    with pytest.raises(PlanError, match=r"\.dt\.round\('decade'\)"):
        bt.col("d").dt.round("decade")


def test_a_null_instant_stays_null():
    ds = bt.from_pydict({"d": [None, dt.datetime(2024, 2, 15, 13, 45)]})
    assert ds.select(r=bt.col("d").dt.ceil("hour")).to_pydict()["r"] == [
        None,
        dt.datetime(2024, 2, 15, 14, 0),
    ]
    assert ds.select(r=bt.col("d").dt.round("hour")).to_pydict()["r"] == [
        None,
        dt.datetime(2024, 2, 15, 14, 0),
    ]


# --- clock-time filters ---------------------------------------------------------------
#
# `time_of_day`/`is_between_time` answer "when in the day", which comparing timestamps
# cannot: the date dominates the ordering. pandas spells the filter `between_time` on the
# index, and it is the oracle here — including for the window that wraps past midnight,
# which is the case a naive `hour() >= 22 & hour() <= 2` silently returns nothing for.


def _clock_frame():
    stamps = [
        dt.datetime(2024, 2, 15, 0, 0),
        dt.datetime(2024, 2, 15, 1, 0),
        dt.datetime(2024, 2, 15, 9, 0),
        dt.datetime(2024, 2, 15, 9, 30),
        dt.datetime(2024, 2, 15, 17, 0),
        dt.datetime(2024, 2, 15, 20, 0),
        dt.datetime(2024, 2, 16, 23, 30),
    ]
    return stamps, pa.table({"d": pa.array(stamps), "i": pa.array(range(len(stamps)))})


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("09:00", "17:00"),  # an ordinary session
        ("00:00", "23:59"),  # everything
        ("09:30", "09:30"),  # a single instant
        ("22:00", "02:00"),  # wraps past midnight
        ("23:30", "00:00"),  # wraps by half an hour
        ("17:00", "09:00"),  # the complement of the session
    ],
    ids=["session", "all_day", "instant", "wrap", "narrow_wrap", "complement"],
)
def test_is_between_time_matches_pandas_between_time(start, end):
    stamps, table = _clock_frame()
    got = (
        bt.from_arrow(table)
        .select(hit=bt.col("d").dt.is_between_time(start, end))
        .to_pydict()["hit"]
    )
    index = pd.DatetimeIndex(stamps)
    kept = set(pd.Series(range(len(stamps)), index=index).between_time(start, end).tolist())
    want = [i in kept for i in range(len(stamps))]
    assert got == want, f"between_time({start!r}, {end!r})"


def test_a_wrapping_window_is_not_empty():
    """The whole reason this is a method: the obvious hour comparison returns nothing."""
    _, table = _clock_frame()
    ds = bt.from_arrow(table)
    wrapped = ds.filter(bt.col("d").dt.is_between_time("22:00", "02:00")).count()
    naive = ds.filter((bt.col("d").dt.hour() >= 22) & (bt.col("d").dt.hour() <= 2)).count()
    assert naive == 0, "the naive comparison is empty, which is the trap"
    assert wrapped == 3, "00:00, 01:00 and 23:30 are all inside the window"


def test_time_of_day_discards_the_date():
    stamps, table = _clock_frame()
    got = bt.from_arrow(table).select(t=bt.col("d").dt.time_of_day()).to_pydict()["t"]
    want = [(s.hour * 3600 + s.minute * 60 + s.second) * 1_000_000 + s.microsecond for s in stamps]
    assert got == want
    # Two instants on different days at the same clock time agree, which is the point.
    same = bt.from_pydict({"d": [dt.datetime(2024, 1, 1, 6, 0), dt.datetime(2030, 7, 4, 6, 0)]})
    values = same.select(t=bt.col("d").dt.time_of_day()).to_pydict()["t"]
    assert values[0] == values[1] == 6 * 3600 * 1_000_000


@pytest.mark.parametrize("bound", ["9", "9:5", "25:00", "09:60", "noon", ""])
def test_a_bad_clock_time_is_rejected_at_plan_time(bound):
    with pytest.raises(PlanError, match="clock time"):
        bt.col("d").dt.is_between_time(bound, "17:00")
