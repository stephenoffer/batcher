"""Time zones on tz-aware columns, held to DuckDB (ICU) and `zoneinfo`.

Three defects lived here and every one of them passed its tests, because the tests only
ever handed the engine a *naive* column:

* `dt.convert_timezone` read the stored UTC micros of an aware column as a wall clock in
  `from_tz`, so 07:30Z with ``from_tz="Asia/Tokyo"`` came back as a different instant.
* `convert_timezone`, `offset_by` and `truncate` declared one type and returned another
  (aware declared, naive returned, or the reverse), which is the schema-contract hole the
  device tier relies on being closed.
* `truncate` and `offset_by` worked on the UTC calendar, so a New York evening truncated to
  the next day and "one day later" across a DST change was 24 elapsed hours, where DuckDB
  (and Polars) give the local day.

DuckDB is run with ``SET TimeZone`` to the column's zone, which is the session reading of a
TIMESTAMPTZ that matches a zoned Arrow column. Instants are compared as epoch microseconds
so the comparison cannot be fooled by how either side renders a zone.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_tables_equal
from batcher import col

pytestmark = pytest.mark.differential

NY = "America/New_York"
_UTC = dt.UTC

# Instants either side of both 2024 New York DST changes, one late evening (which the UTC
# calendar puts on the next day), a pre-1970 one, and a null.
_INSTANTS = [
    dt.datetime(2024, 3, 9, 17, 0, tzinfo=_UTC),  # 12:00 EST, the eve of spring-forward
    dt.datetime(2024, 3, 10, 7, 30, tzinfo=_UTC),  # 03:30 EDT, just after the gap
    dt.datetime(2024, 3, 11, 3, 30, tzinfo=_UTC),  # 23:30 EDT on 03-10
    dt.datetime(2024, 11, 3, 5, 30, tzinfo=_UTC),  # 01:30 EDT, first pass of the overlap
    dt.datetime(2024, 11, 3, 6, 30, tzinfo=_UTC),  # 01:30 EST, second pass
    dt.datetime(1969, 7, 20, 20, 17, 40, tzinfo=_UTC),
    None,
]


def _aware(tz: str) -> bt.Dataset:
    return bt.from_arrow(pa.table({"ts": pa.array(_INSTANTS, pa.timestamp("us", tz=tz))}))


def _duck(duck, tz: str):
    duck.sql(f"SET TimeZone='{tz}'")
    rows = [(None if t is None else t.isoformat(),) for t in _INSTANTS]
    values = ", ".join("(NULL)" if r[0] is None else f"(TIMESTAMPTZ '{r[0]}')" for r in rows)
    duck.sql(f"CREATE OR REPLACE TABLE t AS SELECT * FROM (VALUES {values}) v(ts)")
    return duck


@pytest.mark.parametrize("to_tz", ["Asia/Tokyo", "UTC", NY, "Asia/Kolkata"])
def test_convert_timezone_on_an_aware_column_keeps_the_instant(duck, to_tz):
    got = _aware("UTC").select(r=col("ts").dt.convert_timezone("UTC", to_tz)).collect()
    expected = _duck(duck, "UTC").sql(f"SELECT ts AT TIME ZONE '{to_tz}' AS r FROM t")
    assert_same(got, expected)


def test_convert_timezone_tokyo_regression():
    """07:30Z is 16:30 in Tokyo whatever the column's zone label is spelled as."""
    utc = bt.from_arrow(
        pa.table({"ts": pa.array([dt.datetime(2024, 3, 10, 7, 30)], pa.timestamp("us", "UTC"))})
    )
    out = utc.select(r=col("ts").dt.convert_timezone("UTC", "Asia/Tokyo")).to_pydict()
    assert out == {"r": [dt.datetime(2024, 3, 10, 16, 30)]}
    # An equivalent spelling of the column's zone is the same zone.
    same = utc.select(r=col("ts").dt.convert_timezone("Etc/UTC", "Asia/Tokyo")).to_pydict()
    assert same == out
    # Before the fix this silently answered 2024-03-09 17:30, a different instant.
    with pytest.raises(Exception, match="already tz-aware"):
        utc.select(r=col("ts").dt.convert_timezone("Asia/Tokyo", NY)).collect()


@pytest.mark.parametrize(
    "unit", ["year", "quarter", "month", "week", "day", "hour", "minute", "second"]
)
def test_truncate_on_an_aware_column_uses_the_local_calendar(duck, unit):
    # Row 3 is 01:30 EDT, the first pass through the November overlap, and is held out:
    # see the next test for the one place the two engines deliberately differ.
    keep = [i for i in range(len(_INSTANTS)) if i != 3]
    got = _aware(NY).select(r=col("ts").dt.truncate(unit).dt.epoch_us()).collect()
    expected = _duck(duck, NY).sql(f"SELECT epoch_us(date_trunc('{unit}', ts)) AS r FROM t")
    # Both sides keep the fixture's row order (a projection of one scan), so the held-out
    # row is the same row on each.
    expected_rows = expected.to_arrow_table().take(keep)
    assert_tables_equal(got.take(keep), expected_rows, ordered=True)


def test_truncating_inside_a_dst_overlap_keeps_the_instants_own_offset():
    """A pinned divergence from DuckDB, not an accident.

    01:30 EDT (05:30Z) truncated to the hour is 01:00, a wall clock that happens twice. This
    engine keeps the offset the instant had, so the answer is 01:00 EDT (05:00Z), at or
    before the input like every other truncation. DuckDB (ICU) answers 01:00 EST (06:00Z),
    which is *after* the instant it truncated. The second pass, 01:30 EST, agrees on both.
    """
    first, second = _INSTANTS[3], _INSTANTS[4]
    ds = bt.from_arrow(pa.table({"ts": pa.array([first, second], pa.timestamp("us", tz=NY))}))
    out = ds.select(r=col("ts").dt.truncate("hour").dt.epoch_us()).to_pydict()["r"]
    hour = 3_600_000_000
    assert out == [
        int(first.timestamp()) * 10**6 // hour * hour,
        int(second.timestamp()) * 10**6 // hour * hour,
    ]


@pytest.mark.parametrize(
    ("offset", "interval"),
    [
        ("1d", "INTERVAL 1 DAY"),
        ("-1d", "INTERVAL '-1 day'"),
        ("1mo", "INTERVAL 1 MONTH"),
        ("1w", "INTERVAL 7 DAY"),
        ("1h", "INTERVAL 1 HOUR"),
        ("1d2h", "INTERVAL 1 DAY + INTERVAL 2 HOUR"),
    ],
)
def test_offset_by_on_an_aware_column_shifts_local_calendar_days(duck, offset, interval):
    got = _aware(NY).select(r=col("ts").dt.offset_by(offset).dt.epoch_us()).collect()
    expected = _duck(duck, NY).sql(f"SELECT epoch_us(ts + {interval}) AS r FROM t")
    assert_same(got, expected)


@pytest.mark.parametrize(
    "fn", ["year", "month", "day", "hour", "minute", "dayofyear", "quarter", "week"]
)
def test_fields_of_an_aware_column_are_read_in_its_zone(duck, fn):
    duck_fn = {"dayofyear": "dayofyear"}.get(fn, fn)
    got = _aware(NY).select(r=getattr(col("ts").dt, fn)()).collect()
    expected = _duck(duck, NY).sql(f"SELECT {duck_fn}(ts) AS r FROM t")
    assert_same(got, expected)


def test_dayname_and_strftime_agree_with_hour_about_the_day():
    """`dayname`/`last_day`/`strftime` cast to a naive timestamp and read the UTC day."""
    out = (
        _aware(NY)
        .select(
            name=col("ts").dt.dayname(),
            text=col("ts").dt.strftime("%Y-%m-%d %H:%M"),
            last=col("ts").dt.last_day(),
        )
        .to_pydict()
    )
    zone = ZoneInfo(NY)
    local = [None if t is None else t.astimezone(zone) for t in _INSTANTS]
    assert out["name"] == [None if t is None else t.strftime("%A") for t in local]
    assert out["text"] == [None if t is None else t.strftime("%Y-%m-%d %H:%M") for t in local]
    assert out["last"][2] == dt.date(2024, 3, 31)


def test_declared_types_match_the_engine_on_aware_columns():
    ds = _aware(NY)
    exprs = {
        "conv": col("ts").dt.convert_timezone(NY, "UTC"),
        "repl": col("ts").dt.replace_timezone("Asia/Tokyo"),
        "strip": col("ts").dt.replace_timezone(None),
        "trunc": col("ts").dt.truncate("day"),
        "off": col("ts").dt.offset_by("1d"),
        "off_h": col("ts").dt.offset_by("1h"),
        "win": bt.window(col("ts"), "1h"),
        "months": col("ts").dt.add_months(1),
        "days": bt.date_add(col("ts"), col("ts").dt.day()),
    }
    q = ds.select(**exprs)
    declared = q.schema
    actual = q.to_arrow().schema
    for name in exprs:
        assert declared.field(name).type == actual.field(name).type, name
    assert actual.field("trunc").type == pa.timestamp("us", tz=NY)
    assert actual.field("conv").type == pa.timestamp("us")
    assert actual.field("repl").type == pa.timestamp("us", tz="Asia/Tokyo")


def _localize(naive: dt.datetime, tz: str, fold: int) -> dt.datetime:
    return naive.replace(tzinfo=ZoneInfo(tz), fold=fold).astimezone(_UTC).replace(tzinfo=None)


_GAP = dt.datetime(2024, 3, 10, 2, 30)  # never happens in New York
_OVERLAP = dt.datetime(2024, 11, 3, 1, 30)  # happens twice
_PLAIN = dt.datetime(2024, 6, 1, 9, 0)


def _utc_of(expr) -> list:
    ds = bt.from_pydict({"d": [_OVERLAP, _PLAIN, None]})
    return ds.select(r=expr.dt.convert_timezone(NY, "UTC")).to_pydict()["r"]


def test_replace_timezone_localizes_the_wall_clock_like_zoneinfo():
    early = _utc_of(col("d").dt.replace_timezone(NY, ambiguous="earliest"))
    late = _utc_of(col("d").dt.replace_timezone(NY, ambiguous="latest"))
    nulled = _utc_of(col("d").dt.replace_timezone(NY, ambiguous="null"))
    assert early == [_localize(_OVERLAP, NY, 0), _localize(_PLAIN, NY, 0), None]
    assert late == [_localize(_OVERLAP, NY, 1), _localize(_PLAIN, NY, 0), None]
    assert nulled == [None, _localize(_PLAIN, NY, 0), None]
    with pytest.raises(Exception, match="ambiguous"):
        bt.from_pydict({"d": [_OVERLAP]}).select(r=col("d").dt.replace_timezone(NY)).collect()


def test_nonexistent_policies_on_a_dst_gap():
    gap = bt.from_pydict({"d": [_GAP]})
    shifted = gap.select(
        r=col("d").dt.replace_timezone(NY, nonexistent="shift_forward").dt.epoch_us()
    ).to_pydict()["r"]
    # The first instant after the gap: 03:00 EDT, 07:00Z.
    assert shifted == [int(dt.datetime(2024, 3, 10, 7, 0, tzinfo=_UTC).timestamp()) * 10**6]
    assert gap.select(r=col("d").dt.replace_timezone(NY, nonexistent="null")).to_pydict() == {
        "r": [None]
    }
    with pytest.raises(Exception, match="does not exist"):
        gap.select(r=col("d").dt.replace_timezone(NY)).collect()
    # convert_timezone keeps its historical null default, and takes the same policies.
    assert gap.select(r=col("d").dt.convert_timezone(NY, "UTC")).to_pydict() == {"r": [None]}
    assert gap.select(
        r=col("d").dt.convert_timezone(NY, "UTC", nonexistent="shift_forward")
    ).to_pydict() == {"r": [dt.datetime(2024, 3, 10, 7, 0)]}


def test_replace_timezone_matches_duckdb_on_unambiguous_wall_clocks(duck):
    clocks = [dt.datetime(2024, 1, 15, 9, 0), dt.datetime(2024, 7, 4, 23, 59, 59), None]
    ds = bt.from_pydict({"d": clocks})
    got = ds.select(r=col("d").dt.replace_timezone(NY).dt.epoch_us()).collect()
    duck.sql("SET TimeZone='UTC'")
    rows = ", ".join("(NULL)" if c is None else f"(TIMESTAMP '{c.isoformat()}')" for c in clocks)
    expected = duck.sql(f"SELECT epoch_us(d AT TIME ZONE '{NY}') AS r FROM (VALUES {rows}) v(d)")
    assert_same(got, expected)


def test_replace_then_strip_round_trips_the_wall_clock():
    clocks = [dt.datetime(2024, 1, 15, 9, 0), None]
    ds = bt.from_pydict({"d": clocks})
    out = ds.select(
        r=col("d").dt.replace_timezone("Asia/Kolkata").dt.replace_timezone(None)
    ).to_pydict()
    assert out == {"r": clocks}


def test_policy_vocabulary_is_validated_at_plan_time():
    with pytest.raises(bt.PlanError, match="ambiguous"):
        col("d").dt.replace_timezone(NY, ambiguous="first")
    with pytest.raises(bt.PlanError, match="nonexistent"):
        col("d").dt.convert_timezone("UTC", NY, nonexistent="later")
