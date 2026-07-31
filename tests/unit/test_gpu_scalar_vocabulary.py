"""The GPU translator's scalar vocabulary, checked against the CPU engine on the host backend.

Every entry in the translated vocabulary is an independent claim that one named function means
the same thing here as in the engine, so every entry needs its own case. These cover the ones
whose absence used to send a whole plan to the CPU engine — the calendar `date_trunc` units,
the year-derived date fields, the reciprocal trigonometry, and the two-argument math functions
— plus the two places the translation returned the right number in the *wrong column*.

The oracle is `ds.collect()`, the native engine, and the backend under test is pandas, which
models the device faithfully enough to be worth running here (`gpu_plan.backend`). A GPU is
only *where* a translated plan runs, never *what* it computes, so a host replay checks the same
code a device executes.
"""

from __future__ import annotations

import contextlib
import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.core.gpu_plan import DfBackend, gpu_plan_ops
from batcher.core.gpu_plan.execute import run_chain

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def be():
    import pandas as pd

    return DfBackend(pd)


def _timestamps() -> pa.Table:
    """Instants chosen so every calendar rule has a case that would fail without it.

    A leap year and a common year past February (the quarter-start day-of-year shifts by one),
    a January 1 and a December 31 (the ISO year disagrees with the calendar year at both ends),
    a Sunday and a Monday (the ISO week starts Monday), a month end, and a pre-1970 instant,
    where flooring and truncating disagree about which midnight an instant belongs to.
    """
    return pa.table(
        {
            "t": pa.array(
                [
                    dt.datetime(2024, 2, 29, 13, 45, 30, 123_456),  # leap day
                    dt.datetime(2023, 3, 1, 0, 0, 0),  # common year, past February
                    dt.datetime(2024, 1, 1, 0, 0, 0),  # a Monday, and an ISO year boundary
                    dt.datetime(2023, 12, 31, 23, 59, 59, 999_999),  # a Sunday, ISO year 2023
                    dt.datetime(2021, 7, 4, 6, 30, 0),  # a Sunday mid-quarter
                    dt.datetime(1969, 7, 20, 20, 17, 40),  # pre-epoch
                    dt.datetime(2000, 10, 31, 12, 0, 0),  # month end, century boundary
                    None,
                ],
                pa.timestamp("us"),
            )
        }
    )


def _calendar_sweep() -> pa.Table:
    """Every month-end and mid-month across years chosen to break a naive calendar rule.

    1900 is not a leap year and 2000 is, which is the rule most hand-rolled leap tests get
    wrong; 1583 is just inside the Gregorian calendar; 1969 and 1970 straddle the epoch, where
    flooring and truncating disagree; 2100 and 2400 are the next two century turns.
    """
    instants = []
    for year in (1583, 1899, 1900, 1969, 1970, 1999, 2000, 2023, 2024, 2100, 2400):
        for month in range(1, 13):
            for day in (1, 15, 28, 29, 30, 31):
                # A day the month does not have is simply not a case.
                with contextlib.suppress(ValueError):
                    instants.append(dt.datetime(year, month, day, 13, 45, 30, 123_456))
    instants.append(None)
    return pa.table({"t": pa.array(instants, pa.timestamp("us"))})


def _numbers() -> pa.Table:
    return pa.table(
        {
            "x": pa.array([1.5, -2.5, 0.25, -0.75, 3.0, 0.0, None], pa.float64()),
            "y": pa.array([2.0, 4.0, -1.0, 0.5, 1.0, 3.0, 1.0], pa.float64()),
            "i": pa.array([12, -18, 7, 100, 2**53 + 1, 0, None], pa.int64()),
            "j": pa.array([18, 12, 3, 250, 3, 5, 2], pa.int64()),
        }
    )


def _rows(table: pa.Table) -> list[tuple]:
    """Row tuples with floats canonicalized, so a last-bit difference is not a failure.

    The transcendentals are a different `libm` here than in the engine and are not required by
    IEEE to be correctly rounded, so agreeing to the last bit is not something the two promise
    each other. Twelve significant figures is far tighter than any real disagreement and far
    looser than one unit in the last place.
    """

    def canon(v):
        return float(f"{v:.12e}") if isinstance(v, float) and v == v else v

    return [tuple(canon(v) for v in r) for r in zip(*table.to_pydict().values(), strict=True)]


def _translated(ds, table: pa.Table, be) -> pa.Table:
    """`ds` replayed through the GPU translator on `be`, as Arrow."""
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None, "the translator declined a plan it is supposed to match"
    return be.to_arrow(run_chain(table, spec[1], be))


def _assert_matches_engine(ds, table: pa.Table, be) -> None:
    """The translated result equals the engine's, column types included.

    The type is asserted, not just the values, because a fan-out concatenates its shards and a
    shard that fell back to the CPU engine contributes the engine's type beside this one's. A
    translation that returns a double where the engine returns an int64 is not a wrong number,
    but it is a column that cannot be concatenated with its own peers.
    """
    expected = ds.collect()
    got = _translated(ds, table, be).select(expected.column_names)
    assert _rows(got) == _rows(expected)
    assert got.schema.types == expected.schema.types


# --- date_trunc ---------------------------------------------------------------------------


#: Every `date_trunc` unit the engine accepts. The translator covers all of them, so the list
#: is written out rather than sampled — a unit that silently stopped translating would be a
#: whole plan back on the CPU engine, with nothing in the result to say so.
TRUNC_UNITS = [
    "millennium",
    "century",
    "decade",
    "year",
    "quarter",
    "month",
    "week",
    "day",
    "hour",
    "minute",
    "second",
    "millisecond",
    "microsecond",
]


@pytest.mark.parametrize("unit", TRUNC_UNITS)
def test_date_trunc_matches_the_engine_for_every_unit(unit, be):
    """A `GROUP BY date_trunc(...)` is the most ordinary shape a time series has, and the
    calendar units used to send the whole plan to the CPU engine."""
    table = _timestamps()
    ds = bt.from_arrow(table).select(r=col("t").dt.truncate(unit))
    _assert_matches_engine(ds, table, be)


@pytest.mark.parametrize("unit", TRUNC_UNITS)
def test_date_trunc_matches_the_engine_across_four_centuries(unit, be):
    """Every month-end and mid-month across leap years, century turns and the epoch.

    The calendar units are built from `days_from_civil` rather than from a distance, so a
    single off-by-one in that construction would be systematic and quiet — right for most of a
    year and wrong at a boundary. A handful of hand-picked instants would not find it.
    """
    table = _calendar_sweep()
    ds = bt.from_arrow(table).select(r=col("t").dt.truncate(unit))
    _assert_matches_engine(ds, table, be)


def test_date_trunc_to_a_month_groups_on_the_device(be):
    """The shape the gap actually cost: a monthly rollup, translated end to end."""
    table = _timestamps()
    ds = (
        bt.from_arrow(table)
        .filter(col("t").is_not_null())
        .group_by(m=col("t").dt.truncate("month"))
        .agg(n=bt.count())
    )
    expected = ds.collect()
    got = _translated(ds, table, be).select(expected.column_names)
    assert sorted(_rows(got), key=repr) == sorted(_rows(expected), key=repr)


# --- the year-derived date fields ---------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda e: e.dt.isodow(),
        lambda e: e.dt.century(),
        lambda e: e.dt.decade(),
        lambda e: e.dt.millennium(),
        lambda e: e.dt.iso_year(),
        lambda e: e.dt.last_day(),
    ],
    ids=["isodow", "century", "decade", "millennium", "iso_year", "last_day"],
)
def test_date_field_matches_the_engine(build, be):
    table = _timestamps()
    ds = bt.from_arrow(table).select(r=build(col("t")))
    _assert_matches_engine(ds, table, be)


def test_a_date_column_truncates_from_its_own_midnight(be):
    """`date_trunc` over a DATE, whose integer representation counts days rather than micros.

    Reading those bits without casting to `timestamp[us]` first would scale every arithmetic
    by 86.4 billion. The result is far enough out to look corrupt rather than off-by-a-unit,
    which is exactly why it is worth a case: nothing else here would catch it.
    """
    table = pa.table(
        {
            "d": pa.array(
                [dt.date(2024, 2, 29), dt.date(1969, 7, 20), dt.date(2021, 1, 1), None],
                pa.date32(),
            )
        }
    )
    ds = bt.from_arrow(table).select(r=col("d").dt.truncate("month"))
    _assert_matches_engine(ds, table, be)


def test_epoch_reads_a_date_column_rather_than_declining_it(be):
    """`epoch()` over a DATE, which has no direct cast to int64 at all.

    Reading the bits without casting to `timestamp[us]` first raised, and a raise here is a
    silent CPU fallback for the whole plan — the accelerated path disappearing over an ordinary
    date column, with nothing in the result to say it had.
    """
    table = pa.table(
        {"d": pa.array([dt.date(2024, 2, 29), dt.date(1969, 7, 20), None], pa.date32())}
    )
    ds = bt.from_arrow(table).select(r=col("d").dt.epoch())
    _assert_matches_engine(ds, table, be)


# --- math -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda e: e.cot(),
        lambda e: e.sec(),
        lambda e: e.csc(),
        lambda e: e.rint(),
        lambda e: e.even(),
    ],
    ids=["cot", "sec", "csc", "rint", "even"],
)
def test_unary_math_matches_the_engine(build, be):
    table = _numbers()
    ds = bt.from_arrow(table).select(r=build(col("x")))
    _assert_matches_engine(ds, table, be)


def test_rint_and_round_disagree_on_a_half_and_both_are_right(be):
    """`rint` breaks halves to even and `round` breaks them away from zero.

    They are different functions in the engine, and translating either as the other is a
    half-unit error on exactly the values anyone rounds. `1.5` and `-2.5` are the witnesses:
    ties-to-even sends both to `2.0` and `-2.0`, ties-away sends them to `2.0` and `-3.0`.
    """
    table = _numbers()
    ds = bt.from_arrow(table).select(a=col("x").rint(), b=col("x").round())
    _assert_matches_engine(ds, table, be)
    got = _translated(ds, table, be).to_pydict()
    assert (got["a"][1], got["b"][1]) == (-2.0, -3.0)


@pytest.mark.parametrize(
    "build",
    [
        lambda a, b: bt.atan2(a, b),
        lambda a, b: bt.hypot(a, b),
        lambda a, b: bt.next_after(a, b),
    ],
    ids=["atan2", "hypot", "next_after"],
)
def test_binary_math_matches_the_engine(build, be):
    table = _numbers()
    ds = bt.from_arrow(table).select(r=build(col("x"), col("y")))
    _assert_matches_engine(ds, table, be)


@pytest.mark.parametrize("build", [bt.gcd, bt.lcm], ids=["gcd", "lcm"])
def test_integer_math_stays_integer(build, be):
    """`gcd`/`lcm` return a BIGINT in the engine, and the row above 2^53 is why it matters.

    A double round-trip would return `1.0` for `gcd(2^53+1, 3)` instead of `3` — a value the
    float path cannot represent, arriving as a plausible number rather than as an error.
    """
    table = _numbers()
    ds = bt.from_arrow(table).select(r=build(col("i"), col("j")))
    _assert_matches_engine(ds, table, be)


def test_rounding_an_integer_leaves_it_an_integer(be):
    """`round(bigint, n)` is a BIGINT in the engine; the translation used to widen it to double.

    That is the right number in the wrong column until 2^53, and the wrong number after it —
    and a fan-out cannot concatenate a shard that widened with one that fell back to the engine.
    """
    table = _numbers()
    ds = bt.from_arrow(table).select(a=col("i").round(), b=col("i").round(-2))
    _assert_matches_engine(ds, table, be)


def test_nulls_propagate_from_either_side_of_a_binary_function(be):
    """A row is null when *either* argument is, which one input's mask cannot express alone."""
    table = pa.table(
        {
            "a": pa.array([1.0, None, 3.0, None], pa.float64()),
            "b": pa.array([1.0, 2.0, None, None], pa.float64()),
        }
    )
    ds = bt.from_arrow(table).select(r=bt.hypot(col("a"), col("b")))
    _assert_matches_engine(ds, table, be)
    assert _translated(ds, table, be).to_pydict()["r"] == [pytest.approx(2**0.5), None, None, None]


# --- LIKE -----------------------------------------------------------------------------------


def _strings() -> pa.Table:
    values = ["foobar", "foo", "bar", "", "xfooy", None, "FOO"]
    return pa.table({"s": pa.array(values, pa.string())})


@pytest.mark.parametrize("pattern", ["foo", "foo%", "%foo", "%foo%", "%", "%%"])
def test_like_matches_the_engine_for_the_literal_patterns(pattern, be):
    """`LIKE` without `_` reduces to a literal substring test, which is what the engine itself
    reduces it to rather than a second regex dialect that happens to agree."""
    table = _strings()
    ds = bt.from_arrow(table).select(r=col("s").str.like(pattern))
    _assert_matches_engine(ds, table, be)


@pytest.mark.parametrize("pattern", ["f_o", "a%b", "%foo%bar%"])
def test_like_declines_the_patterns_that_need_a_regex_or_a_segment_scan(pattern, be):
    """These are the cases the engine itself needs more than a substring test for.

    A regex is the one construction that could not be checked here: the engine compiles Rust's,
    the host backend Python's and the device cuDF's, and the three disagree on exactly the
    classes a test over ASCII data would never reach.
    """
    from batcher.core.gpu_plan.backend import Unsupported

    table = _strings()
    ds = bt.from_arrow(table).select(r=col("s").str.like(pattern))
    with pytest.raises(Unsupported):
        _translated(ds, table, be)


def test_like_leaves_a_null_unknown(be):
    """`LIKE` on an unknown is unknown, including under a pattern that matches everything."""
    table = _strings()
    ds = bt.from_arrow(table).select(r=col("s").str.like("%"))
    _assert_matches_engine(ds, table, be)
    assert _translated(ds, table, be).to_pydict()["r"][5] is None
