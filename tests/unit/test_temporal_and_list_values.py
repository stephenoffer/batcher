"""What the `.dt` and `.list` accessors compute, against independent implementations.

The value half of `test_temporal_and_list_accessor_invariants.py`, and the reason it exists
is that the property half is not enough on its own. Those tests check null-preservation and
boundary handling, and a metric that computes the wrong number satisfies all of them -- which
is not hypothetical: `mean_line_length` passed every property test written about it while
reading up to 80% high (`docs/architecture/internals/text_metric_audit.md`).

The oracles are Python's `datetime` and `calendar` for the temporal accessors, and plain
Python for the list ones. Both are independent of the engine and of Arrow.

The timestamps are chosen to break things rather than to be typical: a leap day, the epoch,
the last second of 1999, the first day of 2000, and 2023-01-01 -- which is an ISO week-year
boundary, where the ISO year is 2022 and a naive implementation returns 2023.

Two conventions are pinned explicitly because a reasonable reference gets them wrong, and
both were wrong in this file's first draft:

* `std` is the **sample** standard deviation (n-1), matching SQL's `stddev` and Python's
  `statistics.stdev`. Python's `pstdev` is the population form and disagrees on every input.
* `sum` of an **empty list** is null, not 0. DuckDB's `list_sum([])` is null too; Python's
  `sum([])` is 0, and following Python here would have been following the wrong oracle.
"""

from __future__ import annotations

import calendar
import datetime as dtm
import statistics
from collections.abc import Callable

import pytest

import batcher as bt

pytestmark = pytest.mark.unit

#: Leap day, the epoch, the last second of 1999, the first instant of 2000, null, a
#: year-end, an ISO week-year boundary, and a mid-year afternoon.
TIMESTAMPS = [
    dtm.datetime(2024, 2, 29, 13, 45, 30),
    dtm.datetime(1970, 1, 1, 0, 0, 0),
    dtm.datetime(1999, 12, 31, 23, 59, 59),
    dtm.datetime(2000, 1, 1, 0, 0, 0),
    None,
    dtm.datetime(2024, 12, 31, 12, 0, 0),
    dtm.datetime(2023, 1, 1, 0, 0, 0),
    dtm.datetime(2024, 6, 15, 6, 30, 0),
]

#: Unsorted, empty, null, single, all-zero, signed, all-equal, and summing to zero.
LISTS = [
    [3.0, 1.0, 2.0],
    [],
    None,
    [1.0],
    [0.0, 0.0],
    [-1.0, 1.0],
    [2.0, 2.0, 2.0],
    [5.0, -5.0, 0.0],
]


def _empty_is_null(fn: Callable[[list], object]):
    """These reduce an empty list to null, as SQL does and Python does not."""
    return lambda values: None if not values else fn(values)


TEMPORAL_REFERENCE: dict[str, Callable[[dtm.datetime], object]] = {
    "year": lambda d: d.year,
    "month": lambda d: d.month,
    "day": lambda d: d.day,
    "hour": lambda d: d.hour,
    "minute": lambda d: d.minute,
    "second": lambda d: d.second,
    "day_of_year": lambda d: d.timetuple().tm_yday,
    "quarter": lambda d: (d.month - 1) // 3 + 1,
    "is_leap_year": lambda d: calendar.isleap(d.year),
    "days_in_month": lambda d: calendar.monthrange(d.year, d.month)[1],
    "week_of_year": lambda d: d.isocalendar()[1],
    "iso_year": lambda d: d.isocalendar()[0],
    "date": lambda d: d.date(),
}

LIST_REFERENCE: dict[str, Callable[[list], object]] = {
    "len": len,
    "lengths": len,
    # SQL, not Python: `list_sum([])` is null in DuckDB and 0 in Python.
    "sum": _empty_is_null(sum),
    "mean": _empty_is_null(lambda values: sum(values) / len(values)),
    "min": _empty_is_null(min),
    "max": _empty_is_null(max),
    "n_unique": lambda values: len(set(values)),
    "reverse": lambda values: values[::-1],
    "sort": sorted,
    "cum_sum": lambda values: [sum(values[: i + 1]) for i in range(len(values))],
    # Sample, not population, and undefined for a single value rather than 0.0: the sample
    # form divides by n-1. DuckDB's `stddev` over one row is NULL and Python's
    # `statistics.stdev([1.0])` raises, so 0.0 would have been neither engine's answer.
    "std": lambda values: statistics.stdev(values) if len(values) > 1 else None,
    "arg_max": _empty_is_null(lambda values: values.index(max(values))),
    "arg_min": _empty_is_null(lambda values: values.index(min(values))),
    "drop_nulls": lambda values: [v for v in values if v is not None],
}


def _temporal(name: str) -> list:
    column = getattr(bt.col("t").dt, name)()
    return bt.from_pydict({"t": TIMESTAMPS}).select(r=column).to_pydict()["r"]


def _listwise(name: str) -> list:
    column = getattr(bt.col("v").list, name)()
    return bt.from_pydict({"v": LISTS}).select(r=column).to_pydict()["r"]


def _equal(got: object, want: object) -> bool:
    if got is None or want is None:
        return got is None and want is None
    if isinstance(got, list) and isinstance(want, list):
        return len(got) == len(want) and all(_equal(a, b) for a, b in zip(got, want, strict=True))
    if isinstance(got, bool) or isinstance(want, bool):
        return got == want
    if isinstance(got, int | float) and isinstance(want, int | float):
        return abs(float(got) - float(want)) < 1e-9
    return str(got)[:10] == str(want)[:10] if hasattr(want, "isoformat") else got == want


@pytest.mark.parametrize("name", sorted(TEMPORAL_REFERENCE))
def test_a_temporal_accessor_matches_python_datetime(name):
    got, want = (
        _temporal(name),
        [None if t is None else TEMPORAL_REFERENCE[name](t) for t in TIMESTAMPS],
    )
    bad = [(t, g, w) for t, g, w in zip(TIMESTAMPS, got, want, strict=True) if not _equal(g, w)]
    assert bad == [], f"dt.{name} differs from Python's datetime on: {bad}"


@pytest.mark.parametrize("name", sorted(LIST_REFERENCE))
def test_a_list_accessor_matches_python(name):
    got, want = _listwise(name), [None if v is None else LIST_REFERENCE[name](v) for v in LISTS]
    bad = [(v, g, w) for v, g, w in zip(LISTS, got, want, strict=True) if not _equal(g, w)]
    assert bad == [], f"list.{name} differs from Python on: {bad}"


def test_the_fixtures_exercise_each_reference():
    """A reference that is constant over the fixture agrees with an engine that is too."""
    flat = {}
    for name, reference in TEMPORAL_REFERENCE.items():
        answers = {repr(reference(t)) for t in TIMESTAMPS if t is not None}
        if len(answers) < 2:
            flat[f"dt.{name}"] = answers
    for name, reference in LIST_REFERENCE.items():
        answers = {repr(reference(v)) for v in LISTS if v is not None}
        if len(answers) < 2:
            flat[f"list.{name}"] = answers
    assert flat == {}, f"constant over the fixture, so they check nothing: {flat}"


class TestTheTwoConventions:
    """Both were wrong in this file's first draft, so both are stated rather than assumed."""

    def test_the_std_of_a_single_value_is_null(self):
        """Not 0.0. The sample form divides by n-1, so one value has no answer -- which is
        DuckDB's result and is why `statistics.stdev([1.0])` raises."""
        by_value = dict(zip(map(repr, LISTS), _listwise("std"), strict=True))
        assert by_value[repr([1.0])] is None
        assert by_value[repr([0.0, 0.0])] == 0.0, "two equal values do have an answer, and it is 0"

    def test_std_is_the_sample_not_the_population_form(self):
        by_value = dict(zip(map(repr, LISTS), _listwise("std"), strict=True))
        assert by_value[repr([3.0, 1.0, 2.0])] == pytest.approx(statistics.stdev([3.0, 1.0, 2.0]))
        assert statistics.pstdev([3.0, 1.0, 2.0]) == pytest.approx(0.816496580927726), (
            "the population form is a different number, which is what makes this worth pinning"
        )

    def test_summing_an_empty_list_is_null_not_zero(self):
        by_value = dict(zip(map(repr, LISTS), _listwise("sum"), strict=True))
        assert by_value[repr([])] is None
        assert sum([]) == 0, "Python's answer is 0; SQL's is null, and the engine follows SQL"


class TestTheBoundaryTimestamps:
    """The dates chosen to break a naive implementation, asserted individually."""

    def test_the_iso_week_year_of_the_first_of_january_2023(self):
        """2023-01-01 is a Sunday in ISO week 52 of **2022**. A naive `iso_year` returns 2023."""
        by_value = dict(zip(map(str, TIMESTAMPS), _temporal("iso_year"), strict=True))
        assert by_value["2023-01-01 00:00:00"] == 2022

    def test_february_has_twenty_nine_days_in_a_leap_year(self):
        by_value = dict(zip(map(str, TIMESTAMPS), _temporal("days_in_month"), strict=True))
        assert by_value["2024-02-29 13:45:30"] == 29
