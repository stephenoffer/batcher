"""`.dt.truncate` validates its unit at the API edge, and speaks one unit vocabulary.

Two defects met in this one argument.

*Fail-late.* The unit string was handed straight to `bc-expr`, so a typo built a perfectly
valid plan and only failed once the query ran -- after the scan -- as a bare `RuntimeError`
from Rust rather than a typed `PlanError`. Every neighbouring method (`ceil`, `round`,
`offset_by`) already rejected a bad unit at build time; `truncate`/`floor` was the outlier.

*Two vocabularies.* `truncate` took only the long names (``"month"``), while `offset_by`,
`ceil` and `round` take the duration spellings (``"1mo"``). The two sets did not overlap at
all, so `truncate("1mo")` failed on a string the neighbouring method accepts and
`offset_by("1month")` failed on the string `truncate` required. The duration spellings are
also what Polars' `dt.truncate` takes.

A multiplier other than 1 is refused rather than approximated: ``"5d"`` is a bucket width,
which flooring to a calendar boundary cannot express, and silently flooring to one day
would return plausible, wrong timestamps.
"""

from __future__ import annotations

import datetime as dt

import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit

_WHEN = {"d": [dt.datetime(2024, 3, 15, 13, 45, 30)]}


def _trunc(unit):
    return bt.from_pydict(_WHEN).select(r=bt.col("d").dt.truncate(unit)).to_pydict()["r"][0]


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("year", dt.datetime(2024, 1, 1)),
        ("quarter", dt.datetime(2024, 1, 1)),
        ("month", dt.datetime(2024, 3, 1)),
        ("week", dt.datetime(2024, 3, 11)),
        ("day", dt.datetime(2024, 3, 15)),
        ("hour", dt.datetime(2024, 3, 15, 13)),
        ("minute", dt.datetime(2024, 3, 15, 13, 45)),
    ],
)
def test_long_unit_names_still_work(unit, expected):
    assert _trunc(unit) == expected


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("1y", "year"),
        ("y", "year"),
        ("years", "year"),
        ("1q", "quarter"),
        ("q", "quarter"),
        ("1mo", "month"),
        ("mo", "month"),
        ("months", "month"),
        ("1w", "week"),
        ("w", "week"),
        ("1d", "day"),
        ("d", "day"),
        ("1h", "hour"),
        ("h", "hour"),
        ("1m", "minute"),
        ("m", "minute"),
        ("mins", "minute"),
        ("1s", "second"),
        ("s", "second"),
    ],
)
def test_duration_spellings_match_their_long_name(alias, canonical):
    """The `offset_by`/Polars spelling must mean exactly the long name it aliases."""
    assert _trunc(alias) == _trunc(canonical)


def test_mo_is_months_and_m_is_minutes():
    """The one ambiguity in the vocabulary, resolved as `offset_by` resolves it."""
    assert _trunc("mo") == dt.datetime(2024, 3, 1)
    assert _trunc("m") == dt.datetime(2024, 3, 15, 13, 45)


@pytest.mark.parametrize("unit", ["MONTH", " 1mo ", "Month"])
def test_case_and_surrounding_space_are_tolerated(unit):
    assert _trunc(unit) == dt.datetime(2024, 3, 1)


@pytest.mark.parametrize("bad", ["bogus", "1mox", "", "moo"])
def test_unknown_unit_raises_planerror_at_build_time(bad):
    """Typed, and raised while building the plan -- not a RuntimeError mid-scan."""
    with pytest.raises(PlanError, match="not a known unit"):
        bt.col("d").dt.truncate(bad)


@pytest.mark.parametrize("bad", [7, None, 1.5])
def test_non_string_unit_raises_planerror(bad):
    with pytest.raises(PlanError, match="must be a string"):
        bt.col("d").dt.truncate(bad)


@pytest.mark.parametrize("bad", ["5d", "0mo", "2h", "15m"])
def test_multiplier_is_refused_not_silently_floored(bad):
    """`5d` is a bucket width; flooring it to one day would be plausible and wrong."""
    with pytest.raises(PlanError, match="no multiplier other than 1"):
        bt.col("d").dt.truncate(bad)


def test_floor_alias_validates_the_same_way():
    """`.dt.floor` is the pandas spelling of `truncate`; it must not bypass the check."""
    with pytest.raises(PlanError, match="not a known unit"):
        bt.col("d").dt.floor("bogus")
    got = bt.from_pydict(_WHEN).select(r=bt.col("d").dt.floor("1h")).to_pydict()["r"][0]
    assert got == dt.datetime(2024, 3, 15, 13)


def test_truncate_and_offset_by_now_share_a_vocabulary():
    """The regression this closes: neither method understood the other's spelling."""
    assert _trunc("1mo") == dt.datetime(2024, 3, 1)
    # `offset_by` has always taken the duration spelling; it must keep doing so.
    got = bt.from_pydict(_WHEN).select(r=bt.col("d").dt.offset_by("1mo")).to_pydict()["r"][0]
    assert got == dt.datetime(2024, 4, 15, 13, 45, 30)
