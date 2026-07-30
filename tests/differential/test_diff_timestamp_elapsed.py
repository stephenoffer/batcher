"""`dt.*_between` against DuckDB's `date_diff`, in both directions.

The elapsed-time accessors lowered to a floor division, so a partial unit rounded *down*
rather than toward zero. That is invisible while both timestamps are the right way round —
2.52 days forward truncates to 2 either way — and wrong the moment the arguments are
reversed, where floor gives -3 and DuckDB gives -2. An SLA computed as
``shipped.days_between(placed)`` and the same SLA computed as
``placed.days_between(shipped)`` therefore disagreed by a whole day in magnitude.

DuckDB is the oracle for the sign convention *and* the rounding, so both directions are
compared against it here rather than only the forward one.
"""

from __future__ import annotations

import datetime as dt

import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential

# Two instants 2 days 12.5 hours apart, so every unit below has a fractional part and the
# rounding direction is observable.
_LATE = dt.datetime(2024, 1, 3, 20, 30)
_EARLY = dt.datetime(2024, 1, 1, 8, 0)

_UNITS = [
    ("day", "days_between"),
    ("hour", "hours_between"),
    ("minute", "minutes_between"),
    ("second", "seconds_between"),
    ("week", "weeks_between"),
]


def _duckdb_diff(unit: str, start: dt.datetime, end: dt.datetime) -> int:
    duckdb = pytest.importorskip("duckdb")
    sql = (
        f"select date_diff('{unit}', TIMESTAMP '{start:%Y-%m-%d %H:%M:%S}', "
        f"TIMESTAMP '{end:%Y-%m-%d %H:%M:%S}') as d"
    )
    return duckdb.sql(sql).fetchall()[0][0]


@pytest.mark.parametrize(("unit", "method"), _UNITS)
@pytest.mark.parametrize("reversed_", [False, True], ids=["forward", "backward"])
def test_elapsed_matches_duckdb_date_diff(unit: str, method: str, reversed_: bool) -> None:
    """`a.<unit>_between(b)` equals DuckDB `date_diff(unit, b, a)`, forward and backward."""
    a, b = (_EARLY, _LATE) if reversed_ else (_LATE, _EARLY)
    ds = bt.from_pydict({"a": [a], "b": [b]})
    got = ds.select(d=getattr(col("a").dt, method)(col("b"))).to_pydict()["d"][0]
    assert got == _duckdb_diff(unit, b, a)


@pytest.mark.parametrize(("unit", "method"), _UNITS)
def test_elapsed_is_antisymmetric(unit: str, method: str) -> None:
    """Swapping the operands negates the answer, for every unit.

    Flooring broke this: the magnitude gained a unit in the negative direction.
    """
    ds = bt.from_pydict({"a": [_LATE], "b": [_EARLY]})
    forward = ds.select(d=getattr(col("a").dt, method)(col("b"))).to_pydict()["d"][0]
    backward = ds.select(d=getattr(col("b").dt, method)(col("a"))).to_pydict()["d"][0]
    assert forward == -backward, f"{method} is not antisymmetric ({forward} vs {backward})"


def test_an_exact_multiple_is_unaffected() -> None:
    """A whole number of units has no fractional part, so both roundings agree."""
    ds = bt.from_pydict(
        {"a": [dt.datetime(2024, 1, 3, 8, 0)], "b": [dt.datetime(2024, 1, 1, 8, 0)]}
    )
    assert ds.select(d=col("a").dt.days_between(col("b"))).to_pydict()["d"] == [2]
    assert ds.select(d=col("b").dt.days_between(col("a"))).to_pydict()["d"] == [-2]
