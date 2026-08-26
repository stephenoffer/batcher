"""`lowering.intervals.interval_parts` — the SQL INTERVAL literal vocabulary.

The expected triples are read off DuckDB, not assumed: each was measured with
``datepart('year'/'month'/'day'/'hour'/'minute'/'microsecond', INTERVAL '<text>')`` so the
front end and the oracle agree on what a literal *means* before any query runs. The
differential half — that a shift by these parts lands on the same instant DuckDB lands on
— is `tests/differential/test_diff_sql_interval_literals.py`.

Unit-level because the parse is a pure function of the text: it needs no engine, and a
table of (text, months, days, micros) is the shape that makes a wrong unit obvious.
"""

from __future__ import annotations

import pytest

from batcher._sql.parser.expressions.lowering.intervals import interval_parts

pytestmark = pytest.mark.unit

_HOUR = 3_600_000_000
_MINUTE = 60_000_000
_SECOND = 1_000_000


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Single unit, integer count.
        ("1 day", (0, 1, 0)),
        ("3 days", (0, 3, 0)),
        ("1 month", (1, 0, 0)),
        ("1 quarter", (3, 0, 0)),
        ("1 year", (12, 0, 0)),
        ("1 decade", (120, 0, 0)),
        ("1 century", (1200, 0, 0)),
        ("1 week", (0, 7, 0)),
        ("2 hours", (0, 0, 2 * _HOUR)),
        ("90 minutes", (0, 0, 90 * _MINUTE)),
        ("500 milliseconds", (0, 0, 500_000)),
        ("1000 microseconds", (0, 0, 1000)),
        # PostgreSQL/DuckDB abbreviations. `m` is a MINUTE and `mon` is a MONTH — the one
        # collision in the vocabulary, and the reason units are matched as whole tokens.
        ("1 mon", (1, 0, 0)),
        ("2 mons", (2, 0, 0)),
        ("1 y", (12, 0, 0)),
        ("1 yr", (12, 0, 0)),
        ("3 d", (0, 3, 0)),
        ("2 w", (0, 14, 0)),
        ("2 h", (0, 0, 2 * _HOUR)),
        ("2 hrs", (0, 0, 2 * _HOUR)),
        ("2 m", (0, 0, 2 * _MINUTE)),
        ("2 mins", (0, 0, 2 * _MINUTE)),
        ("2 s", (0, 0, 2 * _SECOND)),
        ("2 secs", (0, 0, 2 * _SECOND)),
        ("2 ms", (0, 0, 2000)),
        ("2 us", (0, 0, 2)),
        # Fractional counts spill into the next finer component (a month is 30 days for
        # that purpose, a day is 86,400 seconds) rather than rounding.
        ("1.5 months", (1, 15, 0)),
        ("0.5 years", (6, 0, 0)),
        ("1.5 days", (0, 1, 12 * _HOUR)),
        ("0.25 days", (0, 0, 6 * _HOUR)),
        ("2.5 weeks", (0, 17, 12 * _HOUR)),
        ("1.5 hours", (0, 0, 90 * _MINUTE)),
        # Compound: the terms add, each with its own sign.
        ("1 day 3 hours", (0, 1, 3 * _HOUR)),
        ("2 years 3 months", (27, 0, 0)),
        ("1 y 2 mons 3 d", (14, 3, 0)),
        ("1 day -3 hours", (0, 1, -3 * _HOUR)),
        (
            "1 week 2 days 3 hours 4 minutes 5 seconds",
            (0, 9, 3 * _HOUR + 4 * _MINUTE + 5 * _SECOND),
        ),
        # Signed single term.
        ("-1 day", (0, -1, 0)),
        ("-2 hours", (0, 0, -2 * _HOUR)),
        # Clock form, which names no units at all.
        ("04:05:06", (0, 0, 4 * _HOUR + 5 * _MINUTE + 6 * _SECOND)),
        ("00:00:01.5", (0, 0, 1_500_000)),
        ("-01:30:00", (0, 0, -90 * _MINUTE)),
    ],
)
def test_interval_parts_matches_duckdb(text, expected):
    assert interval_parts(text) == expected


def test_default_unit_applies_to_a_bare_count():
    """`INTERVAL 3 DAY` parks the unit beside the number, so it arrives separately."""
    assert interval_parts("3", "DAY") == (0, 3, 0)
    assert interval_parts("90", "MINUTE") == (0, 0, 90 * _MINUTE)


def test_an_explicit_unit_in_the_text_wins_over_the_default():
    assert interval_parts("2 hours", "DAY") == (0, 0, 2 * _HOUR)


@pytest.mark.parametrize("text", ["", "   ", "day", "not an interval"])
def test_a_non_interval_is_rejected(text):
    with pytest.raises(ValueError):
        interval_parts(text)


def test_an_unknown_unit_names_itself():
    with pytest.raises(ValueError, match="fortnight"):
        interval_parts("2 fortnights")


def test_a_bare_count_with_no_default_unit_is_rejected():
    """A number alone is not an interval; the caller must say what unit it is in."""
    with pytest.raises(ValueError):
        interval_parts("3")
