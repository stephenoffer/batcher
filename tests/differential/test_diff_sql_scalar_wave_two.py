"""A second wave: the function-catalogue sweep's remaining divergences.

Each was found by calling every DuckDB scalar/aggregate signature with type-appropriate
arguments and comparing. They are unrelated to one another except in how they were found,
and each returned a plausible value rather than raising.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _table() -> pa.Table:
    return pa.table(
        {
            "i": pa.array([5, -3, 0, 255, -1, None], pa.int64()),
            "g": pa.array(["x", "x", "y", "y", "z", "z"], pa.string()),
            "b": pa.array([True, False, True, None, None, None], pa.bool_()),
            "s": pa.array(["ab", "cd", "", None, "x", "y"], pa.string()),
        }
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT i, bin(i) AS r FROM t",
        "SELECT i, hex(i) AS r FROM t",
        "SELECT format_bytes(9223372036854775807) AS r",
        "SELECT format_bytes(-9223372036854775808) AS r",
        "SELECT formatReadableDecimalSize(9223372036854775807) AS r",
        "SELECT count_if(b) AS r FROM t",
        "SELECT g, count_if(b) AS r FROM t GROUP BY g",
    ],
)
def test_a_catalogue_divergence_is_gone(duck, sql):
    table = _table()
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_bin_of_a_negative_is_sixty_four_bits_not_a_sign():
    """`to_base(x, 2)` and `bin(x)` are different functions below zero."""
    table = _table()
    got = bt.sql("SELECT bin(-3) AS r", t=table).collect().to_pydict()
    assert got["r"] == ["1" * 62 + "01"]


def test_count_if_over_an_all_unknown_group_is_null_not_zero():
    """0 reads as "no rows matched"; the truth is "nothing was known"."""
    table = _table()
    got = bt.sql("SELECT g, count_if(b) AS r FROM t GROUP BY g", t=table).collect().to_pydict()
    by_group = dict(zip(got["g"], got["r"], strict=True))
    assert by_group["z"] is None


@pytest.mark.parametrize("unit", ["decade", "century", "millennium", "isoyear"])
def test_date_diff_takes_the_calendar_period_units(duck, unit):
    """Each was refused outright, and each has a `.dt` accessor or a year scale already."""
    import datetime as dt

    table = pa.table(
        {
            "a": pa.array([dt.date(1995, 3, 4), dt.date(2019, 12, 31), dt.date(2024, 1, 1)]),
            "b": pa.array([dt.date(2024, 6, 15), dt.date(2020, 1, 1), dt.date(1990, 1, 1)]),
        }
    )
    sql = f"SELECT date_diff('{unit}', a, b) AS r FROM t"
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_the_three_argument_jaro_winkler_is_refused_not_approximated():
    """The prefix scale factor has no kernel parameter; dropping it answers another query."""
    table = _table()
    with pytest.raises(NotImplementedError, match="prefix scale"):
        bt.sql("SELECT jaro_winkler_similarity(s, 'ab', 0.2) AS r FROM t", t=table).collect()


@pytest.mark.parametrize(
    ("sql", "match"),
    [
        ("SELECT max(i, 2) AS r FROM t", "top-N"),
        ("SELECT min(i, 2) AS r FROM t", "top-N"),
        ("SELECT arg_max(i, i, 2) AS r FROM t", "top-N"),
        ("SELECT string_agg(s, g) AS r FROM t", "constant separator"),
        ("SELECT any_value(i) OVER (ORDER BY i) AS r FROM t", "any_value"),
    ],
)
def test_a_shape_with_no_faithful_lowering_is_refused(sql, match):
    """Each used to answer a *different* question, silently, with a plausible value."""
    table = _table()
    with pytest.raises(NotImplementedError, match=match):
        bt.sql(sql, t=table).collect()
