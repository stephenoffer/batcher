"""A range predicate whose literal sits on no ordinal axis must plan, not raise.

`date_col >= '2013-07-01'` is ordinary SQL: the engine casts the string at execution. The
*estimator* does not cast — it places a literal on a number line with `_ordinal`, which
answers `None` for a string, while the column's `[min, max]` bounds are dates. Two of the
three call sites handed that unplaced `None` straight into the interpolation, which compared
it against a float and raised `TypeError: '<' not supported between 'NoneType' and 'float'`
**at plan time, before a row was read**.

Seven ClickBench queries (q36-q42) died that way — a whole family, because they share the
`EventDate >= '...' AND EventDate <= '...'` idiom, and the crash needs *two* comparisons on
one column (a single one takes a different branch). That pairing is what the tests below
pin: one bound alone, then both, then both with the sort and limit the real queries carry.

The estimate itself is allowed to be poor here — an uninterpolable literal falls back to the
structural default — but the *answer* must match DuckDB, and the query must run.
"""

from __future__ import annotations

import datetime

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

_DAYS = [datetime.date(2013, 7, 1) + datetime.timedelta(days=i) for i in range(60)]
_T = pa.table(
    {
        "d": _DAYS,
        "ts": [datetime.datetime(2013, 7, 1) + datetime.timedelta(hours=i) for i in range(60)],
        "counter": [i % 7 for i in range(60)],
        "v": list(range(60)),
    }
)


def _both(duck, q):
    duck.register("t", _T)
    assert_same(bt.sql(q, t=_T).collect(), duck.sql(q))


@pytest.mark.differential
@pytest.mark.parametrize("op", [">=", ">", "<=", "<", "="])
def test_one_string_literal_against_a_date_column(duck, op):
    """A single comparison of a `date32` column against a string literal."""
    _both(duck, f"SELECT count(*) AS n FROM t WHERE d {op} '2013-07-15'")


@pytest.mark.differential
def test_two_string_literals_bracket_a_date_column(duck):
    """Two comparisons on one column — the interval branch that raised."""
    _both(
        duck,
        "SELECT count(*) AS n FROM t WHERE d >= '2013-07-01' AND d <= '2013-07-31'",
    )


@pytest.mark.differential
def test_between_with_string_literals_on_a_date_column(duck):
    """`BETWEEN` lowers to the same conjunction of two comparisons."""
    _both(duck, "SELECT count(*) AS n FROM t WHERE d BETWEEN '2013-07-05' AND '2013-07-20'")


@pytest.mark.differential
def test_string_literals_against_a_timestamp_column(duck):
    """The same shape on `timestamp`, whose bounds sit on a different axis again."""
    _both(
        duck,
        "SELECT count(*) AS n FROM t "
        "WHERE ts >= '2013-07-01 06:00:00' AND ts <= '2013-07-02 06:00:00'",
    )


@pytest.mark.differential
def test_the_clickbench_shape_end_to_end(duck):
    """The full q36 shape: the bracketed date, another equality, a group-by and a top-N."""
    _both(
        duck,
        "SELECT counter, count(*) AS c FROM t "
        "WHERE counter = 3 AND d >= '2013-07-01' AND d <= '2013-07-31' "
        "GROUP BY counter ORDER BY c DESC, counter LIMIT 10",
    )


@pytest.mark.differential
def test_a_date_bracket_drives_a_join(duck):
    """The bracket above a join, where a raise would take the whole plan down."""
    dim = pa.table({"counter": list(range(7)), "label": [f"c{i}" for i in range(7)]})
    duck.register("t", _T)
    duck.register("dim", dim)
    q = (
        "SELECT label, sum(v) AS s FROM t JOIN dim USING (counter) "
        "WHERE d >= '2013-07-10' AND d < '2013-08-01' GROUP BY label"
    )
    assert_same(bt.sql(q, t=_T, dim=dim).collect(), duck.sql(q))
