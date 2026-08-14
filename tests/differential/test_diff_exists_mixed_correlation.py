"""A correlated `EXISTS` carrying an equality **and** an inequality matches DuckDB.

`EXISTS (SELECT 1 FROM b WHERE b.k = a.k AND b.v > a.v)` raised
`NotImplementedError: correlated subqueries not supported` before ``subquery.mixed``: the
equality became a correlation key, the inequality stayed in the inner relation's `WHERE` still
naming the outer table, and `_reject_correlated` refused it. Every neighbouring shape already
worked, which is what made the gap easy to miss -- so the equality-only and inequality-only
cases are here too, to pin that the new path did not take a query the old ones were answering.

The shapes are chosen around the two ways this rewrite can go wrong: it must not collapse
duplicate outer rows (it reduces over a row tag rather than over values, and the tables below
carry a repeated outer row on purpose), and it must not confuse an inner column with an outer
one of the same name (both tables use `g` and `v`).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

# `t1` repeats the row (1, 10): EXISTS must keep both copies, and a rewrite that reduced over
# the outer *values* instead of over a row identity would return one.
_T1 = pa.table(
    {
        "g": [1, 1, 2, 2, 3, 1, None],
        "v": [10, 20, 5, 30, 7, 10, 4],
        "tag": ["a", "b", "c", "d", "e", "f", "g"],
    }
)
# A null key and a null value, so the three-valued comparison is exercised on both sides.
_T2 = pa.table({"g": [1, 2, 2, 4, None, 1], "v": [15, 1, 25, 99, 50, None]})
# Intervals, for the two-inequality shape: `lo <= v < hi` beside an equality on `g`.
_T3 = pa.table({"g": [1, 1, 2, 3, 2], "lo": [0, 18, 0, 90, 4], "hi": [15, 40, 3, 99, 31]})


@pytest.fixture
def sess(duck):
    duck.register("t1", _T1)
    duck.register("t2", _T2)
    duck.register("t3", _T3)
    s = bt.Session()
    s.register("t1", bt.from_arrow(_T1))
    s.register("t2", bt.from_arrow(_T2))
    s.register("t3", bt.from_arrow(_T3))
    return s


def _check(sess, duck, sql: str) -> None:
    assert_same(sess.sql(sql).collect(), duck.sql(sql))


def test_mixed_exists(sess, duck):
    _check(
        sess,
        duck,
        "SELECT * FROM t1 WHERE EXISTS (SELECT 1 FROM t2 x WHERE x.g = t1.g AND x.v > t1.v)",
    )


def test_mixed_not_exists(sess, duck):
    _check(
        sess,
        duck,
        "SELECT * FROM t1 WHERE NOT EXISTS (SELECT 1 FROM t2 x WHERE x.g = t1.g AND x.v > t1.v)",
    )


@pytest.mark.parametrize("op", ["<", "<=", ">", ">="])
def test_every_inequality_direction(sess, duck, op):
    _check(
        sess,
        duck,
        f"SELECT * FROM t1 WHERE EXISTS (SELECT 1 FROM t2 x WHERE x.g = t1.g AND x.v {op} t1.v)",
    )


def test_inequality_written_the_other_way_round(sess, duck):
    """`t1.v < x.v` is the same correlation as `x.v > t1.v`, and must read as one."""
    _check(
        sess,
        duck,
        "SELECT * FROM t1 WHERE EXISTS (SELECT 1 FROM t2 x WHERE x.g = t1.g AND t1.v < x.v)",
    )


def test_mixed_with_a_local_predicate(sess, duck):
    """A predicate on the inner relation alone stays the inner relation's own filter."""
    _check(
        sess,
        duck,
        "SELECT * FROM t1 WHERE EXISTS ("
        "  SELECT 1 FROM t2 x WHERE x.g = t1.g AND x.v > t1.v AND x.v < 90)",
    )


def test_two_inequalities_bound_an_interval(sess, duck):
    """An equality and *two* inequalities: `g` matches and `v` falls inside `[lo, hi)`."""
    _check(
        sess,
        duck,
        "SELECT * FROM t1 WHERE EXISTS ("
        "  SELECT 1 FROM t3 x WHERE x.g = t1.g AND x.lo <= t1.v AND x.hi > t1.v)",
    )


def test_two_inequalities_not_exists(sess, duck):
    _check(
        sess,
        duck,
        "SELECT * FROM t1 WHERE NOT EXISTS ("
        "  SELECT 1 FROM t3 x WHERE x.g = t1.g AND x.lo <= t1.v AND x.hi > t1.v)",
    )


def test_a_correlation_on_an_expression_is_still_declined(sess):
    """The boundary, stated rather than assumed.

    Both this path and the `range` one read a correlation as two plain *columns*, because
    `RangeCondition` carries column names. `x.v < t1.v + 100` correlates on an expression, so
    it is refused rather than silently mis-planned -- and refusing is the right answer until
    the join carries an expression, not something to work around in the front end.
    """
    sql = (
        "SELECT * FROM t1 WHERE EXISTS ("
        "  SELECT 1 FROM t2 x WHERE x.g = t1.g AND x.v > t1.v AND x.v < t1.v + 100)"
    )
    with pytest.raises(NotImplementedError):
        sess.sql(sql).collect()


def test_duplicate_outer_rows_are_both_kept(sess, duck):
    """The trap this rewrite is shaped around: reducing over values would return one row."""
    sql = "SELECT g, v FROM t1 WHERE EXISTS (  SELECT 1 FROM t2 x WHERE x.g = t1.g AND x.v > t1.v)"
    out = sess.sql(sql).collect()
    assert out.num_rows == len(duck.sql(sql).fetchall())
    assert_same(out, duck.sql(sql))


def test_mixed_exists_over_an_empty_inner(sess, duck):
    _check(
        sess,
        duck,
        "SELECT * FROM t1 WHERE EXISTS ("
        "  SELECT 1 FROM t2 x WHERE x.g = t1.g AND x.v > t1.v AND x.v > 1000)",
    )


def test_mixed_not_exists_over_an_empty_inner(sess, duck):
    _check(
        sess,
        duck,
        "SELECT * FROM t1 WHERE NOT EXISTS ("
        "  SELECT 1 FROM t2 x WHERE x.g = t1.g AND x.v > t1.v AND x.v > 1000)",
    )


def test_mixed_exists_beside_an_ordinary_predicate(sess, duck):
    _check(
        sess,
        duck,
        "SELECT * FROM t1 WHERE t1.v > 6 AND EXISTS ("
        "  SELECT 1 FROM t2 x WHERE x.g = t1.g AND x.v > t1.v)",
    )


def test_equality_only_still_takes_the_semi_join(sess, duck):
    """A regression guard: the new path must decline a correlation with no inequality."""
    _check(sess, duck, "SELECT * FROM t1 WHERE EXISTS (SELECT 1 FROM t2 x WHERE x.g = t1.g)")


def test_inequality_only_still_takes_the_range_join(sess, duck):
    _check(sess, duck, "SELECT * FROM t1 WHERE EXISTS (SELECT 1 FROM t2 x WHERE x.v > t1.v)")


def test_not_exists_equality_only_still_takes_the_anti_join(sess, duck):
    _check(sess, duck, "SELECT * FROM t1 WHERE NOT EXISTS (SELECT 1 FROM t2 x WHERE x.g = t1.g)")


def test_mixed_exists_matches_distributed(sess, duck):
    """The rewrite is joins and a distinct, so many machines must answer as one does."""
    sql = "SELECT * FROM t1 WHERE EXISTS (SELECT 1 FROM t2 x WHERE x.g = t1.g AND x.v > t1.v)"
    expected = duck.sql(sql)
    assert_same(sess.sql(sql).collect(), expected)
    assert_same(sess.sql(sql).collect(distributed=True), expected)
