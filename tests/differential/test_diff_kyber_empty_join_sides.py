"""Folding a join whose side is provably empty — vs DuckDB.

`propagate_empty_relation` folded through unary operators and unions but stopped at a
join, so a join with a provably-empty side still built a hash table and probed it. It now
folds, and the whole risk is that **an empty side does not make every join empty**:

* an empty *left* empties `inner`, `left`, `semi` and `anti` — every output row of those
  is a left row;
* an empty *left* leaves `right` and `full` **non-empty**, because they keep the right
  side's rows padded with nulls. Treating them as empty is a wrong answer, not a slow
  one, and it is the first thing this file checks;
* an empty *right* empties `inner` and `semi`, but an `anti` join keeps **all** of the
  left (nothing left to exclude) and a `left` join keeps all of it padded.

So the fixture runs all six join types against both an empty left and an empty right, and
compares rows against DuckDB. The asymmetry is not something a reader can eyeball from the
rule, which is why it is enumerated here rather than argued.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col, lit

JOIN_TYPES = ["inner", "left", "right", "full", "semi", "anti"]

# DuckDB spells semi/anti as SEMI/ANTI joins; the rest map directly.
DUCK_JOIN = {
    "inner": "INNER JOIN",
    "left": "LEFT JOIN",
    "right": "RIGHT JOIN",
    "full": "FULL JOIN",
    "semi": "SEMI JOIN",
    "anti": "ANTI JOIN",
}


@pytest.fixture
def tables(duck):
    left = pa.table({"k": [1, 2, 3], "a": [10, 20, 30]})
    right = pa.table({"k": [2, 3, 4], "b": [200, 300, 400]})
    duck.register("l", left)
    duck.register("r", right)
    return left, right


def _columns(how: str) -> str:
    """Batcher emits one merged key column, so `SELECT *` (which duplicates `k`) will not
    line up. `right` and `full` take the key from whichever side has it."""
    if how in ("semi", "anti"):
        return "l.k AS k, l.a AS a"
    if how == "right":
        return "r.k AS k, l.a AS a, r.b AS b"
    if how == "full":
        return "coalesce(l.k, r.k) AS k, l.a AS a, r.b AS b"
    return "l.k AS k, l.a AS a, r.b AS b"


@pytest.mark.differential
@pytest.mark.parametrize("how", JOIN_TYPES)
def test_an_empty_left_side_matches_duckdb(duck, tables, how):
    left, right = tables
    got = bt.from_arrow(left).filter(lit(False)).join(bt.from_arrow(right), on="k", how=how)
    expected = duck.sql(
        f"SELECT {_columns(how)} FROM (SELECT * FROM l WHERE false) l "
        f"{DUCK_JOIN[how]} r ON l.k = r.k"
    )
    assert_same(got.collect(), expected)


@pytest.mark.differential
@pytest.mark.parametrize("how", JOIN_TYPES)
def test_an_empty_right_side_matches_duckdb(duck, tables, how):
    left, right = tables
    got = bt.from_arrow(left).join(bt.from_arrow(right).filter(lit(False)), on="k", how=how)
    expected = duck.sql(
        f"SELECT {_columns(how)} FROM l {DUCK_JOIN[how]} "
        "(SELECT * FROM r WHERE false) r ON l.k = r.k"
    )
    assert_same(got.collect(), expected)


@pytest.mark.differential
def test_a_right_join_over_an_empty_left_is_not_empty(duck, tables):
    """The case that makes the rule asymmetric, asserted on its own.

    A `RIGHT JOIN` keeps every right row padded with nulls, so an empty left leaves three
    rows, not zero. Folding it to empty would pass a plan-shape test and return the wrong
    answer.
    """
    left, right = tables
    got = bt.from_arrow(left).filter(lit(False)).join(bt.from_arrow(right), on="k", how="right")
    assert got.collect().num_rows == 3


@pytest.mark.differential
def test_an_anti_join_over_an_empty_right_keeps_every_left_row(duck, tables):
    """The rewrite that removes the join outright: with nothing to exclude, an anti join
    is its own left input."""
    left, right = tables
    got = bt.from_arrow(left).join(bt.from_arrow(right).filter(lit(False)), on="k", how="anti")
    assert_same(got.collect(), duck.sql("SELECT k, a FROM l"))


@pytest.mark.differential
def test_a_join_with_neither_side_empty_is_untouched(duck, tables):
    """Guard against the rule firing on an ordinary join."""
    left, right = tables
    got = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how="inner")
    assert_same(
        got.collect(),
        duck.sql(f"SELECT {_columns('inner')} FROM l INNER JOIN r ON l.k = r.k"),
    )


@pytest.mark.differential
def test_a_constant_false_conjunct_still_empties_the_side(duck, tables):
    """The reason the fold reaches a join at all.

    `filter_null_join_keys` runs after constant folding and rewrites a `false` predicate
    under a join into `false AND k IS NOT NULL`. Nothing folds after that, so the side was
    no longer recognizable as empty and the whole conjunction shipped to the engine to be
    evaluated per row. A boolean literal now decides itself in the tri-state walker.
    """
    left, right = tables
    got = bt.from_arrow(left).join(
        bt.from_arrow(right).filter(lit(False) & col("k").is_not_null()), on="k", how="inner"
    )
    assert got.collect().num_rows == 0
