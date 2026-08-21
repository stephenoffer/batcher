"""An outer join's ``ON`` residual that reads only one side.

``A LEFT JOIN B ON a.k = b.k AND <residual>`` does not filter the *result*: B's columns
are null wherever nothing matched, and those A rows must survive. Two one-sided shapes are
exactly expressible and both are now taken:

* a residual reading only the null-extended side pre-filters that side;
* a residual reading only a **preserved** side becomes an extra join *key* — the predicate
  on the side it reads, the constant TRUE opposite — so a row failing it carries a key no
  row opposite holds and the outer join null-extends it. That covers ``FULL``, where both
  sides are preserved, and it is the shape that used to be refused outright.

A residual reading *both* sides is a real theta join and still raises; that refusal is
pinned here too, because turning it into a post-join filter would drop rows.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _tables() -> tuple[pa.Table, pa.Table]:
    left = pa.table(
        {
            "i": pa.array([1, 2, 3, None], pa.int64()),
            "j": pa.array([10, 2, 30, 4], pa.int64()),
            "g": pa.array(["a", "b", "c", "d"], pa.string()),
        }
    )
    right = pa.table(
        {
            "i": pa.array([1, 2, 9], pa.int64()),
            "k": pa.array(["x", "y", "z"], pa.string()),
            "v": pa.array([5, 6, 7], pa.int64()),
        }
    )
    return left, right


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT t.i AS ti, t.j AS tj, u.k AS uk FROM t LEFT JOIN u ON t.i = u.i AND t.j > 5",
        "SELECT t.i AS ti, u.k AS uk FROM t LEFT JOIN u ON t.i = u.i AND u.v > 5",
        "SELECT t.i AS ti, u.k AS uk FROM t RIGHT JOIN u ON t.i = u.i AND u.v > 5",
        "SELECT t.i AS ti, u.k AS uk FROM t RIGHT JOIN u ON t.i = u.i AND t.j > 5",
        "SELECT t.i AS ti, u.k AS uk FROM t FULL OUTER JOIN u ON t.i = u.i AND t.j > 5",
        "SELECT t.i AS ti, u.k AS uk FROM t FULL OUTER JOIN u ON t.i = u.i AND u.v > 5",
        "SELECT t.i AS ti, u.k AS uk FROM t LEFT JOIN u ON t.i = u.i AND t.g = 'a'",
        "SELECT count(*) AS c FROM t LEFT JOIN u ON t.i = u.i AND t.j > 5",
        "SELECT count(*) AS c FROM t INNER JOIN u ON t.i = u.i AND t.j > 5",
    ],
)
def test_a_one_sided_on_residual_matches_duckdb(duck, sql):
    left, right = _tables()
    duck.register("t", left)
    duck.register("u", right)
    assert_same(bt.sql(sql, t=left, u=right).collect(), duck.sql(sql))


def test_the_preserved_side_keeps_its_unmatched_rows():
    """The defect a post-join filter would cause, stated directly.

    Every left row must appear exactly once, whether or not its residual held — three
    rows null-extended and one matched, not the single matching row a filter would leave.
    """
    left, right = _tables()
    sql = "SELECT t.i AS ti, u.k AS uk FROM t LEFT JOIN u ON t.i = u.i AND t.j > 5"
    got = bt.sql(sql, t=left, u=right).collect().to_pydict()
    assert len(got["ti"]) == 4
    assert sorted(x for x in got["uk"] if x) == ["x"]


def test_a_residual_reading_both_sides_still_raises():
    """A genuine theta join: refused rather than answered with the wrong row set."""
    left, right = _tables()
    with pytest.raises(NotImplementedError, match="reads both sides"):
        bt.sql(
            "SELECT t.i AS ti FROM t LEFT JOIN u ON t.i = u.i AND t.j > u.v", t=left, u=right
        ).collect()


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) AS c FROM t JOIN u ON t.i = u.i OR t.g = u.k",
        "SELECT count(*) AS c FROM t a JOIN u b ON a.i = b.i OR a.g = b.k",
        "SELECT count(*) AS c FROM t LEFT JOIN u ON t.i = u.i OR t.g = u.k",
    ],
)
def test_an_equality_under_or_is_not_a_join_key(duck, sql):
    """An `=` buried under `OR` is part of a theta predicate, not a merged key.

    Treating it as one marked its column "merged, keep the bare name", so both sides kept
    the name `i` and the cross-join-plus-filter lowering refused the query as ambiguous.
    """
    left, right = _tables()
    duck.register("t", left)
    duck.register("u", right)
    assert_same(bt.sql(sql, t=left, u=right).collect(), duck.sql(sql))
