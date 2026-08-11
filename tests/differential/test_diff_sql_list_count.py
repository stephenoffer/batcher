"""`list_count` counts non-null elements, not list length.

DuckDB's `list_count` is a COUNT: it ignores nulls, so `list_count([NULL, 4])` is 1 and
`list_count([NULL, NULL])` is 0. It was translated straight to `.list.len()`, which returns
the element count including nulls -- so a list of four values and a list of four nulls
reported the same number, and the one function whose entire job is to not count nulls
counted them.

This is invisible on data without nulls, which is why it needs the null-shaped cases below
rather than a general list test. `array_length`, the function that *is* a length, is
asserted alongside so the fix cannot have been to make both the same thing again.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _lists() -> pa.Table:
    """Every null shape a list can take, including a null list and an empty one."""
    return pa.table(
        {
            "l": pa.array([[None, 4], [None, None], [1, None, 2], [], [3], None, [None]]),
            "s": pa.array([["a", None], [None], [], ["b", "c"], None, [None, None], ["d"]]),
        }
    )


@pytest.mark.parametrize("column", ["l", "s"])
def test_list_count_ignores_nulls_like_duckdb(duck, column):
    """Asserted on an int and a string element type: the fix must not be type-specific."""
    table = _lists()
    sql = f"SELECT list_count({column}) AS r FROM t"
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_the_counts_are_the_expected_values(duck):
    """Stated as values, because a match alone does not show which behaviour won."""
    table = _lists()
    sql = "SELECT list_count(l) AS r FROM t"
    duck.register("t", table)
    assert bt.sql(sql, t=table).to_pydict()["r"] == [1, 0, 2, 0, 1, None, 0]
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_array_length_still_counts_every_element(duck):
    """The neighbouring function that really is a length, so the two stay distinguishable.

    Without this, mapping `list_count` back onto `len` would look correct again the moment
    someone tidied the dispatch table.
    """
    table = _lists()
    sql = "SELECT array_length(l) AS r FROM t"
    duck.register("t", table)
    assert bt.sql(sql, t=table).to_pydict()["r"] == [2, 2, 3, 0, 1, None, 1]
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_a_list_of_nulls_and_a_list_of_values_do_not_agree():
    """The bug in one line: same length, different count."""
    table = pa.table({"l": pa.array([[1, 2, 3, 4], [None, None, None, None]])})
    counts = bt.sql("SELECT list_count(l) AS r FROM t", t=table).to_pydict()["r"]
    assert counts == [4, 0]


def test_it_survives_a_partitioned_collect(duck):
    """A per-element predicate has to hold however the rows are split."""
    table = _lists()
    sql = "SELECT list_count(l) AS r FROM t"
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).repartition(3).collect(), duck.sql(sql))
