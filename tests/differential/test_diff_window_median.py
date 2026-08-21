"""`median(x) OVER (...)` — the one order statistic the window layer computes.

Every other window aggregate here is a fold, so it reaches its running form by applying an
associative operator one row at a time. A median is not a fold, and that is why SQL's
`median(x) OVER (...)` used to raise `unsupported window function: median` in every spelling,
including the two most common ones.

Two structures make it answerable, and both are checked here rather than assumed:

* whole-partition, it is `agg/median.rs`'s own quickselect, so a window median and a
  `GROUP BY` median over the same rows agree because they are the same code;
* along an ordered partition, it is a two-heap (a max-heap of the lower half, a min-heap of
  the upper), which is `O(log n)` per row.

An *explicit* frame still declines. That case has its own test in
`test_diff_window_agg_vocabulary.py`, because the dangerous failure there is answering the
unframed question with a perfectly plausible number rather than raising.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered

pytestmark = pytest.mark.differential


def _table() -> pa.Table:
    """Ties, nulls, negatives and both parities of group size, in one table.

    The tie rows exist for the peer-group rule: `ORDER BY v` puts equal values in one peer
    group, and every row of a peer group must see the aggregate *through the end* of that
    group, not through its own position.
    """
    return pa.table(
        {
            "g": pa.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3], pa.int64()),
            "i": pa.array(list(range(12)), pa.int64()),
            "v": pa.array(
                [1.0, 3.0, 3.0, 8.0, -2.0, 4.5, None, 0.0, 7.0, 7.0, 7.0, -1.5],
                pa.float64(),
            ),
            "n": pa.array([5, 1, 1, 9, None, 3, 4, 2, 8, 8, 6, 7], pa.int64()),
        }
    )


_QUERIES = [
    # The two spellings that were unreachable and are the ones people write.
    "SELECT i, median(v) OVER (PARTITION BY g) AS w FROM %s ORDER BY i",
    "SELECT i, median(v) OVER (PARTITION BY g ORDER BY i) AS w FROM %s ORDER BY i",
    # No partition at all: one whole-table median, and one running over the whole table.
    "SELECT i, median(v) OVER () AS w FROM %s ORDER BY i",
    "SELECT i, median(v) OVER (ORDER BY i) AS w FROM %s ORDER BY i",
    # An integer input must widen — the median of two integers is generally not one.
    "SELECT i, median(n) OVER (PARTITION BY g) AS w FROM %s ORDER BY i",
    "SELECT i, median(n) OVER (PARTITION BY g ORDER BY i) AS w FROM %s ORDER BY i",
    # ORDER BY the measured column itself, so ties become peer groups.
    "SELECT i, median(v) OVER (PARTITION BY g ORDER BY v) AS w FROM %s ORDER BY i",
]


@pytest.mark.parametrize("sql", _QUERIES, ids=range(len(_QUERIES)))
def test_a_window_median_matches_duckdb(duck, sql):
    t = _table()
    duck.register("t", t)
    assert_same_ordered(bt.from_arrow(t).sql(sql % "self").collect(), duck.sql(sql % "t"))


def test_the_window_median_is_the_group_by_median(duck):
    """The whole-partition form must equal the `GROUP BY` one, because it is the same kernel.

    Stated as its own test rather than inferred from the DuckDB comparisons: the two spellings
    sharing `quickselect_median` is the property that keeps them from drifting apart, and it
    would drift silently — both answers stay plausible numbers.
    """
    t = _table()
    duck.register("t", t)
    sql = "SELECT g, median(v) AS w FROM %s GROUP BY g ORDER BY g"
    grouped = bt.from_arrow(t).sql(sql % "self").collect()
    assert_same_ordered(grouped, duck.sql(sql % "t"))

    windowed = bt.from_arrow(t).sql(
        "SELECT DISTINCT g, median(v) OVER (PARTITION BY g) AS w FROM self ORDER BY g"
    )
    assert windowed.collect().to_pydict() == grouped.to_pydict()


def test_an_all_null_partition_is_null_not_zero(duck):
    """No values to take a median of is NULL, the rule every other aggregate here follows."""
    t = pa.table(
        {
            "g": pa.array([1, 1, 2, 2], pa.int64()),
            "i": pa.array([0, 1, 2, 3], pa.int64()),
            "v": pa.array([None, None, 4.0, 6.0], pa.float64()),
        }
    )
    duck.register("t", t)
    sql = "SELECT i, median(v) OVER (PARTITION BY g) AS w FROM %s ORDER BY i"
    assert_same_ordered(bt.from_arrow(t).sql(sql % "self").collect(), duck.sql(sql % "t"))


def test_the_declared_type_is_the_type_a_run_produces():
    """`Dataset.schema` answered `None` for a window median before it had a rule.

    The kernel builds a `Float64Array` whatever the input's width, so an integer column must
    be declared `double` and not left uncertain — an unknown here is what lets a downstream
    plan mis-declare, and it is the device tier's characteristic defect.
    """
    ds = bt.from_arrow(_table()).sql(
        "SELECT median(n) OVER (PARTITION BY g) AS w,"
        " median(v) OVER (PARTITION BY g) AS x FROM self"
    )
    assert ds.schema.field("w").type == pa.float64()
    assert ds.schema.field("x").type == pa.float64()
    out = ds.collect()
    assert out.schema.field("w").type == pa.float64()
    assert out.schema.field("x").type == pa.float64()
