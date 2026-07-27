"""The `relational/windows` rewrites must return DuckDB's rows after optimization.

The unit suite asserts these two rules produce the right *plan*; this asserts the plan
produces the right *rows*. Both are needed: a structural rule that fires correctly on a
shape test can still be semantically wrong, and a result test alone cannot tell whether
the rule fired at all.

The top-N cases use `assert_same_ordered`: the pushed-down sort makes the prefix a
defined set, so which rows come back and in what order is the whole contract, and the
order-independent comparison would happily accept a wrong prefix.

The refusal cases deliberately do *not* compare under a limit. When the rule declines,
the query is an unordered `LIMIT`, which returns an arbitrary prefix on either engine --
a comparison there would be non-deterministic rather than strict. They assert the window
*values* over the whole relation instead, and the refusal itself is asserted structurally
in `tests/unit/test_relational_window_rules.py`.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "a": [5, 1, 4, 2, 3, None],
            "g": ["x", "y", "x", "y", "x", "x"],
            "v": [10, 20, 30, 40, 50, 60],
        }
    )
    duck.register("t", tbl)
    return tbl


def test_ranking_window_under_limit_matches_duckdb(t, duck):
    """`ROW_NUMBER() OVER (ORDER BY a)` with a `LIMIT` keeps DuckDB's prefix."""
    got = bt.from_arrow(t).with_columns(rn=bt.row_number().over(order_by="a")).limit(3)
    assert_same_ordered(
        got.collect(),
        duck.sql("select a, g, v, row_number() over (order by a) as rn from t limit 3"),
    )


def test_ranking_window_under_limit_with_offset(t, duck):
    """The offset must be added to the pushed cap, or rows 4 and 5 would be missing."""
    got = bt.from_arrow(t).with_columns(rn=bt.row_number().over(order_by="a")).slice(2, 2)
    assert_same_ordered(
        got.collect(),
        duck.sql("select a, g, v, row_number() over (order by a) as rn from t limit 2 offset 2"),
    )


@pytest.mark.parametrize("func", ["rank", "dense_rank"])
def test_other_prefix_stable_rankings_under_limit(t, duck, func):
    got = bt.from_arrow(t).with_columns(r=getattr(bt, func)().over(order_by="a")).limit(3)
    assert_same_ordered(
        got.collect(),
        duck.sql(f"select a, g, v, {func}() over (order by a) as r from t limit 3"),
    )


def test_partition_size_dependent_ranking_values_match_duckdb(t, duck):
    """`percent_rank` divides by the partition size, so the input must not be truncated.

    Were the top-N pushed under this window, the denominator would become the truncated
    row count and every value would be wrong. Asserted over the *whole* relation rather
    than under a limit: an unordered `LIMIT` returns an arbitrary prefix on both sides,
    so a limited comparison would be non-deterministic and would prove nothing. The
    refusal itself is asserted structurally in `tests/unit/test_relational_window_rules.py`.
    """
    got = bt.from_arrow(t).with_columns(p=bt.percent_rank().over(order_by="a"))
    assert_same(
        got.collect(),
        duck.sql("select a, g, v, percent_rank() over (order by a) as p from t"),
    )


def test_partitioned_ranking_values_match_duckdb(t, duck):
    """With partition keys, rank restarts per partition -- the values must be per-group.

    Compared over the whole relation, for the same reason as the test above.
    """
    got = bt.from_arrow(t).with_columns(rn=bt.row_number().over(partition_by="g", order_by="a"))
    assert_same(
        got.collect(),
        duck.sql("select a, g, v, row_number() over (partition by g order by a) as rn from t"),
    )


def test_transposed_independent_windows_match_duckdb(t, duck):
    """Two windows over different specs must give the same columns in either order."""
    got = bt.from_arrow(t).with_columns(
        r1=bt.row_number().over(partition_by="g", order_by="a"),
        r2=bt.row_number().over(order_by="v"),
    )
    assert_same(
        got.collect(),
        duck.sql(
            "select a, g, v, "
            "row_number() over (partition by g order by a) as r1, "
            "row_number() over (order by v) as r2 from t"
        ),
    )


def test_dependent_windows_are_not_transposed(t, duck):
    """A window aggregating another window's output must keep its position."""
    got = (
        bt.from_arrow(t)
        .with_columns(r1=bt.row_number().over(partition_by="g", order_by="a"))
        .with_columns(s=bt.col("r1").sum().over(order_by="v"))
    )
    assert_same(
        got.collect(),
        duck.sql(
            "select a, g, v, r1, sum(r1) over ("
            "order by v range between unbounded preceding and current row) as s "
            "from (select a, g, v, row_number() over ("
            "partition by g order by a) as r1 from t) q"
        ),
    )


def test_transpose_lets_the_collapse_rule_delete_a_window(t, duck):
    """The point of the transposition: three interleaved window specs become two nodes.

    Specs are written A, B, A. `collapse_adjacent_windows` only looks at a node and its
    immediate child, so without the transposition the two A windows are never neighbours
    and all three survive. Asserting the node count is what proves the rule paid off --
    the results would match either way.
    """
    got = (
        bt.from_arrow(t)
        .with_columns(r1=bt.row_number().over(order_by="v"))
        .with_columns(r2=bt.row_number().over(partition_by="g", order_by="a"))
        .with_columns(r3=bt.rank().over(order_by="v"))
    )
    assert got.explain().count("window") == 2
    assert_same(
        got.collect(),
        duck.sql(
            "select a, g, v, "
            "row_number() over (order by v) as r1, "
            "row_number() over (partition by g order by a) as r2, "
            "rank() over (order by v) as r3 from t"
        ),
    )
