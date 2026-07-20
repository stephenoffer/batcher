"""`QUALIFY` vs DuckDB, including the form where the window function is only in QUALIFY.

`QUALIFY <window expr> <cmp>` is the idiomatic "top-N per group" filter — and the usual
spelling never mentions the window function in the SELECT list. That form previously
raised: QUALIFY was supported only when the SELECT itself computed the window function
and QUALIFY referenced its alias.

The window column has to exist before it can be filtered, so it is now materialized under
a `__qualify<n>` alias, filtered, and dropped by the projection. The tests that matter are
the ones proving the helper column does not leak into the output and that several window
functions in one predicate each get their own.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered


@pytest.fixture
def t(duck):
    table = pa.table(
        {
            "k": ["a", "a", "b", "b", "b"],
            "v": [3, 1, 2, 5, 4],
            "w": [10, 20, 30, 40, 50],
        }
    )
    duck.register("t", table)
    return table


@pytest.mark.differential
@pytest.mark.parametrize(
    "predicate",
    [
        "row_number() OVER (PARTITION BY k ORDER BY v) = 1",
        "rank() OVER (PARTITION BY k ORDER BY v) <= 2",
        "dense_rank() OVER (PARTITION BY k ORDER BY v) = 1",
        "sum(v) OVER (PARTITION BY k) > 4",
        "row_number() OVER (ORDER BY v) <= 3",
    ],
)
def test_qualify_window_only_in_qualify(duck, t, predicate):
    """The window function appears solely in QUALIFY — the idiomatic spelling."""
    query = f"SELECT k, v FROM t QUALIFY {predicate}"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


@pytest.mark.differential
def test_qualify_does_not_leak_helper_columns(duck, t):
    """The materialized window column must not appear in the output."""
    query = "SELECT k, v FROM t QUALIFY row_number() OVER (PARTITION BY k ORDER BY v) = 1"
    out = bt.sql(query, t=t).collect()
    assert out.column_names == ["k", "v"]
    assert not any(c.startswith("__qualify") for c in out.column_names)
    assert_same(out, duck.sql(query))


@pytest.mark.differential
def test_qualify_with_two_window_functions(duck, t):
    """Two window expressions in one predicate each get their own hidden column."""
    query = (
        "SELECT k, v FROM t "
        "QUALIFY row_number() OVER (PARTITION BY k ORDER BY v) = 1 "
        "AND sum(v) OVER (PARTITION BY k) > 3"
    )
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


@pytest.mark.differential
def test_qualify_with_order_by(duck, t):
    """ORDER BY after a QUALIFY still applies to the filtered rows."""
    query = (
        "SELECT k, v FROM t QUALIFY row_number() OVER (PARTITION BY k ORDER BY v) = 1 ORDER BY k"
    )
    assert_same_ordered(bt.sql(query, t=t).collect(), duck.sql(query))


@pytest.mark.differential
def test_qualify_referencing_a_select_alias_still_works(duck, t):
    """The previously-supported form — window computed in SELECT — must not regress."""
    query = "SELECT k, v, row_number() OVER (PARTITION BY k ORDER BY v) AS rn FROM t QUALIFY rn = 1"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


@pytest.mark.differential
def test_qualify_with_where(duck, t):
    """WHERE filters rows *before* the window sees them; QUALIFY filters after."""
    query = (
        "SELECT k, v FROM t WHERE v > 1 QUALIFY row_number() OVER (PARTITION BY k ORDER BY v) = 1"
    )
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


def test_qualify_with_group_by_rejects(t):
    """QUALIFY over an aggregate is not supported and must say so rather than mislead."""
    query = "SELECT k, sum(v) AS s FROM t GROUP BY k QUALIFY sum(v) > 3"
    with pytest.raises(NotImplementedError, match="QUALIFY"):
        bt.sql(query, t=t).collect()
