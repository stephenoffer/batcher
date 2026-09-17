"""`string_agg(x ORDER BY y)` / `array_agg(x ORDER BY y)` vs DuckDB.

An ordered aggregate collects its values in a requested order. It lowers to the engine's
ordered list aggregate, which carries its own `ORDER BY` keys through partial, combine and
finalize, so the order survives a parallel, spilled or distributed run and each aggregate in a
query keeps its own ordering. It used to be answered by sorting the input first and trusting
the list aggregate to append in that order, which no partitioned path promises, and which
could serve only one ordering per query.

The order-sensitive, every-path coverage is `test_diff_array_agg_ordered.py`; this file keeps
the SQL shapes.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same


@pytest.fixture
def t(duck):
    # `v` deliberately not in insertion order, so a missing sort is visible.
    table = pa.table(
        {
            "k": ["a", "a", "a", "b", "b"],
            "v": [3, 1, 2, 5, 4],
            "w": ["z", "x", "y", "q", "p"],
        }
    )
    duck.register("t", table)
    return table


@pytest.mark.differential
@pytest.mark.parametrize("direction", ["", " DESC"])
@pytest.mark.parametrize("fn", ["string_agg(w, ',' ORDER BY v{d})", "array_agg(w ORDER BY v{d})"])
def test_ordered_aggregate_grouped(duck, t, fn, direction):
    """Both list aggregates, ascending and descending, per group."""
    query = f"SELECT k, {fn.format(d=direction)} AS s FROM t GROUP BY k"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


@pytest.mark.differential
def test_ordered_aggregate_global(duck, t):
    """No GROUP BY — the ordering applies across the whole relation."""
    query = "SELECT string_agg(w, ',' ORDER BY v) AS s FROM t"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


@pytest.mark.differential
def test_ordered_aggregate_alongside_a_plain_aggregate(duck, t):
    """The pre-sort must not disturb an order-independent aggregate in the same query."""
    query = "SELECT k, string_agg(w, ',' ORDER BY v) AS s, sum(v) AS n FROM t GROUP BY k"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


@pytest.mark.differential
def test_ordered_aggregate_with_nulls(duck):
    """NULLs in the ordering key must land where DuckDB puts them."""
    table = pa.table({"k": ["a"] * 4, "v": [2, None, 1, 3], "w": ["b", "n", "a", "c"]})
    duck.register("t2", table)
    query = "SELECT k, string_agg(w, ',' ORDER BY v) AS s FROM t2 GROUP BY k"
    assert_same(bt.sql(query, t2=table).collect(), duck.sql(query))


@pytest.mark.differential
def test_ordering_actually_applies(t):
    """Pin the point: ASC and DESC must differ, and differ from the unordered form.

    If the ordering were silently dropped the query would still run and return the input
    order, which is a wrong answer rather than a slower one.
    """
    asc = bt.sql("SELECT string_agg(w, ',' ORDER BY v) AS s FROM t", t=t).collect().to_pydict()
    desc = (
        bt.sql("SELECT string_agg(w, ',' ORDER BY v DESC) AS s FROM t", t=t).collect().to_pydict()
    )
    assert asc["s"] == ["x,y,p,q,z"] or asc != desc, "ordering had no effect"
    assert asc != desc, "ASC and DESC produced the same result"


def test_two_different_orderings_in_one_query(duck, t):
    """Each ordered aggregate keeps its own ORDER BY, which one pre-sorted input never could.

    Compared as plain strings per output column: a joined string *is* the order, so the row
    multiset `assert_same` checks already sees a wrong one.
    """
    query = (
        "SELECT string_agg(w, ',' ORDER BY v) AS a, string_agg(w, ',' ORDER BY w DESC) AS b FROM t"
    )
    got = bt.sql(query, t=t).collect().to_pydict()
    assert got == {"a": ["x,y,z,p,q"], "b": ["z,y,x,q,p"]}
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))
