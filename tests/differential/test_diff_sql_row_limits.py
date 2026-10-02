"""``FETCH FIRST n ROWS ONLY`` is the ANSI row limit, and it crashed; PERCENT was ignored.

sqlglot parses ``FETCH`` into an `exp.Fetch` (count in ``count``) rather than an
`exp.Limit` (count in ``expression``), and both land in the same ``limit`` slot. Reading
only the `Limit` shape meant standard SQL failed with an internal
``AttributeError: 'NoneType' object has no attribute 'this'``.

The modifiers on that slot were worse than a crash. ``LIMIT n PERCENT`` and
``FETCH ... WITH TIES`` were parsed, the modifier dropped, and the bare row count applied
— so ``LIMIT 20 PERCENT`` over five rows returned **all five** where DuckDB returns one.
Both were then declined, because a silently different row count is the failure mode
nothing downstream can detect; both are now answered exactly, over ranking windows
(`_sql/parser/windowing/limits.py`). DuckDB is the oracle for PERCENT. It has no WITH TIES,
so that one is held to the SQL-standard definition written out in `_with_ties`.

Ordering is part of the contract for every case here, so these use `assert_same_ordered`:
an order-independent comparison cannot tell ``LIMIT 3`` over a sort from ``LIMIT 3`` over
an arbitrary three rows.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.differential


def _t() -> pa.Table:
    return pa.table({"x": pa.array([5, 1, 4, 2, 3], pa.int64())})


@pytest.fixture
def tables(duck):
    duck.register("t", _t())
    return {"t": bt.from_arrow(_t())}


@pytest.mark.parametrize(
    "query",
    [
        "SELECT x FROM t ORDER BY x FETCH FIRST 3 ROWS ONLY",
        "SELECT x FROM t ORDER BY x FETCH NEXT 2 ROWS ONLY",
        # An omitted count means one row, and sqlglot leaves the bare `ROW` keyword in
        # the count slot as an identifier rather than a number.
        "SELECT x FROM t ORDER BY x FETCH FIRST ROW ONLY",
        "SELECT x FROM t ORDER BY x FETCH NEXT ROWS ONLY",
        "SELECT x FROM t ORDER BY x OFFSET 1 ROWS FETCH NEXT 2 ROWS ONLY",
        # The LIMIT spellings must keep working — the two share one code path now.
        "SELECT x FROM t ORDER BY x LIMIT 2",
        "SELECT x FROM t ORDER BY x LIMIT 2 OFFSET 1",
        "SELECT x FROM t ORDER BY x OFFSET 2",
        # A set operation applies the row limit to the combined result, through the same
        # helper, so it is covered here too.
        "SELECT x FROM t UNION ALL SELECT x FROM t ORDER BY x FETCH FIRST 3 ROWS ONLY",
    ],
)
def test_row_limits_match_duckdb(tables, duck, query):
    assert_same_ordered(bt.sql(query, **tables).collect(), duck.sql(query))


#: A relation with ties on `k` and a unique `id`, for the two modifiers.
_TIED = pa.table(
    {
        "id": pa.array([1, 2, 3, 4, 5, 6, 7], pa.int64()),
        "k": pa.array([1, 2, 2, 2, 3, 4, 4], pa.int64()),
    }
)


@pytest.mark.parametrize("percent", [0, 10, 14.29, 20, 42.86, 50, 99.99, 100])
@pytest.mark.parametrize(
    "shape",
    [
        "SELECT id FROM u ORDER BY id DESC LIMIT {p} PERCENT",
        "SELECT id AS i FROM u ORDER BY k, i LIMIT {p} PERCENT OFFSET 2",
        "SELECT DISTINCT k FROM u ORDER BY k LIMIT {p} PERCENT",
        "SELECT k, count(*) AS c FROM u GROUP BY k ORDER BY 2 DESC, k LIMIT {p} PERCENT",
        "SELECT id FROM u UNION ALL SELECT id + 10 FROM u ORDER BY 1 LIMIT {p} PERCENT OFFSET 6",
    ],
)
def test_limit_percent_matches_duckdb(duck, shape, percent):
    """``floor(p * n / 100)`` rows of the ordered result, after the OFFSET, as DuckDB keeps."""
    duck.register("u", _TIED)
    query = shape.format(p=percent)
    assert_same_ordered(bt.sql(query, u=_TIED).collect(), duck.sql(query))


def test_limit_percent_outside_0_to_100_is_refused():
    with pytest.raises(PlanError, match="PERCENT"):
        bt.sql("SELECT id FROM u LIMIT 150 PERCENT", u=_TIED).collect()


def _with_ties(n: int, skip: int, descending: bool) -> list[int]:
    """The `k` values FETCH ... WITH TIES keeps: rows skip+1..skip+n plus the last one's peers.

    DuckDB has no WITH TIES, so this is the SQL-standard definition written out directly.
    """
    keys = sorted(_TIED.column("k").to_pylist(), reverse=descending)
    kept = keys[skip : skip + n]
    return kept + [k for k in keys[skip + n :] if kept and k == kept[-1]]


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("skip", [0, 1, 2, 5])
@pytest.mark.parametrize("n", [0, 1, 2, 4, 7, 9])
def test_fetch_with_ties_keeps_the_peers_of_the_last_row(n, skip, descending):
    """Which *rows* of a tie the OFFSET skips is open, so the ordered keys are compared."""
    order = "k DESC" if descending else "k"
    query = (
        f"SELECT id, k FROM u ORDER BY {order} OFFSET {skip} ROWS FETCH FIRST {n} ROWS WITH TIES"
    )
    got = bt.sql(query, u=_TIED).collect().column("k").to_pylist()
    assert got == _with_ties(n, skip, descending)


def test_fetch_with_ties_without_offset_keeps_whole_rows():
    query = "SELECT id FROM u ORDER BY k LIMIT 2 WITH TIES"
    assert sorted(bt.sql(query, u=_TIED).to_pydict()["id"]) == [1, 2, 3, 4]


def test_fetch_percent_with_ties_combines_both():
    # 30% of 7 rows is 2, and the second row (k = 2) has two more peers.
    query = "SELECT id FROM u ORDER BY k FETCH FIRST 30 PERCENT ROWS WITH TIES"
    assert sorted(bt.sql(query, u=_TIED).to_pydict()["id"]) == [1, 2, 3, 4]


def test_with_ties_needs_an_order_by():
    with pytest.raises(NotImplementedError, match="ORDER BY"):
        bt.sql("SELECT id FROM u FETCH FIRST 2 ROWS WITH TIES", u=_TIED).collect()
