"""``FETCH FIRST n ROWS ONLY`` is the ANSI row limit, and it crashed; PERCENT was ignored.

sqlglot parses ``FETCH`` into an `exp.Fetch` (count in ``count``) rather than an
`exp.Limit` (count in ``expression``), and both land in the same ``limit`` slot. Reading
only the `Limit` shape meant standard SQL failed with an internal
``AttributeError: 'NoneType' object has no attribute 'this'``.

The modifiers on that slot were worse than a crash. ``LIMIT n PERCENT`` and
``FETCH ... WITH TIES`` were parsed, the modifier dropped, and the bare row count applied
— so ``LIMIT 20 PERCENT`` over five rows returned **all five** where DuckDB returns one.
Both are now declined, because a silently different row count is the failure mode nothing
downstream can detect.

Ordering is part of the contract for every case here, so these use `assert_same_ordered`:
an order-independent comparison cannot tell ``LIMIT 3`` over a sort from ``LIMIT 3`` over
an arbitrary three rows.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered

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


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ("SELECT x FROM t ORDER BY x LIMIT 20 PERCENT", "PERCENT"),
        ("SELECT x FROM t ORDER BY x FETCH FIRST 2 ROWS WITH TIES", "WITH TIES"),
    ],
)
def test_unsupported_limit_modifiers_are_declined_not_dropped(tables, query, message):
    """Dropping the modifier returns a different number of rows and reports success."""
    with pytest.raises(NotImplementedError, match=message):
        bt.sql(query, **tables).collect()
