"""A NULL inside an ``IN`` list is not a comparable value, and must not be compared.

``x IN (a, NULL)`` is TRUE when ``x = a`` and NULL otherwise — never FALSE — because a
comparison against NULL is unknown. The lowering built one equality per list item, so the
NULL became ``x = NULL``; the untyped NULL literal lowers as Int64, and the query died
with ``Invalid comparison operation: Utf8 == Int64`` on any text column. Ordinary SQL
(``WHERE country IN ('US', NULL)``, which a generated IN list produces routinely) could
not run at all.

The three-valued cases are the point, and they are the ones an order-independent value
comparison *can* see, because they change which rows survive a WHERE:

    WHERE g IN ('a', NULL)      -- keeps only g = 'a'
    WHERE g NOT IN ('a', NULL)  -- keeps nothing at all, for any g

The second is the one that looks wrong and is right: ``NOT IN`` with a NULL in the list is
NULL for every row that is not an outright match, and NULL does not pass a WHERE.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _t() -> pa.Table:
    return pa.table(
        {
            "g": pa.array(["a", "b", None]),
            "n": pa.array([1, 2, None], pa.int64()),
        }
    )


@pytest.fixture
def tables(duck):
    duck.register("t", _t())
    return {"t": bt.from_arrow(_t())}


@pytest.mark.parametrize(
    "query",
    [
        # Filters: the NULL changes which rows survive.
        "SELECT g FROM t WHERE g IN ('a', NULL)",
        "SELECT g FROM t WHERE g NOT IN ('a', NULL)",
        "SELECT g FROM t WHERE g IN (NULL)",
        "SELECT g FROM t WHERE g NOT IN (NULL)",
        "SELECT n FROM t WHERE n IN (1, NULL)",
        "SELECT n FROM t WHERE n NOT IN (1, NULL)",
        # Projections: the three-valued result itself, TRUE/NULL and FALSE/NULL.
        "SELECT g, g IN ('a', NULL) AS r FROM t",
        "SELECT g, g NOT IN ('a', NULL) AS r FROM t",
        "SELECT g, g IN (NULL) AS r FROM t",
        "SELECT n, n IN (1, NULL) AS r FROM t",
        # A NULL-free list must keep behaving exactly as before.
        "SELECT g FROM t WHERE g IN ('a', 'b')",
        "SELECT g FROM t WHERE g NOT IN ('a')",
        "SELECT g, g IN ('a', 'b') AS r FROM t",
    ],
)
def test_null_in_list_matches_duckdb(tables, duck, query):
    assert_same(bt.sql(query, **tables).collect(), duck.sql(query))


def test_not_in_with_a_null_keeps_no_rows(tables):
    """Pinned on its own: the answer is empty for every row, which is easy to "fix" wrongly."""
    got = bt.sql("SELECT g FROM t WHERE g NOT IN ('a', NULL)", **tables).collect()
    assert got.num_rows == 0


def test_in_with_a_null_is_true_or_null_but_never_false(tables):
    got = bt.sql("SELECT g IN ('a', NULL) AS r FROM t", **tables).collect().to_pydict()["r"]
    assert got == [True, None, None]
