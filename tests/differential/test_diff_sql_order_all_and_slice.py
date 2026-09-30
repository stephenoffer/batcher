"""Two DuckDB spellings the translator rejected as unsupported expressions.

``ORDER BY ALL`` sorts by every output column, left to right, and carries one direction
for all of them. sqlglot leaves the bare ``ALL`` keyword as a `Var`, which reached the
scalar path and raised ``unsupported SQL expression: Var``.

``a[lo:hi]`` is a list slice, 1-based and inclusive at both ends. The subscript handler
covered the single-index form and returned None for a `Slice`, which surfaced as
``unsupported SQL expression: Bracket``.

Ordering is the whole contract for the first group, so those use `assert_same_ordered`;
`assert_same` is order-independent by design and would pass on an unsorted result.

A *negative* slice bound counts back from each row's end. ``list.slice`` clamps a negative
offset to the start, so handing it one answered the wrong elements; the translator now
gathers the per-row positions instead (`sql_list_slice`), which is pinned below over lists
of several lengths.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered

pytestmark = pytest.mark.differential


def _t() -> pa.Table:
    return pa.table(
        {
            "g": pa.array(["b", "a", "b", None]),
            "x": pa.array([2, 3, 1, 4], pa.int64()),
        }
    )


@pytest.fixture
def tables(duck):
    duck.register("t", _t())
    return {"t": bt.from_arrow(_t())}


@pytest.mark.parametrize(
    "query",
    [
        "SELECT g, x FROM t ORDER BY ALL",
        "SELECT g, x FROM t ORDER BY ALL DESC",
        "SELECT x FROM t ORDER BY ALL",
        "SELECT x, g FROM t ORDER BY ALL",  # column order decides the sort order
        "SELECT g, sum(x) AS s FROM t GROUP BY g ORDER BY ALL",
        "SELECT DISTINCT g FROM t ORDER BY ALL",
        "SELECT g FROM t UNION SELECT g FROM t ORDER BY ALL",
        "SELECT g, x FROM t ORDER BY ALL LIMIT 2",
        # ROLLUP/CUBE/GROUPING SETS sort the *union* of their levels through a separate
        # path, which resolves each ORDER BY term against the SELECT list. `ALL` names no
        # term at all, so it fell into that resolution and was rejected as unresolvable —
        # found only by combining the two features, which no single-feature test does.
        "SELECT g, sum(x) AS s FROM t GROUP BY ROLLUP(g) ORDER BY ALL",
        "SELECT g, sum(x) AS s FROM t GROUP BY CUBE(g) ORDER BY ALL DESC",
    ],
)
def test_order_by_all_matches_duckdb(tables, duck, query):
    assert_same_ordered(bt.sql(query, **tables).collect(), duck.sql(query))


@pytest.mark.parametrize(
    "query",
    [
        "SELECT [10, 20, 30, 40][1:2] AS a",
        "SELECT [10, 20, 30, 40][2:] AS a",
        "SELECT [10, 20, 30, 40][:2] AS a",
        "SELECT [10, 20, 30][2:99] AS a",  # upper bound past the end
        "SELECT [10, 20, 30][3:2] AS a",  # empty range
        "SELECT [10, 20, 30][0:2] AS a",  # DuckDB reads a 0 lower bound as 1
        "SELECT s[2:3] AS a FROM (SELECT [1, 2, 3, 4] AS s)",
        "SELECT [1, 2, 3][2] AS a",  # the single-index form must keep working
    ],
)
def test_list_slice_matches_duckdb(query, duck):
    assert_same(bt.sql(query).collect(), duck.sql(query))


#: Every bound shape against lists of length 4, 0, 1 and NULL, in both spellings. The row
#: lengths matter because a negative bound is resolved per row.
_SLICE_ROWS = pa.table(
    {
        "id": pa.array([1, 2, 3, 4, 5], pa.int64()),
        "a": pa.array([[1, 2, 3, 4], [], None, [5], [None, 7, 8]], pa.list_(pa.int64())),
    }
)


@pytest.mark.parametrize(
    "form",
    [
        "a[-2:]",
        "a[-10:]",
        "a[1:-1]",
        "a[-3:-2]",
        "a[:-3]",
        "a[-1:1]",
        "a[-3:2]",
        "a[2:-10]",
        "a[0:-1]",
        "list_slice(a, -2, 4)",
        "list_slice(a, -3, -2)",
        "list_slice(a, 2, -1)",
    ],
)
def test_negative_slice_bounds_count_from_the_end_like_duckdb(duck, form):
    """A negative bound counts back from each row's end; `list_slice` follows the same rule.

    Both used to be wrong or refused: the bracket form raised, and `list_slice(a, -2, 4)`
    handed the negative offset to `list.slice`, which clamped it and returned `[2, 3, 4]`.
    Compared positionally on `id`, so a list with the right elements in the wrong row fails.
    """
    duck.register("t", _SLICE_ROWS)
    query = f"SELECT id, {form} AS r FROM t ORDER BY id"
    got = bt.sql(query, t=_SLICE_ROWS).collect()
    assert [tuple(r.values()) for r in got.to_pylist()] == duck.sql(query).fetchall()
