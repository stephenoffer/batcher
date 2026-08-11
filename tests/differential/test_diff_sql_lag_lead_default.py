"""``lag(x, n, default)`` / ``lead(x, n, default)`` — the third argument fills the edges.

The window operator has no default parameter, so the translator refused the three-argument
form outright rather than honour the offset and drop the default. But "the offset falls
outside the partition" is expressible with window functions that *are* supported, so the
whole thing is an AST rewrite:

    lag(x, n, d)  OVER w  ->  CASE WHEN row_number() OVER w <= n THEN d ELSE lag(x, n) END
    lead(x, n, d) OVER w  ->  CASE WHEN row_number() OVER w > count(*) OVER p - n
                                   THEN d ELSE lead(x, n) END

``COALESCE(lag(x, n), d)`` is the obvious one-liner and it is **wrong**: it also replaces a
NULL that `x` genuinely holds inside the partition, where SQL keeps the NULL. The data here
puts a real NULL at ``id = 2`` precisely so that a coalesce-based implementation fails —
``lag(x, 1, -1)`` at ``id = 3`` must be NULL, not ``-1``, and only the row's *position*
decides.
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
            "id": pa.array([1, 2, 3, 4, 5], pa.int64()),
            "g": pa.array(["a", "a", "a", "b", "b"]),
            # The NULL at id=2 is load-bearing; see the module docstring.
            "x": pa.array([10, None, 30, 40, 50], pa.int64()),
        }
    )


@pytest.fixture
def tables(duck):
    duck.register("t", _t())
    return {"t": bt.from_arrow(_t())}


@pytest.mark.parametrize(
    "query",
    [
        "SELECT id, lag(x, 1, -1) OVER (ORDER BY id) AS s FROM t",
        "SELECT id, lag(x, 2, -1) OVER (ORDER BY id) AS s FROM t",
        "SELECT id, lead(x, 1, -1) OVER (ORDER BY id) AS s FROM t",
        "SELECT id, lead(x, 2, -9) OVER (ORDER BY id) AS s FROM t",
        # Per partition: the fill applies at each partition's edge, not only the first.
        "SELECT id, lag(x, 1, 0) OVER (PARTITION BY g ORDER BY id) AS s FROM t",
        "SELECT id, lead(x, 1, 0) OVER (PARTITION BY g ORDER BY id) AS s FROM t",
        # An offset larger than the partition: every row is out of range.
        "SELECT id, lag(x, 9, -1) OVER (PARTITION BY g ORDER BY id) AS s FROM t",
        # The default may be an expression rather than a literal.
        "SELECT id, lag(x, 1, id * 100) OVER (ORDER BY id) AS s FROM t",
        # Combined with the expression-argument hoisting.
        "SELECT id, lag(x * 2, 1, -1) OVER (ORDER BY id) AS s FROM t",
        # The two-argument and one-argument forms must be untouched by the rewrite.
        "SELECT id, lag(x, 1) OVER (ORDER BY id) AS s FROM t",
        "SELECT id, lead(x) OVER (ORDER BY id) AS s FROM t",
    ],
)
def test_offset_default_matches_duckdb(tables, duck, query):
    assert_same(bt.sql(query, **tables).collect(), duck.sql(query))


def test_a_real_null_inside_the_partition_is_not_replaced_by_the_default(tables):
    """The assertion a COALESCE-based implementation fails: NULL at id=3, not -1."""
    got = (
        bt.sql("SELECT id, lag(x, 1, -1) OVER (ORDER BY id) AS s FROM t", **tables)
        .collect()
        .to_pydict()
    )
    by_id = dict(zip(got["id"], got["s"], strict=True))
    assert by_id[1] == -1  # out of range: the default applies
    assert by_id[3] is None  # x[id=2] is genuinely NULL: the default must NOT apply
