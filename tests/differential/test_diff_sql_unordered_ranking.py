"""Ranking windows with no ``ORDER BY`` — valid SQL the translator refused outright.

`ds.window` needs an ordering for every ranking function, so the translator raised
*"window ranking function 'rownumber' requires ORDER BY"* and the query never ran. Two
shapes do not need one, and both are ordinary SQL:

- ``row_number() OVER ()`` numbers the relation. That is `with_row_index`, a single
  counter, so the sequential and parallel paths agree on an order.
- ``rank()``, ``dense_rank()``, ``cume_dist()`` and ``percent_rank()`` over an unordered
  window are *constants*: with nothing to order by, every row is a peer of every other,
  so they all rank together. This holds with or without a ``PARTITION BY``.

``row_number() OVER (PARTITION BY g)`` is genuinely different — it has to number within
each partition, which needs the operator — and still raises. The last test pins that,
so the decline is not quietly widened into a wrong answer later.

Row order is not asserted: none of these queries has an ORDER BY, so neither engine
promises one. `assert_same` compares the row multiset, which is what is actually
specified, and it still sees a wrong *value* on any row.
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
            "id": pa.array([1, 2, 3, 4], pa.int64()),
            "g": pa.array(["a", "a", "b", "b"]),
            "x": pa.array([10, 20, 30, 40], pa.int64()),
        }
    )


@pytest.fixture
def tables(duck):
    duck.register("t", _t())
    return {"t": bt.from_arrow(_t())}


@pytest.mark.parametrize(
    "query",
    [
        "SELECT id, row_number() OVER () AS r FROM t",
        "SELECT id, rank() OVER () AS r FROM t",
        "SELECT id, dense_rank() OVER () AS r FROM t",
        "SELECT id, percent_rank() OVER () AS r FROM t",
        "SELECT id, cume_dist() OVER () AS r FROM t",
        # A PARTITION BY without an ORDER BY: still all peers, per partition.
        "SELECT id, rank() OVER (PARTITION BY g) AS r FROM t",
        "SELECT id, dense_rank() OVER (PARTITION BY g) AS r FROM t",
        # Mixed with an ordinary ordered window in the same query.
        "SELECT id, row_number() OVER () AS r, sum(x) OVER (ORDER BY id) AS s FROM t",
        # The ordered forms must be untouched.
        "SELECT id, row_number() OVER (ORDER BY id DESC) AS r FROM t",
        "SELECT id, rank() OVER (PARTITION BY g ORDER BY x) AS r FROM t",
    ],
)
def test_unordered_ranking_matches_duckdb(tables, duck, query):
    assert_same(bt.sql(query, **tables).collect(), duck.sql(query))


def test_row_number_over_nothing_numbers_every_row_once(tables):
    got = bt.sql("SELECT row_number() OVER () AS r FROM t", **tables).collect().to_pydict()
    assert sorted(got["r"]) == [1, 2, 3, 4]


def test_partitioned_row_number_without_an_order_still_declines(tables):
    """It must number *within* each partition, which needs an ordering to be defined."""
    with pytest.raises(NotImplementedError, match="requires ORDER BY"):
        bt.sql("SELECT row_number() OVER (PARTITION BY g) AS r FROM t", **tables).collect()
