"""Regression: SQL `ROUND(x, n)` and `SELECT DISTINCT ... ORDER BY`, vs DuckDB.

Both of these were silent wrong answers rather than missing features, which is the worst
shape a bug can take: the query ran, returned rows, and the rows were wrong.

`ROUND(x, n)` dropped `n` on the floor and rounded to a whole number, so `ROUND(2.0/3, 3)`
came back `1.0` instead of `0.667`. The DataFrame `.round(3)` was correct the whole time,
which is exactly why nothing caught it.

`SELECT DISTINCT ... ORDER BY` applied the sort *before* the dedup. The dedup is a hash
operation and does not preserve input order, so the ORDER BY was silently discarded and the
rows came back in hash order. The ordered assertions below are the point of this file: an
order-*independent* comparison cannot see either bug, and `assert_same` would have passed
against the broken engine.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from conftest import assert_same, assert_same_ordered

pytestmark = pytest.mark.differential


@pytest.fixture
def nums(duck):
    t = pa.table(
        {
            "x": [2.0 / 3, 1.0 / 4, 1.23456, -2.5, 0.0, 12345.6789],
            "g": ["a", "b", "a", "b", "a", "b"],
        }
    )
    duck.register("nums", t)
    return t


@pytest.fixture
def cities(duck):
    t = pa.table(
        {
            "city": ["c", "a", "b", "a", "c", "a"],
            "n": [3, 1, 2, 1, 3, 9],
        }
    )
    duck.register("cities", t)
    return t


@pytest.mark.parametrize(
    "query",
    [
        # The bug: `n` was discarded and every value rounded to a whole number.
        "SELECT ROUND(x, 3) AS r FROM nums",
        "SELECT ROUND(x, 1) AS r FROM nums",
        "SELECT ROUND(x, 0) AS r FROM nums",
        # A bare ROUND must keep working.
        "SELECT ROUND(x) AS r FROM nums",
        # Negative digits round to tens/hundreds.
        "SELECT ROUND(x, -2) AS r FROM nums",
        # In an expression, and through an aggregate.
        "SELECT ROUND(x * 2, 2) AS r FROM nums",
        "SELECT g, ROUND(SUM(x), 2) AS r FROM nums GROUP BY g",
    ],
)
def test_round_honors_the_digit_count(duck, nums, query):
    assert_same(bt.sql(query, nums=nums).collect(), duck.sql(query))


@pytest.mark.parametrize(
    "query",
    [
        # The bug: the sort ran before the dedup and was discarded by it.
        "SELECT DISTINCT city FROM cities ORDER BY city",
        "SELECT DISTINCT city FROM cities ORDER BY city DESC",
        "SELECT DISTINCT city, n FROM cities ORDER BY city, n",
        "SELECT DISTINCT city, n FROM cities ORDER BY n DESC, city",
        "SELECT DISTINCT city FROM cities ORDER BY city LIMIT 2",
        "SELECT DISTINCT city FROM cities ORDER BY city DESC LIMIT 2 OFFSET 1",
        # DISTINCT over a computed column, still ordered.
        "SELECT DISTINCT n * 2 AS d FROM cities ORDER BY d",
        # DISTINCT after a filter.
        "SELECT DISTINCT city FROM cities WHERE n > 1 ORDER BY city",
    ],
)
def test_distinct_then_order_by(duck, cities, query):
    # Ordered on purpose: an order-independent comparison cannot see this bug.
    assert_same_ordered(bt.sql(query, cities=cities).collect(), duck.sql(query))


def test_distinct_without_order_by_is_still_a_set(duck, cities):
    query = "SELECT DISTINCT city FROM cities"
    assert_same(bt.sql(query, cities=cities).collect(), duck.sql(query))
