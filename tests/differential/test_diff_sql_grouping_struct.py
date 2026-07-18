"""Differential coverage for SQL grouping/aggregation *structure* vs DuckDB.

Pins four fixes in ``_sql/parser``:

* ``GROUP BY ALL`` — group by every non-aggregate SELECT item (was collapsing to a
  grand total / raising ``PlanError``).
* ``GROUPING_ID(...)`` — the bit-vector spelling of ``GROUPING(...)`` (was
  ``unsupported aggregate: groupingid``).
* ``BOOL_AND`` / ``BOOL_OR`` NULL semantics — a group with no non-null input yields
  NULL, not a spurious TRUE/FALSE from the old ``COUNT(*) FILTER`` rewrite.
* Duplicate GROUP BY keys (``GROUP BY region, region``) — deduped, not a duplicate
  output-column error.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from conftest import assert_same

pytestmark = pytest.mark.differential


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "region": ["us", "us", "eu", "eu", "us", None],
            "product": ["a", "b", "a", "b", "a", "a"],
            "amt": [10, 20, 30, 40, 50, 5],
            "qty": [1, 2, 3, 4, 5, None],
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.mark.parametrize(
    "q",
    [
        "SELECT region, SUM(amt) AS s FROM t GROUP BY ALL",
        "SELECT region, product, SUM(amt) AS s FROM t GROUP BY ALL",
        "SELECT amt % 2 AS parity, COUNT(*) AS c FROM t GROUP BY ALL",
        "SELECT region FROM t GROUP BY ALL",
        "SELECT SUM(amt) AS s FROM t GROUP BY ALL",
        "SELECT region, product, SUM(amt) AS s FROM t GROUP BY ALL HAVING SUM(amt) > 20",
        "SELECT amt + qty AS e, COUNT(*) AS c FROM t WHERE qty IS NOT NULL GROUP BY ALL",
    ],
)
def test_group_by_all(duck, t, q):
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT region, product, GROUPING_ID(region, product) AS g, SUM(amt) AS s "
        "FROM t GROUP BY CUBE(region, product)",
        "SELECT region, GROUPING_ID(region) AS g, SUM(amt) AS s FROM t GROUP BY ROLLUP(region)",
        # GROUPING_ID and GROUPING agree on the same set.
        "SELECT region, product, GROUPING(region, product) AS g, "
        "GROUPING_ID(region, product) AS gid, SUM(amt) AS s FROM t GROUP BY CUBE(region, product)",
        # In a plain GROUP BY nothing is rolled up, so GROUPING_ID is 0.
        "SELECT region, GROUPING_ID(region) AS g, SUM(amt) AS s FROM t GROUP BY region",
    ],
)
def test_grouping_id(duck, t, q):
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        # The all-null group (region NULL, single row, qty NULL) must yield NULL, not
        # TRUE/FALSE.
        "SELECT region, BOOL_AND(qty > 0) AS a, BOOL_OR(qty > 3) AS o FROM t GROUP BY region",
        # Empty input → NULL.
        "SELECT BOOL_AND(qty > 0) AS a, BOOL_OR(qty > 3) AS o FROM t WHERE region = 'zz'",
        # FILTER pushes the guard inside; a guarded-away group is still NULL.
        "SELECT region, BOOL_AND(qty > 0) FILTER (WHERE amt > 5) AS a FROM t GROUP BY region",
        # Ordinary non-null cases still match.
        "SELECT region, BOOL_AND(amt > 5) AS a, BOOL_OR(amt > 45) AS o FROM t GROUP BY region",
    ],
)
def test_bool_and_or_null_semantics(duck, t, q):
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT region, SUM(amt) AS s FROM t GROUP BY region, region",
        "SELECT region, SUM(amt) AS s FROM t GROUP BY 1, region",
        "SELECT amt % 2 AS m, SUM(amt) AS s FROM t GROUP BY amt % 2, amt % 2",
    ],
)
def test_duplicate_group_keys(duck, t, q):
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))
