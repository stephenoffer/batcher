"""Join orders chosen past the DP's budget, and over a date dimension's ascending columns.

Two changes move plans here and neither may move a result. When the connected-subset DP
cannot afford a join graph, the order is the cheaper of a left-deep greedy tree and a bushy
greedy-operator-ordering one (`kyber.rules.joins.order_goo`); and a filter on one of a
dimension's ascending columns narrows the estimates of the others
(`kyber.stats.comonotone`), which is what re-orders a q72-shaped query. Each query runs with
the DP forced to bail as well as with its normal budget, against DuckDB both times, with
NULL keys, duplicate matches and a week join that fans out sevenfold.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.joins.order as order
from _harness import assert_same_for_query

pytestmark = pytest.mark.differential

_DAYS = 3 * 364
_DATES = pa.table(
    {
        "d_sk": pa.array([100 + i for i in range(_DAYS)], pa.int64()),
        "d_week": pa.array([i // 7 for i in range(_DAYS)], pa.int64()),
        "d_year": pa.array([2000 + i // 364 for i in range(_DAYS)], pa.int64()),
        "d_day": pa.array(list(range(_DAYS)), pa.int64()),
    }
)
_INV = pa.table(
    {
        "inv_item": pa.array([None if i % 101 == 0 else i % 40 for i in range(3_000)], pa.int64()),
        "inv_date": pa.array([100 + 7 * (i % 150) for i in range(3_000)], pa.int64()),
        "inv_wh": pa.array([i % 3 for i in range(3_000)], pa.int64()),
        "inv_qty": pa.array([i % 13 for i in range(3_000)], pa.int64()),
    }
)
_SALES = pa.table(
    {
        "s_item": pa.array([None if i % 97 == 0 else i % 40 for i in range(2_000)], pa.int64()),
        "s_date": pa.array([100 + (i * 5) % _DAYS for i in range(2_000)], pa.int64()),
        "s_ship": pa.array([100 + (i * 5 + 9) % _DAYS for i in range(2_000)], pa.int64()),
        "s_cust": pa.array([i % 50 for i in range(2_000)], pa.int64()),
        "s_qty": pa.array([i % 11 for i in range(2_000)], pa.int64()),
    }
)
_WH = pa.table({"w_sk": pa.array([0, 1, 2], pa.int64()), "w_name": ["a", "b", "c"]})
_CUST = pa.table(
    {"c_sk": pa.array(list(range(50)), pa.int64()), "c_flag": [int(i % 5 == 1) for i in range(50)]}
)

_TABLES = {"dates": _DATES, "inv": _INV, "sales": _SALES, "wh": _WH, "cust": _CUST}

_QUERIES = [
    # q72's spine: two facts on item, the inventory date joined to the sale date by week.
    "SELECT w_name, d1.d_week AS wk, count(*) AS n, sum(s_qty) AS q "
    "FROM sales JOIN inv ON s_item = inv_item JOIN wh ON inv_wh = w_sk "
    "JOIN cust ON s_cust = c_sk JOIN dates d1 ON s_date = d1.d_sk "
    "JOIN dates d2 ON inv_date = d2.d_sk JOIN dates d3 ON s_ship = d3.d_sk "
    "WHERE d1.d_week = d2.d_week AND inv_qty < s_qty AND d3.d_day > d1.d_day + 5 "
    "AND c_flag = 1 AND d1.d_year = 2001 GROUP BY w_name, d1.d_week",
    # The week join alone above a year filter: the sevenfold fan-out itself.
    "SELECT count(*) AS n, sum(inv_qty) AS q FROM inv "
    "JOIN dates d2 ON inv_date = d2.d_sk JOIN dates d1 ON d1.d_week = d2.d_week "
    "JOIN wh ON inv_wh = w_sk WHERE d1.d_year = 2000",
    # A range on the surrogate key that the year's run satisfies entirely.
    "SELECT count(*) AS n FROM sales JOIN dates ON s_date = d_sk JOIN cust ON s_cust = c_sk "
    "JOIN inv ON s_item = inv_item WHERE d_year = 2002 AND d_sk BETWEEN 828 AND 1191",
]


def _session() -> bt.Session:
    s = bt.Session()
    for name, table in _TABLES.items():
        s.register(name, table)
    return s


@pytest.mark.parametrize("dp", ["budgeted", "bailed"])
@pytest.mark.parametrize("query", _QUERIES)
def test_join_order_fallback_matches_duckdb(duck, query, dp, monkeypatch):
    if dp == "bailed":
        monkeypatch.setattr(order, "search_pair_budget", lambda region, ctx: 0)
    for name, table in _TABLES.items():
        duck.register(name, table)
    expected = duck.sql(query)
    got = _session().sql(query).collect()
    assert got.num_rows > 0, "control: every query must find rows, or it compares nothing"
    assert_same_for_query(got, expected, query)
