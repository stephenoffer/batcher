"""An outer join beneath a join keyed on its null-supplied side, checked against DuckDB.

`outer_to_inner_under_join` strengthens the outer join when the join above it is keyed on a
column the outer join null-extends: an equi-join key never matches NULL, so the padded rows
cannot reach the output. TPC-DS q93 is the shape -- `store_sales LEFT JOIN store_returns`
under a comma join to `reason` whose `WHERE` becomes the inner join's key.

The fixture holds every row the rewrite has to get right: a sale with no return (null-extended
by the left join), a return whose reason is NULL, a reason no return names, and a duplicated
key on each side. The anti form must keep the rewrite away -- an anti join keeps the
null-keyed rows -- and the semi form must take it.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_SALES = {
    "item": [1, 1, 2, 3, 4, 5],
    "tick": [1, 2, 1, 1, 1, 1],
    "qty": [5, 6, 7, 8, 9, 10],
}
# (1,1) and (1,2) return for reason 10; (2,1) has a NULL reason; (3,1) and (4,1) never return.
_RETS = {
    "r_item": [1, 1, 2, 5, 5],
    "r_tick": [1, 2, 1, 1, 1],
    "reason": [10, 10, None, 20, 30],
}
_REASONS = {"reason_id": [10, 20, 40, 10], "descr": ["late", "broken", "late", "dup"]}


def _session(reasons: dict) -> bt.Session:
    s = bt.Session()
    s.register("sales", bt.from_pydict(_SALES))
    s.register("rets", bt.from_pydict(_RETS))
    s.register("reasons", bt.from_pydict(reasons))
    return s


@pytest.fixture
def duck_tables(duck):
    duck.register("sales", pa.table(_SALES))
    duck.register("rets", pa.table(_RETS))
    duck.register("reasons", pa.table(_REASONS))
    return duck


_CASES = {
    # TPC-DS q93's form: the WHERE over the comma join becomes the inner join's key.
    "q93_shape": (
        "SELECT item, qty, reason FROM sales LEFT OUTER JOIN rets "
        "ON (r_item = item AND r_tick = tick), reasons "
        "WHERE reason = reason_id AND descr = 'late'"
    ),
    "aggregated": (
        "SELECT item, sum(CASE WHEN reason IS NOT NULL THEN qty ELSE 0 END) AS s "
        "FROM sales LEFT OUTER JOIN rets ON (r_item = item AND r_tick = tick), reasons "
        "WHERE reason = reason_id GROUP BY item"
    ),
    "semi_via_in": (
        "SELECT item, qty FROM sales LEFT JOIN rets ON (r_item = item AND r_tick = tick) "
        "WHERE reason IN (SELECT reason_id FROM reasons WHERE descr = 'late')"
    ),
    # An anti join keeps the padded rows: (3,1) and (4,1) have no return at all.
    "anti_keeps_padded_rows": (
        "SELECT item, qty FROM sales LEFT JOIN rets ON (r_item = item AND r_tick = tick) "
        "WHERE NOT EXISTS (SELECT 1 FROM reasons WHERE reason_id = reason)"
    ),
    "right_join_mirror": (
        "SELECT item, qty, reason FROM rets RIGHT OUTER JOIN sales "
        "ON (r_item = item AND r_tick = tick), reasons WHERE reason = reason_id"
    ),
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_matches_duckdb(duck_tables, name: str) -> None:
    sql = _CASES[name]
    assert_same(_session(_REASONS).sql(sql).collect(), duck_tables.sql(sql))


def test_an_empty_dimension_returns_nothing(duck) -> None:
    empty = {"reason_id": pa.array([], pa.int64()), "descr": pa.array([], pa.string())}
    duck.register("sales", pa.table(_SALES))
    duck.register("rets", pa.table(_RETS))
    duck.register("reasons", pa.table(empty))
    sql = _CASES["q93_shape"]
    got = _session(empty).sql(sql).collect()
    assert got.num_rows == 0
    assert_same(got, duck.sql(sql))
