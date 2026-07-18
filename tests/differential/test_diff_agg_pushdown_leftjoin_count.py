"""Aggregate-through-join pushdown must preserve results whenever it fires.

`pre_aggregate_join_measures` pre-aggregates the measure side of a join and merges the
partials in the outer aggregate. For a LEFT join a `COUNT` merge is `sum(coalesce(__pm,
0))`, so a group with *no* matching rows counts 0. That `coalesce` used to survive only a
single application: a second firing (the outer aggregate is a `sum` on re-entry, not a
`count`) dropped it, flipping a fully-unmatched group's answer from 0 to NULL — the exact
TPC-H Q13 shape (`COUNT` of a customer's orders, for customers with none).

The rule's cost gate normally declines that second firing, so the bug hid behind an
*estimate*. These tests force the gate open — a rewrite must be correct **whenever** it
fires, not only when the estimator happens to stop it — and check the result against
DuckDB. They failed before the structural idempotency guard landed and pass after.
"""

from __future__ import annotations

import pytest

from batcher.kyber.rules import agg_pushdown

bt = pytest.importorskip("batcher")
duckdb = pytest.importorskip("duckdb")

from batcher import col  # noqa: E402
from conftest import assert_same  # noqa: E402


@pytest.fixture
def _force_pushdown(monkeypatch):
    """Force the aggregate-pushdown cost gate open (it is a perf heuristic, not a
    correctness guard), so the rewrite itself is exercised rather than the estimate."""
    monkeypatch.setattr(agg_pushdown, "_reduces_enough", lambda ctx, pushed, source: True)


_CUST = {"custkey": [1, 2, 3, 4], "nation": [10, 10, 20, 20]}
_ORDERS = {"okey": [100, 101, 102, 103, 104], "custkey": [1, 1, 1, 2, 2], "price": [5, 6, 7, 8, 9]}


def _duck():
    con = duckdb.connect()
    con.execute(
        "create table cust as select * from (values (1,10),(2,10),(3,20),(4,20)) t(custkey,nation)"
    )
    con.execute(
        "create table orders as select * from (values "
        "(100,1,5),(101,1,6),(102,1,7),(103,2,8),(104,2,9)) t(okey,custkey,price)"
    )
    return con


def test_left_join_count_by_key(_force_pushdown):
    # custkeys 3 and 4 have no orders → COUNT must be 0, not NULL.
    ds = (
        bt.from_pydict(_CUST)
        .join(bt.from_pydict(_ORDERS), on="custkey", how="left")
        .group_by("custkey")
        .agg(n=col("okey").count())
    )
    con = _duck()
    duck = con.sql(
        "select c.custkey, count(o.okey) n "
        "from cust c left join orders o using(custkey) group by c.custkey"
    )
    assert_same(ds.collect(), duck)


def test_left_join_count_fully_unmatched_group(_force_pushdown):
    # nation 20 = custkeys 3,4, both order-less → the whole group is unmatched → COUNT 0.
    ds = (
        bt.from_pydict(_CUST)
        .join(bt.from_pydict(_ORDERS), on="custkey", how="left")
        .group_by("nation")
        .agg(n=col("okey").count())
    )
    con = _duck()
    duck = con.sql(
        "select c.nation, count(o.okey) n "
        "from cust c left join orders o using(custkey) group by c.nation"
    )
    assert_same(ds.collect(), duck)


def test_left_join_sum_and_count_mixed(_force_pushdown):
    ds = (
        bt.from_pydict(_CUST)
        .join(bt.from_pydict(_ORDERS), on="custkey", how="left")
        .group_by("nation")
        .agg(n=col("okey").count(), s=col("price").sum())
    )
    con = _duck()
    duck = con.sql(
        "select c.nation, count(o.okey) n, sum(o.price) s "
        "from cust c left join orders o using(custkey) group by c.nation"
    )
    assert_same(ds.collect(), duck)
