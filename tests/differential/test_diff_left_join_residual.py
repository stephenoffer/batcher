"""Outer-join ON residual vs DuckDB — the residual filters match eligibility, not the result.

``A LEFT JOIN B ON A.k = B.k AND <cond on B>`` must keep every A row (B columns null
where nothing matched); the residual filters which B rows are eligible, so it pre-
filters B rather than post-filtering the joined result. Regression test for the bug
where the residual was applied as a post-join filter and dropped the null-extended
rows (TPC-H Q13 shape).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def cust_orders(duck):
    customer = pa.table({"c_custkey": [1, 2, 3, 4]})
    orders = pa.table(
        {
            "o_orderkey": [10, 11, 12],
            "o_custkey": [1, 1, 2],
            "o_comment": ["normal", "special requests here", "normal"],
        }
    )
    duck.register("customer", customer)
    duck.register("orders", orders)
    return customer, orders


def test_left_join_residual_on_right_side(duck, cust_orders):
    """Q13 shape: customers with no eligible (non-special) order keep a 0 count."""
    from conftest import assert_same

    customer, orders = cust_orders
    query = (
        "SELECT c_custkey, count(o_orderkey) AS cnt "
        "FROM customer LEFT OUTER JOIN orders "
        "ON c_custkey = o_custkey AND o_comment NOT LIKE '%special%requests%' "
        "GROUP BY c_custkey"
    )
    out = bt.sql(query, customer=customer, orders=orders).collect()
    assert_same(out, duck.sql(query))


def test_left_join_residual_keeps_unmatched_rows(duck, cust_orders):
    """The null-extended left rows survive the residual (not silently filtered out)."""
    from conftest import assert_same

    customer, orders = cust_orders
    query = (
        "SELECT c_custkey, o_orderkey "
        "FROM customer LEFT OUTER JOIN orders "
        "ON c_custkey = o_custkey AND o_comment NOT LIKE '%special%requests%'"
    )
    out = bt.sql(query, customer=customer, orders=orders).collect()
    assert_same(out, duck.sql(query))


def test_right_join_residual_on_left_side(duck):
    """RIGHT JOIN residual on the (nullable) left side pre-filters the left input."""
    from conftest import assert_same

    a = pa.table({"k": [1, 2, 3], "tag": ["keep", "drop", "keep"]})
    b = pa.table({"k": [1, 2, 4], "bv": [10, 20, 40]})
    duck.register("a", a)
    duck.register("b", b)
    query = "SELECT b.k, a.tag, b.bv FROM a RIGHT JOIN b ON a.k = b.k AND a.tag = 'keep'"
    out = bt.sql(query, a=a, b=b).collect()
    assert_same(out, duck.sql(query))


def test_inner_join_residual_unchanged(duck, cust_orders):
    """Inner-join residuals are unaffected (still a correct post-join filter)."""
    from conftest import assert_same

    customer, orders = cust_orders
    query = (
        "SELECT c_custkey, o_orderkey "
        "FROM customer JOIN orders "
        "ON c_custkey = o_custkey AND o_comment NOT LIKE '%special%requests%'"
    )
    out = bt.sql(query, customer=customer, orders=orders).collect()
    assert_same(out, duck.sql(query))


def test_left_join_residual_on_preserved_side_rejected(cust_orders):
    """A residual touching the preserved side can't be expressed — reject, don't mis-answer."""
    customer, orders = cust_orders
    query = (
        "SELECT c_custkey, o_orderkey "
        "FROM customer LEFT OUTER JOIN orders "
        "ON c_custkey = o_custkey AND c_custkey > 1"
    )
    with pytest.raises(NotImplementedError):
        bt.sql(query, customer=customer, orders=orders).collect()
