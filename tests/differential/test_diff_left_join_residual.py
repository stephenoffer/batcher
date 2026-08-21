"""Outer-join ON residual vs DuckDB — the residual filters match eligibility, not the result.

``A LEFT JOIN B ON A.k = B.k AND <cond>`` must keep every A row (B columns null where
nothing matched), so the residual decides *which rows may match* — it is not a predicate
on the result. Two one-sided shapes are expressible and both are taken: a residual reading
only the null-extended side pre-filters that side (the TPC-H Q13 shape, where applying it
as a post-join filter dropped the null-extended rows), and a residual reading only a
*preserved* side becomes an extra join key, so a row failing it matches nothing and is
null-extended — which covers FULL, where both sides are preserved.

A residual reading *both* sides is a real theta join and still raises; that refusal is
pinned here too, because turning it into a filter would drop rows.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same


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
    a = pa.table({"k": [1, 2, 3], "tag": ["keep", "drop", "keep"]})
    b = pa.table({"k": [1, 2, 4], "bv": [10, 20, 40]})
    duck.register("a", a)
    duck.register("b", b)
    query = "SELECT b.k, a.tag, b.bv FROM a RIGHT JOIN b ON a.k = b.k AND a.tag = 'keep'"
    out = bt.sql(query, a=a, b=b).collect()
    assert_same(out, duck.sql(query))


def test_inner_join_residual_unchanged(duck, cust_orders):
    """Inner-join residuals are unaffected (still a correct post-join filter)."""
    customer, orders = cust_orders
    query = (
        "SELECT c_custkey, o_orderkey "
        "FROM customer JOIN orders "
        "ON c_custkey = o_custkey AND o_comment NOT LIKE '%special%requests%'"
    )
    out = bt.sql(query, customer=customer, orders=orders).collect()
    assert_same(out, duck.sql(query))


def test_left_join_residual_on_preserved_side(duck, cust_orders):
    """A residual reading only the *preserved* side becomes an extra join key.

    A left row failing it must match nothing, and "matches nothing" is what an equi-join
    already does to a key value the other side does not hold — so the predicate joins
    against the constant TRUE, and the outer join null-extends the rest. This used to be
    refused outright, which is the ordinary
    ``LEFT JOIN ... ON a.k = b.k AND a.active`` shape.
    """
    customer, orders = cust_orders
    query = (
        "SELECT c_custkey, o_orderkey "
        "FROM customer LEFT OUTER JOIN orders "
        "ON c_custkey = o_custkey AND c_custkey > 1"
    )
    out = bt.sql(query, customer=customer, orders=orders).collect()
    assert_same(out, duck.sql(query))


def test_a_full_join_residual_on_one_side(duck, cust_orders):
    """FULL preserves both sides, so the same marker rule is what makes it expressible."""
    customer, orders = cust_orders
    query = (
        "SELECT c_custkey, o_orderkey "
        "FROM customer FULL OUTER JOIN orders "
        "ON c_custkey = o_custkey AND c_custkey > 1"
    )
    out = bt.sql(query, customer=customer, orders=orders).collect()
    assert_same(out, duck.sql(query))


# --- residuals over a column present on BOTH sides --------------------------
#
# `ON a.k = b.k AND a.v < b.v` compares a column that exists on both inputs. The two
# `v`s must stay distinct across the join (the right one takes the `_right` suffix);
# collapsing them would compare `v` with itself — always-false for `<`, always-true for
# `=` — and silently return the wrong rows. This shape once raised
# `NotImplementedError`; these pin the semantics now that it is supported.


@pytest.fixture
def both_sides_v(duck):
    """`v` on both sides, with the residual true / false / equal / null across the rows."""
    a = pa.table({"k": [1, 2, 3, 4, 5], "v": [10, 20, 5, 99, None]})
    b = pa.table({"k": [1, 2, 3, 4, 5], "v": [15, 20, 7, 1, 50]})
    duck.register("a", a)
    duck.register("b", b)
    return a, b


@pytest.mark.parametrize("op", ["<", "<=", ">", ">=", "=", "<>"])
def test_inner_join_residual_on_column_present_on_both_sides(duck, both_sides_v, op):
    """Every comparison must discriminate the two sides, not fold to a self-comparison."""
    a, b = both_sides_v
    query = f"SELECT a.k AS k, a.v AS av, b.v AS bv FROM a JOIN b ON a.k = b.k AND a.v {op} b.v"
    out = bt.sql(query, a=a, b=b).collect()
    assert_same(out, duck.sql(query))


def test_inner_join_residual_both_sides_null_row_is_dropped(duck, both_sides_v):
    """The `k=5` row has a NULL `a.v`: `NULL < 50` is NULL, so the join must drop it."""
    a, b = both_sides_v
    query = "SELECT a.k AS k FROM a JOIN b ON a.k = b.k AND a.v < b.v"
    out = bt.sql(query, a=a, b=b).collect()
    assert_same(out, duck.sql(query))


def test_left_join_residual_across_both_sides_rejected(both_sides_v):
    """A residual reading *both* sides is a real theta join — refuse, don't mis-answer.

    The marker rewrite that handles a one-sided residual cannot express this one: the
    eligibility of a left row depends on which right row it is paired with, so there is no
    key to compute before the join.
    """
    a, b = both_sides_v
    query = "SELECT a.k AS k, b.v AS bv FROM a LEFT JOIN b ON a.k = b.k AND a.v < b.v"
    with pytest.raises(NotImplementedError, match="reads both sides"):
        bt.sql(query, a=a, b=b).collect()
