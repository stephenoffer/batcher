"""Differential tests for SQL name resolution in the post-SELECT scope.

Two TPC-DS failures share this area, and both were hard errors rather than wrong
answers, so they are pinned here against DuckDB:

* **ORDER BY could not see the columns a SELECT aliased over.** SQL resolves an
  ORDER BY term against the select-list aliases *and* the input columns underneath
  them; the aggregate path projected first, destroying the latter, so TPC-DS q55/q19
  (``SELECT i_brand_id AS brand_id ... ORDER BY ext_price DESC, i_brand_id``) raised
  ``sort key references unknown column``.
* **A window could not be computed over an aggregate.** SQL runs window functions
  *after* GROUP BY, so ``sum(sum(x)) OVER (...)`` is a window sum over the grouped
  sum. The outer aggregate was instead registered as a group aggregate, so TPC-DS q98
  raised ``aggregate '__agg0' references unknown column``.

Every ORDER BY assertion uses ``assert_same_ordered``: ``assert_same`` is
order-independent by design and therefore cannot see a sort bug at all.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered


@pytest.fixture
def sales(duck):
    """A small grouped-sales table, registered with DuckDB under the same name."""
    t = pa.table(
        {
            "brand_key": [1, 2, 1, 3, 2, 3],
            "brand_name": ["ace", "bolt", "ace", "cog", "bolt", "cog"],
            "klass": ["x", "x", "y", "y", "x", "y"],
            "price": [10.0, 20.0, 5.0, 40.0, 2.5, 7.5],
        }
    )
    duck.register("sales", t)
    return t


def _both(duck, sales, query):
    """Run `query` through Batcher and DuckDB, asserting identical *ordered* rows."""
    out = bt.sql(query, sales=sales).collect()
    assert_same_ordered(out, duck.sql(query))
    return out


# --- ORDER BY name resolution ------------------------------------------------


def test_order_by_original_name_of_an_aliased_column(duck, sales):
    """TPC-DS q55/q19: ORDER BY names the *input* column that SELECT aliased away."""
    _both(
        duck,
        sales,
        """
        SELECT brand_key AS brand_id, brand_name AS brand, SUM(price) AS ext_price
        FROM sales
        GROUP BY brand_name, brand_key
        ORDER BY ext_price DESC, brand_key
        """,
    )


def test_order_by_the_select_alias(duck, sales):
    _both(
        duck,
        sales,
        """
        SELECT brand_key AS brand_id, SUM(price) AS ext_price
        FROM sales GROUP BY brand_key ORDER BY brand_id DESC
        """,
    )


def test_order_by_ordinal(duck, sales):
    _both(
        duck,
        sales,
        """
        SELECT brand_key AS brand_id, SUM(price) AS ext_price
        FROM sales GROUP BY brand_key ORDER BY 2 DESC, 1
        """,
    )


def test_order_by_aggregate_not_in_the_select_list(duck, sales):
    """`ORDER BY MIN(price)` — an aggregate the SELECT list never projects."""
    _both(
        duck,
        sales,
        """
        SELECT brand_key AS brand_id, SUM(price) AS ext_price
        FROM sales GROUP BY brand_key ORDER BY MIN(price) DESC, brand_id
        """,
    )


def test_order_by_expression_not_in_the_select_list(duck, sales):
    """An ungrouped query ordering by an expression over columns it does not project."""
    _both(
        duck,
        sales,
        "SELECT brand_name AS brand FROM sales ORDER BY price * 2 DESC, brand_key",
    )


def test_order_by_expression_over_an_aliased_input_column(duck, sales):
    _both(
        duck,
        sales,
        """
        SELECT brand_key AS brand_id, SUM(price) AS ext_price
        FROM sales GROUP BY brand_key ORDER BY SUM(price) / 2 DESC, brand_key
        """,
    )


# --- alias / input-column collisions -----------------------------------------
#
# DuckDB resolves an ORDER BY name against the select-list alias *first*, and only
# then against the input columns. These pin that precedence: in each case the alias
# shadows a real, differently-valued input column of the same name, so a resolver
# that preferred the input column would produce a visibly different order.


def test_order_by_alias_shadowing_a_different_input_column(duck, sales):
    """`SELECT a AS b, b AS a ... ORDER BY a` sorts by the *alias* a (the old b)."""
    _both(
        duck,
        sales,
        """
        SELECT brand_key AS brand_name, brand_name AS brand_key
        FROM sales ORDER BY brand_key, brand_name
        """,
    )


def test_order_by_alias_collision_in_a_grouped_query(duck, sales):
    _both(
        duck,
        sales,
        """
        SELECT SUM(price) AS brand_key, brand_key AS price
        FROM sales GROUP BY brand_key ORDER BY brand_key DESC
        """,
    )


def test_alias_matching_its_own_column_still_orders_by_that_column(duck, sales):
    """`SELECT brand_key AS brand_key` — alias and input agree; no ambiguity to resolve."""
    _both(
        duck,
        sales,
        "SELECT brand_key AS brand_key, SUM(price) AS s FROM sales "
        "GROUP BY brand_key ORDER BY brand_key DESC",
    )


# --- windows over aggregates and over select-list aliases --------------------


def test_window_over_an_aggregate_in_the_same_select(duck, sales):
    """TPC-DS q98: `sum(x) * 100 / sum(sum(x)) OVER (PARTITION BY k)`."""
    _both(
        duck,
        sales,
        """
        SELECT klass, brand_key,
               SUM(price) AS itemrevenue,
               SUM(price) * 100 / SUM(SUM(price)) OVER (PARTITION BY klass) AS revenueratio
        FROM sales
        GROUP BY klass, brand_key
        ORDER BY klass, brand_key, revenueratio
        """,
    )


def test_window_over_an_aggregate_whole_projection(duck, sales):
    """The window *is* the whole SELECT item, rather than nested in arithmetic."""
    _both(
        duck,
        sales,
        """
        SELECT klass, brand_key, SUM(price) AS rev,
               SUM(SUM(price)) OVER (PARTITION BY klass) AS classrev
        FROM sales GROUP BY klass, brand_key ORDER BY klass, brand_key
        """,
    )


def test_window_ordered_by_an_aggregate(duck, sales):
    """A window whose ORDER BY is an aggregate computed by the same GROUP BY."""
    _both(
        duck,
        sales,
        """
        SELECT klass, brand_key, SUM(price) AS rev,
               ROW_NUMBER() OVER (PARTITION BY klass ORDER BY SUM(price) DESC) AS rn
        FROM sales GROUP BY klass, brand_key ORDER BY klass, rn
        """,
    )


def test_window_nested_in_arithmetic_without_grouping(duck, sales):
    """A window buried in a larger expression in an ungrouped query."""
    _both(
        duck,
        sales,
        """
        SELECT brand_key, price, SUM(price) OVER (PARTITION BY klass) + 1 AS bumped
        FROM sales ORDER BY brand_key, price
        """,
    )


def test_order_by_a_window_alias(duck, sales):
    _both(
        duck,
        sales,
        """
        SELECT brand_key, klass,
               ROW_NUMBER() OVER (PARTITION BY klass ORDER BY price DESC) AS rn
        FROM sales ORDER BY rn, klass, brand_key
        """,
    )


def test_window_over_a_grouped_relation_does_not_leak_helper_columns(duck, sales):
    """The synthetic `__bc_win<n>` column must never reach the output."""
    out = bt.sql(
        """
        SELECT klass, SUM(price) * 2 / SUM(SUM(price)) OVER () AS share
        FROM sales GROUP BY klass ORDER BY klass
        """,
        sales=sales,
    ).collect()
    assert out.column_names == ["klass", "share"]
