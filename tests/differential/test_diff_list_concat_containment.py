"""`list_concat`, `list_has_all` and `list_has_any` — vs DuckDB.

Three DuckDB list functions the engine did not have. They are tested together because
what separates them from the set operations next to them is the same thing in each case:
**how a null list and an empty list behave**, and none of the three follows the set ops'
rule.

* `list_concat` keeps duplicates and treats a null list as *empty*, so
  `list_concat(NULL, [1])` is `[1]` — where `list_union(NULL, [1])` is NULL.
* `list_has_all`/`list_has_any` are null when *either* side is null, even though
  `list_intersect([1,2], NULL)` is `[]` rather than NULL in both engines. That asymmetry
  is why they are composed to make both operands load-bearing rather than written the
  obvious way.

Every row of the fixture is one of those cases.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered
from batcher import col


@pytest.fixture
def lists(duck):
    """Both operands crossed over: disjoint, overlapping, contained, null and empty."""
    t = pa.table(
        {
            "k": [0, 1, 2, 3, 4, 5, 6, 7],
            "a": [[1, 2], [1, 2, 3], [1, 2], None, [], [1, 2], [1, 2], None],
            "b": [[2, 3], [1, 2], [5], [1], [1], None, [], None],
        }
    )
    duck.register("lists", t)
    return t


@pytest.mark.differential
@pytest.mark.parametrize(
    ("method", "duck_fn"),
    [("concat", "list_concat"), ("has_all", "list_has_all"), ("has_any", "list_has_any")],
)
def test_matches_duckdb(duck, lists, method, duck_fn):
    out = (
        bt.from_arrow(lists)
        .select(k=col("k"), r=getattr(col("a").list, method)(col("b")))
        .sort("k")
        .collect()
    )
    assert_same_ordered(out, duck.sql(f"SELECT k, {duck_fn}(a, b) r FROM lists ORDER BY k"))


@pytest.mark.differential
@pytest.mark.parametrize(
    ("query", "_name"),
    [
        ("SELECT k, list_concat(a, b) r FROM lists ORDER BY k", "list_concat"),
        ("SELECT k, array_cat(a, b) r FROM lists ORDER BY k", "array_cat"),
        ("SELECT k, list_has_all(a, b) r FROM lists ORDER BY k", "list_has_all"),
        ("SELECT k, array_has_all(a, b) r FROM lists ORDER BY k", "array_has_all"),
        ("SELECT k, list_has_any(a, b) r FROM lists ORDER BY k", "list_has_any"),
        ("SELECT k, array_has_any(a, b) r FROM lists ORDER BY k", "array_has_any"),
    ],
)
def test_reachable_from_sql(duck, lists, query, _name):
    assert_same_ordered(bt.sql(query, lists=lists).collect(), duck.sql(query))


@pytest.mark.differential
def test_concat_keeps_duplicates_where_union_removes_them(duck, lists):
    """The one-line difference between `concat` and `union`, asserted directly."""
    out = bt.from_arrow(lists).select(
        k=col("k"),
        c=col("a").list.concat(col("b")),
        u=col("a").list.union(col("b")),
    )
    rows = out.sort("k").to_pydict()
    assert rows["c"][0] == [1, 2, 2, 3]  # the shared 2 appears twice
    assert rows["u"][0] == [1, 2, 3]  # and once
    # A null list is empty for concat and absorbing for union.
    assert rows["c"][3] == [1]
    assert rows["u"][3] is None


@pytest.mark.differential
def test_negative_inner_product_is_the_sign_flipped_dot(duck):
    """DuckDB's `list_negative_dot_product` is the form a maximum-inner-product search
    minimizes; it is exactly `-dot`."""
    t = pa.table({"a": [[1.0, 2.0]], "b": [[3.0, 4.0]]})
    duck.register("v", t)
    for fn in ("list_negative_dot_product", "list_negative_inner_product"):
        query = f"SELECT {fn}(a, b) r FROM v"
        assert_same_ordered(bt.sql(query, v=t).collect(), duck.sql(query))
