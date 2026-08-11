"""Differential tests for a qualified `SELECT x.*` over a join.

A qualified star used to ignore its qualifier entirely: `x.*` expanded to *every* column
of the joined relation, and — where the two sides shared a column name — under the
`alias__col` names the join disambiguator invents internally. So

    SELECT x.*, small.id4 AS small_id4, v2 FROM x JOIN small USING (id1)

returned `x__id4` **and** `small__id4` where SQL requires x's `id4` alone, leaking an
implementation detail into the result schema and silently answering a different query.

This is the shape the H2O.ai db-benchmark's five join queries are written in
(`benchmarks/suites/h2o/join.py`), which is how it was found: every one of them failed the
benchmark's cross-engine correctness gate on column names.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_tables_equal

# `x` and `y` deliberately share `k` (the join key) and `dup` (a plain colliding column),
# and each has one column the other lacks, so a correct `x.*` is not the same set as `*`.
X = pa.table({"k": [1, 2, 3], "dup": ["a", "b", "c"], "xonly": [10, 20, 30]})
Y = pa.table({"k": [1, 3, 4], "dup": ["p", "q", "r"], "yonly": [1.5, 2.5, 3.5]})


@pytest.fixture
def joined(duck):
    duck.register("x", X)
    duck.register("y", Y)


@pytest.mark.differential
@pytest.mark.parametrize(
    "query",
    [
        # The h2o-join shape: one side's star plus a named column from the other.
        "SELECT x.*, y.dup AS y_dup, yonly FROM x JOIN y USING (k)",
        # Either side's star alone, under both join spellings.
        "SELECT x.* FROM x JOIN y USING (k)",
        "SELECT y.* FROM x JOIN y USING (k)",
        "SELECT x.* FROM x JOIN y ON x.k = y.k",
        "SELECT y.* FROM x JOIN y ON x.k = y.k",
        # An outer join: the star must still carry only its own side's columns, and the
        # unmatched rows' right-hand values must be null rather than the coalesced key.
        "SELECT x.* FROM x LEFT JOIN y USING (k)",
        "SELECT x.*, yonly FROM x LEFT JOIN y USING (k)",
        # Star modifiers apply to the qualified source's columns, not the whole relation.
        "SELECT x.* EXCLUDE (dup) FROM x JOIN y USING (k)",
        "SELECT x.* RENAME (dup AS xdup) FROM x JOIN y USING (k)",
        "SELECT x.* REPLACE (xonly * 2 AS xonly) FROM x JOIN y USING (k)",
        # A star over a three-way join, where the middle source collides with both.
        "SELECT x.* FROM x JOIN y USING (k) JOIN y AS z ON x.k = z.k",
        # Every other join kind, since each takes a different path through the
        # disambiguator: semi/anti emit the left side alone, a comma join has no ON to
        # protect the key, FULL coalesces it, and CROSS shares nothing.
        "SELECT x.* FROM x SEMI JOIN y USING (k)",
        "SELECT x.* FROM x ANTI JOIN y USING (k)",
        "SELECT x.* FROM x, y WHERE x.k = y.k",
        # RIGHT and FULL are where the merged key and x's own key actually differ: on a
        # row with no x match, SQL requires `x.k` to be NULL while the bare `k` carries
        # y's. Both spellings of the star must read x's copy, not the coalesced one.
        "SELECT x.* FROM x FULL JOIN y USING (k)",
        "SELECT x.* FROM x RIGHT JOIN y USING (k)",
        "SELECT y.* FROM x FULL JOIN y USING (k)",
        "SELECT x.* FROM x CROSS JOIN y",
    ],
)
def test_qualified_star_projects_only_its_own_source(duck, joined, query):
    out = bt.sql(query, x=X, y=Y).collect()
    expected = duck.sql(query).to_arrow_table()
    # The schema is the whole point of this test: the bug produced the right *values*
    # under wrong column names, which an order-independent value comparison alone
    # would not have caught.
    assert out.column_names == expected.column_names
    assert_same(out, duck.sql(query))


@pytest.mark.differential
def test_qualified_star_still_expands_a_single_table(duck):
    """With one source, `x.*` is every column — the qualifier narrows nothing."""
    duck.register("x", X)
    query = "SELECT x.* FROM x WHERE k > 1"
    out = bt.sql(query, x=X).collect()
    assert out.column_names == ["k", "dup", "xonly"]
    assert_same(out, duck.sql(query))


@pytest.mark.differential
@pytest.mark.parametrize(
    ("name", "left", "right"),
    [
        # An empty side, one row, all-null keys, and duplicate keys — the edge inputs a
        # projection-list fix is least likely to have been exercised against.
        # `slice(0, 0)` rather than a dict of empty lists: the latter builds `null`-typed
        # columns, which makes the join fail on key types and tests nothing about the star.
        ("empty_left", X.slice(0, 0), Y),
        ("empty_right", X, Y.slice(0, 0)),
        ("one_row", X.slice(0, 1), Y.slice(0, 1)),
        (
            "null_keys",
            pa.table({"k": [None, 1], "dup": ["a", "b"], "xonly": [1, 2]}),
            pa.table({"k": [None, 1], "dup": ["p", "q"], "yonly": [1.0, 2.0]}),
        ),
        (
            "duplicate_keys",
            pa.table({"k": [1, 1, 2], "dup": ["a", "b", "c"], "xonly": [1, 2, 3]}),
            pa.table({"k": [1, 1, 3], "dup": ["p", "q", "r"], "yonly": [1.0, 2.0, 3.0]}),
        ),
    ],
)
def test_qualified_star_on_edge_inputs(duck, name, left, right):
    query = "SELECT x.*, y.dup AS y_dup FROM x JOIN y USING (k)"
    duck.register("x", left)
    duck.register("y", right)
    out = bt.sql(query, x=left, y=right).collect()
    expected = duck.sql(query).to_arrow_table()
    assert out.column_names == expected.column_names
    assert_same(out, duck.sql(query))


@pytest.mark.differential
def test_qualified_star_is_the_same_through_iter_batches():
    """`collect()` and `iter_batches()` take different paths; the star must not differ.

    A green `collect()` is not evidence for the streaming path — that is the cross-product
    the repo's silent-failure guard exists for.
    """
    query = "SELECT x.*, y.dup AS y_dup FROM x JOIN y USING (k)"
    collected = bt.sql(query, x=X, y=Y).collect()
    streamed = pa.Table.from_batches(
        list(bt.sql(query, x=X, y=Y).iter_batches()), schema=collected.schema
    )
    assert streamed.column_names == collected.column_names
    assert_tables_equal(streamed, collected)
