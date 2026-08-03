"""A top-N seeded from its own learned k-th best value still matches DuckDB.

The optimization rewrites the *second* run of a top-N into a filtered one, so a differential
test that runs each query once would never exercise the seeded path at all. Every case here
therefore runs the query twice and checks **both** results against DuckDB — the first
answering "did we learn the right bound", the second "does the seeded plan return the same
rows".

`assert_same` is order-independent, so it cannot see an ordering bug in a sort. That matters
more than usual here, because the whole optimization is about which rows a sort returns. So
the returned key column is additionally compared **positionally** against DuckDB's, which is
the sort's actual contract.

Cases follow the operator matrix the contract asks for: nulls (both orderings), ties dense
enough that the k-th value is not unique, `k` larger than the relation, an empty relation, a
single row, and negative/duplicate values.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered

pytestmark = pytest.mark.differential


def _cases():
    return {
        "plain": pa.table({"x": [5, 3, 9, 1, 7, 2, 8], "p": [1, 2, 3, 4, 5, 6, 7]}),
        "ties": pa.table({"x": [5, 5, 5, 5, 5, 1, 9], "p": [1, 2, 3, 4, 5, 6, 7]}),
        "nulls": pa.table({"x": [5, None, 9, None, 7], "p": [1, 2, 3, 4, 5]}),
        "negatives": pa.table({"x": [-5, -3, 0, -9, 4], "p": [1, 2, 3, 4, 5]}),
        "single": pa.table({"x": [42], "p": [1]}),
        "empty": pa.table({"x": pa.array([], pa.int64()), "p": pa.array([], pa.int64())}),
    }


@pytest.mark.parametrize("name", sorted(_cases()))
@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("k", [1, 3, 100])
def test_seeded_topn_matches_duckdb_on_both_runs(duck, name, descending, k):
    table = _cases()[name]
    direction = "DESC" if descending else "ASC"
    duck.register("t", table)
    # `p` is a unique tiebreaker, making the order total. Without it the "ties" case is
    # genuinely ambiguous — Batcher and DuckDB each return a *valid* top-k that disagrees on
    # which tied rows it picked — and the test would be asserting something no sort promises.
    # It also exercises the multi-key path, where only the leading key carries a bound.
    sql = f"SELECT x, p FROM t ORDER BY x {direction} NULLS LAST, p {direction} LIMIT {k}"

    query = bt.from_arrow(table).sort("x", "p", descending=descending).limit(k)

    # Run 1 learns the bound; run 2 is the one the optimization rewrites.
    first = query.collect()
    second = query.collect()

    assert_same(first, duck.sql(sql))
    assert_same(second, duck.sql(sql))

    # `assert_same` is order-independent by design, so it cannot see a sort bug. The ordered
    # comparison is what actually pins a top-N, and it is the point of this file.
    assert_same_ordered(first, duck.sql(sql))
    assert_same_ordered(second, duck.sql(sql))


def test_a_grown_relation_re_learns_rather_than_returning_a_stale_top_k():
    """The bound must not pin the answer to what was true when it was learned.

    A second, larger relation of the same shape has different top rows. Because the bound is
    keyed on the plan (and the plan names its source), this is a different shape — but the
    check that matters is the one stated: whatever the bound does, the answer is the answer.
    """
    small = pa.table({"x": [1, 2, 3], "p": [1, 2, 3]})
    big = pa.table({"x": [1, 2, 3, 400, 500], "p": [1, 2, 3, 4, 5]})

    bt.from_arrow(small).sort("x", descending=True).limit(2).collect()
    got = bt.from_arrow(big).sort("x", descending=True).limit(2).collect()

    assert got.to_pydict()["x"] == [500, 400]
