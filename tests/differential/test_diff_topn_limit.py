"""Correctness for the `topn_limit` rules — each rewrite is semantics-preserving.

Importing the rule module registers it into `DEFAULT_REGISTRY` so the rules fire in
the full `Optimizer` that `.collect()` runs. Unordered `LIMIT` results are arbitrary,
so the deterministic cases use `ORDER BY` (or, for the order-arbitrary union case, a
Batcher self-consistency oracle: the optimized limit must select exactly the rows the
unoptimized union produces at ``[offset : offset+n]``). Plan-shape assertions live in
tests/unit/test_topn_limit.py.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
import batcher.kyber.rules.extra.topn_limit
from _harness import assert_same, assert_same_ordered


def _tbl(n=5):
    return pa.table(
        {
            "k": pa.array(list(range(n)), pa.int64()),
            "v": pa.array([i * 10 for i in range(n)], pa.int64()),
        }
    )


# --- drop_redundant_limit -----------------------------------------------------


def test_drop_redundant_limit(duck):
    t = _tbl(5)
    duck.register("t", t)
    out = bt.from_arrow(t).limit(100).collect()  # keep-all limit is dropped
    assert_same(out, duck.sql("SELECT * FROM t LIMIT 100"))


def test_drop_redundant_limit_empty_input(duck):
    t = _tbl(0)
    duck.register("t0", t)
    out = bt.from_arrow(t).limit(100).collect()
    assert_same(out, duck.sql("SELECT * FROM t0 LIMIT 100"))


# --- empty_limit_past_cardinality ---------------------------------------------


def test_empty_limit_past_cardinality(duck):
    t = _tbl(5)
    duck.register("te", t)
    out = bt.from_arrow(t).limit(3, offset=100).collect()  # offset past the end → empty
    assert_same(out, duck.sql("SELECT * FROM te LIMIT 3 OFFSET 100"))


# --- push_limit_through_row_index ---------------------------------------------


def test_push_limit_through_row_index(duck):
    # Distinct keys make the row_number ordering unambiguous; sort first so the pushed
    # top-N (and its 0-based index) is deterministic and maps onto DuckDB row_number.
    t = _tbl(6)
    duck.register("tr", t)
    out = bt.from_arrow(t).sort("k").with_row_index("idx").limit(3).collect()
    assert_same_ordered(
        out,
        duck.sql(
            "SELECT k, v, idx FROM ("
            "  SELECT k, v, row_number() OVER (ORDER BY k) - 1 AS idx FROM tr"
            ") WHERE idx < 3"
        ),
    )


def test_push_limit_through_row_index_with_offset(duck):
    t = _tbl(6)
    duck.register("tr2", t)
    out = bt.from_arrow(t).sort("k").with_row_index("idx", offset=10).limit(2, offset=1).collect()
    # idx = 10 + row-position; the window [offset 1, +2) is the 2nd and 3rd sorted rows.
    assert_same_ordered(
        out,
        duck.sql(
            "SELECT k, v, idx FROM ("
            "  SELECT k, v, (row_number() OVER (ORDER BY k) - 1) + 10 AS idx FROM tr2"
            ") WHERE idx >= 11 AND idx < 13"
        ),
    )


# --- push_offset_limit_into_union ---------------------------------------------


def test_push_offset_limit_into_union(duck):
    # UNION ALL + OFFSET is order-arbitrary, so the oracle is Batcher's own unoptimized
    # union sliced at [offset : offset+n]: the optimized (branch-capped) plan must
    # select exactly those rows.
    a = bt.from_pydict({"x": [1, 2, 3]})
    b = bt.from_pydict({"x": [10, 20, 30, 40]})
    expected = a.union(b).collect().slice(1, 2)  # rows [1, 2, 3, 10, 20, 30, 40][1:3]
    duck.register("exp", expected)
    out = a.union(b).limit(2, offset=1).collect()
    assert_same(out, duck.sql("SELECT * FROM exp"))


def test_push_offset_limit_into_union_window_past_first_branch(duck):
    a = bt.from_pydict({"x": [1, 2, 3]})
    b = bt.from_pydict({"x": [10, 20, 30, 40]})
    expected = a.union(b).collect().slice(2, 3)  # spans the branch boundary
    duck.register("exp2", expected)
    out = a.union(b).limit(3, offset=2).collect()
    assert_same(out, duck.sql("SELECT * FROM exp2"))
