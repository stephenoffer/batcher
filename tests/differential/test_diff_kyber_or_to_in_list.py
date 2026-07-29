"""`or_equalities_to_in_list` preserves results — vs DuckDB.

The rewrite folds `c = v1 OR c = v2 OR …` into `c IN (v1, v2, …)`, which is DuckDB's
`contains_to_in_clause`. The plan-shape assertions live in
`tests/unit/test_kyber_or_to_in_list.py`; this file is the half that matters, because a
rewrite is only worth having if the answer does not move.

The fixture leans on the case that could: **NULL**. `x = 1 OR x = 2` is NULL for a null
`x` — not false — and `x IN (1, 2)` must be NULL too, or the rewrite silently changes
which rows survive a `NOT`, an `OR`, or a `CASE`. So the predicate is exercised bare, and
negated, and as a projected value where NULL and FALSE are distinguishable.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

PREDICATES = [
    ("a = 1 OR a = 2 OR a = 3", lambda: (col("a") == 1) | (col("a") == 2) | (col("a") == 3)),
    ("a = 1 OR a = 1 OR a = 5", lambda: (col("a") == 1) | (col("a") == 1) | (col("a") == 5)),
    ("s = 'x' OR s = 'y'", lambda: (col("s") == "x") | (col("s") == "y")),
    ("NOT (a = 1 OR a = 2)", lambda: ~((col("a") == 1) | (col("a") == 2))),
    ("(a = 1 OR a = 2) AND a < 3", lambda: ((col("a") == 1) | (col("a") == 2)) & (col("a") < 3)),
    (
        "(a = 1 OR a = 2 OR a = 5) AND a < 3",
        lambda: ((col("a") == 1) | (col("a") == 2) | (col("a") == 5)) & (col("a") < 3),
    ),
    ("a = 1 OR s = 'y'", lambda: (col("a") == 1) | (col("s") == "y")),
]


@pytest.fixture
def table(duck):
    """Nulls in both columns, a value that matches no disjunct, and a repeated value."""
    t = pa.table(
        {
            "a": [1, 2, 3, None, 5, 1],
            "s": ["x", "y", "z", None, "x", "w"],
        }
    )
    duck.register("t", t)
    return t


@pytest.mark.differential
@pytest.mark.parametrize(("sql", "build"), PREDICATES)
def test_filter_result_is_unchanged(duck, table, sql, build):
    out = bt.from_arrow(table).filter(build()).collect()
    assert_same(out, duck.sql(f"SELECT a, s FROM t WHERE {sql}"))


@pytest.mark.differential
@pytest.mark.parametrize(("sql", "build"), PREDICATES)
def test_the_predicate_as_a_projected_value_keeps_its_nulls(duck, table, sql, build):
    """Under a filter, NULL and FALSE both drop the row, so a filter test cannot tell a
    three-valued mistake from a correct rewrite. Projecting the predicate can."""
    out = bt.from_arrow(table).select(a=col("a"), s=col("s"), r=build()).collect()
    assert_same(out, duck.sql(f"SELECT a, s, ({sql}) r FROM t"))


@pytest.mark.differential
def test_a_null_row_survives_the_fold_as_null_not_false(duck, table):
    """The specific value the whole rewrite turns on, asserted directly."""
    got = bt.from_arrow(table).select(r=(col("a") == 1) | (col("a") == 2)).to_pydict()["r"]
    assert got == [True, True, False, None, False, True]
    assert got[3] is None, "a null input must stay null, not become false"
