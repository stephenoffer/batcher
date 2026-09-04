"""The optimizer's post-FUSION canonicalization round preserves results vs DuckDB.

`Optimizer._run_cleanup` re-runs the rules that declared `Rule.recanonicalize` — the
*contracting* rewrites (`merge_adjacent_filters`, `merge_projections`,
`merge_projection_renames`, `skip_sort_of_single_row`) — once more after FUSION, because
pushdown, join reordering and fusion all re-create the shapes those rules exist to collapse.
It removes 70 operator nodes across the 99 TPC-DS queries (execution effect measured at
-1.5%, within noise; see `_run_cleanup` for why the round is kept anyway).

Every one of those rules is semantics-preserving, so removing the operator must not move a
row. That is what this file checks, on the shapes the round actually fires on: stacked
filters, stacked and renaming projections, a projection feeding an aggregate, and a sort over
a provably single-row input. The plan-level properties (contraction, phase placement) are
pinned in `tests/unit/test_kyber_canonicalization_round.py`; this is the row-level half.

The edges are the ones a filter/projection merge can get wrong: NULLs on both sides of a
conjunction (Kleene logic), a column referenced twice through a rename, an empty relation, a
single row, and a duplicate key.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered
from batcher import col

pytestmark = pytest.mark.differential

# `g` carries a NULL group and duplicates; `x` and `y` each carry a NULL, so a merged
# conjunction has to agree with DuckDB on three-valued logic rather than on non-null rows.
_DATA = pa.table(
    {
        "id": [1, 2, 3, 4, 5, 6],
        "g": ["a", "a", "b", "b", None, "c"],
        "x": [10, 20, None, 40, 50, 60],
        "y": [1.5, None, 3.5, 4.5, 5.5, 6.5],
    }
)
_ONE = _DATA.slice(0, 1)
_EMPTY = _DATA.slice(0, 0)


@pytest.fixture
def tables(duck):
    for name, tbl in (("t", _DATA), ("one", _ONE), ("e", _EMPTY)):
        duck.register(name, tbl)
    return {"t": _DATA, "one": _ONE, "e": _EMPTY}


def _ds(tables, name):
    return bt.from_arrow(tables[name])


# --- stacked filters: the shape `merge_adjacent_filters` collapses ------------------------


@pytest.mark.parametrize("name", ["t", "one", "e"])
def test_a_chain_of_filters_matches_duckdb(duck, tables, name):
    """Merging into one conjunction must agree on NULLs, which drop under both spellings."""
    got = (
        _ds(tables, name)
        .filter(col("x") > 15)
        .filter(col("y") < 6.0)
        .filter(col("g") == "b")
        .to_arrow()
    )
    want = duck.sql(f"SELECT * FROM {name} WHERE x > 15 AND y < 6.0 AND g = 'b'")
    assert_same(got, want)


def test_a_filter_chain_whose_conjuncts_are_all_null_matches_duckdb(duck, tables):
    """A NULL conjunct makes the row drop, merged or stacked — the Kleene case."""
    got = _ds(tables, "t").filter(col("x") > 15).filter(col("y") > 0).to_arrow()
    want = duck.sql("SELECT * FROM t WHERE x > 15 AND y > 0")
    assert_same(got, want)


# --- stacked projections: `merge_projections` / `merge_projection_renames` ----------------


def test_a_projection_stack_matches_duckdb(duck, tables):
    got = (
        _ds(tables, "t")
        .select(a=col("x"), b=col("y"), k=col("g"))
        .select(a2=col("a"), b2=col("b"))
        .select(s=col("a2"))
        .to_arrow()
    )
    want = duck.sql("SELECT x AS s FROM t")
    assert_same(got, want)


def test_a_rename_referenced_twice_matches_duckdb(duck, tables):
    """The case `merge_projections` declines and `merge_projection_renames` takes.

    A bare-column inner item referenced more than once is free to inline; a computed one is
    not, and inlining it twice would double the work. Both spellings must produce the same
    rows.
    """
    got = (
        _ds(tables, "t")
        .select(r=col("x"), keep=col("id"))
        .select(lo=col("r"), hi=col("r"), keep=col("keep"))
        .to_arrow()
    )
    want = duck.sql("SELECT x AS lo, x AS hi, id AS keep FROM t")
    assert_same(got, want)


def test_a_projection_feeding_an_aggregate_matches_duckdb(duck, tables):
    """`projection_inlining_into_agg` now runs in FUSION, over a pushed-down projection."""
    got = (
        _ds(tables, "t")
        .select(key=col("g"), val=col("x"))
        .group_by("key")
        .agg(total=col("val").sum(), n=col("val").count())
        .to_arrow()
    )
    want = duck.sql("SELECT g AS key, sum(x) AS total, count(x) AS n FROM t GROUP BY g")
    assert_same(got, want)


# --- a sort the round may drop -----------------------------------------------------------


def test_a_sort_over_a_global_aggregate_matches_duckdb(duck, tables):
    """`skip_sort_of_single_row` drops this sort; the one row must be unchanged.

    Compared **ordered** even though a global aggregate is one row: the rule this covers
    *removes a sort*, and an order-blind comparison is precisely what could not tell a
    correctly-dropped sort from a wrongly-dropped one.
    """
    got = _ds(tables, "t").agg(total=col("x").sum()).sort("total").to_arrow()
    want = duck.sql("SELECT sum(x) AS total FROM t ORDER BY total")
    assert_same_ordered(got, want)


# --- the combined shape, ordered ----------------------------------------------------------


def test_the_full_shape_matches_duckdb_in_order(duck, tables):
    """Filters, projections, an aggregate and a real sort in one plan.

    Compared **ordered**: this query ends in `ORDER BY`, and `assert_same` is
    order-independent by design, so it could not see a rewrite that disturbed the ordering.
    """
    got = (
        _ds(tables, "t")
        .filter(col("x") > 5)
        .select(key=col("g"), val=col("x"), w=col("y"))
        .filter(col("w") > 1.0)
        .group_by("key")
        .agg(total=col("val").sum())
        .filter(col("total") > 0)
        .sort("total", descending=True)
        .to_arrow()
    )
    want = duck.sql(
        "SELECT g AS key, sum(x) AS total FROM t "
        "WHERE x > 5 AND y > 1.0 GROUP BY g HAVING sum(x) > 0 ORDER BY total DESC"
    )
    assert_same_ordered(got, want)
