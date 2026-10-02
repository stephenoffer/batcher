"""A conjunction the literals prove empty is estimated at zero, not by backoff.

Exponential backoff assumes positively correlated conjuncts, so on predicates that exclude
each other it answered 3% for `x = 1 AND x = 2` and 19% for `x > 5 AND x < 3`. The proof
needs no statistics, and the estimate must stay a *proof*: every satisfiable neighbour below
keeps its non-zero estimate, and the rows the engine returns agree with DuckDB.
"""

from __future__ import annotations

import datetime

import pytest

import batcher as bt
from batcher import col
from batcher.config import CardinalityConfig
from batcher.kyber.stats.selectivity import predicate_selectivity
from batcher.kyber.stats.selectivity.contradiction import provably_unsatisfiable

pytestmark = pytest.mark.unit

_EMPTY = {
    "two_equalities": (col("x") == 1) & (col("x") == 2),
    "equality_outside_range": (col("x") == 1) & (col("x") > 5),
    "crossed_range": (col("x") > 5) & (col("x") < 3),
    "strict_touching_range": (col("x") > 3) & (col("x") <= 3),
    "equality_and_its_negation": (col("x") == 1) & (col("x") != 1),
    "literal_on_the_left": (bt.lit(5) < col("x")) & (col("x") < 3),
    "strings": (col("s") == "a") & (col("s") == "b"),
    "dates": (col("d") > datetime.date(2024, 1, 1)) & (col("d") < datetime.date(2023, 1, 1)),
}

_SATISFIABLE = {
    "repeated_equality": (col("x") == 1) & (col("x") == 1),
    "closed_point_range": (col("x") >= 3) & (col("x") <= 3),
    "two_columns": (col("x") == 1) & (col("y") == 2),
    "open_range": (col("x") > 3) & (col("x") < 9),
    "int_equals_float": (col("x") == 1) & (col("x") == 1.0),
    # Through a cast the comparisons are not bounds on `x`: x = 1.5 satisfies both.
    "through_a_cast": (col("x").cast("int64") == 1) & (col("x") == 1.5),
}


def test_mixed_literal_families_are_never_a_proof():
    """A number and a string on one column are not ordered here, so no proof is attempted."""
    assert not provably_unsatisfiable([col("x") == 1, col("x") == "1"])


@pytest.mark.parametrize("name", sorted(_EMPTY))
def test_a_provably_empty_conjunction_estimates_zero(name):
    expr = _EMPTY[name]
    assert provably_unsatisfiable([expr.left, expr.right])
    assert predicate_selectivity(expr, {"x": 10.0, "y": 10.0}, CardinalityConfig()) == 0.0


@pytest.mark.parametrize("name", sorted(_SATISFIABLE))
def test_a_satisfiable_conjunction_keeps_its_estimate(name):
    """The positive control: without it, a detector that always fires would pass the above."""
    expr = _SATISFIABLE[name]
    assert not provably_unsatisfiable([expr.left, expr.right])
    assert predicate_selectivity(expr, {"x": 10.0, "y": 10.0}, CardinalityConfig()) > 0.0


def test_the_engine_agrees_the_proven_filters_keep_nothing():
    """The estimate is only right if the rows are: each proven filter returns what DuckDB does."""
    duckdb = pytest.importorskip("duckdb")
    table = {"x": [1, 2, 3, 4, 5, 6, None], "y": [2, 2, 2, 2, 2, 2, 2]}
    ds = bt.from_pydict(table)
    con = duckdb.connect()
    import pyarrow as pa

    con.register("t", pa.table(table))
    for sql, expr in [
        ("x = 1 AND x = 2", _EMPTY["two_equalities"]),
        ("x > 5 AND x < 3", _EMPTY["crossed_range"]),
        ("x > 3 AND x <= 3", _EMPTY["strict_touching_range"]),
        ("x >= 3 AND x <= 3", _SATISFIABLE["closed_point_range"]),
    ]:
        want = con.execute(f"SELECT count(*) FROM t WHERE {sql}").fetchone()[0]
        got = ds.filter(expr).collect().num_rows
        assert got == want, sql
