"""The four `normalize/predicates.py` rewrites preserve results — vs DuckDB.

Each rule turns a correct-but-opaque predicate into a shape Kyber's other rules can use.
`tests/unit/test_kyber_predicate_normalizations.py` checks the shape; this file checks
that the answer did not move.

Three of the four are sound only because a filter cannot distinguish NULL from FALSE, so
the fixture is built to expose exactly that: every column carries a null, and every
predicate is run **twice** — once as a filter, and once *projected as a value*, which is
the only way a three-valued mistake becomes visible. A rule that leaked into the
projection path would pass the filter half and fail here.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col, lit

PREDICATES = [
    ("a = a", lambda: col("a") == col("a")),
    ("a <= a", lambda: col("a") <= col("a")),
    ("a >= a", lambda: col("a") >= col("a")),
    ("a = b", lambda: col("a") == col("b")),
    (
        "CASE WHEN a > 2 THEN true ELSE false END",
        lambda: bt.when(col("a") > lit(2)).then(lit(True)).otherwise(lit(False)),
    ),
    (
        "CASE WHEN a > 2 THEN false ELSE true END",
        lambda: bt.when(col("a") > lit(2)).then(lit(False)).otherwise(lit(True)),
    ),
    (
        "CASE WHEN b IS NULL THEN true ELSE false END",
        lambda: bt.when(col("b").is_null()).then(lit(True)).otherwise(lit(False)),
    ),
    (
        "a IN (1,2,3) AND a IN (2,3,4)",
        lambda: col("a").is_in([1, 2, 3]) & col("a").is_in([2, 3, 4]),
    ),
    ("a IN (1,2) AND b IN (2,3)", lambda: col("a").is_in([1, 2]) & col("b").is_in([2, 3])),
    (
        "a = a AND (CASE WHEN a > 2 THEN true ELSE false END)",
        lambda: (
            (col("a") == col("a"))
            & bt.when(col("a") > lit(2)).then(lit(True)).otherwise(lit(False))
        ),
    ),
]


@pytest.fixture
def nullable(duck):
    """Every column carries a null, because every rule here is null-sensitive."""
    t = pa.table(
        {
            "a": pa.array([1, 2, 3, 4, None], type=pa.int64()),
            "b": pa.array([2, None, 3, 9, 5], type=pa.int64()),
            "g": ["p", "q", "p", "q", "p"],
        }
    )
    duck.register("t", t)
    return t


@pytest.mark.differential
@pytest.mark.parametrize(("sql", "build"), PREDICATES)
def test_the_filtered_rows_are_unchanged(duck, nullable, sql, build):
    assert_same(
        bt.from_arrow(nullable).filter(build()).collect(),
        duck.sql(f"SELECT a, b, g FROM t WHERE {sql}"),
    )


@pytest.mark.differential
@pytest.mark.parametrize(("sql", "build"), PREDICATES)
def test_the_predicate_as_a_projected_value_keeps_its_three_valued_answer(
    duck, nullable, sql, build
):
    """The half that catches a rule leaking out of the filter context.

    `CASE WHEN c THEN true ELSE false END` is `false` for a NULL `c` while `c` is NULL,
    so a rewrite that fired here would return the wrong value while every filter test
    above stayed green.
    """
    out = bt.from_arrow(nullable).select(a=col("a"), r=build()).collect()
    assert_same(out, duck.sql(f"SELECT a, ({sql}) r FROM t"))


@pytest.mark.differential
def test_a_constant_group_key_keeps_its_column_and_its_groups(duck, nullable):
    out = (
        bt.from_arrow(nullable)
        .with_columns(k=lit(1))
        .group_by("g", "k")
        .agg(n=col("a").sum())
        .collect()
    )
    assert_same(out, duck.sql("SELECT g, 1 AS k, sum(a) n FROM t GROUP BY g, 1"))
    assert out.schema.names == ["g", "k", "n"]


@pytest.mark.differential
def test_an_all_constant_grouping_still_produces_one_row_over_an_empty_input(duck):
    """The reason the rule declines the all-constant case.

    `GROUP BY 1` over no rows produces no rows; a global aggregate over no rows produces
    one. Removing the only key would have swapped one for the other.
    """
    empty = pa.table({"a": pa.array([], type=pa.int64())})
    duck.register("e", empty)
    out = bt.from_arrow(empty).with_columns(k=lit(1)).group_by("k").agg(n=col("a").sum()).collect()
    assert_same(out, duck.sql("SELECT 1 AS k, sum(a) n FROM e GROUP BY 1"))
    assert out.num_rows == 0


@pytest.mark.differential
def test_the_intersected_in_list_selects_exactly_the_intersection(duck, nullable):
    got = bt.from_arrow(nullable).filter(col("a").is_in([1, 2, 3]) & col("a").is_in([2, 3, 4]))
    assert sorted(got.to_pydict()["a"]) == [2, 3]
