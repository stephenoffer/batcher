"""Differential: every form `ds.filter` takes selects the rows a SQL `WHERE` does.

`filter` dispatches on its argument: an `Expr`, a SQL predicate string, or a callable batch
predicate. They are three spellings of one relational filter, so the only thing worth proving
is that each is *exactly* that filter: the same surviving rows, columns and types, on nulls and
empties, and unchanged when the optimizer sinks a vectorized predicate below the callable.
DuckDB is the oracle for every case, and the SQL-string form is also held equal to the `Expr`
form it spells.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import batcher as bt
from _harness import assert_same, assert_tables_equal
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.plan.logical import Filter, MapBatches

pytestmark = pytest.mark.differential

_ROWS = "(1,10,'aa'),(2,NULL,'bbbb'),(3,30,'c'),(4,40,NULL),(5,50,'ee'),(6,60,'f')"


def _t() -> bt.Dataset:
    return bt.from_arrow(
        pa.table(
            {
                "id": [1, 2, 3, 4, 5, 6],
                "v": [10, None, 30, 40, 50, 60],
                "s": ["aa", "bbbb", "c", None, "ee", "f"],
            }
        )
    )


def _register(duck) -> None:
    duck.execute(f"CREATE TABLE t AS SELECT * FROM (VALUES {_ROWS}) AS x(id, v, s)")


#: (SQL predicate, the same predicate as an Expr, as a callable batch predicate).
CASES = {
    "modulo": (
        "id % 2 = 0",
        col("id") % 2 == 0,
        lambda b: pc.equal(pc.bit_wise_and(b["id"], 1), 0),
    ),
    "null_comparison": ("v > 20", col("v") > 20, lambda b: pc.greater(b["v"], 20)),
    "string_prefix": (
        "s LIKE 'a%'",
        col("s").str.starts_with("a"),
        lambda b: pc.starts_with(b["s"], "a"),
    ),
    "nothing": ("1 = 0", bt.lit(False), lambda b: pa.array([False] * b.num_rows)),
    "everything": ("1 = 1", bt.lit(True), lambda b: pa.array([True] * b.num_rows)),
    "conjunction": (
        "id >= 2 AND v < 50",
        (col("id") >= 2) & (col("v") < 50),
        lambda b: pc.and_kleene(pc.greater_equal(b["id"], 2), pc.less(b["v"], 50)),
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_expression_form_matches_where(duck, case):
    sql, expr, _ = CASES[case]
    _register(duck)
    assert_same(_t().filter(expr).collect(), duck.sql(f"SELECT * FROM t WHERE {sql}"))


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_sql_string_form_matches_where(duck, case):
    sql, _, _ = CASES[case]
    _register(duck)
    assert_same(_t().filter(sql).collect(), duck.sql(f"SELECT * FROM t WHERE {sql}"))


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_callable_form_matches_where(duck, case):
    sql, _, fn = CASES[case]
    _register(duck)
    assert_same(_t().filter(fn).collect(), duck.sql(f"SELECT * FROM t WHERE {sql}"))


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_sql_string_equals_the_expression_it_spells(case):
    """Same rows, same column names, same column types: one filter, two spellings."""
    sql, expr, _ = CASES[case]
    by_sql, by_expr = _t().filter(sql).collect(), _t().filter(expr).collect()
    assert by_sql.schema == by_expr.schema
    assert_tables_equal(by_sql, by_expr)


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_callable_keeps_the_input_types(case):
    _, _, fn = CASES[case]
    assert _t().filter(fn).collect().schema == _t().collect().schema


def test_sql_strings_and_expressions_are_anded(duck):
    _register(duck)
    got = _t().filter("v IS NOT NULL", col("id") < 5, s="c").collect()
    assert_same(got, duck.sql("SELECT * FROM t WHERE v IS NOT NULL AND id < 5 AND s = 'c'"))


def test_a_whole_query_is_not_a_predicate():
    with pytest.raises(bt.PlanError, match="not a SQL predicate"):
        _t().filter("SELECT * FROM self")


def test_a_vectorized_filter_sinks_below_the_callable_without_changing_the_answer(duck):
    """The rewrite this stage's `preserves_columns` declaration enables, proven harmless."""
    ds = _t().filter(CASES["modulo"][2]).filter(col("id") < 5)

    optimized = Optimizer().logical_rewrite(ds._plan)
    assert isinstance(optimized, MapBatches)  # the rewrite fired
    assert isinstance(optimized.input, Filter)

    _register(duck)
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE id % 2 = 0 AND id < 5"))


def test_it_composes_with_a_projection_and_an_aggregate(duck):
    ds = (
        _t()
        .filter(lambda b: pc.greater_equal(b["id"], 3), input_columns=["id"])
        .select("id", "v")
        .agg(total=col("v").sum(), n=col("id").count())
    )
    _register(duck)
    assert_same(
        ds.collect(),
        duck.sql("SELECT sum(v) AS total, count(id) AS n FROM t WHERE id >= 3"),
    )


def test_an_empty_input_matches_for_every_form(duck):
    empty = bt.from_arrow(_t().collect().slice(0, 0))
    duck.execute("CREATE TABLE e (id BIGINT, v BIGINT, s VARCHAR)")
    oracle = duck.sql("SELECT * FROM e WHERE id > 1")
    for form in ("id > 1", col("id") > 1, lambda b: pc.greater(b["id"], 1)):
        assert_same(empty.filter(form).collect(), oracle)
