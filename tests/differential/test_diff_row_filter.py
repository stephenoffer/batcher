"""Differential: a Python row predicate selects the same rows a SQL `WHERE` does.

`ds.ml.filter` is a relational filter that happens to be evaluated in Python, so the only
thing worth proving is that it is *exactly* a filter: same surviving rows, same columns, same
types, and no change to that when the optimizer sinks a vectorized predicate below it. DuckDB
is the oracle for every case, including the null and empty ones where a Python truth value and
SQL three-valued logic could plausibly disagree.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from _harness import assert_same
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.plan.logical import Filter, MapBatches

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


def test_row_predicate_matches_where(duck):
    ds = _t().ml.filter(lambda row: row["id"] % 2 == 0)
    _register(duck)
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE id % 2 = 0"))


def test_a_null_column_is_falsy_the_same_way_sql_drops_it(duck):
    """``row["v"] > 20`` cannot be written in Python against a NULL, so the predicate must
    handle it — and the SQL that keeps the same rows is the plain comparison, because SQL
    drops a NULL comparison too. Same rows out, from opposite treatments of null."""
    ds = _t().ml.filter(lambda row: row["v"] is not None and row["v"] > 20)
    _register(duck)
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE v > 20"))


def test_a_string_predicate_matches_like(duck):
    ds = _t().ml.filter(lambda row: row["s"] is not None and row["s"].startswith("a"))
    _register(duck)
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE s LIKE 'a%'"))


def test_keeping_nothing_matches_a_false_where(duck):
    ds = _t().ml.filter(lambda row: False)
    _register(duck)
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE 1 = 0"))


def test_keeping_everything_matches_no_where(duck):
    ds = _t().ml.filter(lambda row: True)
    _register(duck)
    assert_same(ds.collect(), duck.sql("SELECT * FROM t"))


def test_a_vectorized_filter_sinks_below_it_without_changing_the_answer(duck):
    """The rewrite this stage's `preserves_columns` declaration enables, proven harmless."""
    ds = _t().ml.filter(lambda row: row["id"] % 2 == 0).filter(col("id") < 5)

    optimized = Optimizer().logical_rewrite(ds._plan)
    assert isinstance(optimized, MapBatches)  # the rewrite fired
    assert isinstance(optimized.input, Filter)

    _register(duck)
    assert_same(ds.collect(), duck.sql("SELECT * FROM t WHERE id % 2 = 0 AND id < 5"))


def test_it_composes_with_a_projection_and_an_aggregate(duck):
    ds = (
        _t()
        .ml.filter(lambda row: row["id"] >= 3, input_columns=["id"])
        .select("id", "v")
        .agg(total=col("v").sum(), n=col("id").count())
    )
    _register(duck)
    assert_same(
        ds.collect(),
        duck.sql("SELECT sum(v) AS total, count(id) AS n FROM t WHERE id >= 3"),
    )
