"""Top-level aggregate shorthands (`bt.sum("x")`, `bt.mean("x")`, …) vs DuckDB.

These are the Polars ``pl.sum('x')`` convention — ``bt.sum('x')`` reads as
``col('x').sum()`` — so each must equal both the method form and DuckDB.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def sales(duck):
    t = pa.table(
        {
            "dept": ["eng", "eng", "sales", "sales"],
            "amount": [100, 120, 90, 80],
            "score": [1.5, 2.5, 3.5, 4.5],
        }
    )
    duck.register("sales", t)
    return t


@pytest.mark.differential
@pytest.mark.parametrize(
    ("shorthand", "method", "sql"),
    [
        (bt.sum, "sum", "SUM"),
        (bt.mean, "mean", "AVG"),
        (bt.min, "min", "MIN"),
        (bt.max, "max", "MAX"),
        (bt.median, "median", "MEDIAN"),
        (bt.std, "std", "STDDEV_SAMP"),
        (bt.var, "var", "VAR_SAMP"),
        (bt.n_unique, "n_unique", "COUNT(DISTINCT amount)"),
    ],
)
def test_shorthand_equals_method_and_duckdb(duck, sales, shorthand, method, sql):
    from conftest import assert_same

    ds = bt.from_arrow(sales)
    short = ds.group_by("dept").agg(shorthand("amount")).sort("dept")
    method_form = ds.group_by("dept").agg(getattr(bt.col("amount"), method)()).sort("dept")
    # `bt.sum("amount")` == `col("amount").sum()`.
    assert short.to_pydict() == method_form.to_pydict()
    # And both match DuckDB.
    agg = sql if "(" in sql else f"{sql}(amount)"
    assert_same(short.collect(), duck.sql(f"SELECT dept, {agg} AS amount FROM sales GROUP BY dept"))


@pytest.mark.differential
def test_shorthand_accepts_expression_and_names_output(sales):
    ds = bt.from_arrow(sales)
    # A str is a column; an Expr passes through; a keyword renames.
    out = ds.group_by("dept").agg(
        bt.sum("amount"),
        avg_score=bt.mean(bt.col("score")),
    ).sort("dept")
    assert out.columns == ["dept", "amount", "avg_score"]


@pytest.mark.differential
def test_global_agg_takes_positional_shorthands(sales):
    ds = bt.from_arrow(sales)
    out = ds.agg(bt.sum("amount"), max_score=bt.max("score"))
    assert out.to_pydict() == {"amount": [390], "max_score": [4.5]}
