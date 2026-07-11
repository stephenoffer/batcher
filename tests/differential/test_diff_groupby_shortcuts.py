"""Differential tests for the GroupBy shortcut reductions vs DuckDB.

``group_by(...).sum()`` and friends reduce every non-key column the same way; each
must equal the DuckDB aggregate over the same columns.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError


@pytest.fixture
def sales(duck):
    t = pa.table(
        {
            "dept": ["eng", "eng", "sales", "sales", "eng"],
            "region": ["us", "eu", "us", "eu", "us"],
            "amount": [100, 120, 90, 80, 110],
            "score": [1.5, 2.5, 3.5, 4.5, 5.5],
        }
    )
    duck.register("sales", t)
    return t


@pytest.mark.differential
@pytest.mark.parametrize(
    ("method", "sql_fn"),
    [
        ("sum", "SUM"),
        ("mean", "AVG"),
        ("min", "MIN"),
        ("max", "MAX"),
        ("median", "MEDIAN"),
        ("n_unique", "COUNT(DISTINCT {c})"),
    ],
)
def test_reduction_matches_duckdb(duck, sales, method, sql_fn):
    from conftest import assert_same

    # Group by both string columns so the only value columns left to reduce are the
    # two numerics — the bare reduction then targets exactly `amount` and `score`.
    out = getattr(bt.from_arrow(sales).group_by("dept", "region"), method)()
    assert out.columns == ["dept", "region", "amount", "score"]
    agg = sql_fn if "{c}" in sql_fn else f"{sql_fn}({{c}})"
    cols = ", ".join(f"{agg.format(c=c)} AS {c}" for c in ("amount", "score"))
    duck_out = duck.sql(f"SELECT dept, region, {cols} FROM sales GROUP BY dept, region")
    assert_same(out.collect(), duck_out)


@pytest.mark.differential
def test_len_matches_count_star(duck, sales):
    from conftest import assert_same

    out = bt.from_arrow(sales).group_by("dept").len()
    assert out.columns == ["dept", "len"]
    assert_same(out.collect(), duck.sql("SELECT dept, COUNT(*) AS len FROM sales GROUP BY dept"))


@pytest.mark.differential
def test_two_group_keys_and_column_subset(duck, sales):
    from conftest import assert_same

    out = bt.from_arrow(sales).group_by("dept", "region").sum("amount")
    assert out.columns == ["dept", "region", "amount"]
    assert_same(
        out.collect(),
        duck.sql("SELECT dept, region, SUM(amount) AS amount FROM sales GROUP BY dept, region"),
    )


@pytest.mark.differential
def test_selector_selects_reduction_columns(duck, sales):
    from conftest import assert_same

    # `floating()` picks only `score`; `amount` (int) is left out.
    out = bt.from_arrow(sales).group_by("dept").mean(bt.floating())
    assert out.columns == ["dept", "score"]
    assert_same(
        out.collect(), duck.sql("SELECT dept, AVG(score) AS score FROM sales GROUP BY dept")
    )


@pytest.mark.differential
def test_positional_and_mixed_agg(duck, sales):
    from conftest import assert_same

    out = bt.from_arrow(sales).group_by("dept").agg(bt.col("amount").sum(), n=bt.count())
    assert out.columns == ["dept", "amount", "n"]
    assert_same(
        out.collect(),
        duck.sql("SELECT dept, SUM(amount) AS amount, COUNT(*) AS n FROM sales GROUP BY dept"),
    )


@pytest.mark.differential
def test_count_is_nonnull_per_column(duck):
    """group_by().count() counts non-null values per column (pandas/SQL COUNT(col))."""
    from conftest import assert_same

    t = pa.table(
        {
            "g": ["a", "a", "b", "b"],
            "x": [1, None, 3, 4],
            "s": ["p", "q", None, "r"],
        }
    )
    duck.register("nulls", t)
    out = bt.from_arrow(t).group_by("g").count()
    assert out.columns == ["g", "x", "s"]
    assert_same(
        out.collect(),
        duck.sql("SELECT g, COUNT(x) AS x, COUNT(s) AS s FROM nulls GROUP BY g"),
    )


@pytest.mark.differential
def test_len_and_count_are_distinct(duck):
    """len() counts rows; count() counts non-null values — they differ when nulls exist."""
    t = pa.table({"g": ["a", "a", "a"], "x": [1, None, None]})
    duck.register("nn", t)
    ds = bt.from_arrow(t)
    assert ds.group_by("g").len().to_pydict() == {"g": ["a"], "len": [3]}
    assert ds.group_by("g").count().to_pydict() == {"g": ["a"], "x": [1]}


@pytest.mark.differential
def test_quantile_matches_duckdb(duck, sales):
    from conftest import assert_same

    out = bt.from_arrow(sales).group_by("dept").quantile(0.5)
    # numeric-only: the string `region` is excluded.
    assert out.columns == ["dept", "amount", "score"]
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT dept, QUANTILE_CONT(amount, 0.5) AS amount, "
            "QUANTILE_CONT(score, 0.5) AS score FROM sales GROUP BY dept"
        ),
    )


@pytest.mark.differential
def test_quantile_out_of_range_raises(sales):
    with pytest.raises(PlanError, match=r"quantile q must be in \[0, 1\]"):
        bt.from_arrow(sales).group_by("dept").quantile(2.0)


@pytest.mark.differential
def test_numeric_only_default_skips_string_columns(sales):
    """A bare sum()/mean() reduces only numeric columns, like pandas numeric_only."""
    # `region` is a non-key string column; it must not break or appear in sum().
    out = bt.from_arrow(sales).group_by("dept").sum()
    assert out.columns == ["dept", "amount", "score"]
    # min()/max()/n_unique() are not numeric-only and keep the string column.
    assert bt.from_arrow(sales).group_by("dept").max().columns == [
        "dept",
        "region",
        "amount",
        "score",
    ]


@pytest.mark.differential
def test_reduction_errors(sales):
    ds = bt.from_arrow(sales)
    # Every column is a group key → nothing left to reduce.
    with pytest.raises(PlanError, match="no numeric value columns to reduce"):
        ds.group_by("dept", "region", "amount", "score").sum().collect()
    # max() is not numeric-only, so its message is the plain form.
    with pytest.raises(PlanError, match="no value columns to reduce"):
        ds.group_by("dept", "region", "amount", "score").max().collect()
    with pytest.raises(PlanError, match="must be a single-column aggregate"):
        ds.group_by("dept").agg(bt.col("amount") + 1).collect()
    with pytest.raises(PlanError, match="requires at least one aggregate"):
        ds.group_by("dept").agg().collect()
