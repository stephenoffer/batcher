"""Filtered, distinct-list and multi-quantile aggregates vs DuckDB, in both front ends.

Three aggregate forms that each have a SQL spelling and a DataFrame spelling sharing one
lowering (`plan.functions.aggregate_semantics`):

* ``agg FILTER (WHERE p)`` and ``AggExpr.filter(p)``. A row the predicate rejects must be
  *absent* from the aggregate, not merely masked to NULL: ``array_agg`` keeps NULL
  elements, so masking alone returned ``[NULL, 2, 2, NULL]`` where DuckDB answers
  ``[2, 2]``. A group with no matching row counts 0 and sums (or collects) to NULL.
* ``array_agg(DISTINCT x ORDER BY x)`` / ``string_agg(DISTINCT ...)`` and
  ``array_agg(distinct=True)``: one NULL element survives, as in DuckDB.
* ``quantile_cont(x, [...])`` and ``quantile([...])``: a list in the order given, and a NULL
  list (not a list of NULLs) for a group with no value.

The fixture has duplicates within a group, NULL values, a NULL group key, a group whose
values are all NULL, and a single-row group. Cell values that are lists are compared element
by element, so a list in the wrong order fails even though rows are compared as a multiset.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_T = pa.table(
    {
        "k": pa.array([1, 1, 1, 2, 2, None, None, 3, 4, 4], pa.int64()),
        "x": pa.array([1, 2, 2, None, 5, 3, 3, 7, None, None], pa.int64()),
        "y": pa.array([1.0, 2.0, 3.0, 0.5, 2.5, None, 4.0, 1.5, 9.0, 0.0], pa.float64()),
        "s": pa.array(["a", "b", "b", None, "c", "a", None, "z", None, "q"], pa.string()),
    }
)


@pytest.fixture
def session(duck):
    duck.register("t", _T)
    s = bt.Session()
    s.register("t", _T)
    return s


@pytest.mark.parametrize(
    "agg",
    [
        "count(*) FILTER (WHERE y > 1)",
        "count(x) FILTER (WHERE y > 1)",
        "sum(x) FILTER (WHERE y > 1)",
        "count(DISTINCT x) FILTER (WHERE y > 1)",
        "sum(DISTINCT x) FILTER (WHERE y > 1)",
        "avg(x) FILTER (WHERE y > 100)",
        "array_agg(x ORDER BY y) FILTER (WHERE y > 1)",
        "array_agg(x ORDER BY y DESC) FILTER (WHERE y > 1)",
        "string_agg(s, ',' ORDER BY s) FILTER (WHERE y > 1)",
        "product(x) FILTER (WHERE y > 1)",
        "quantile_disc(x, 0.5) FILTER (WHERE y > 1)",
        "stddev_pop(y) FILTER (WHERE x IS NOT NULL)",
        "min_by(x, y) FILTER (WHERE y > 1)",
    ],
)
def test_sql_filter_clause(session, duck, agg):
    q = f"SELECT k, {agg} AS r FROM t GROUP BY k"
    assert_same(session.sql(q).collect(), duck.sql(q))


def test_sql_filter_beside_the_same_aggregate_unfiltered(session, duck):
    """The filtered and unfiltered forms are two aggregates, not one cached under one key."""
    q = (
        "SELECT k, array_agg(x ORDER BY y) AS a, array_agg(x ORDER BY y) FILTER (WHERE y > 1) "
        "AS b, sum(DISTINCT x) AS c, sum(DISTINCT x) FILTER (WHERE y > 1) AS d FROM t GROUP BY k"
    )
    assert_same(session.sql(q).collect(), duck.sql(q))


def test_sql_filter_ungrouped_empty_input(session, duck):
    q = (
        "SELECT count(*) FILTER (WHERE y > 1) AS a, sum(x) FILTER (WHERE y > 1) AS b, "
        "array_agg(x) FILTER (WHERE y > 1) AS c FROM t WHERE x > 100"
    )
    assert_same(session.sql(q).collect(), duck.sql(q))


def test_dataframe_filter_matches_the_sql_filter(duck):
    """`AggExpr.filter` against DuckDB's FILTER, including composites and an empty match."""
    duck.register("t", _T)
    x, y, p = bt.col("x"), bt.col("y"), bt.col("y") > 1
    got = (
        bt.from_arrow(_T)
        .group_by("k")
        .agg(
            n=bt.count().filter(p),
            c=x.count().filter(p),
            s=x.sum().filter(p),
            none=x.sum().filter(y > 100),
            zero=bt.coalesce(x.sum().filter(y > 100), 0),
            d=x.count_distinct().filter(p),
            a=x.array_agg(order_by="y").filter(p),
            first=x.first(order_by="y", ignore_nulls=False).filter(p),
        )
    )
    want = duck.sql(
        "SELECT k, count(*) FILTER (WHERE y > 1) AS n, count(x) FILTER (WHERE y > 1) AS c, "
        "sum(x) FILTER (WHERE y > 1) AS s, sum(x) FILTER (WHERE y > 100) AS none, "
        "coalesce(sum(x) FILTER (WHERE y > 100), 0) AS zero, "
        "count(DISTINCT x) FILTER (WHERE y > 1) AS d, "
        "array_agg(x ORDER BY y) FILTER (WHERE y > 1) AS a, "
        "first(x ORDER BY y) FILTER (WHERE y > 1) AS first FROM t GROUP BY k"
    )
    assert_same(got.collect(), want)


def test_dataframe_filter_keeps_the_alias():
    got = bt.from_arrow(_T).agg(bt.count().filter(bt.col("y") > 1).alias("n"))
    assert got.to_pydict() == {"n": [6]}


@pytest.mark.parametrize(
    "agg",
    [
        "array_agg(DISTINCT x ORDER BY x)",
        "array_agg(DISTINCT x ORDER BY x DESC)",
        "array_agg(DISTINCT x ORDER BY x NULLS FIRST)",
        "array_agg(DISTINCT x ORDER BY x DESC NULLS FIRST)",
        "string_agg(DISTINCT s, ',' ORDER BY s)",
        "string_agg(DISTINCT s, '|' ORDER BY s DESC)",
    ],
)
def test_sql_distinct_list_aggregates(session, duck, agg):
    q = f"SELECT k, {agg} AS r FROM t GROUP BY k"
    assert_same(session.sql(q).collect(), duck.sql(q))


def test_sql_distinct_list_beside_another_distinct(session, duck):
    q = "SELECT k, array_agg(DISTINCT x ORDER BY x) AS a, sum(DISTINCT y) AS b FROM t GROUP BY k"
    assert_same(session.sql(q).collect(), duck.sql(q))


def test_sql_distinct_list_ordered_by_another_column_is_refused(session):
    with pytest.raises(NotImplementedError, match="ORDER BY may only order by"):
        session.sql("SELECT array_agg(DISTINCT x ORDER BY y) FROM t").collect()


def test_dataframe_distinct_array_agg(duck):
    duck.register("t", _T)
    x = bt.col("x")
    got = (
        bt.from_arrow(_T)
        .group_by("k")
        .agg(
            up=x.array_agg(distinct=True),
            down=x.array_agg(order_by="x", descending=True, distinct=True),
            first=x.array_agg(order_by="x", nulls_last=False, distinct=True),
            nonull=x.array_agg(distinct=True, ignore_nulls=True),
        )
    )
    want = duck.sql(
        "SELECT k, array_agg(DISTINCT x ORDER BY x) AS up, "
        "array_agg(DISTINCT x ORDER BY x DESC) AS down, "
        "array_agg(DISTINCT x ORDER BY x NULLS FIRST) AS first, "
        "coalesce(array_agg(DISTINCT x ORDER BY x) FILTER (WHERE x IS NOT NULL), []) AS nonull "
        "FROM t GROUP BY k"
    )
    assert_same(got.collect(), want)


def test_dataframe_distinct_array_agg_shortcuts_agree():
    ds = bt.from_arrow(_T)
    via_expr = ds.group_by("k").agg(x=bt.col("x").array_agg(distinct=True)).sort("k")
    via_groupby = ds.group_by("k").array_agg("x", distinct=True).sort("k")
    via_function = ds.group_by("k").agg(x=bt.array_agg("x", distinct=True)).sort("k")
    assert via_expr.to_pydict() == via_groupby.to_pydict() == via_function.to_pydict()


def test_dataframe_distinct_array_agg_refuses_another_order_key():
    with pytest.raises(bt.PlanError, match="ordered by the value itself"):
        bt.col("x").array_agg(order_by="y", distinct=True)


def test_dataframe_distinct_array_agg_over_zero_rows_is_null():
    empty = bt.from_arrow(_T).filter(bt.col("x") > 100)
    assert empty.agg(r=bt.col("x").array_agg(distinct=True)).to_pydict() == {"r": [None]}


def test_quantile_list_matches_duckdb(duck):
    duck.register("t", _T)
    x = bt.col("x")
    got = (
        bt.from_arrow(_T)
        .group_by("k")
        .agg(
            cont=x.quantile([0.1, 0.5, 0.9]),
            given_order=x.quantile([0.9, 0.0]),
            lower=x.quantile([0.25, 0.75], "equiprobable"),
        )
    )
    want = duck.sql(
        "SELECT k, quantile_cont(x, [0.1, 0.5, 0.9]) AS cont, "
        "quantile_cont(x, [0.9, 0.0]) AS given_order, "
        "quantile_disc(x, [0.25, 0.75]) AS lower FROM t GROUP BY k"
    )
    assert_same(got.collect(), want)


def test_quantile_list_shortcuts():
    ds = bt.from_arrow(_T)
    assert ds.quantile("x", [0.5, 0.0]) == [3.0, 1.0]
    assert ds.filter(bt.col("x") > 100).quantile("x", [0.5]) is None
    per_group = ds.group_by("k").quantile([0.0, 1.0], "x").sort("k", nulls_first=True)
    assert per_group.to_pydict()["x"] == [[3.0, 3.0], [1.0, 2.0], [5.0, 5.0], [7.0, 7.0], None]


def test_quantile_list_rejects_an_empty_list_and_a_bad_fraction():
    with pytest.raises(bt.PlanError, match="at least one fraction"):
        bt.col("x").quantile([])
    with pytest.raises(bt.PlanError, match=r"q must be in \[0, 1\]"):
        bt.col("x").quantile([0.5, 2.0])


def test_count_distinct_tuple_counts_null_fields_as_values(session, duck):
    """DuckDB counts distinct row tuples; a tuple with NULL fields is still one value."""
    for q in (
        "SELECT count(DISTINCT (x, s)) AS n FROM t",
        "SELECT k, count(DISTINCT (x, s)) AS n FROM t GROUP BY k",
        "SELECT k, count(DISTINCT (x, s)) AS n, sum(DISTINCT x) AS m FROM t GROUP BY k",
        "SELECT count(DISTINCT (x, s)) AS n FROM t WHERE x > 100",
    ):
        assert_same(session.sql(q).collect(), duck.sql(q))
    struct = bt.struct(f0=bt.col("x"), f1=bt.col("s"))
    got = bt.from_arrow(_T).group_by("k").agg(n=bt.count_distinct(struct))
    assert_same(
        got.collect(),
        duck.sql("SELECT k, count(DISTINCT (x, s)) AS n FROM t GROUP BY k"),
    )


def test_struct_aggregates_over_different_fields_stay_apart():
    """Two aggregates over different structs are two aggregates.

    An expression over aggregates deduplicates its leaves by their rendering, and a struct
    used to render as a bare ``MakeStruct()``, so the second distinct count silently
    answered the first one's value.
    """
    t = bt.from_pydict({"x": [1, 2, 3], "y": [1, 1, 1]})
    a = bt.count_distinct(bt.struct(f=bt.col("x"))) + 0
    b = bt.count_distinct(bt.struct(f=bt.col("y"))) + 0
    assert t.agg(a=a, b=b).to_pydict() == {"a": [3], "b": [1]}


def test_count_distinct_of_several_arguments_points_at_the_tuple_form(session):
    with pytest.raises(NotImplementedError, match=r"COUNT\(DISTINCT \(a, b\)\)"):
        session.sql("SELECT count(DISTINCT x, s) FROM t").collect()
