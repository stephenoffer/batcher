"""Strong behavioral tests for null handling and empty/edge inputs vs DuckDB.

These lock in *actual* semantics (which aggregates ignore nulls, how null group
and join keys behave, what empty inputs produce) so they can't silently regress.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count


@pytest.fixture
def nulls(duck):
    # group 3 is entirely null; a null group key is present; v has scattered nulls.
    tbl = pa.table(
        {
            "k": pa.array([1, 1, 2, 2, 3, 3, None, None], pa.int64()),
            "v": pa.array([10, None, 20, 30, None, None, 5, 7], pa.int64()),
            "f": pa.array([1.5, None, 2.5, 3.5, None, None, 0.5, 4.5], pa.float64()),
        }
    )
    duck.register("t", tbl)
    return tbl


def test_null_aggregates_ignore_nulls(duck, nulls):
    """sum/avg/min/max/count(v) skip nulls; an all-null group yields NULL (count 0)."""
    from conftest import assert_same

    out = (
        bt.from_arrow(nulls)
        .group_by("k")
        .agg(
            s=col("v").sum(),
            a=col("v").mean(),
            mn=col("v").min(),
            mx=col("v").max(),
            cv=col("v").count(),
            cs=count(),
        )
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT k, SUM(v) s, AVG(v) a, MIN(v) mn, MAX(v) mx, "
            "COUNT(v) cv, COUNT(*) cs FROM t GROUP BY k"
        ),
    )


def test_null_median_and_count_distinct(duck, nulls):
    from conftest import assert_same

    out = (
        bt.from_arrow(nulls)
        .group_by("k")
        .agg(m=col("v").median(), nd=col("v").n_unique())
        .collect()
    )
    assert_same(out, duck.sql("SELECT k, median(v) m, COUNT(DISTINCT v) nd FROM t GROUP BY k"))


def test_null_var_stddev(duck, nulls):
    from conftest import assert_same

    out = (
        bt.from_arrow(nulls)
        .group_by("k")
        .agg(vv=col("v").var(), sd=col("v").std(), fv=col("f").var())
        .collect()
    )
    assert_same(
        out,
        duck.sql("SELECT k, var_samp(v) vv, stddev_samp(v) sd, var_samp(f) fv FROM t GROUP BY k"),
    )


def test_null_group_key_is_its_own_group(duck, nulls):
    from conftest import assert_same

    out = bt.from_arrow(nulls).group_by("k").agg(c=count()).collect()
    assert_same(out, duck.sql("SELECT k, COUNT(*) c FROM t GROUP BY k"))


@pytest.fixture
def join_nulls(duck):
    left = pa.table({"k": pa.array([1, 2, None, 3], pa.int64()), "lv": [10, 20, 30, 40]})
    right = pa.table({"k": pa.array([2, 3, None], pa.int64()), "rv": [200, 300, 999]})
    duck.register("l", left)
    duck.register("r", right)
    return left, right


@pytest.mark.parametrize(
    "how,sql",
    [
        ("inner", "JOIN"),
        ("left", "LEFT JOIN"),
        ("right", "RIGHT JOIN"),
    ],
)
def test_null_join_keys_never_match(duck, join_nulls, how, sql):
    """NULL keys must not match (SQL NULL != NULL), on either side of any join."""
    from conftest import assert_same

    left, right = join_nulls
    out = (
        bt.from_arrow(left)
        .join(bt.from_arrow(right), on="k", how=how)
        .select("k", "lv", "rv")
        .collect()
    )
    assert_same(out, duck.sql(f"SELECT l.k, l.lv, r.rv FROM l {sql} r ON l.k = r.k"))


def test_empty_filter_result_preserves_schema():
    t = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out = bt.from_arrow(t).filter(col("a") > 1000).collect()
    assert out.num_rows == 0
    assert out.column_names == ["a", "b"]


def test_global_aggregate_over_empty_input(duck):
    t = pa.table({"v": [1, 2, 3]})
    duck.register("e", t)
    out = (
        bt.from_arrow(t)
        .filter(col("v") > 1000)
        .group_by()
        .agg(s=col("v").sum(), c=count(), a=col("v").mean())
        .collect()
    )
    # SUM/AVG over no rows → NULL; COUNT(*) → 0.
    assert out.to_pylist() == [{"s": None, "c": 0, "a": None}]
    from conftest import assert_same

    assert_same(out, duck.sql("SELECT SUM(v) s, COUNT(*) c, AVG(v) a FROM e WHERE v > 1000"))


def test_all_null_column_aggregates(duck):
    t = pa.table({"k": [1, 1, 2], "v": pa.array([None, None, None], pa.int64())})
    duck.register("a", t)
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .group_by("k")
        .agg(s=col("v").sum(), c=col("v").count(), mn=col("v").min(), m=col("v").median())
        .collect()
    )
    assert_same(
        out, duck.sql("SELECT k, SUM(v) s, COUNT(v) c, MIN(v) mn, median(v) m FROM a GROUP BY k")
    )


def test_single_row_dataset():
    t = pa.table({"k": [7], "v": [42]})
    out = bt.from_arrow(t).group_by("k").agg(s=col("v").sum(), m=col("v").median()).collect()
    assert out.to_pylist() == [{"k": 7, "s": 42, "m": 42.0}]


def test_kleene_and_or_three_valued_logic(duck):
    """`FALSE AND NULL` is FALSE and `TRUE OR NULL` is TRUE (SQL Kleene logic),
    not NULL — the difference between arrow `and`/`or` and `and_kleene`/`or_kleene`."""
    from conftest import assert_same

    t = pa.table(
        {
            "a": pa.array([True, True, False, False, None, None, True, False], pa.bool_()),
            "b": pa.array([True, False, True, None, True, None, None, None], pa.bool_()),
        }
    )
    duck.register("kb", t)
    out = bt.from_arrow(t).select("a", "b", x=col("a") & col("b"), y=col("a") | col("b")).collect()
    assert_same(out, duck.sql("SELECT a, b, a AND b AS x, a OR b AS y FROM kb"))


def test_substr_zero_and_negative_start(duck):
    """SQL substring is 1-based; a start < 1 shrinks the result (out-of-range
    positions are clipped but still consume length), it does not shift the window."""
    from conftest import assert_same

    t = pa.table({"s": ["abcdef", "hello", "x", ""]})
    duck.register("ss", t)
    out = (
        bt.from_arrow(t)
        .select(
            a=col("s").str.substr(0, 3),
            b=col("s").str.substr(-2, 4),
            c=col("s").str.substr(2, 3),
            d=col("s").str.substr(2),
            e=col("s").str.substr(10, 2),
            f=col("s").str.substr(-1, 3),
            g=col("s").str.substr(4, -2),
            h=col("s").str.substr(-2, -1),
        )
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT substring(s,0,3) a, substring(s,-2,4) b, substring(s,2,3) c, "
            "substring(s,2) d, substring(s,10,2) e, substring(s,-1,3) f, "
            "substring(s,4,-2) g, substring(s,-2,-1) h FROM ss"
        ),
    )


def test_cast_float_to_int_rounds_half_to_even(duck):
    """`CAST(float AS BIGINT)` rounds half-to-even like DuckDB (2.5→2, 3.5→4,
    2.7→3), not truncates toward zero like the raw arrow cast."""
    from conftest import assert_same

    t = pa.table(
        {
            "f": [2.7, -2.7, 2.5, 3.5, 2.4, -2.5, 0.5],
            "s": ["10", "-5", "100", "0", "7", "8", "9"],
        }
    )
    duck.register("ct", t)
    out = bt.from_arrow(t).select(fi=col("f").cast("int64"), si=col("s").cast("int64")).collect()
    assert_same(out, duck.sql("SELECT CAST(f AS BIGINT) fi, CAST(s AS BIGINT) si FROM ct"))


def test_is_in_and_between(duck):
    """`is_in`/`between` match SQL IN/BETWEEN, including NULL three-valued logic."""
    from conftest import assert_same

    t = pa.table({"v": pa.array([1, 2, 3, 4, 5, None], pa.int64())})
    duck.register("ib", t)
    out = (
        bt.from_arrow(t)
        .select(
            "v",
            inx=col("v").is_in([2, 4]),
            btw=col("v").between(2, 4),
        )
        .collect()
    )
    assert_same(
        out,
        duck.sql("SELECT v, v IN (2, 4) AS inx, v BETWEEN 2 AND 4 AS btw FROM ib"),
    )


def test_jit_null_propagating_exprs_vs_duckdb(duck, nulls):
    """Projections/filters over nullable columns now take the JIT null-propagating
    path (arithmetic + comparison); the result must still match DuckDB exactly,
    nulls included."""
    from conftest import assert_same

    out = (
        bt.from_arrow(nulls)
        .select(
            "k",
            sum_kv=col("k") + col("v"),  # int + int, both nullable
            scaled=col("v") * 2,  # int * literal
            diff=col("v") - col("k"),
            fadj=col("f") + 1.0,  # float + literal, nullable
            big=col("v") > 15,  # nullable comparison → nullable bool
        )
        .collect()
    )
    expected = duck.sql(
        "SELECT k, k + v AS sum_kv, v * 2 AS scaled, v - k AS diff, "
        "f + 1.0 AS fadj, v > 15 AS big FROM t"
    )
    assert_same(out, expected)


def test_jit_null_filter_predicate_vs_duckdb(duck, nulls):
    """A filter whose predicate is null on null rows (excluded) over the JIT path."""
    from conftest import assert_same

    out = bt.from_arrow(nulls).filter(col("v") > 8).collect()
    expected = duck.sql("SELECT * FROM t WHERE v > 8")
    assert_same(out, expected)
