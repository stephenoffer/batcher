"""Differential coverage for the row-wise ``*_horizontal`` family, ``between(closed=)``,
and the ``.list.first()``/``.list.last()`` conveniences.

Each addition desugars to existing supported IR (``greatest``/``least``, ``&``/``|``,
range comparisons, ``list_get``), so DuckDB computing the same logic is the oracle —
the tests prove the engine executes the desugared plan identically, including under
nulls and empties.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential


def _nums():
    return pa.table(
        {
            "a": pa.array([1, None, 3, 9], type=pa.int64()),
            "b": pa.array([10, 20, None, 2], type=pa.int64()),
            "c": pa.array([5, 5, 5, 5], type=pa.int64()),
        }
    )


def _bools():
    return pa.table(
        {
            "p": pa.array([True, True, False, None], type=pa.bool_()),
            "q": pa.array([True, False, False, True], type=pa.bool_()),
            "r": pa.array([True, None, True, True], type=pa.bool_()),
        }
    )


def test_min_horizontal_matches_duckdb(duck):
    out = bt.from_arrow(_nums()).select(m=bt.min_horizontal(col("a"), col("b"), col("c"))).collect()
    duck.register("t", _nums())
    assert_same(out, duck.sql("SELECT least(a, b, c) AS m FROM t"))


def test_max_horizontal_matches_duckdb(duck):
    out = bt.from_arrow(_nums()).select(m=bt.max_horizontal(col("a"), col("b"), col("c"))).collect()
    duck.register("t", _nums())
    assert_same(out, duck.sql("SELECT greatest(a, b, c) AS m FROM t"))


def test_horizontal_min_max_agree_with_sql_aliases(duck):
    # The Polars-named spellings are the SQL greatest/least — same result.
    out = (
        bt.from_arrow(_nums())
        .select(
            lo=bt.min_horizontal(col("a"), col("b")),
            hi=bt.max_horizontal(col("a"), col("b")),
        )
        .collect()
    )
    duck.register("t", _nums())
    assert_same(out, duck.sql("SELECT least(a, b) AS lo, greatest(a, b) AS hi FROM t"))


def test_all_horizontal_matches_duckdb(duck):
    x = bt.all_horizontal(col("p"), col("q"), col("r"))
    out = bt.from_arrow(_bools()).select(x=x).collect()
    duck.register("t", _bools())
    assert_same(out, duck.sql("SELECT p AND q AND r AS x FROM t"))


def test_any_horizontal_matches_duckdb(duck):
    x = bt.any_horizontal(col("p"), col("q"), col("r"))
    out = bt.from_arrow(_bools()).select(x=x).collect()
    duck.register("t", _bools())
    assert_same(out, duck.sql("SELECT p OR q OR r AS x FROM t"))


def test_all_horizontal_single_arg(duck):
    out = bt.from_arrow(_bools()).select(x=bt.all_horizontal(col("p"))).collect()
    duck.register("t", _bools())
    assert_same(out, duck.sql("SELECT p AS x FROM t"))


@pytest.mark.parametrize(
    ("closed", "lower_op", "upper_op"),
    [
        ("both", ">=", "<="),
        ("left", ">=", "<"),
        ("right", ">", "<="),
        ("none", ">", "<"),
    ],
)
def test_between_closed_matches_duckdb(duck, closed, lower_op, upper_op):
    data = pa.table({"x": pa.array([1, 3, 5, 8, 10, None], type=pa.int64())})
    out = bt.from_arrow(data).select(r=col("x").between(3, 8, closed=closed)).collect()
    duck.register("t", data)
    assert_same(out, duck.sql(f"SELECT (x {lower_op} 3 AND x {upper_op} 8) AS r FROM t"))


def test_between_default_is_inclusive(duck):
    data = pa.table({"x": pa.array([1, 3, 5, 8, 10], type=pa.int64())})
    out = bt.from_arrow(data).select(r=col("x").between(3, 8)).collect()
    duck.register("t", data)
    assert_same(out, duck.sql("SELECT x BETWEEN 3 AND 8 AS r FROM t"))


def test_between_rejects_bad_closed():
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="closed"):
        col("x").between(1, 2, closed="inclusive")


def _lists():
    return pa.table({"a": pa.array([[3, 1, 2], [], None, [7]], type=pa.list_(pa.int64()))})


def test_list_first_matches_get_0(duck):
    out = bt.from_arrow(_lists()).select(r=col("a").list.first()).collect()
    duck.register("t", _lists())
    # DuckDB lists are 1-indexed; element [1] is the first (NULL for empty/null list).
    assert_same(out, duck.sql("SELECT a[1] AS r FROM t"))


def test_list_last_matches_get_neg1(duck):
    out = bt.from_arrow(_lists()).select(r=col("a").list.last()).collect()
    duck.register("t", _lists())
    assert_same(out, duck.sql("SELECT a[-1] AS r FROM t"))


def test_list_first_last_agree_with_get():
    # first()/last() are the idiomatic spellings of get(0)/get(-1): identical IR.
    assert col("a").list.first().to_ir() == col("a").list.get(0).to_ir()
    assert col("a").list.last().to_ir() == col("a").list.get(-1).to_ir()
