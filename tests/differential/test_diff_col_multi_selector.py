"""Multi-column, regex and dtype `bt.col`: one expression per matched column at bind time.

``col("b", "a")`` and ``col(["b", "a"])`` expand in the order named and require every name to
exist; ``col("^x_.*$")`` is a regular expression (a name wrapped in ``^...$``, as in Polars)
and ``col(pa.int64())`` matches by Arrow type, both in the dataset's column order. The
expansion works in `select`, `with_columns`, and as the input of a positional aggregate.

DuckDB's ``COLUMNS(...)`` is the oracle for values; Polars 1.40's ``pl.col`` for names and
order, compared positionally.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher._internal.errors import ColumnNotFoundError, PlanError

pytestmark = pytest.mark.differential

_TABLES = {
    "nulls": pa.table(
        {
            "x_a": pa.array([1, None, 3], pa.int64()),
            "k": pa.array(["p", "q", None]),
            "x_b": pa.array([0.5, 2.0, None]),
            "y": pa.array([10, 20, None], pa.int64()),
        }
    ),
    "empty": pa.table(
        {
            "x_a": pa.array([], pa.int64()),
            "k": pa.array([], pa.string()),
            "x_b": pa.array([], pa.float64()),
            "y": pa.array([], pa.int64()),
        }
    ),
    "one_row": pa.table({"x_a": [4], "k": ["p"], "x_b": [1.5], "y": [3]}),
    "duplicates": pa.table(
        {"x_a": [2, 2, 2], "k": ["p", "p", "p"], "x_b": [1.0, 1.0, 1.0], "y": [5, 5, 5]}
    ),
}


@pytest.fixture(params=sorted(_TABLES))
def table(request) -> pa.Table:
    return _TABLES[request.param]


def test_named_columns_expand_in_the_order_given(duck, table):
    duck.register("t", table)
    ours = bt.from_arrow(table).select(bt.col("y", "x_a") * 2)
    assert ours.columns == ["y", "x_a"]
    assert_same(ours.collect(), duck.sql("SELECT y * 2 AS y, x_a * 2 AS x_a FROM t"))
    listed = bt.from_arrow(table).select(bt.col(["y", "x_a"]) * 2)
    assert listed.to_pydict() == ours.to_pydict()


def test_regex_matches_in_column_order(duck, table):
    duck.register("t", table)
    ours = bt.from_arrow(table).with_columns(bt.col("^x_.*$").fill_null(0))
    assert ours.columns == ["x_a", "k", "x_b", "y"]
    expected = duck.sql("SELECT coalesce(x_a, 0) AS x_a, k, coalesce(x_b, 0) AS x_b, y FROM t")
    assert_same(ours.collect(), expected)


def test_dtype_matches_in_column_order(duck, table):
    duck.register("t", table)
    ours = bt.from_arrow(table).select(bt.col([pa.int64(), pa.string()]))
    assert ours.columns == ["x_a", "k", "y"]
    assert_same(ours.collect(), duck.sql("SELECT x_a, k, y FROM t"))


def test_positional_aggregate_over_several_columns(duck, table):
    duck.register("t", table)
    ours = bt.from_arrow(table).group_by("k").agg(bt.col("x_a", "y").sum())
    assert ours.columns == ["k", "x_a", "y"]
    assert_same(
        ours.collect(), duck.sql("SELECT k, sum(x_a) AS x_a, sum(y) AS y FROM t GROUP BY k")
    )


def test_a_single_plain_name_is_still_a_column():
    """One name, bare or listed, stays a plain column so it can sit in a filter or a key."""
    assert type(bt.col("x_a")).__name__ == "Col"
    assert type(bt.col(["x_a"])).__name__ == "Col"
    assert type(bt.col("x_a", "y")).__name__ == "Selector"
    assert type(bt.col("^x_a$")).__name__ == "Selector"


def test_unknown_names_and_mixed_arguments_are_refused():
    ds = bt.from_arrow(_TABLES["nulls"])
    with pytest.raises(ColumnNotFoundError, match="nope"):
        ds.select(bt.col("x_a", "nope"))
    with pytest.raises(PlanError, match="not both"):
        bt.col("x_a", pa.int64())
    with pytest.raises(PlanError, match="column names or Arrow types"):
        bt.col(3)


def test_polars_names_and_order_agree(table):
    pl = pytest.importorskip("polars")
    frame = pl.from_arrow(table)
    for ours_expr, their_expr in [
        (bt.col("y", "x_a"), pl.col("y", "x_a")),
        (bt.col(["x_b", "k"]), pl.col(["x_b", "k"])),
        (bt.col("^x_.*$"), pl.col("^x_.*$")),
        (bt.col(pa.int64()), pl.col(pl.Int64)),
    ]:
        theirs = frame.select(their_expr).to_dict(as_series=False)
        assert bt.from_arrow(table).select(ours_expr).to_pydict() == theirs
