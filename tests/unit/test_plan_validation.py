"""Logical-plan validation fails fast on unknown columns."""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError


def test_filter_unknown_column_raises():
    ds = bt.from_pydict({"x": [1, 2, 3]})
    with pytest.raises(PlanError, match="unknown column"):
        ds.filter(bt.col("nope") > 0)


def test_select_unknown_column_raises():
    ds = bt.from_pydict({"x": [1, 2, 3]})
    with pytest.raises(PlanError, match="unknown column"):
        ds.select(total=bt.col("missing") + 1)


def test_positional_expr_in_select_is_named_after_its_leftmost_column():
    # An unnamed derived expression takes its leftmost column's name, as in Polars; two that
    # land on one name are refused rather than silently overwriting each other.
    ds = bt.from_pydict({"x": [1, 2, 3]})
    assert ds.select(bt.col("x") + 1).to_pydict() == {"x": [2, 3, 4]}
    with pytest.raises(PlanError, match="duplicate output column 'x'"):
        ds.select(bt.col("x") + 1, bt.col("x") * 2)


def test_positional_aliased_and_col_in_select_accepted():
    ds = bt.from_pydict({"x": [1, 2, 3]})
    out = ds.select(bt.col("x"), (bt.col("x") + 1).alias("y")).to_pydict()
    assert out == {"x": [1, 2, 3], "y": [2, 3, 4]}


def test_filter_requires_expression():
    ds = bt.from_pydict({"x": [1, 2, 3]})
    with pytest.raises(PlanError, match="requires an expression"):
        ds.filter("x > 0")  # type: ignore[arg-type]


def test_columns_reflect_projection():
    ds = bt.from_pydict({"x": [1], "y": [2]}).select("x", z=bt.col("y") + 1)
    assert ds.columns == ["x", "z"]
