"""`Expr.meta`: expression introspection that reads the tree and never a row.

The Polars side of the same answers is `tests/differential/test_diff_frame_verbs_polars.py`. This
file pins what Polars has no opinion on: that the walk sees through the nodes a rewrite treats as
opaque (an aggregate, a window's partition and order keys, a `when` chain, a mid-expression
alias), that a selector reports several outputs wherever it sits, and the exact tree drawing.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.unit


def test_root_names_see_through_windows_cases_and_aliases():
    windowed = (bt.col("x").alias("y") * 2).sum().over(partition_by="g", order_by="t")
    assert windowed.meta.root_names() == ["x", "g", "t"]
    case = bt.when(bt.col("a") > 1).then(bt.col("b")).otherwise(bt.col("a"))
    assert case.meta.root_names() == ["a", "b", "a"]
    assert bt.count().meta.root_names() == []


def test_a_selector_anywhere_has_multiple_outputs():
    assert (bt.numeric() * 2).meta.has_multiple_outputs()
    assert not bt.col("a").meta.has_multiple_outputs()


def test_output_name_refuses_a_selector_unless_asked_not_to():
    with pytest.raises(bt.PlanError, match="no single output name"):
        bt.numeric().meta.output_name()
    assert bt.numeric().meta.output_name(raise_if_undetermined=False) is None


def test_is_column_is_false_for_everything_but_a_bare_column():
    assert bt.col("a").meta.is_column()
    assert not bt.col("a").alias("a").meta.is_column()
    assert not bt.col("a", "b").meta.is_column()
    assert not bt.col("a").sum().meta.is_column()


def test_tree_format_draws_the_engine_tree(capsys):
    expr = bt.when(bt.col("a") > 1).then(bt.col("s").str.upper()).otherwise(bt.lit("y"))
    drawing = expr.alias("q").meta.tree_format(return_as_string=True)
    assert drawing == (
        "alias(q)\n"
        "└─ case\n"
        "   ├─ binary(gt)\n"
        "   │  ├─ col(a)\n"
        "   │  └─ lit(1)\n"
        "   ├─ str(upper)\n"
        "   │  └─ col(s)\n"
        "   └─ lit('y')"
    )
    assert expr.alias("q").meta.tree_format() is None
    assert capsys.readouterr().out == drawing + "\n"


def test_tree_format_shows_a_node_parameters_on_its_line():
    drawing = bt.col("x").cast("int32").meta.tree_format(return_as_string=True)
    assert drawing == "cast(int32, try_cast=False)\n└─ col(x)"
