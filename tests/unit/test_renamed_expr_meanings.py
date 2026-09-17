"""The collision renames: a spelling whose meaning moved, and the one whose meaning left.

`Expr.arg_max(by)` / `arg_min(by)` returned the value at another column's extreme and are
now `max_by` / `min_by`, freeing `arg_max()` / `arg_min()` for the position, as in Polars.
`Expr.top_k(k)` returned the most frequent values and is now `mode_top_k`, freeing
`top_k(k)` for the largest. `Expr.rint()` and `.list.sort_desc()` are second spellings of
`round(mode="half_to_even")` and `.list.sort(descending=True)` and are gone. None of these
leaves a second name behind, so each test proves the old meaning is unreachable under the
old call and that the guidance names the new one.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import col
from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import AggExpr


def test_value_by_extreme_moved_to_max_by_and_min_by():
    ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10], "t": [3, 1, 5]})
    got = ds.group_by("g").agg(hi=col("x").max_by("t"), lo=bt.min_by("x", "t")).sort("g")
    assert got.to_pydict() == {"g": ["a", "b"], "hi": [1, 10], "lo": [2, 10]}
    assert isinstance(col("x").max_by("t"), AggExpr)
    # The removed meaning: the old call shape no longer builds a value-by aggregate.
    with pytest.raises(TypeError):
        col("x").arg_max(col("t"))
    with pytest.raises(TypeError):
        col("x").arg_min("t")


def test_arg_max_and_arg_min_are_positions():
    ds = bt.from_pydict({"x": [None, 5, 2, 5, -1]}).with_row_index("_row")
    got = ds.agg(i=col("x").arg_max(order_by="_row"), j=col("x").arg_min(order_by="_row"))
    assert got.to_pydict() == {"i": [1], "j": [4]}
    # A position needs an order: the unordered call is refused, not numbered by arrival.
    with pytest.raises(PlanError, match="requires order_by"):
        col("x").arg_max()


def test_top_level_value_by_names_are_gone_with_guidance():
    for old, new in (("arg_max", "bt.max_by"), ("arg_min", "bt.min_by")):
        with pytest.raises(AttributeError, match=new.replace(".", r"\.")):
            getattr(bt, old)


def test_top_k_is_largest_and_mode_top_k_is_most_frequent():
    ds = bt.from_pydict({"x": [1, 2, 2, 3, 3, 3, 9]})
    got = ds.agg(largest=col("x").top_k(2), frequent=col("x").mode_top_k(2)).to_pydict()
    assert got == {"largest": [[9, 3]], "frequent": [[3, 2]]}


@pytest.mark.parametrize(
    ("build", "hint"),
    [
        (lambda: col("x").rint(), "round(mode='half_to_even')"),
        (lambda: col("l").list.sort_desc(), "sort(descending=True)"),
    ],
)
def test_removed_second_spellings_name_the_one_that_stays(build, hint):
    with pytest.raises(AttributeError) as err:
        build()
    assert hint in str(err.value)


def test_the_kept_spellings_compute_what_the_removed_ones_did():
    ds = bt.from_pydict({"x": [0.5, 1.5, 2.5, -2.5], "l": [[1, None, 3], [], None, [2, 1]]})
    got = ds.select(
        r=col("x").round(mode="half_to_even"), s=col("l").list.sort(descending=True)
    ).to_pydict()
    assert got == {"r": [0.0, 2.0, 2.0, -2.0], "s": [[3, 1, None], [], None, [2, 1]]}
