"""When `set_membership_to_join` replaces INTERSECT/EXCEPT's aggregate with a join, and when not.

Every test runs the *real* optimizer, so a rule that stops being reached fails here rather than
keeping a green test that calls it by hand. The rewrite is only exact when each column is
proven null-free on at least one side and the branch types agree, so the negatives are the
point: NULLs on both sides of one column, a type that only the union can widen, the ALL
forms, and a column whose null count is not known. Results against DuckDB for the same
matrix live in `tests/differential/test_diff_set_membership_join.py`.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.source_stats import collect_source_stats
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.plan.logical import Aggregate, Distinct, Join
from batcher.plan.visitor import walk


def _optimized(ds: bt.Dataset):
    # The statistics a real query plans with: the conductor collects them and hands them in,
    # and without them no null count is exact and the rule rightly never fires.
    stats = collect_source_stats(ds._sources, None)
    return Optimizer(sources=ds._sources, source_stats=stats).logical_rewrite(ds._plan)


def _join_kinds(ds: bt.Dataset) -> list[str]:
    return [n.join_type for n in walk(_optimized(ds)) if isinstance(n, Join)]


def _has_aggregate(ds: bt.Dataset) -> bool:
    return any(isinstance(n, Aggregate) for n in walk(_optimized(ds)))


def _pair(left: dict, right: dict) -> tuple[bt.Dataset, bt.Dataset]:
    return bt.from_arrow(pa.table(left)), bt.from_arrow(pa.table(right))


def test_the_rule_is_registered():
    assert "set_membership_to_join" in {r.name for r in DEFAULT_REGISTRY.rules()}


@pytest.mark.parametrize(
    ("left", "right"),
    [
        pytest.param({"x": [1, 2, 3]}, {"x": [2, 9]}, id="no-nulls"),
        pytest.param({"x": [1, None, 3]}, {"x": [2, 9]}, id="nulls-left-only"),
        pytest.param({"x": [1, 2, 3]}, {"x": [None, 2]}, id="nulls-right-only"),
        pytest.param(
            {"x": [1, None], "y": ["a", "b"]},
            {"x": [1, 2], "y": ["a", None]},
            id="nulls-split-across-columns",
        ),
    ],
)
def test_fires_when_every_column_is_null_free_on_one_side(left, right):
    a, b = _pair(left, right)
    assert _join_kinds(a.except_(b)) == ["anti"]
    assert _join_kinds(a.intersect(b)) == ["semi"]
    for ds in (a.except_(b), a.intersect(b)):
        plan = _optimized(ds)
        assert not _has_aggregate(ds)
        assert any(isinstance(n, Distinct) for n in walk(plan))
        assert plan.available_columns() == list(left)


def test_declines_when_one_column_holds_nulls_on_both_sides():
    # (NULL, 'b') is on both sides: grouping pairs the two rows, a join cannot.
    a, b = _pair({"x": [None, 1], "y": ["b", "c"]}, {"x": [None, 5], "y": ["b", "d"]})
    for ds in (a.except_(b), a.intersect(b)):
        assert _join_kinds(ds) == []
        assert _has_aggregate(ds)


def test_declines_when_only_the_union_can_reconcile_the_types():
    a, b = _pair({"x": [1, 2, 3]}, {"x": [1.0, 2.5]})
    for ds in (a.except_(b), a.intersect(b)):
        assert _join_kinds(ds) == []
        assert ds.collect().schema.field("x").type == pa.float64()


def test_declines_the_all_forms():
    a, b = _pair({"x": [1, 1, 2]}, {"x": [1]})
    assert _join_kinds(a.except_(b, distinct=False)) == []
    assert _join_kinds(a.intersect(b, distinct=False)) == []


def test_declines_when_the_null_count_is_unknown():
    # Both sides really hold a NULL in `x`, behind a filter on another column: the schema is
    # still known, but a filter leaves `x`'s null count an estimate, so nothing is proven
    # null-free and the aggregate must stay. Were the rule to fire on an inexact count, the
    # join would pair nothing where SQL pairs the two NULLs.
    a, b = _pair({"x": [1, None, 3], "y": [1, 2, 3]}, {"x": [None, 3, 9], "y": [1, 2, 3]})
    a = a.filter(bt.col("y") > 0).select("x")
    b = b.filter(bt.col("y") > 0).select("x")
    assert _join_kinds(a.except_(b)) == []
    assert a.except_(b).to_pydict() == {"x": [1]}


def test_results_through_the_rewrite():
    # Left holds a NULL, right holds none, so the rewrite fires; SQL answers spelled out by hand.
    a, b = _pair({"x": [1, 1, None, 3, 4]}, {"x": [3, 5, 1]})
    assert sorted(a.except_(b).to_pydict()["x"], key=str) == [4, None]
    assert sorted(a.intersect(b).to_pydict()["x"]) == [1, 3]
