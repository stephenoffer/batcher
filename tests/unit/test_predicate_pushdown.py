"""Predicate pushdown moves filters below joins (semantics-preserving)."""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.rules.pushdown import rewrite_predicate
from batcher.plan.logical import Filter, Join


def _emp_dept():
    emp = bt.from_pydict({"id": [1, 2, 3], "name": ["a", "b", "c"], "dept_id": [10, 20, 10]})
    dept = bt.from_pydict({"dept_id": [10, 20], "dept": ["eng", "sales"], "budget": [100, 200]})
    return emp, dept


def test_single_side_filter_pushed_below_join():
    emp, dept = _emp_dept()
    ds = emp.join(dept, on="dept_id").filter(col("id") > 1)
    rewritten = rewrite_predicate(ds._plan)
    # Top node is now the Join (filter pushed under it), not a Filter.
    assert isinstance(rewritten, Join)
    # The left input of the join is now a Filter (the pushed predicate).
    assert isinstance(rewritten.left, Filter)


def test_cross_side_filter_stays_above_join():
    emp, dept = _emp_dept()
    # id is left-only, budget is right-only -> spans both sides, can't be pushed.
    ds = emp.join(dept, on="dept_id").filter(col("id") < col("budget"))
    rewritten = rewrite_predicate(ds._plan)
    assert isinstance(rewritten, Filter)  # stays above
    assert isinstance(rewritten.input, Join)


def test_conjunction_splits_across_sides():
    emp, dept = _emp_dept()
    ds = emp.join(dept, on="dept_id").filter((col("id") > 1) & (col("dept") == "eng"))
    rewritten = rewrite_predicate(ds._plan)
    # Both conjuncts are single-side -> fully pushed, no Filter remains on top.
    assert isinstance(rewritten, Join)
    assert isinstance(rewritten.left, Filter)  # id > 1
    assert isinstance(rewritten.right, Filter)  # dept == 'eng'


def test_pushdown_preserves_results():
    emp, dept = _emp_dept()
    out = emp.join(dept, on="dept_id").filter(col("id") > 1).sort("id").collect().to_pydict()
    assert out == {
        "dept_id": [20, 10],
        "id": [2, 3],
        "name": ["b", "c"],
        "dept": ["sales", "eng"],
        "budget": [200, 100],
    }
