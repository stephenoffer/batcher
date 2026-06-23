"""Cost-based join reordering: fires on >=3-way inner joins, no-op otherwise (W3)."""

from __future__ import annotations

import batcher as bt
from batcher.config import active_config
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.rules.join_order import reorder_joins


def _tables():
    # region (2 rows) << dept (3) << emp (8): sizes differ enough to force a reorder.
    emp = bt.from_pydict(
        {
            "emp_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "dept_id": [10, 10, 20, 20, 30, 30, 10, 20],
            "emp_name": ["a", "b", "c", "d", "e", "f", "g", "h"],
        }
    )
    dept = bt.from_pydict(
        {"dept_id": [10, 20, 30], "region_id": [1, 1, 2], "dept_name": ["x", "y", "z"]}
    )
    region = bt.from_pydict({"region_id": [1, 2], "region_name": ["west", "east"]})
    return emp, dept, region


def _ctx(ds):
    return OptimizerContext(
        config=active_config(),
        sources=ds._sources,
        hub=None,
        estimator=CardinalityEstimator(ds._sources, {}),
    )


def test_two_way_join_is_not_reordered():
    emp, dept, _ = _tables()
    ds = emp.join(dept, on="dept_id")
    out = reorder_joins(ds._plan, _ctx(ds))
    assert out.to_ir() == ds._plan.to_ir()  # unchanged for a 2-way join


def test_three_way_join_is_reordered():
    emp, dept, region = _tables()
    ds = emp.join(dept, on="dept_id").join(region, on="region_id")
    out = reorder_joins(ds._plan, _ctx(ds))
    # Reorder wraps the rebuilt tree in a final Project pinning the original schema.
    assert out.to_ir()["op"] == "project"
    # Schema (column set) is preserved exactly.
    assert set(out.available_columns()) == set(ds._plan.available_columns())


def test_reorder_with_no_sources_is_noop():
    emp, dept, region = _tables()
    ds = emp.join(dept, on="dept_id").join(region, on="region_id")
    ctx = OptimizerContext(
        config=active_config(),
        sources=[],
        hub=None,
        estimator=CardinalityEstimator([], {}),
    )
    assert reorder_joins(ds._plan, ctx) is ds._plan  # no sources → cannot cost → no-op


def test_three_way_join_result_is_correct():
    # End-to-end through the optimizer (reorder + build-side + projection pruning):
    # the result is the same rows as the unambiguous hand-computed join.
    emp, dept, region = _tables()
    ds = emp.join(dept, on="dept_id").join(region, on="region_id")
    out = ds.select("emp_id", "dept_name", "region_name").sort("emp_id").collect().to_pydict()
    # emp 1 -> dept 10 -> region 1 (west); emp 3 -> dept 20 -> region 1 (west);
    # emp 5 -> dept 30 -> region 2 (east); etc.
    assert out["emp_id"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert out["region_name"] == [
        "west",
        "west",
        "west",
        "west",
        "east",
        "east",
        "west",
        "west",
    ]
