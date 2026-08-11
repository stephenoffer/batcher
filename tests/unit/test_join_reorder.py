"""Cost-based join reordering: fires on >=3-way inner joins, no-op otherwise (W3)."""

from __future__ import annotations

import batcher as bt
from batcher.config import active_config
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.optimizer import optimize_logical
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.rules.joins.order import reorder_joins
from batcher.plan.logical import Join, is_cartesian_key_pair
from batcher.plan.visitor import walk


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


def _cartesian_joins(plan) -> list[Join]:
    """Every join in `plan` whose keys are *all* cartesian pseudo-keys — a cross product.

    A join left in this state multiplies its two inputs' row counts, so finding one in an
    optimized plan over a fully connected join graph is a defect, not a cost trade-off.
    """
    found = []
    for node in walk(plan):
        if not isinstance(node, Join) or node.join_type != "inner" or not node.left_keys:
            continue
        if all(
            is_cartesian_key_pair(node.left, lk, node.right, rk)
            for lk, rk in zip(node.left_keys, node.right_keys, strict=True)
        ):
            found.append(node)
    return found


# JOB q7c's `FROM` list and its full equi-join graph, over eight one-row-ish tables.
# The graph is connected, but two of its tables (`it`, whose only partner is `pi`, and
# `lt`, whose only partner is `ml`) sit early in the `FROM` list while their partners
# arrive several joins later — so the left-deep comma lowering cross-joins them into
# whatever has accumulated so far. On the real IMDb data that intermediate is 36 M x 11
# rows and the query dies by SIGKILL, which is why this graph is reproduced verbatim
# rather than reduced: the smaller shapes above are all repaired by reordering already.
_Q7C_GRAPH = """
SELECT MIN(n.gender) AS g, MIN(pi.info) AS i
FROM an, ci, it, lt, ml, n, pi, t
WHERE it.info = 'mini biography'
  AND lt.link IN ('references', 'features')
  AND t.production_year BETWEEN 1980 AND 2010
  AND n.id = an.person_id
  AND n.id = pi.person_id
  AND ci.person_id = n.id
  AND t.id = ci.movie_id
  AND ml.linked_movie_id = t.id
  AND lt.id = ml.link_type_id
  AND it.id = pi.info_type_id
  AND pi.person_id = an.person_id
  AND pi.person_id = ci.person_id
  AND an.person_id = ci.person_id
  AND ci.movie_id = ml.linked_movie_id
"""


def _q7c_shaped_query():
    """JOB q7c's join graph over tiny stand-ins for the eight IMDb tables."""
    sess = bt.Session()
    sess.register("an", bt.from_pydict({"person_id": [1, 2], "name": ["a", "b"]}))
    sess.register("ci", bt.from_pydict({"person_id": [1, 2], "movie_id": [10, 20]}))
    sess.register("it", bt.from_pydict({"id": [5], "info": ["mini biography"]}))
    sess.register("lt", bt.from_pydict({"id": [3], "link": ["features"]}))
    sess.register("ml", bt.from_pydict({"linked_movie_id": [10], "link_type_id": [3]}))
    sess.register("n", bt.from_pydict({"id": [1, 2], "gender": ["m", "f"]}))
    sess.register("pi", bt.from_pydict({"person_id": [1], "info_type_id": [5], "info": ["bio"]}))
    sess.register("t", bt.from_pydict({"id": [10, 20], "production_year": [1990, 2000]}))
    return sess.sql(_Q7C_GRAPH)


def test_comma_join_leaves_no_cross_product():
    ds = _q7c_shaped_query()
    # The unoptimized lowering *does* contain cross products — otherwise this test would
    # pass without the optimizer doing anything.
    assert _cartesian_joins(ds._plan), "the lowering should start with cross products"
    opt = optimize_logical(ds._plan, sources=ds._sources)
    assert _cartesian_joins(opt) == [], "reordering left a cross product in the plan"


def test_comma_join_out_of_order_still_returns_the_join():
    # Semantics-preserving: reordering away the cross products must not change a row.
    assert _q7c_shaped_query().collect().to_pydict() == {"g": ["m"], "i": ["bio"]}
