"""Key-constraint mirroring through outer and semi joins, and through a group key.

`infer_join_predicates` mirrors a `key OP literal` constraint across a join's key pairs. For an
inner join both directions hold; for an outer join only from the preserved side onto the
null-extended one (whose rows failing the constraint could never have matched); for a
semi/anti join from the left onto the right. The constraint is also traced *down* through an
aggregate's group key, which is how TPC-DS q78's one-year `store_sales` aggregate scopes the
`web_sales` and `catalog_sales` aggregates it is left-joined to.

The answers these rewrites must preserve are checked against DuckDB in
`tests/differential/test_diff_infer_join_predicates_outer.py`.
"""

from __future__ import annotations

import logging

import pytest

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.rules.pushdown import _column_constraints, infer_join_predicates
from batcher.plan.logical import Filter, Join

pytestmark = pytest.mark.unit


def _left():
    return bt.from_pydict({"k": [1, 2, 3, 4], "a": [10, 20, 30, 40]})


def _right():
    return bt.from_pydict({"k": [1, 2, 5, 6], "b": [1, 2, 3, 4]})


def _key_filtered_left():
    return _left().filter(col("k") <= 2)


@pytest.mark.parametrize("how", ["left", "semi", "anti"])
def test_a_preserved_side_constraint_filters_the_other_side(how):
    join = _key_filtered_left().join(_right(), on="k", how=how)._plan
    assert isinstance(join, Join) and join.join_type == how
    out = infer_join_predicates(join, None)
    assert isinstance(out, Join)
    assert isinstance(out.right, Filter), f"a {how} join's right side may be filtered"
    assert out.left is join.left, "the preserved side is never touched"


def test_a_null_extended_side_constraint_does_not_filter_the_preserved_side():
    join = _left().join(_right().filter(col("k") <= 2), on="k", how="left")._plan
    assert infer_join_predicates(join, None) is None


def test_a_full_join_mirrors_nothing():
    from batcher.plan.visitor import walk

    plan = _key_filtered_left().join(_right(), on="k", how="full")._plan
    join = next(n for n in walk(plan) if isinstance(n, Join))
    assert join.join_type == "full"
    assert infer_join_predicates(join, None) is None


def test_a_constraint_is_traced_through_a_group_key():
    grouped = _key_filtered_left().group_by("k").agg(s=col("a").sum())
    found = _column_constraints(grouped._plan, "k")
    assert found, "a constraint below a group key holds for every group"
    assert _column_constraints(grouped._plan, "s") == [], "an aggregate output proves nothing"


def test_a_group_key_constraint_mirrors_across_an_inner_join():
    grouped = _key_filtered_left().group_by("k").agg(s=col("a").sum())
    join = grouped.join(_right(), on="k")._plan
    out = infer_join_predicates(join, None)
    assert isinstance(out, Join) and isinstance(out.right, Filter)


def test_a_constraint_already_proven_below_is_not_added_again():
    """The confluence half: once sunk under the right side's aggregate it is not re-added."""
    right = _right().filter(col("k") <= 2).group_by("k").agg(t=col("b").sum())
    join = _key_filtered_left().join(right, on="k", how="left")._plan
    assert infer_join_predicates(join, None) is None


def test_the_q78_shape_reaches_a_fixpoint(caplog):
    """Mirrored onto a left-joined aggregate, pushed beneath it, and not mirrored again."""
    years = bt.from_pydict({"d": [1, 2, 3, 4], "y": [2000, 2000, 2001, 2001]})
    fact = bt.from_pydict({"d": [1, 2, 3, 4] * 3, "i": [1, 2] * 6, "q": list(range(12))})

    def per_year(f):
        return f.join(years, on="d").group_by("y", "i").agg(q=col("q").sum())

    left = per_year(fact).filter(col("y") == 2001)
    other = per_year(fact).select(col("y").alias("y2"), col("i").alias("i2"), col("q").alias("q2"))
    q = left.join(other, left_on=["y", "i"], right_on=["y2", "i2"], how="left")
    with caplog.at_level(logging.WARNING, logger="batcher.kyber"):
        plan = Optimizer().optimize(q._plan)
    assert not [r for r in caplog.records if "fixpoint" in r.getMessage()]
    filters = _filters(plan.ir)
    assert sum('"y"' in f or "2001" in f for f in filters) >= 2, (
        "the year constraint must reach the left-joined side's aggregate input too"
    )


def _filters(ir) -> list[str]:
    import json

    out: list[str] = []
    if isinstance(ir, dict):
        if ir.get("op") == "filter":
            out.append(json.dumps(ir.get("predicate")))
        for v in ir.values():
            out.extend(_filters(v))
    elif isinstance(ir, list):
        for v in ir:
            out.extend(_filters(v))
    return out
