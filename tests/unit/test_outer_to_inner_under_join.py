"""Plan-shape unit tests for `outer_to_inner_under_join`.

An inner or semi equi-join drops every row whose key is null, so when its key comes from an
outer join's null-supplied side the outer join's padded rows cannot reach the output and the
outer join may be strengthened. Each test builds the plan through the public API, calls the
rule directly, and checks the join type it leaves.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.joins import outer_to_inner_under_join
from batcher.plan.logical import Join

pytestmark = pytest.mark.unit


def _sales():
    return bt.from_pydict({"item": [1, 2, 3], "qty": [5, 6, 7]})


def _returns():
    return bt.from_pydict({"r_item": [1, 2], "reason": [10, None]})


def _reasons():
    return bt.from_pydict({"reason_id": [10, 20], "desc": ["late", "broken"]})


def _left_then(how: str, on_right_side: bool = True) -> Join:
    """`(sales LEFT JOIN returns) <how> reasons`, keyed on a returns or a sales column."""
    joined = _sales().join(_returns(), left_on="item", right_on="r_item", how="left")
    key = "reason" if on_right_side else "item"
    return joined.join(_reasons(), left_on=key, right_on="reason_id", how=how)._plan


def _outer_type(plan) -> str:
    """The join type of the outer join beneath `plan`'s top join (through one projection)."""
    child = plan.left
    child = child.input if not isinstance(child, Join) else child
    assert isinstance(child, Join)
    return child.join_type


def test_rule_registered():
    assert "outer_to_inner_under_join" in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_an_inner_join_on_the_null_supplied_side_makes_the_left_join_inner():
    plan = _left_then("inner")
    assert _outer_type(plan) == "left"
    out = outer_to_inner_under_join(plan, None)
    assert out is not None and _outer_type(out) == "inner"


def test_a_semi_join_on_the_null_supplied_side_strengthens_it_too():
    out = outer_to_inner_under_join(_left_then("semi"), None)
    assert out is not None and _outer_type(out) == "inner"


def test_a_key_on_the_preserved_side_changes_nothing():
    # `item` is the left join's own preserved column: a null-extended row still carries it.
    assert outer_to_inner_under_join(_left_then("inner", on_right_side=False), None) is None


def test_an_anti_join_is_left_alone():
    # An anti join keeps the rows whose key is null, so the padded rows can survive it.
    assert outer_to_inner_under_join(_left_then("anti"), None) is None


def test_idempotent():
    once = outer_to_inner_under_join(_left_then("inner"), None)
    assert outer_to_inner_under_join(once, None) is None


_Q93_SHAPE = (
    "SELECT item, qty FROM sales LEFT OUTER JOIN rets ON (r_item = item AND r_tick = tick), "
    "reasons WHERE reason = reason_id AND descr = 'late'"
)


def _q93_session() -> bt.Session:
    s = bt.Session()
    s.register("sales", bt.from_pydict({"item": [1, 2, 3], "tick": [1, 1, 2], "qty": [5, 6, 7]}))
    s.register("rets", bt.from_pydict({"r_item": [1, 2], "r_tick": [1, 1], "reason": [10, None]}))
    s.register("reasons", bt.from_pydict({"reason_id": [10, 20], "descr": ["late", "broken"]}))
    return s


def _optimized_join_types(plan) -> list[str]:
    from batcher.kyber import plan_cache
    from batcher.kyber.optimizer import optimize_logical
    from batcher.plan.visitor import walk

    # A cached plan would answer for whichever registry optimized it first, so each call
    # optimizes afresh -- otherwise the control below would read the other test's plan.
    plan_cache.clear()
    return [n.join_type for n in walk(optimize_logical(plan)) if isinstance(n, Join)]


def test_the_optimizer_reaches_it_on_the_q93_shape():
    """TPC-DS q93's form: the null-rejecting predicate is a `WHERE` over a comma join.

    It reaches the outer join only once pushdown has made it the inner join's key, so this is
    the shape the rule exists for. The control below shows the left join surviving without it.
    """
    plan = _q93_session().sql(_Q93_SHAPE)._plan
    assert _optimized_join_types(plan) == ["inner", "inner"]


def test_without_the_rule_the_q93_shape_keeps_its_left_join(monkeypatch):
    """The positive control: the assertion above is about this rule, not about some other one."""
    rules = [r for r in DEFAULT_REGISTRY._rules if r.name != "outer_to_inner_under_join"]
    monkeypatch.setattr(DEFAULT_REGISTRY, "_rules", rules)
    # The registry partitions its rules by phase once and reuses that; drop the partition so
    # the optimizer walks the edited list (monkeypatch restores all three afterwards).
    monkeypatch.setattr(DEFAULT_REGISTRY, "_phase_cache", None)
    monkeypatch.setattr(DEFAULT_REGISTRY, "_recanonicalize_cache", None)
    plan = _q93_session().sql(_Q93_SHAPE)._plan
    assert "left" in _optimized_join_types(plan)
