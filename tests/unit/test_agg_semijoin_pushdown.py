"""Semi-join pushdown into a decorrelated aggregate's input (plan-shape + safety).

The rule restricts a joined-to `Aggregate`'s input to the keys the other side can
produce. These tests pin the plan shape it emits, the join types it refuses (the ones
that must keep unmatched right rows), the shapes it cannot prove safe, and that the
optimizer stays idempotent with it registered. The result-equivalence half of the
proof lives in `tests/differential/test_diff_agg_semijoin_pushdown.py`.
"""

from __future__ import annotations

import batcher as bt

# The rule refuses to fire below a row floor and an input ratio; the unit tests use
# tiny in-memory tables, so they lower both to exercise the rewrite itself. The
# thresholds are benchmarked separately (they are a cost decision, not a semantic one).
import batcher.kyber.rules.joins.agg_semijoin as _mod
from batcher.config import active_config
from batcher.kyber.optimizer import optimize_logical
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.joins.agg_semijoin import (
    push_semijoin_into_decorrelated_aggregate as push_semijoin,
)
from batcher.kyber.stats.estimator import StatsEstimator
from batcher.plan.expr_ir import col
from batcher.plan.logical import Aggregate, Join, JoinOutputCol, Project


def _ctx(ds):
    est = StatsEstimator(ds._sources, learned={})
    return OptimizerContext(config=active_config(), sources=ds._sources, hub=None, estimator=est)


def _outer():
    """The 'outer query' side — few keys, standing in for Q21's filtered stream."""
    return bt.from_pydict({"k": [1, 2], "tag": ["a", "b"]})


def _inner():
    """The 'subquery' side — many keys, only some of which the outer side can use."""
    return bt.from_pydict({"k": [1, 2, 3, 4, 5, 6], "v": [10, 20, 30, 40, 50, 60]})


def _decorrelated(how: str = "left"):
    """`Join(outer, Aggregate(inner GROUP BY k), on k)` — the decorrelation shape."""
    agg = _inner().group_by("k").agg(mn=col("v").min(), mx=col("v").max())
    return _outer().join(agg, on="k", how=how)


def _fire(ds, *, reduction=0.0, floor=1.0):
    """Run the rule on the join at the root of `ds`, with the cost gates relaxed."""
    old_ratio, old_floor = _mod._MIN_GROUP_REDUCTION, _mod._MIN_AGG_INPUT_ROWS
    _mod._MIN_GROUP_REDUCTION, _mod._MIN_AGG_INPUT_ROWS = reduction, floor
    try:
        node = ds._plan
        while not isinstance(node, Join):
            node = node.input
        return push_semijoin(node, _ctx(ds))
    finally:
        _mod._MIN_GROUP_REDUCTION, _mod._MIN_AGG_INPUT_ROWS = old_ratio, old_floor


def test_rule_registered():
    assert "push_semijoin_into_decorrelated_aggregate" in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_inserts_semijoin_below_the_aggregate():
    ds = _decorrelated("left")
    out = _fire(ds)
    assert out is not None, "the rule should fire on the decorrelation shape"
    assert isinstance(out, Join) and out.join_type == "left", "outer join type is preserved"

    agg = out.right
    assert isinstance(agg, Aggregate), "the aggregate stays directly under the join"
    semi = agg.input
    assert isinstance(semi, Join) and semi.join_type == "semi", "a semi-join is inserted below it"
    # The semi-join restricts the *inner* relation by the *outer* side's keys.
    assert semi.left_keys == ("k",) and semi.right_keys == ("k",)
    # Its build side is the outer relation narrowed to the key columns only.
    assert isinstance(semi.right, Project)
    assert [i.alias for i in semi.right.items] == ["k"]
    # The aggregate is otherwise untouched — same grouping, same aggregates.
    assert [s.alias for s in agg.aggregates] == ["mn", "mx"]
    # The semi-join preserves every column the aggregate's input offered.
    assert [o.alias for o in semi.output] == list(semi.left.available_columns())


def test_changes_the_plan_but_not_the_columns():
    ds = _decorrelated("left")
    out = _fire(ds)
    assert out.to_ir() != ds._plan.to_ir(), "the plan must actually change"
    assert out.available_columns() == ds._plan.available_columns(), "the schema must not"


def test_fires_for_every_join_type_that_discards_unmatched_right_rows():
    # inner/left/semi/anti all drop right rows that found no match, so restricting the
    # right side to keys the left can produce is invisible to them.
    for how in ("inner", "left", "semi", "anti"):
        ds = _decorrelated(how)
        assert _fire(ds) is not None, f"{how} join should accept the pushdown"


def test_refuses_join_types_that_preserve_unmatched_right_rows():
    # A right/full join emits right rows with no left match; deleting those groups
    # would delete output rows. This is the case the rule must never get wrong.
    for how in ("right", "full"):
        ds = _decorrelated(how)
        assert _fire(ds) is None, f"{how} join must refuse the pushdown"


def test_refuses_a_computed_group_key():
    # GROUP BY an expression: there is no input column whose membership the semi-join
    # could test, and the rule may not assume the expression is invertible.
    agg = (
        _inner()
        .with_columns(k2=col("k") * 2)
        .group_by("k2")
        .agg(mn=col("v").min())
        .with_columns(k=col("k2"))
    )
    ds = _outer().join(agg, on="k", how="left")
    assert _fire(ds) is None


def test_no_op_when_the_right_side_is_not_an_aggregate():
    ds = _outer().join(_inner(), on="k", how="left")
    assert _fire(ds) is None


def test_cost_gates_refuse_a_small_or_unprofitable_aggregate():
    ds = _decorrelated("left")
    # Below the row floor the second evaluation of the probe side cannot pay for itself.
    assert _fire(ds, floor=10**9) is None
    # A restricting side with as many rows as the aggregate has groups deletes nothing —
    # this is the TPC-H Q13 shape, where an input-size gate wrongly said yes.
    assert _fire(ds, reduction=10**9) is None


def test_refuses_when_recomputing_the_restricting_side_costs_more_than_the_aggregate():
    # Q21's failure mode. The semi-join recomputes `node.left` a second time (once for its
    # build, once for the outer join it already fed). When that side is a cheap dimension the
    # cost is nothing; when it is *itself* the expensive relation — Q21's `supplier ⋈ lineitem
    # ⋈ orders ⋈ nation` spine — recomputing it costs more than the aggregate the semi-join
    # shrinks, so the plan the group-reduction ratio approves is a ~2x wall-time loss. The cost
    # gate refuses exactly when `cost(node.left) >= cost(agg)`, independent of how favorable the
    # cardinality looks. A stub cost model makes the two costs unambiguous.
    from batcher.kyber.cost import Cost

    ds = _decorrelated("left")
    node = ds._plan
    while not isinstance(node, Join):
        node = node.input

    def _ctx_with(cost_of_left: float):
        est = StatsEstimator(ds._sources, learned={})

        class _Costs:
            def cost(self, n):
                return Cost(cpu=cost_of_left) if n is node.left else Cost(cpu=1.0)

        return OptimizerContext(
            config=active_config(),
            sources=ds._sources,
            hub=None,
            estimator=est,
            cost_model=_Costs(),
        )

    old_ratio, old_floor = _mod._MIN_GROUP_REDUCTION, _mod._MIN_AGG_INPUT_ROWS
    _mod._MIN_GROUP_REDUCTION, _mod._MIN_AGG_INPUT_ROWS = 0.0, 1.0
    try:
        # An expensive restricting side (cost 10^12 ≫ the aggregate's 1.0) refuses...
        assert push_semijoin(node, _ctx_with(10**12)) is None
        # ...while a cheap one (cost 0.1 < 1.0) still fires — the gate discriminates on cost,
        # not merely on the plan shape both share.
        assert push_semijoin(node, _ctx_with(0.1)) is not None
    finally:
        _mod._MIN_GROUP_REDUCTION, _mod._MIN_AGG_INPUT_ROWS = old_ratio, old_floor


def test_optimizer_is_idempotent_with_the_rule_registered():
    # The rewrite puts a Join above the Aggregate it matched, so a rule that re-fired
    # on its own output would not converge. ENFORCE runs once; prove the whole
    # optimizer is stable anyway.
    ds = _decorrelated("left")
    once = optimize_logical(ds._plan, sources=ds._sources)
    twice = optimize_logical(once, sources=ds._sources)
    assert once.to_ir() == twice.to_ir()


def test_semijoin_output_is_constructible():
    # Join.__post_init__ validates keys, key counts, key types and unique aliases —
    # a malformed rewrite raises here rather than at execution.
    ds = _decorrelated("left")
    out = _fire(ds)
    semi = out.right.input
    rebuilt = Join(
        semi.left, semi.right, semi.left_keys, semi.right_keys, "semi", semi.output, semi.strategy
    )
    assert rebuilt.to_ir() == semi.to_ir()
    assert all(isinstance(o, JoinOutputCol) and o.side == "left" for o in semi.output)
