"""Plan-shape and cardinality tests for the range-join rewrite.

`tests/differential/test_diff_range_join.py` proves the *results* match DuckDB. This file
proves the things a result comparison cannot see: that the rewrite fires on the shapes it
should and stands down on the ones it should not, and that the cardinality it reports is a
function of the inputs rather than the fixed `unknown_rows` default a missing estimator
leaves behind.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.kyber.optimizer import optimize_logical
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.stats.estimator import _uniform_p_less
from batcher.plan.logical import RangeJoin
from batcher.plan.stats import ColumnStat, Provenance, RelStats


def _find(plan, cls):
    """Depth-first search for the first node of `cls` in a logical plan."""
    if isinstance(plan, cls):
        return plan
    for name in ("input", "left", "right"):
        child = getattr(plan, name, None)
        if child is not None:
            found = _find(child, cls)
            if found is not None:
                return found
    for child in getattr(plan, "inputs", ()) or ():
        found = _find(child, cls)
        if found is not None:
            return found
    return None


@pytest.fixture
def ab():
    a = pa.table({"x": [1, 2, 3], "z": [7, 8, 9], "lab": ["a", "b", "c"]})
    b = pa.table({"lo": [0, 2], "hi": [5, 9], "k": [1, 2], "tag": ["p", "q"]})
    return a, b


@pytest.mark.unit
def test_the_rule_is_registered():
    assert "derive_range_join" in {r.name for r in DEFAULT_REGISTRY.rules()}


@pytest.mark.unit
def test_two_inequalities_become_one_range_join(ab):
    a, b = ab
    plan = optimize_logical(
        bt.sql("SELECT lab, tag FROM a, b WHERE a.x > b.lo AND a.z < b.hi", a=a, b=b)._plan
    )
    node = _find(plan, RangeJoin)
    assert node is not None
    # Conjunct order is the optimizer's to choose (it orders by measured cost), so the
    # set is the contract, not the sequence.
    assert {(c.left_key, c.op, c.right_key) for c in node.conditions} == {
        ("x", "gt", "lo"),
        ("z", "lt", "hi"),
    }


@pytest.mark.unit
def test_a_reversed_comparison_flips_the_operator_not_the_sides(ab):
    a, b = ab
    plan = optimize_logical(bt.sql("SELECT lab, tag FROM a, b WHERE b.hi > a.z", a=a, b=b)._plan)
    node = _find(plan, RangeJoin)
    assert node is not None
    # `b.hi > a.z` is `a.z < b.hi`: the left key stays on the left, the operator flips.
    assert [(c.left_key, c.op, c.right_key) for c in node.conditions] == [("z", "lt", "hi")]


@pytest.mark.unit
def test_a_third_inequality_stays_in_the_filter(ab):
    """Two axes is IEJoin's ceiling; the rest is a post-check on surviving pairs."""
    a, b = ab
    plan = optimize_logical(
        bt.sql(
            "SELECT lab, tag FROM a, b WHERE a.x > b.lo AND a.z < b.hi AND a.x < b.k",
            a=a,
            b=b,
        )._plan
    )
    node = _find(plan, RangeJoin)
    assert node is not None and len(node.conditions) == 2
    from batcher.plan.logical import Filter

    assert _find(plan, Filter) is not None, "the third conjunct must survive as a filter"


@pytest.mark.unit
def test_an_equality_conjunct_stands_the_rule_down(ab):
    """A hash join beats any range algorithm, so the equality must win the join keys."""
    a, b = ab
    plan = optimize_logical(
        bt.sql("SELECT lab, tag FROM a, b WHERE a.x = b.k AND a.z < b.hi", a=a, b=b)._plan
    )
    assert _find(plan, RangeJoin) is None


@pytest.mark.unit
def test_a_single_side_predicate_alone_does_not_fire(ab):
    a, b = ab
    plan = optimize_logical(bt.sql("SELECT lab, tag FROM a, b WHERE a.x > 1", a=a, b=b)._plan)
    assert _find(plan, RangeJoin) is None


@pytest.mark.unit
def test_a_computed_operand_is_materialized_as_a_hidden_key(ab):
    """`a.x + 1 < b.hi` has no bare column to sort on, so the rule computes one.

    The expression is materialized in a `Project` beneath the join — the same per-row work
    the filter over the cartesian product was already doing, on the same rows — and the
    join's key names it. Without this the most common temporal-join shape
    (`a.ts - w < b.ts`) stayed quadratic.
    """
    a, b = ab
    plan = optimize_logical(
        bt.sql("SELECT lab, tag FROM a, b WHERE a.x + 1 < b.hi", a=a, b=b)._plan
    )
    node = _find(plan, RangeJoin)
    assert node is not None
    ((left_key, op, right_key),) = [(c.left_key, c.op, c.right_key) for c in node.conditions]
    assert left_key.startswith("__rj_key") and (op, right_key) == ("lt", "hi")
    assert left_key in node.left.available_columns()
    # The hidden key is an implementation detail and must not reach the output.
    assert not [c for c in node.available_columns() if c.startswith("__rj_key")]


@pytest.mark.unit
def test_a_raising_computed_operand_does_not_fire(ab):
    """Integer division can raise, and the cartesian plan never runs it on an empty side."""
    a, b = ab
    plan = optimize_logical(
        bt.sql("SELECT lab, tag FROM a, b WHERE a.x / 2 < b.hi", a=a, b=b)._plan
    )
    assert _find(plan, RangeJoin) is None


@pytest.mark.unit
def test_mismatched_key_types_do_not_fire():
    a = pa.table({"x": pa.array([1, 2], type=pa.int64()), "lab": ["a", "b"]})
    b = pa.table({"y": pa.array([1.5], type=pa.float64()), "tag": ["p"]})
    plan = optimize_logical(bt.sql("SELECT lab, tag FROM a, b WHERE a.x < b.y", a=a, b=b)._plan)
    assert _find(plan, RangeJoin) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ranges", "expected"),
    [
        # X entirely below Y: every pair satisfies `X < Y`.
        ((0.0, 1.0, 2.0, 3.0), 1.0),
        # X entirely above Y: none does.
        ((2.0, 3.0, 0.0, 1.0), 0.0),
        # Identical ranges, both uniform: half the pairs, by symmetry.
        ((0.0, 1.0, 0.0, 1.0), 0.5),
        # Y's range sits in the middle of X's: `P(X < Y)` is the mass of X below Y's mean.
        ((0.0, 4.0, 1.0, 3.0), 0.5),
        # Two point masses.
        ((1.0, 1.0, 2.0, 2.0), 1.0),
        ((2.0, 2.0, 1.0, 1.0), 0.0),
    ],
)
def test_uniform_p_less_closed_form(ranges, expected):
    a1, b1, a2, b2 = ranges
    assert _uniform_p_less(a1, b1, a2, b2) == pytest.approx(expected, abs=1e-9)


@pytest.mark.unit
def test_uniform_p_less_agrees_with_a_monte_carlo_draw():
    """The closed form is only worth having if it is the integral it claims to be."""
    import random

    rng = random.Random(11)
    for a1, b1, a2, b2 in [(0.0, 10.0, 5.0, 20.0), (-3.0, 3.0, 0.0, 1.0), (0.0, 1.0, 0.9, 5.0)]:
        n = 200_000
        hits = sum(1 for _ in range(n) if rng.uniform(a1, b1) < rng.uniform(a2, b2))
        assert _uniform_p_less(a1, b1, a2, b2) == pytest.approx(hits / n, abs=0.01)


@pytest.mark.unit
def test_cardinality_tracks_the_inputs_not_a_constant():
    """The whole point of the estimator: doubling an input must move the estimate.

    Before it existed every range join reported the fixed `unknown_rows` default, so join
    ordering and memory sizing above one could not tell a small range join from a huge one.
    """
    from batcher.config import active_config
    from batcher.kyber.stats import StatsEstimator

    def rows_for(n_left: int, n_right: int) -> float:
        a = pa.table({"x": list(range(n_left)), "lab": ["a"] * n_left})
        b = pa.table({"y": list(range(n_right)), "tag": ["p"] * n_right})
        ds = bt.sql("SELECT lab, tag FROM a, b WHERE a.x < b.y", a=a, b=b)
        node = _find(optimize_logical(ds._plan), RangeJoin)
        assert node is not None
        est = StatsEstimator(ds._sources, {}, active_config().optimizer.cardinality)
        return est.estimate(node).rows

    small = rows_for(10, 10)
    big = rows_for(20, 20)
    assert big > small, f"{big} must exceed {small}"
    assert small < 1e9, "a range join must not report the unknown-rows default"


@pytest.mark.unit
def test_known_bounds_beat_the_fallback_constant():
    """With both ranges known the estimate is the closed form, not System R's 1/3."""
    from batcher.kyber.stats import StatsEstimator
    from batcher.plan.logical import RangeCondition

    a = pa.table({"x": [1, 2], "lab": ["a", "b"]})
    b = pa.table({"y": [1, 2], "tag": ["p", "q"]})
    plan = optimize_logical(bt.sql("SELECT lab, tag FROM a, b WHERE a.x < b.y", a=a, b=b)._plan)
    node = _find(plan, RangeJoin)
    assert node is not None

    est = StatsEstimator([a, b])
    left = RelStats(1000.0, Provenance.DEFAULT, {"x": ColumnStat(min=0, max=10)})
    right = RelStats(1000.0, Provenance.DEFAULT, {"y": ColumnStat(min=100, max=110)})
    cond = RangeCondition("x", "y", "lt")
    # Disjoint ranges with X entirely below Y: every pair satisfies `x < y`.
    assert est._inequality_selectivity(left, right, cond) == pytest.approx(1.0)
    # And the reverse comparison over the same bounds admits nothing (floored, not zero).
    flipped = RangeCondition("x", "y", "gt")
    assert est._inequality_selectivity(left, right, flipped) < 1e-5


@pytest.mark.unit
def test_the_plan_cache_cannot_hand_one_schema_the_other_schemas_plan():
    """A scan's IR is only its `source_id`, so its *schema* has to be in the plan key.

    Without it, two runs of one query text over sources with the same column names and
    different column types collided in `kyber.plan_cache`, and the second run was handed
    the plan optimized for the first one's types. Silently: every schema-dependent
    rewrite had already decided. Found while adding the range-join rule, whose own
    key-type check is exactly such a decision.
    """
    query = "SELECT lab, tag FROM a, b WHERE a.x < b.y"
    mixed = (
        pa.table({"x": pa.array([1, 2], type=pa.int64()), "lab": ["a", "b"]}),
        pa.table({"y": pa.array([1.5], type=pa.float64()), "tag": ["p"]}),
    )
    same = (
        pa.table({"x": pa.array([1, 2], type=pa.int64()), "lab": ["a", "b"]}),
        pa.table({"y": pa.array([1, 2], type=pa.int64()), "tag": ["p", "q"]}),
    )
    mixed_plan = bt.sql(query, a=mixed[0], b=mixed[1])._plan
    same_plan = bt.sql(query, a=same[0], b=same[1])._plan
    assert mixed_plan.content_key() != same_plan.content_key()
    # And the key still *hits* for a genuinely identical plan, or the cache is pointless.
    assert same_plan.content_key() == bt.sql(query, a=same[0], b=same[1])._plan.content_key()

    # The observable consequence: matched types get the range join, mixed ones do not,
    # in either order.
    assert _find(optimize_logical(same_plan), RangeJoin) is not None
    assert _find(optimize_logical(mixed_plan), RangeJoin) is None
