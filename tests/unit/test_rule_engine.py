"""The Kyber rule engine: phased rules, pattern indexing, fixpoint, scaling.

These tests cover the *framework* (W0), not individual optimizations — that the
builtin rules are registered into the right phases, that the driver only attempts
rules whose node types are present (the indexing that lets the rule set scale to
thousands), that rewrite phases reach a fixpoint, and that the migrated pipeline
still produces the optimized plans the old ordered-pass pipeline did.
"""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer, _applicable
from batcher.kyber.registry import DEFAULT_REGISTRY, RuleRegistry, register_builtin_rules
from batcher.kyber.rule import Phase, RuleCategory, node_rule
from batcher.plan.logical import Filter, Join, Scan, Window

_BUILTINS = {
    "constant_folding",
    "expr_simplification",
    "predicate_pushdown",
    "projection_rewrite",
    "topn_fusion",
    "adaptive_build_side",
}


def _emp_dept():
    emp = bt.from_pydict({"id": [1, 2, 3], "name": ["a", "b", "c"], "dept_id": [10, 20, 10]})
    dept = bt.from_pydict({"dept_id": [10, 20], "dept": ["eng", "sales"], "budget": [100, 200]})
    return emp, dept


# --- Registry / phase assignment ---------------------------------------------


def test_builtin_rules_registered_in_expected_phases():
    by_name = {r.name: r for r in DEFAULT_REGISTRY.rules()}
    assert set(by_name) >= _BUILTINS
    assert by_name["constant_folding"].phase is Phase.NORMALIZE
    assert by_name["expr_simplification"].phase is Phase.NORMALIZE
    assert by_name["predicate_pushdown"].phase is Phase.PUSHDOWN
    assert by_name["projection_rewrite"].phase is Phase.PUSHDOWN
    assert by_name["topn_fusion"].phase is Phase.FUSION
    assert by_name["adaptive_build_side"].phase is Phase.SELECTION
    assert by_name["adaptive_build_side"].category is RuleCategory.SELECTION


def test_registry_dedupes_by_name():
    reg = RuleRegistry()
    r = node_rule("dup", Phase.REWRITE, lambda n, c: None, matches=(Filter,))
    reg.add(r)
    reg.add(r)
    assert sum(1 for x in reg.rules() if x.name == "dup") == 1


# --- Pattern indexing (the scaling property) ---------------------------------


def test_applicable_skips_rules_whose_types_are_absent():
    join_rule = node_rule("j", Phase.REWRITE, lambda n, c: None, matches=(Join,))
    any_rule = node_rule("f", Phase.REWRITE, lambda n, c: None, matches=(Filter,))
    present = frozenset({Filter, Scan})
    applicable = _applicable([join_rule, any_rule], present)
    assert join_rule not in applicable  # no Join in the plan
    assert any_rule in applicable


def test_many_inapplicable_rules_never_fire():
    # Register the builtins plus 500 rules that only match Window — a node type
    # absent from the plan. They must add nothing: pattern indexing prunes them.
    reg = RuleRegistry()
    register_builtin_rules(reg)
    fired = {"n": 0}

    def mark(_node, _ctx):
        fired["n"] += 1
        return None

    for i in range(500):
        reg.add(node_rule(f"noop_{i}", Phase.REWRITE, mark, matches=(Window,)))

    emp, dept = _emp_dept()
    plan = emp.join(dept, on="dept_id").filter(col("id") > 1).sort("id").head(2)._plan
    ir = Optimizer(rules=reg.rules()).optimize(plan).ir

    assert fired["n"] == 0  # none fired — the plan has no Window node
    # Still correctly optimized: top-N fused into the Sort, join below, filter pushed.
    assert ir["op"] == "sort"
    assert ir["limit"] == 2
    assert ir["input"]["op"] == "hash_join"


# --- Migrated optimizations still fire ---------------------------------------


def test_predicate_pushed_below_join_through_optimizer():
    emp, dept = _emp_dept()
    plan = emp.join(dept, on="dept_id").filter(col("id") > 1)._plan
    ir = Optimizer().optimize(plan).ir
    assert ir["op"] == "hash_join"  # filter pushed under the join
    assert ir["left"]["op"] == "filter"


def test_topn_fusion_through_optimizer():
    emp, _ = _emp_dept()
    plan = emp.sort("id").head(2)._plan
    ir = Optimizer().optimize(plan).ir
    assert ir["op"] == "sort"
    assert ir["limit"] == 2  # Limit fused into the Sort as a top-N


def test_constant_folding_applied_through_optimizer():
    # NORMALIZE must actually fold expression *content* (not just shape-changing
    # rewrites). 2 + 3 should disappear, folded to the literal 5. This guards the
    # fixpoint change-detection: a structural `==` over Expr-bearing plans is
    # meaningless (Expr.__eq__ builds a comparison), so the driver compares IR.
    import json

    emp, _ = _emp_dept()
    plan = emp.filter(col("id") > (bt.lit(2) + bt.lit(3)))._plan
    ir = Optimizer().optimize(plan).ir
    assert '"add"' not in json.dumps(ir)  # the constant add was folded away


def test_optimization_is_deterministic():
    emp, dept = _emp_dept()
    plan = emp.join(dept, on="dept_id").filter(col("id") > 1).sort("id").head(2)._plan
    assert Optimizer().optimize(plan).ir == Optimizer().optimize(plan).ir


# --- Rule mechanics ----------------------------------------------------------


def test_node_rule_returning_none_leaves_plan_unchanged():
    emp, _ = _emp_dept()
    plan = emp.filter(col("id") > 1)._plan
    noop = node_rule("noop", Phase.REWRITE, lambda n, c: None, matches=(Filter,))
    assert Optimizer(rules=[noop]).optimize(plan).ir == plan.to_ir()


def test_empty_rule_set_is_identity():
    emp, _ = _emp_dept()
    plan = emp.filter(col("id") > 1)._plan
    assert Optimizer(rules=[]).optimize(plan).ir == plan.to_ir()


# --- Fused node-rule traversal (one walk for a run of node rules) ------------


def test_fused_node_rules_equal_sequential():
    """A run of node-local rules applied in one bottom-up pass yields the same plan
    as running each rule's own `transform_up` in sequence."""
    from batcher.kyber.optimizer import _apply_node_rules

    emp, _ = _emp_dept()
    plan = emp.filter(col("id") > 1).filter(col("id") < 9).filter(col("name") > "a")._plan

    # Two node-local rules from different families, exercised together.
    rules = [r for r in DEFAULT_REGISTRY.rules() if r.name == "merge_adjacent_filters"]
    rules += [r for r in DEFAULT_REGISTRY.rules() if r.name == "prune_true_filter"]
    assert all(r.node_fn is not None for r in rules)

    fused = _apply_node_rules(plan, rules, None)

    sequential = plan
    for r in rules:  # each rule's standalone (guarded) whole-plan traversal, in order
        sequential = r.fn(sequential, None)
    assert fused.to_ir() == sequential.to_ir()


def test_builtin_node_rules_carry_node_fn():
    # The driver fuses rules with a `node_fn`; whole-plan rules keep it None.
    by_name = {r.name: r for r in DEFAULT_REGISTRY.rules()}
    assert by_name["merge_adjacent_filters"].node_fn is not None
    assert by_name["predicate_pushdown"].node_fn is None  # a whole-plan rule


# --- Structural sharing + identity-based fixpoint (driver performance) --------


def test_transform_up_preserves_identity_on_noop():
    from batcher.plan.visitor import transform_up

    emp, _ = _emp_dept()
    plan = emp.filter(col("id") > 1).sort("id")._plan
    # A no-op rewrite must return the *same* object (so fixpoint detection is O(1)).
    assert transform_up(plan, lambda n: n) is plan


def test_transform_up_shares_untouched_subtrees():
    from batcher.plan.logical import Filter, Sort
    from batcher.plan.visitor import transform_up

    emp, _ = _emp_dept()
    # Sort(Filter(scan)): rewrite only the Sort; the Filter subtree keeps its identity.
    plan = emp.filter(col("id") > 1).sort("id")._plan
    assert isinstance(plan, Sort) and isinstance(plan.input, Filter)
    original_filter = plan.input

    out = transform_up(plan, lambda n: Sort(n.input, n.keys, 3) if isinstance(n, Sort) else n)
    assert out is not plan  # the Sort changed
    assert out.input is original_filter  # the untouched Filter subtree is shared


def test_noop_phase_returns_same_object_via_optimizer():
    # A plan already at its fixpoint: re-running the optimizer is deterministic, and
    # the driver detects no-change by identity (the to_ir fallback only confirms).
    emp, _ = _emp_dept()
    plan = emp.filter(col("id") > 1)._plan
    once = Optimizer().optimize(plan)
    assert Optimizer().optimize(plan).ir == once.ir


def test_run_phase_identity_fixpoint_no_to_ir_on_stable_plan(monkeypatch):
    # On a no-op phase, the driver must break on `updated is plan` WITHOUT serializing
    # to IR (the fallback path). Use a single node rule that never fires.
    from batcher.kyber.optimizer import _run_phase
    from batcher.kyber.rule import Phase, node_rule
    from batcher.plan.logical import Filter

    emp, _ = _emp_dept()
    plan = emp.filter(col("id") > 1)._plan
    noop = node_rule("noop", Phase.REWRITE, lambda n, c: None, matches=(Filter,))

    calls = {"n": 0}
    orig = type(plan).to_ir

    def counting_to_ir(self):
        calls["n"] += 1
        return orig(self)

    monkeypatch.setattr(type(plan), "to_ir", counting_to_ir)
    _run_phase(plan, [noop], None, 8)
    # No change ever → identity break → zero to_ir() calls in the loop.
    assert calls["n"] == 0
