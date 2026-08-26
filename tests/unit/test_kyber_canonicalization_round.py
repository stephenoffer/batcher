"""The optimizer's post-FUSION canonicalization round, and the contract it rests on.

Kyber runs its phases as a **single forward pass**, so a rule only ever sees the plan as it
stands when its own phase runs. That is wrong for a *canonicalizing* rule — one that
collapses a shape rather than improving it — because the later phases put the shape back:
projection pushdown stacks `Project` on `Project`, and join reordering and fusion re-parent
subtrees so operators that were separated become adjacent again. The canonicalizer has long
since run, nothing runs it again, and the redundant operator ships to the engine.

`Optimizer._run_cleanup` re-runs exactly the rules that declared `Rule.recanonicalize`, once,
after FUSION. These tests pin the three things that make that sound:

* the round actually collapses the shapes the later phases create (`_collapses_*`);
* it is *contracting* — it never hands the engine a bigger plan than it was given;
* it stays **before SELECTION**, because `split_expensive_filter` deliberately emits the
  stacked `Filter` that `merge_adjacent_filters` exists to fuse, and a round after it would
  undo that decision on every query.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.kyber.optimizer.facade import Optimizer, optimize_logical
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rule import Phase, RuleCategory
from batcher.plan.logical import Filter, Project
from batcher.plan.visitor import walk

pytestmark = pytest.mark.unit


def _nodes(plan) -> list[type]:
    return [type(n) for n in walk(plan)]


def _stacked(plan, node_type: type) -> int:
    """How many `node_type` nodes sit directly on top of another one."""
    return sum(1 for n in walk(plan) if isinstance(n, node_type) and isinstance(n.input, node_type))


# --- the round is populated, and only by rules that may safely run twice ----------------


def test_the_cleanup_set_is_not_empty():
    """A silently empty set would make every test below vacuously pass."""
    assert DEFAULT_REGISTRY.recanonicalize_rules(), (
        "no rule declares recanonicalize=True, so the canonicalization round does nothing"
    )


def test_no_cost_based_or_enforcing_rule_joins_the_cleanup_round():
    """Re-running a run-once decision corrupts what it recorded.

    A SELECTION rule makes a cost-based choice and an ENFORCE rule inserts a required
    operator; neither is a semantics-preserving contraction, and `build_side_rule` in
    particular re-derives from the join it already swapped and overwrites the telemetry that
    describes what actually happened.
    """
    offenders = [
        r.name
        for r in DEFAULT_REGISTRY.recanonicalize_rules()
        if r.phase in (Phase.SELECTION, Phase.ENFORCE)
        or r.category in (RuleCategory.SELECTION, RuleCategory.ENFORCE)
    ]
    assert offenders == [], f"run-once rules must not set recanonicalize=True: {offenders}"


def test_the_round_runs_before_selection_so_it_cannot_undo_a_split_filter():
    """`split_expensive_filter` emits the exact shape `merge_adjacent_filters` fuses.

    The two are inverses. Nothing but phase order keeps them from ping-ponging, so the
    canonicalization round must sit strictly before SELECTION. Pinned as a phase comparison
    rather than as behaviour because the failure it guards against is silent: the plan stays
    correct and only gets slower.
    """
    cleanup_phases = {r.phase for r in DEFAULT_REGISTRY.recanonicalize_rules()}
    assert all(p < Phase.SELECTION for p in cleanup_phases)


# --- it collapses what the later phases create ------------------------------------------


def test_it_collapses_projections_that_pushdown_stacked():
    """`Project(Project(...))` left behind after column pruning is folded into one."""
    ds = (
        bt.from_pydict({"a": [1], "b": [2], "c": [3]})
        .select("a", "b", "c")
        .filter(bt.col("a") > 0)
        .select(x=bt.col("a"), y=bt.col("b"))
        .select(x=bt.col("x"))
    )
    out = optimize_logical(ds._plan)
    assert _stacked(out, Project) == 0


def test_it_collapses_filters_that_reordering_made_adjacent():
    ds = (
        bt.from_pydict({"a": [1, 2, 3], "b": [4, 5, 6]})
        .filter(bt.col("a") > 0)
        .filter(bt.col("b") < 10)
        .filter(bt.col("a") < 3)
    )
    out = optimize_logical(ds._plan)
    assert _stacked(out, Filter) == 0


# --- and it never makes a plan worse -----------------------------------------------------


_SHAPES = {
    "filter-chain": lambda L, R: L.filter(bt.col("b") > 0.1).filter(bt.col("a") < 50),
    "project-stack": lambda L, R: L.select("a", "b").select(x=bt.col("a")),
    "join-agg": lambda L, R: L.join(R, on="a").group_by("c").agg(s=bt.col("b").sum()),
    "agg-over-project": lambda L, R: (
        L.select(k=bt.col("c"), v=bt.col("b")).group_by("k").agg(s=bt.col("v").sum())
    ),
    "sort-limit": lambda L, R: L.sort("b").limit(3),
    "deep": lambda L, R: (
        L.filter(bt.col("a") > 1)
        .select("a", "b", "c")
        .filter(bt.col("b") > 0.1)
        .group_by("c")
        .agg(s=bt.col("b").sum())
        .filter(bt.col("s") > 0)
        .sort("s")
        .limit(4)
    ),
}


@pytest.fixture
def sides():
    left = bt.from_pydict({"a": [1, 2, 3, 4], "b": [0.5, 0.2, 0.9, 0.4], "c": [1, 1, 2, 2]})
    right = bt.from_pydict({"a": [1, 2, 3, 4], "e": [1.0, 2.0, 3.0, 4.0]})
    return left, right


def _rules_without_the_round():
    """The same rule set with the cleanup round disabled — and nothing else changed.

    Clearing the flag rather than dropping the rules is what isolates the round: a rule that
    declared `recanonicalize` still runs in its own phase, exactly as it did before the round
    existed, so the only difference between the two optimizers is the second application.
    """
    import dataclasses

    return [dataclasses.replace(r, recanonicalize=False) for r in DEFAULT_REGISTRY.rules()]


@pytest.mark.parametrize("name", sorted(_SHAPES))
def test_the_round_never_grows_a_plan(name, sides):
    """Contraction is the property that makes running these rules a second time safe."""
    left, right = sides
    plan = _SHAPES[name](left, right)._plan
    with_round = Optimizer(rules=list(DEFAULT_REGISTRY.rules())).logical_rewrite(plan)
    without_round = Optimizer(rules=_rules_without_the_round()).logical_rewrite(plan)
    assert len(_nodes(with_round)) <= len(_nodes(without_round)), (
        f"{name}: the canonicalization round added operators"
    )


@pytest.mark.parametrize("name", sorted(_SHAPES))
def test_the_round_preserves_the_output_schema(name, sides):
    """A canonicalizer may remove an operator, never a column or a column's type.

    The cheap half of the equivalence claim, pinned here so a schema-level breakage fails a
    fast unit test; the row-level half is the differential test against DuckDB in
    `tests/differential/test_diff_canonicalization_round.py`.
    """
    left, right = sides
    plan = _SHAPES[name](left, right)._plan
    with_round = Optimizer(rules=list(DEFAULT_REGISTRY.rules())).logical_rewrite(plan)
    without_round = Optimizer(rules=_rules_without_the_round()).logical_rewrite(plan)
    assert with_round.available_schema() == without_round.available_schema()
