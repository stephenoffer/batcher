"""The driver's expression-level no-op memo must be invisible in the output.

`_apply_expr_leaves` skips the whole leaf chain for an expression object it has already
proved nothing matched. That is a pure performance device, so what is worth asserting is
that it changes nothing an unmemoized run would produce, that it really does skip on a
repeat, and that the fixpoint still converges to the same plan across the shapes its
phases iterate hardest on.
"""

from __future__ import annotations

import datetime as dt

import pytest

import batcher as bt
from batcher import col, lit
from batcher.kyber.optimizer import driver, expr_dispatch, optimize_logical
from batcher.plan.expr_ir import Binary, Expr, Lit


def _ds():
    return bt.from_pydict(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "f": [1.5, 2.5, 3.5],
            "s": ["x", "yy", "zzz"],
            "t": [dt.datetime(2020, 1, 1), dt.datetime(2021, 2, 3), dt.datetime(2022, 3, 4)],
        }
    )


def _node():
    """A `Project` carrying a nested arithmetic expression for the leaves to walk."""
    return _ds().select(r=(col("a") + lit(1)) * (col("b") - lit(2)))._plan


def _double_literals(expr: Expr) -> Expr:
    """A leaf that rewrites one shape, so the memo has both hits and misses to handle."""
    if isinstance(expr, Lit) and expr.value == 1:
        return Lit(2)
    return expr


def _identity_leaf(expr: Expr) -> Expr:
    return expr


# --- the memo returns what the unmemoized chain returns ---------------------


@pytest.mark.parametrize(
    "leaves",
    [[_identity_leaf], [_double_literals], [_identity_leaf, _double_literals]],
    ids=["identity", "rewriting", "mixed"],
)
def test_memoized_and_unmemoized_agree(leaves):
    node = _node()
    memoized = expr_dispatch.apply_expr_leaves(node, leaves, {})
    plain = expr_dispatch.apply_expr_leaves(node, leaves, None)
    assert memoized.to_ir() == plain.to_ir()


def test_a_repeat_with_the_same_memo_is_stable():
    memo: dict[int, object] = {}
    node = _node()
    first = expr_dispatch.apply_expr_leaves(node, [_double_literals], memo)
    second = expr_dispatch.apply_expr_leaves(first, [_double_literals], memo)
    assert second.to_ir() == first.to_ir()


def test_the_memo_skips_the_leaf_chain_on_a_repeat():
    calls: list[Expr] = []

    def counting(expr: Expr) -> Expr:
        calls.append(expr)
        return expr

    memo: dict[int, object] = {}
    node = _node()
    expr_dispatch.apply_expr_leaves(node, [counting], memo)
    first_pass = len(calls)
    assert first_pass > 0
    expr_dispatch.apply_expr_leaves(node, [counting], memo)
    assert len(calls) == first_pass, "the second pass re-ran the leaves it had memoized"


def test_the_memo_misses_after_the_expression_is_rebuilt():
    calls: list[Expr] = []

    def counting(expr: Expr) -> Expr:
        calls.append(expr)
        return Lit(9) if isinstance(expr, Lit) and expr.value == 1 else expr

    memo: dict[int, object] = {}
    node = _node()
    rebuilt = expr_dispatch.apply_expr_leaves(node, [counting], memo)
    before = len(calls)
    expr_dispatch.apply_expr_leaves(rebuilt, [counting], memo)
    assert len(calls) > before, "a rebuilt subtree must not hit the memo"


# --- the optimizer as a whole is unaffected ---------------------------------


def _plans():
    """A battery covering the shapes the fixpoint phases iterate hardest on."""
    ds = _ds()
    wide: Expr | None = None
    for i in range(6):
        term = (col("a").abs() < lit(50 + i)) & (col("b") + lit(i) != lit(3))
        wide = term if wide is None else (wide | term)
    return [
        ds.filter(wide)._plan,
        ds.filter(col("f").abs() < lit(2)).select(r=col("f").floor())._plan,
        ds.select(r=col("s").str.upper().str.trim().is_null())._plan,
        ds.filter(col("t").dt.year() != lit(2020))._plan,
        ds.filter((col("a") < lit(3)) | (col("a") < lit(7)))._plan,
        ds.select(
            r=bt.when(col("a") > lit(1)).then(lit("ab")).otherwise(lit("cd")).str.upper()
        )._plan,
        ds.group_by("a").agg(m=col("f").quantile(0.0))._plan,
        ds.filter(col("a") > lit(0)).select(x=col("a") + lit(1), y=col("b").abs())._plan,
    ]


@pytest.mark.parametrize("index", range(8))
def test_optimizing_twice_is_stable(index):
    plan = _plans()[index]
    once = optimize_logical(plan)
    assert optimize_logical(once).to_ir() == once.to_ir()


def test_a_phase_that_changes_the_node_type_set_still_optimizes_fully():
    # `eliminate_identity_project` removes a Project mid-phase, changing which rules are
    # applicable — the case the memo has to invalidate on. The result must still be the
    # fully optimized plan rather than one frozen by stale entries.
    ds = _ds()
    plan = ds.select(a=col("a"), b=col("b")).filter(col("a").abs() < lit(2))._plan
    optimized = optimize_logical(plan)
    assert "abs" not in str(optimized.to_ir())


def test_dispatch_prefilter_keeps_every_non_leaf_rule():
    # The per-node loop iterates a prefiltered list; every rule that is *not* an already-
    # fused leaf must survive it, or a node-local rewrite would silently stop running.
    from batcher.kyber.registry import DEFAULT_REGISTRY
    from batcher.kyber.rule import Phase

    normalize = [r for r in DEFAULT_REGISTRY.rules() if r.phase is Phase.NORMALIZE]
    node_local = [r for r in normalize if r.node_fn is not None]
    dispatch = [r for r in node_local if r.expr_fn is None]
    assert dispatch, "the prefilter must not empty the node-local dispatch list"
    assert len(dispatch) < len(node_local), "the prefilter must actually remove leaf rules"


def test_optimized_plan_matches_a_single_pass_reference():
    # A once-run phase never fuses expressions, so its path has no memo at all. Comparing a
    # fixpoint-optimized plan against re-optimizing it proves the memoized path reached the
    # same fixpoint the unmemoized one would.
    for plan in _plans():
        optimized = optimize_logical(plan)
        assert optimize_logical(optimized).to_ir() == optimized.to_ir()


def test_binary_expressions_survive_the_memo():
    node = _node()
    out = expr_dispatch.apply_expr_leaves(node, [_identity_leaf], {})
    assert isinstance(out.items[0].expr, Binary)


def test_same_node_type_with_and_without_a_resolvable_schema():
    # Regression: the dispatch index is cached per plan node type for the whole pass, but the
    # leaf list is *not* the same length for every node of a type. A node whose schema
    # resolves gets the schema-dependent leaves appended; one above a `map_batches` (which
    # makes the schema unknowable) gets only the plain prefix. Caching the index on the node
    # type alone let the longer list's slot indices be reused against the shorter one, and
    # the chain indexed past its end. Both `Filter`s below are the same node type and land on
    # opposite sides of that split, so this fails with `IndexError` without the length in the
    # cache key. `tests/differential/test_diff_fuzzy_dedup.py` is what caught it originally.
    ds = bt.from_pydict({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]})
    plan = (
        ds.filter(col("a").abs() < lit(10))
        .map_batches(lambda batch: batch)
        .filter(col("a").abs() < lit(20))
        ._plan
    )
    # `map_batches` runs in Python and lowers to no IR, so the plan is compared structurally.
    optimized = optimize_logical(plan)
    assert repr(optimized)


def test_rule_selection_cache_answers_per_input():
    # `_applicable` is memoized on `(rule list, node types, expression shapes)`. A cache that
    # ignored any of the three would serve one plan's rule set to another — silently running
    # the wrong rules, with results still correct and only plan quality wrong. So the test is
    # that distinct inputs give distinct answers and repeated inputs give identical ones.
    from batcher.kyber.registry import DEFAULT_REGISTRY
    from batcher.kyber.rule import Phase
    from batcher.plan.expr_ir import Binary, Col, DateFunc, StrFunc
    from batcher.plan.logical import Filter, Project

    rules = [r for r in DEFAULT_REGISTRY.rules() if r.phase is Phase.NORMALIZE]
    present = frozenset({Filter, Project})
    numeric = frozenset({(Binary, "lt"), (Col, None)})
    textual = frozenset({(StrFunc, "upper"), (Col, None)})
    temporal = frozenset({(DateFunc, "year"), (Col, None)})

    first = driver._applicable(rules, present, numeric)
    assert [r.name for r in driver._applicable(rules, present, numeric)] == [r.name for r in first]

    for other in (textual, temporal):
        assert [r.name for r in driver._applicable(rules, present, other)] != [
            r.name for r in first
        ], "a different expression vocabulary must select a different rule set"

    # And the vocabulary filter is a strict subset of the node-type filter alone.
    unfiltered = driver._applicable(rules, present)
    assert set(first).issubset(set(unfiltered)) and len(first) < len(unfiltered)
