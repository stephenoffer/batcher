"""Every `expr_matches` / `expr_ops` declaration must be a filter, never a behavior change.

A rule declares the expression shapes it needs, and the driver uses that twice: to drop the
rule for a plan whose expressions contain none of them, and to dispatch each expression to
only the leaves that named its type. Both are meant to be *strict filters* — they may skip
only a rule that would have returned its input unchanged.

A declaration that is too narrow breaks that quietly. The rule is semantics-preserving, so
the answer stays correct; the rule simply stops firing, and only plan quality degrades. No
ordinary test notices. `BATCHER_VERIFY_EXPR_MATCHES=1` is the guard: it re-runs the fused
chain undeclared for every expression, and re-runs each whole phase with the vocabulary
filter off, failing on any difference. This module turns that guard on in-process so it runs
in normal CI over a battery covering each declared family, rather than only when someone
remembers the environment variable.
"""

from __future__ import annotations

import datetime as dt

import pytest

import batcher as bt
from batcher import col, lit
from batcher.kyber.optimizer import driver, expr_dispatch, optimize_logical


@pytest.fixture
def verified(monkeypatch):
    """Turn the cross-check on in both modules that read it.

    `driver` imports the flag by value, so patching only `expr_dispatch` would leave the
    phase-level check off and silently verify half of what this module claims to.
    """
    monkeypatch.setattr(expr_dispatch, "VERIFY_EXPR_MATCHES", True)
    monkeypatch.setattr(driver, "VERIFY_EXPR_MATCHES", True)


def _ds():
    return bt.from_pydict(
        {
            "i": [1, 2, 3],
            "j": [-4, 0, 5],
            "f": [1.5, -2.5, 3.5],
            "s": ["Abc", "b_b", "%c%"],
            "t": [
                dt.datetime(2021, 1, 1, 3, 4, 5),
                dt.datetime(2021, 6, 30),
                dt.datetime(2022, 2, 1),
            ],
            "d": [dt.date(2021, 1, 1), dt.date(2021, 6, 30), dt.date(2022, 2, 1)],
            "l": [[1, 2], [3], []],
        }
    )


#: One entry per declared family, named so a failure says which declaration is wrong.
CASES: dict[str, object] = {
    "rounding_ranges": lambda ds: ds.filter(col("f").floor() == lit(2)),
    "abs_range": lambda ds: ds.filter(col("f").abs() < lit(3)),
    "sign_integer": lambda ds: ds.filter(col("j").sign() > lit(0)),
    "cast_unwrap": lambda ds: ds.filter(col("i").cast("float64") >= lit(3.0)),
    "cast_folds": lambda ds: ds.select(r=col("i").cast("int64").cast("int64")),
    "like_family": lambda ds: ds.filter(col("s").str.contains("b")),
    "string_folds": lambda ds: ds.select(r=lit("aB").str.upper(), n=lit("abc").str.len()),
    "string_lengths": lambda ds: ds.filter(col("s").str.len() > lit(0)),
    "temporal_parts": lambda ds: ds.filter(col("t").dt.year() == lit(2021)),
    "temporal_trunc": lambda ds: ds.filter(col("t").dt.truncate("day") < lit(dt.date(2022, 1, 1))),
    "temporal_date": lambda ds: ds.filter(col("d").dt.month() != lit(6)),
    "null_algebra": lambda ds: ds.filter(col("i").is_not_null() & col("s").is_null()),
    "coalesce": lambda ds: ds.select(r=bt.coalesce(col("i"), col("j"), lit(0)).is_null()),
    "boolean_algebra": lambda ds: ds.filter((col("i") > lit(0)) | (col("i") > lit(0))),
    # The `NOT` family declares `(Not,)` rather than `(Binary, Not)`: each of these rewrites
    # the `Not` itself, and its `Binary` is only an operand the traversal reaches on its own.
    # Narrowing a declaration is exactly the move this module exists to police, so every one
    # of them gets a case.
    "not_of_comparison": lambda ds: ds.filter(~(col("i") == lit(1))),
    "de_morgan_and": lambda ds: ds.filter(~((col("i") > lit(1)) & (col("j") < lit(5)))),
    "de_morgan_or": lambda ds: ds.filter(~((col("i") > lit(1)) | (col("j") < lit(5)))),
    "double_negation": lambda ds: ds.filter(~(~(col("i") > lit(1)))),
    "not_of_null_check": lambda ds: ds.filter(~col("i").is_null()),
    "in_list": lambda ds: ds.filter(col("i").is_in([1, 1, 2])),
    "predicate_bounds": lambda ds: ds.filter((col("i") < lit(5)) | (col("i") < lit(9))),
    "sargable": lambda ds: ds.filter(col("i") + lit(2) == lit(7)),
    "conditionals": lambda ds: ds.select(
        r=bt.when(col("i") > lit(1)).then(col("f").abs()).otherwise(lit(0.0))
    ),
    "collections": lambda ds: ds.select(r=col("l").list.len()),
    "greatest_least": lambda ds: ds.select(r=bt.greatest(col("i"), col("j"), col("i"))),
    "mixed_wide": lambda ds: ds.filter(
        (col("i").abs() < lit(50))
        & (col("s").str.len() > lit(0))
        & col("t").dt.year().is_not_null()
    ).select(a=col("i").abs().floor(), b=col("s").str.upper(), c=col("t").dt.month()),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_declarations_are_pure_filters(name, verified):
    # The assertion is that optimizing does not raise: with the cross-check on, the driver
    # itself compares declared against undeclared and raises if they ever disagree.
    optimize_logical(CASES[name](_ds())._plan)


def test_the_cross_check_is_actually_engaged(verified):
    # A guard on the guard: if the flag stopped reaching the driver, every test above would
    # pass while verifying nothing.
    assert driver.VERIFY_EXPR_MATCHES and expr_dispatch.VERIFY_EXPR_MATCHES


def test_a_too_narrow_declaration_is_caught(verified):
    # Prove the cross-check has teeth. This leaf rewrites a `Binary`, but is declared as
    # touching only `Lit` — so the dispatch never offers it the expression it acts on and it
    # silently stops firing, which is exactly the failure the guard exists to surface.
    from batcher.plan.expr_ir import Binary, Expr, Lit

    def rewrites_a_binary(expr: Expr) -> Expr:
        return Lit(True) if isinstance(expr, Binary) else expr

    node = _ds().filter(col("i") > lit(1))._plan

    honest = expr_dispatch.apply_expr_leaves(node, [(rewrites_a_binary, frozenset({Binary}), None)])
    assert honest is not node, "the leaf must actually fire, or this proves nothing"

    with pytest.raises(AssertionError, match="expr_matches"):
        expr_dispatch.apply_expr_leaves(node, [(rewrites_a_binary, frozenset({Lit}), None)])


def test_a_too_narrow_declaration_on_a_whole_rule_is_caught(verified):
    # The leaf-level check above only sees rules that join the fused chain. A rule that walks
    # the plan itself is dropped *entirely* by the vocabulary prefilter, so it needs the
    # phase-level check — and that one has to compare the phase's fixpoint rather than a
    # single iteration, because a rewrite can introduce an expression type the plan did not
    # have and the rules keyed on it legitimately wait a turn.
    from batcher.kyber.rule import Phase, node_rule
    from batcher.kyber.rules.leaf_rewrite import rewrite_node
    from batcher.plan.expr_ir import Lit, StrFunc
    from batcher.plan.logical import Filter, Project

    def leaf(expr):
        return Lit(99) if isinstance(expr, Lit) and expr.value == 1 else expr

    def build(name, declared):
        return node_rule(
            name,
            Phase.NORMALIZE,
            lambda node, _ctx: rewrite_node(node, leaf),
            matches=(Filter, Project),
            expr_matches=declared,
        )

    plan = bt.from_pydict({"a": [1, 2]}).select(r=col("a") + lit(1))._plan

    honest, _ = driver._run_phase(plan, [build("probe_honest", (Lit,))], None, 4)
    assert "99" in str(honest.items[0].expr), "the probe rule must fire, or this proves nothing"

    # The leaf only ever rewrites a `Lit`, so declaring `StrFunc` can never be satisfied.
    with pytest.raises(AssertionError, match="too narrow"):
        driver._run_phase(plan, [build("probe_lying", (StrFunc,))], None, 4)
