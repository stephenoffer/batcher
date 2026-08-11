"""`factor_common_conjuncts` must reach an `OR` nested inside an `AND`, not only a bare one.

The rule exists so an equi-join condition hidden inside a disjunction is pulled out where
join-key derivation can see it. It used to match only a predicate that was *itself* a
top-level `OR`, which covers TPC-H Q19 and little else: TPC-DS q13 and q48 write
`join-preds AND (…OR…) AND (…OR…)`, and the only mention of two of their dimension keys is
inside those disjunctions. There the keys stayed buried and the six-way join planned as a
chain of cartesian products — 1.2e16 estimated rows on q13, which does not raise, it gets
the process killed.

These are plan-shape assertions. That the rewrite preserves *results* is pinned separately
against DuckDB in `tests/differential/test_diff_kyber_factor_disjunction.py`.
"""

from __future__ import annotations

import batcher as bt
from batcher.kyber.rules.algebraic.disjunctions import factor_common_conjuncts
from batcher.plan.expr_rewrite import split_conjuncts, split_disjuncts
from batcher.plan.logical import Filter


def _filter(predicate) -> Filter:
    ds = bt.from_pydict({"a": [1, 2], "b": [1, 2], "x": [1, 2], "y": [1, 2]})
    return Filter(ds._plan, predicate)


def _rewrite(predicate):
    """The rewritten predicate, or `None` when the rule declined."""
    out = factor_common_conjuncts(_filter(predicate), None)
    return None if out is None else out.predicate


def test_factors_a_bare_top_level_or():
    """The original behavior: `(A AND X) OR (A AND Y)` → `A AND (X OR Y)`."""
    a = bt.col("a") == bt.col("b")
    predicate = (a & (bt.col("x") > 1)) | (a & (bt.col("y") > 2))
    conjuncts = split_conjuncts(_rewrite(predicate))
    # `a` is now a conjunct in its own right, and the residual OR holds the rest.
    assert len(conjuncts) == 2
    assert any(len(split_disjuncts(c)) == 2 for c in conjuncts)


def test_factors_an_or_nested_under_an_and():
    """The regression: the same `OR` as one conjunct among several must still factor."""
    a = bt.col("a") == bt.col("b")
    disjunction = (a & (bt.col("x") > 1)) | (a & (bt.col("y") > 2))
    predicate = (bt.col("x") < 100) & disjunction
    conjuncts = split_conjuncts(_rewrite(predicate))
    # `x < 100`, the factored `a`, and the residual `(x > 1) OR (y > 2)`.
    assert len(conjuncts) == 3
    assert sum(len(split_disjuncts(c)) == 2 for c in conjuncts) == 1


def test_factors_every_disjunct_conjunct_in_one_pass():
    """TPC-DS q13's shape: two independent disjunctions, each hiding its own join key."""
    a = bt.col("a") == bt.col("b")
    x = bt.col("x") == bt.col("y")
    predicate = (
        (bt.col("a") > 0)
        & ((a & (bt.col("x") > 1)) | (a & (bt.col("x") > 2)))
        & ((x & (bt.col("y") > 3)) | (x & (bt.col("y") > 4)))
    )
    conjuncts = split_conjuncts(_rewrite(predicate))
    # a > 0, factored `a`, its residual OR, factored `x`, its residual OR.
    assert len(conjuncts) == 5
    assert sum(len(split_disjuncts(c)) == 2 for c in conjuncts) == 2


def test_declines_when_no_conjunct_is_shared():
    """Nothing common means nothing to factor — the rule must not rewrite at all."""
    predicate = ((bt.col("a") > 1) & (bt.col("x") > 1)) | ((bt.col("b") > 2) & (bt.col("y") > 2))
    assert _rewrite(predicate) is None


def test_is_idempotent():
    """A second application must decline, or the NORMALIZE phase would not reach a fixpoint."""
    a = bt.col("a") == bt.col("b")
    disjunction = (a & (bt.col("x") > 1)) | (a & (bt.col("y") > 2))
    once = _rewrite((bt.col("x") < 100) & disjunction)
    assert once is not None
    assert _rewrite(once) is None
