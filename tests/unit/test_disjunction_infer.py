"""Plan-shape, idempotence, and safety unit tests for `disjunction_infer`.

The rule derives per-column ``IN`` predicates implied by a multi-column DNF conjunct
so pushdown can sink them. Each check covers: it fires and adds the intended
membership(s), applying it twice equals once (idempotence — the fixpoint driver
needs it), and it does *not* fire on a plain single-column ``OR`` (left to
``or_to_in_and_range``) or where a column is not pinned in every disjunct.
"""

from __future__ import annotations

import json

import batcher as bt
from batcher import col
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra import disjunction_infer as m
from batcher.plan.expr_ir import Binary, Col, InList, Lit
from batcher.plan.expr_rewrite import split_conjuncts
from batcher.plan.logical import Filter


def _plan(pred) -> Filter:
    ds = bt.from_pydict({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
    return ds.filter(pred)._plan


def _added(node) -> set[str]:
    """The IR of every conjunct on the rewritten filter (the original plus what was added)."""
    return {json.dumps(c.to_ir(), sort_keys=True) for c in split_conjuncts(node.predicate)}


def test_registered():
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert "infer_disjunction_in_lists" in names


def test_two_column_dnf_derives_both_in_lists():
    # (a=1 AND b=2) OR (a=3 AND b=4)  ->  adds a IN (1,3) and b IN (2,4)
    pred = ((col("a") == 1) & (col("b") == 2)) | ((col("a") == 3) & (col("b") == 4))
    out = m.infer_disjunction_in_lists(_plan(pred), None)
    assert out is not None
    added = _added(out)
    import json

    assert json.dumps(InList(Col("a"), (1, 3)).to_ir(), sort_keys=True) in added
    assert json.dumps(InList(Col("b"), (2, 4)).to_ir(), sort_keys=True) in added


def test_idempotent():
    pred = ((col("a") == 1) & (col("b") == 2)) | ((col("a") == 3) & (col("b") == 4))
    once = m.infer_disjunction_in_lists(_plan(pred), None)
    assert once is not None
    again = m.infer_disjunction_in_lists(once, None)
    assert again is None or again.to_ir() == once.to_ir()


def test_column_pinned_to_one_value_becomes_equality():
    # a is 1 in both disjuncts -> a = 1 (sharper than IN); b differs -> b IN (2,4)
    pred = ((col("a") == 1) & (col("b") == 2)) | ((col("a") == 1) & (col("b") == 4))
    out = m.infer_disjunction_in_lists(_plan(pred), None)
    assert out is not None
    added = _added(out)
    import json

    assert json.dumps(Binary("eq", Col("a"), Lit(1)).to_ir(), sort_keys=True) in added


def test_does_not_fire_on_single_column_or():
    # plain c=1 OR c=2 — no disjunct is a conjunction; left to or_to_in_and_range
    pred = (col("c") == 1) | (col("c") == 2)
    assert m.infer_disjunction_in_lists(_plan(pred), None) is None


def test_column_not_in_every_disjunct_is_skipped():
    # b appears only in the first disjunct so it is not derivable; a is pinned in both,
    # and the first disjunct being a conjunction qualifies the DNF, so a IN (1,3) is added
    # while no `b` membership is (a superset of `b` cannot be proven).
    import json

    pred = ((col("a") == 1) & (col("b") == 2)) | (col("a") == 3)
    out = m.infer_disjunction_in_lists(_plan(pred), None)
    assert out is not None
    added = _added(out)
    assert json.dumps(InList(Col("a"), (1, 3)).to_ir(), sort_keys=True) in added
    assert not any('"name": "b"' in a and "in_list" in a for a in added)
